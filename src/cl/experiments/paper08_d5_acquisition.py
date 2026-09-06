"""Learned D5 unary modular-rule acquisition with exact validity gates."""
from __future__ import annotations
import argparse,csv,hashlib,json,random
from collections import Counter
from pathlib import Path
import numpy as np,torch
from cl.common.artifacts import atomic_write_json,write_csv
from cl.common.model_adapter import TinyTransformerLM
from cl.experiments.paper05_predictive_order_phase import resolve_device
from cl.semantic.paper08_d5 import D5Episode,episode_signature,generate_split,split_audit

PAD,BOS,PAIR,ARROW,QUERY,SEP=range(6)

def residue_token(k,x,moduli):return 6+moduli.index(k)*max(moduli)+x
def serialize(row:D5Episode,moduli,max_demos,condition="correct"):
    demos=list(row.demonstrations);rng=random.Random(int(row.episode_id,16)+sum(map(ord,condition)))
    if condition=="shuffled":
        ys=[y for _,y in demos]
        if len(ys)>1:rng.shuffle(ys);ys=ys[1:]+ys[:1] if ys==[y for _,y in demos] else ys
        else:ys=[(ys[0]+1)%row.modulus]
        demos=[(demos[i][0],ys[i]) for i in range(len(demos))]
    elif condition=="wrong_rule":demos=[(x,(y+1)%row.modulus) for x,y in demos]
    elif condition=="none":demos=[]
    elif condition!="correct":raise ValueError(condition)
    tokens=[BOS]+[PAD]*(4*(max_demos-len(demos)))
    for x,y in demos:tokens += [PAIR,residue_token(row.modulus,x,moduli),ARROW,residue_token(row.modulus,y,moduli)]
    tokens += [QUERY,residue_token(row.modulus,row.query,moduli)]
    return tuple(tokens)

def build_data(cfg,seed,smoke=False):
    train_n=1 if smoke else cfg["train_episodes_per_hypothesis"];test_n=1 if smoke else cfg["test_episodes_per_hypothesis"]
    parts={};used=set()
    for s in ("train","seen_parameter","unseen_parameter","unseen_family"):
        repeats=train_n if s=="train" else (1 if smoke else cfg["unseen_family_test_episodes"] if s=="unseen_family" else test_n)
        parts[s]=generate_split(cfg["moduli"],s,repeats,cfg["dataset_seed"],cfg["parameter_test_fraction"],used)
        used.update(episode_signature(r) for r in parts[s])
    return parts,split_audit(parts["train"],parts["seen_parameter"],parts["unseen_parameter"],parts["unseen_family"])

def make_model(cfg,cell,device):
    net=TinyTransformerLM(cfg["vocab_size"],cfg["max_length"],cell["width"],cell["layers"],cell["heads"],cfg["mlp_ratio"]).to(device)
    if cell["variant"]=="sa_only":
        for block in net.blocks:
            for p in block.ff.parameters():p.data.zero_();p.requires_grad=False
            for p in block.norm_ff.parameters():p.requires_grad=False
    return net

def batch(rows,cfg,rng,device):
    chosen=[rows[rng.randrange(len(rows))] for _ in range(cfg["batch_size"])]
    x=torch.tensor([serialize(r,cfg["moduli"],cfg["max_demonstrations"]) for r in chosen],device=device)
    y=torch.tensor([residue_token(r.modulus,r.target,cfg["moduli"]) for r in chosen],device=device)
    return x,y

def train(rows,cfg,cell,seed,device,updates):
    torch.manual_seed(seed);rng=random.Random(seed+805);net=make_model(cfg,cell,device)
    opt=torch.optim.AdamW((p for p in net.parameters() if p.requires_grad),lr=cfg["learning_rate"]);losses=[]
    for step in range(updates):
        x,y=batch(rows,cfg,rng,device);opt.zero_grad(set_to_none=True);logits,_=net(x)
        loss=torch.nn.functional.cross_entropy(logits[:,-1],y);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
        if step==0 or (step+1)%cfg["log_every"]==0 or step+1==updates:losses.append({"step":step+1,"loss":float(loss.detach())})
    return net.eval(),losses

@torch.no_grad()
def evaluate(net,parts,cfg,cell,seed):
    out=[]
    for split in ("seen_parameter","unseen_parameter","unseen_family"):
      for row in parts[split]:
       for condition in ("correct","none","shuffled","wrong_rule"):
        x=torch.tensor([serialize(row,cfg["moduli"],cfg["max_demonstrations"],condition)],device=next(net.parameters()).device)
        logits=net(x)[0][0,-1];target=residue_token(row.modulus,row.target,cfg["moduli"]);other=torch.cat((logits[:target],logits[target+1:]))
        out.append({"split":split,"family":row.family,"modulus":row.modulus,"parameters":";".join(map(str,row.parameters)),
          "episode_id":row.episode_id,"condition":condition,"target":row.target,"top1":int(logits.argmax()==target),
          "target_rank":int((logits>logits[target]).sum())+1,"target_margin":float(logits[target]-other.max()),
          "layers":cell["layers"],"width":cell["width"],"heads":cell["heads"],"variant":cell["variant"],"seed":seed})
    return out

def aggregate(raw,train_rows,cfg,cell,seed):
    majority={k:Counter(r.target for r in train_rows if r.modulus==k).most_common(1)[0][0] for k in cfg["moduli"]};out=[]
    for split in ("seen_parameter","unseen_parameter","unseen_family"):
      selected=[r for r in raw if r["split"]==split];by={c:[r for r in selected if r["condition"]==c] for c in ("correct","none","shuffled","wrong_rule")}
      accuracy={c:float(np.mean([r["top1"] for r in rows])) for c,rows in by.items()};correct=by["correct"]
      majority_accuracy=float(np.mean([int(r["target"]==majority[int(r["modulus"])]) for r in correct]))
      max_control=max(accuracy["none"],accuracy["shuffled"],accuracy["wrong_rule"])
      out.append({**cell,"seed":seed,"split":split,"examples":len(correct),"correct_accuracy":accuracy["correct"],
        "no_context_accuracy":accuracy["none"],"shuffled_accuracy":accuracy["shuffled"],"wrong_rule_accuracy":accuracy["wrong_rule"],
        "selectivity":accuracy["correct"]-max_control,"uniform_baseline":float(np.mean([1/int(r["modulus"]) for r in correct])),
        "majority_baseline":majority_accuracy,"mean_target_margin":float(np.mean([r["target_margin"] for r in correct])),
        "competent":int(accuracy["correct"]>=cfg["competence_accuracy"] and accuracy["correct"]-max_control>=cfg["minimum_selectivity"])})
    return out

def read_csv(path):
    if not path.exists():return []
    with path.open(newline="",encoding="utf-8") as handle:return list(csv.DictReader(handle))

def main(args):
    cfg=json.loads(Path(args.config).read_text());device=resolve_device(args.device or cfg["device"]);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    cells=cfg["first_pass_cells"] if not args.cell else [dict(zip(("layers","width","heads","variant"),[int(x) if i<3 else x for i,x in enumerate(args.cell.split(","))]))]
    seeds=cfg["model_seeds"][:1] if args.smoke else cfg["model_seeds"];updates=cfg["smoke_updates"] if args.smoke else cfg["updates"]
    parts,audit=build_data(cfg,cfg["dataset_seed"],args.smoke)
    raw=read_csv(out/"raw_results.csv") if args.resume else [];summaries=read_csv(out/"cell_summary.csv") if args.resume else [];losses=read_csv(out/"training_loss.csv") if args.resume else []
    done={(int(r["layers"]),int(r["width"]),int(r["heads"]),r["variant"],int(r["seed"])) for r in summaries}
    planned=cells[:1] if args.smoke else cells
    for cell in planned:
      for seed in seeds:
        key=(cell["layers"],cell["width"],cell["heads"],cell["variant"],seed)
        if key in done:continue
        net,loss=train(parts["train"],cfg,cell,seed,device,updates);raw.extend(evaluate(net,parts,cfg,cell,seed));summaries.extend(aggregate(raw[-sum(len(parts[s]) for s in ("seen_parameter","unseen_parameter","unseen_family"))*4:],parts["train"],cfg,cell,seed));losses.extend({**cell,"seed":seed,**r} for r in loss)
        if not args.smoke:torch.save(net.state_dict(),out/f"D5_L{cell['layers']}_W{cell['width']}_H{cell['heads']}_{cell['variant']}_seed{seed}.pt")
        write_csv(out/"raw_results.csv",raw);write_csv(out/"cell_summary.csv",summaries);write_csv(out/"training_loss.csv",losses)
    write_csv(out/"dataset_audit.csv",[{**audit,"split":name,"episodes":len(rows),"mean_identifying_examples":float(np.mean([r.minimum_examples_needed for r in rows]))} for name,rows in parts.items()])
    write_csv(out/"raw_results.csv",raw);write_csv(out/"cell_summary.csv",summaries);write_csv(out/"training_loss.csv",losses)
    atomic_write_json(out/"manifest.json",{"schema_version":"paper08.d5_acquisition.v1","learned_experiment":True,"smoke":args.smoke,"device":str(device),
      "completed_cells":len({(r["layers"],r["width"],r["heads"],r["variant"],r["seed"]) for r in summaries}),"planned_cells":len(planned)*len(seeds),"updates":updates,"model_seeds":seeds,"moduli":cfg["moduli"],
      "train_families":["successor","add_n","multiply_a","affine"],"heldout_family":"square","controls":["none","shuffled","wrong_rule"],
      "validity_gate":audit,"parameter_split_seed":cfg["dataset_seed"],"generic_family_free_serialization":True,"checkpoints_ignored":True})

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--config",default="configs/paper08/d5_acquisition_v1.json");p.add_argument("--output",default="docs/papers/paper0_8/results/d5_acquisition_v1");p.add_argument("--device");p.add_argument("--smoke",action="store_true");p.add_argument("--resume",action="store_true");p.add_argument("--cell",help="layers,width,heads,variant");main(p.parse_args())
