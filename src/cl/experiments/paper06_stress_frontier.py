"""Resumable learned-model frontier stage for the Paper 0.6 v6 stress design."""
from __future__ import annotations
import argparse,csv,json,random,time
from collections import defaultdict
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from cl.common.artifacts import atomic_write_json,stable_hash,write_csv
from cl.common.model_adapter import TinyTransformerLM
from cl.experiments.paper05_predictive_order_phase import resolve_device
from cl.semantic.predicate_stress import axis,stress_design,training_batch,validate_stress

def read(path:Path):
    if not path.exists():return []
    with path.open(newline="",encoding="utf-8") as handle:return list(csv.DictReader(handle))

def tensor(rows,device):
    length=max(map(lambda r:len(r.tokens),rows))
    return torch.tensor([[0]*(length-len(r.tokens))+list(r.tokens) for r in rows],device=device)

def model_for(cfg,arch,device):
    return TinyTransformerLM(cfg["vocab_size"],cfg["max_length"],arch["width"],arch["layers"],
        arch["heads"],position_encoding=cfg.get("position_encoding","sinusoidal")).to(device)

def save_state(path,payload):
    temporary=path.with_suffix(".tmp");torch.save(payload,temporary);temporary.replace(path)

def train(cfg,arch,seed,steps,predicates,device,directory,resume,smoke):
    torch.manual_seed(seed+arch["layers"]*1009);rng=random.Random(seed+arch["layers"]*701)
    model=model_for(cfg,arch,device);optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"])
    state_path=directory/"training_state.pt";losses=[];first=0;started=time.time()
    if resume and state_path.exists():
        state=torch.load(state_path,map_location=device,weights_only=False);model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"]);rng.setstate(state["rng"]);losses=state["losses"];first=state["step"]
    for step in range(first,steps):
        rows=training_batch(cfg,rng,seed,predicates,8 if smoke else cfg["batch_size"])
        x=tensor(rows,device);y=torch.tensor([r.target for r in rows],device=device)
        optimizer.zero_grad(set_to_none=True);logits,_=model(x);loss=F.cross_entropy(logits[:,-1],y)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        if step==first or (step+1)%cfg.get("log_every",100)==0 or step+1==steps:
            losses.append({"step":step+1,"loss":float(loss.detach().cpu()),"elapsed_seconds":time.time()-started})
            print(f"{arch['name']} seed={seed} step={step+1}/{steps} loss={losses[-1]['loss']:.4f}",flush=True)
        if (step+1)%cfg.get("save_every",100)==0 or step+1==steps:
            save_state(state_path,{"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                "rng":rng.getstate(),"losses":losses,"step":step+1})
    torch.save(model.to("cpu").state_dict(),directory/"checkpoint.pt")
    return model.to(device).eval(),losses

@torch.no_grad()
def evaluate(model,rows,arch,seed,batch_size=128):
    device=next(model.parameters()).device;result=[];buckets=defaultdict(list)
    for row in rows:buckets[len(row.tokens)].append(row)
    for length,bucket in sorted(buckets.items()):
      for start in range(0,len(bucket),batch_size):
        batch=bucket[start:start+batch_size];x=torch.tensor([r.tokens for r in batch],device=device)
        y=torch.tensor([r.target for r in batch],device=device);z,_=model(x);final=z[:,-1]
        chosen=final.gather(1,y[:,None]).squeeze(1);other=final.clone();other.scatter_(1,y[:,None],-torch.inf)
        for i,row in enumerate(batch):result.append({"architecture":arch["name"],"layers":arch["layers"],
          "width":arch["width"],"heads":arch["heads"],"model_seed":seed,"tree_seed":row.tree_seed,
          "example_id":row.example_id,"evaluation_axis":axis(row),"predicate":row.predicate,
          "total_depth":row.total_depth,"required_path":row.required_path,"branching":row.branching,
          "distractors":row.distractors,"template_id":row.template_id,"position_mode":row.position_mode,
          "sequence_length":length,"top1_correct":int(final[i].argmax()==y[i]),
          "target_margin":float(chosen[i]-other[i].max()),"target_rank":int(1+(final[i]>chosen[i]).sum())})
    return result

def aggregate(raw):
    keys=("architecture","layers","width","heads","model_seed","evaluation_axis","predicate",
          "total_depth","required_path","branching","distractors")
    groups=defaultdict(list)
    for row in raw:groups[tuple(row[k] for k in keys)].append(row)
    return [{**dict(zip(keys,key)),"accuracy":float(np.mean([int(r["top1_correct"]) for r in rows])),
        "mean_margin":float(np.mean([float(r["target_margin"]) for r in rows])),"n":len(rows)}
        for key,rows in groups.items()]

def main(args):
    cfg=json.loads(Path(args.config).read_text());audit=validate_stress(cfg)
    if not audit["passed"]:raise RuntimeError(audit)
    output=Path(args.output);(output/"models").mkdir(parents=True,exist_ok=True);atomic_write_json(output/"generator_validation.json",audit)
    architectures=cfg["frontier_architectures"][:1] if args.smoke else cfg["frontier_architectures"]
    if args.architectures:
        selected=set(args.architectures.split(","));architectures=[a for a in architectures if a["name"] in selected]
    seeds=cfg["model_seeds"][:1] if args.smoke else cfg["model_seeds"]
    if args.seeds:
        selected={int(x) for x in args.seeds.split(",")};seeds=[s for s in seeds if s in selected]
    predicates=args.predicates.split(",") if args.predicates else cfg["frontier_predicates"]
    steps=4 if args.smoke else (args.steps or cfg["frontier_training_steps"]);examples=2 if args.smoke else (args.eval_examples or cfg["frontier_eval_examples"])
    device=resolve_device(args.device);specs=[(a,s) for a in architectures for s in seeds]
    for index,(arch,seed) in enumerate(specs,1):
        directory=output/"models"/f"{arch['name']}_seed{seed}";directory.mkdir(parents=True,exist_ok=True)
        if args.resume and (directory/"complete.json").exists():print(f"[{index}/{len(specs)}] skip {arch['name']} seed {seed}",flush=True);continue
        print(f"[{index}/{len(specs)}] train {arch['name']} seed {seed} on {device}",flush=True)
        model,losses=train(cfg,arch,seed,steps,predicates,device,directory,args.resume,args.smoke)
        rows=[r for r in stress_design(cfg,seed,examples) if r.predicate in predicates]
        raw=evaluate(model,rows,arch,seed);write_csv(directory/"raw.csv",raw);write_csv(directory/"loss.csv",losses)
        atomic_write_json(directory/"complete.json",{"architecture":arch,"model_seed":seed,"steps":steps,
            "predicates":predicates,"raw_rows":len(raw),"artifact_hash":stable_hash(raw)})
    raw=[];loss=[]
    for complete in sorted((output/"models").glob("*/complete.json")):
        meta=json.loads(complete.read_text());raw+=read(complete.parent/"raw.csv")
        loss += [{"architecture":meta["architecture"]["name"],"model_seed":meta["model_seed"],**r} for r in read(complete.parent/"loss.csv")]
    write_csv(output/"stress_frontier_raw.csv",raw);write_csv(output/"stress_frontier_cells.csv",aggregate(raw));write_csv(output/"stress_frontier_training_loss.csv",loss)
    atomic_write_json(output/"stress_frontier_manifest.json",{"schema_version":"paper06.stress_v6.frontier.v1",
        "device":str(device),"planned_models":len(specs),"completed_models":len(list((output/"models").glob("*/complete.json"))),
        "predicates":predicates,"training_steps":steps,"evaluation_examples":examples,"smoke":args.smoke})

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/paper06/stress_v6.json")
    p.add_argument("--output",default="docs/papers/paper0_6/results/v6/frontier_isancestor")
    p.add_argument("--device",default="auto");p.add_argument("--architectures");p.add_argument("--seeds")
    p.add_argument("--predicates");p.add_argument("--steps",type=int);p.add_argument("--eval-examples",type=int)
    p.add_argument("--resume",action="store_true");p.add_argument("--smoke",action="store_true");main(p.parse_args())
