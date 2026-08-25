"""Dataset/model phase boundary for the inception of ICL."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np,torch
from cl.common.artifacts import atomic_write_json,write_csv
from cl.common.model_adapter import TinyTransformerLM
from cl.experiments.paper05_predictive_order_phase import resolve_device
from cl.semantic.icl import controlled,episodic_examples,validation

CONDITIONS=("none","correct","shuffled","wrong_chain","irrelevant","reversed")

def dataset(stage,cfg,seed,train):
    families=cfg["train_families"] if train else cfg["test_families"]
    episodes=cfg["episodes_per_family"] if train else cfg["test_episodes_per_family"]
    return episodic_examples(stage,families,episodes,cfg["chain_length"],seed,
        exclude_critical=train,critical_only=not train,
        family_offset=0)

def train_model(stage,layers,width,heads,seed,cfg,device,updates):
    torch.manual_seed(seed);rng=random.Random(seed+800);rows=dataset(stage,cfg,seed,True)
    model=TinyTransformerLM(cfg["vocab_size"],cfg["max_length"],width,layers,heads).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"]);losses=[]
    for step in range(updates):
        batch=[rows[rng.randrange(len(rows))] for _ in range(cfg["batch_size"])]
        x=torch.tensor([r.tokens for r in batch],device=device);y=torch.tensor([r.target for r in batch],device=device)
        opt.zero_grad(set_to_none=True);logits,_=model(x);loss=torch.nn.functional.cross_entropy(logits[:,-1],y);loss.backward();opt.step()
        if step%50==0 or step==updates-1:losses.append({"stage":stage,"layers":layers,"width":width,"heads":heads,"model_seed":seed,"step":step+1,"loss":float(loss)})
    return model.eval(),losses,rows

def metrics(logits,target):
    values=logits.detach();rank=int((values>values[target]).sum())+1;margin=float(values[target]-torch.cat((values[:target],values[target+1:])).max());prob=float(values.softmax(-1)[target]);return rank,margin,prob,int(values.argmax()==target)

@torch.no_grad()
def evaluate(model,stage,cfg,seed,layers,width,heads):
    rows=dataset(stage,cfg,seed+1000,False);out=[]
    for i,row in enumerate(rows):
        base={}
        for condition in CONDITIONS:
            r=row if condition=="correct" else controlled(row,condition,seed+i)
            x=torch.tensor([r.tokens],device=next(model.parameters()).device);logits,_=model(x)
            rank,margin,prob,top1=metrics(logits[0,-1],row.target);base[condition]=(rank,margin,prob,top1)
            out.append({"stage":stage,"layers":layers,"width":width,"heads":heads,"model_seed":seed,"example_id":i,"family":row.family,"condition":condition,"target":row.target,"rank":rank,"margin":margin,"probability":prob,"top1":top1})
        out[-len(CONDITIONS)+1]["icl_gain"]=base["correct"][1]-base["none"][1]
    return out

def aggregate(rows,cfg):
    cells=[]
    keys=sorted({(r["stage"],r["layers"],r["width"],r["heads"],r["model_seed"]) for r in rows})
    for key in keys:
        q=[r for r in rows if tuple(r[k] for k in ("stage","layers","width","heads","model_seed"))==key];by={c:[r for r in q if r["condition"]==c] for c in CONDITIONS}
        correct=by["correct"];none=by["none"]
        acc=np.mean([r["top1"] for r in correct]);none_rank=np.mean([r["rank"] for r in none]);gain=np.mean([r["margin"] for r in correct])-np.mean([r["margin"] for r in none]);control=max(np.mean([r["top1"] for r in by[c]]) for c in ("shuffled","wrong_chain","irrelevant"))
        cells.append(dict(zip(("stage","layers","width","heads","model_seed"),key),correct_accuracy=acc,context_free_accuracy=np.mean([r["top1"] for r in none]),context_free_mean_rank=none_rank,icl_margin_gain=gain,max_matched_control_accuracy=control,selectivity=acc-control,competent=int(acc>=cfg["competence_accuracy"] and none_rank>=cfg["minimum_context_free_rank"] and control<acc-.2)))
    return cells

def main(args):
    cfg=json.loads(Path(args.config).read_text());device=resolve_device(args.device or cfg["device"]);out=Path(args.output);(out/"generator_validation").mkdir(parents=True,exist_ok=True);(out/"phase").mkdir(exist_ok=True)
    updates=20 if args.smoke else cfg["updates"];seeds=cfg["model_seeds"][:1] if args.smoke else cfg["model_seeds"]
    settings=[];dm=cfg["dataset_model"]
    for stage in cfg["dataset_stages"]:settings += [(stage,dm["layers"],dm["width"],dm["heads"],s) for s in seeds]
    # Coarse D4 architecture surface; all three seeds are retained for the boundary.
    for l in cfg["layers"]:
        for w in cfg["widths"]:
            for h in cfg["heads"]:
                if w%h==0:settings += [("D4",l,w,h,s) for s in seeds]
    settings=list(dict.fromkeys(settings));raw=[];losses=[];valid=[]
    if args.only_stage:settings=[s for s in settings if s[0]==args.only_stage]
    if args.limit:settings=settings[:args.limit]
    for stage,l,w,h,seed in settings:
        model,lr,train=train_model(stage,l,w,h,seed,cfg,device,updates);test=dataset(stage,cfg,seed+1000,False);valid.append({"stage":stage,"layers":l,"width":w,"heads":h,"model_seed":seed,**validation(train,test)});raw += evaluate(model,stage,cfg,seed,l,w,h);losses += lr
        torch.save(model.state_dict(),out/"phase"/f"{stage}_L{l}_W{w}_H{h}_seed{seed}.pt")
    cells=aggregate(raw,cfg);write_csv(out/"generator_validation/icl_generator_validation.csv",valid);write_csv(out/"phase/icl_rank_promotion.csv",raw);write_csv(out/"phase/icl_phase_grid.csv",cells);write_csv(out/"phase/icl_training_loss.csv",losses)
    atomic_write_json(out/"phase/phase_manifest.json",{"settings":len(settings),"updates":updates,"device":str(device),"competent_cells":sum(r["competent"] for r in cells)})

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/paper08/icl_v1.json");p.add_argument("--output",default="docs/papers/paper0_8/results");p.add_argument("--device");p.add_argument("--smoke",action="store_true");p.add_argument("--limit",type=int);p.add_argument("--only-stage");main(p.parse_args())
