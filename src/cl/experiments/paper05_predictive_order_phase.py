"""Predictive-order x depth x width competence phase diagram."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cl.common.artifacts import atomic_write_json, stable_hash, write_csv
from cl.common.model_adapter import TinyTransformerLM

VALUE_ROLES = tuple(tuple(range(4 + 4*i, 8 + 4*i)) for i in range(8))
ORDER_MARKERS = {1: 48, 2: 49, 3: 50, 4: 51, 6: 52, 8: 53}
LENGTH_MARKERS = {1: 56, 2: 57, 3: 58, 4: 59, 6: 60, 8: 61, 10: 62}
TARGETS = tuple(range(80, 84))
START, QUERY, NEUTRAL = 88, 89, 90


def target_rule(values: list[int]) -> int:
    return sum(values) % 4


def mutual_information(x, y) -> float:
    n=len(x); joint=Counter(zip(x,y)); cx=Counter(x); cy=Counter(y)
    return sum((c/n)*math.log2((c/n)/((cx[a]/n)*(cy[b]/n))) for (a,b),c in joint.items())


def information_audit(config: dict) -> tuple[dict, list[dict]]:
    rows=[]
    for order in config["predictive_orders"]:
        if order <= 4:
            values=list(itertools.product(range(4),repeat=order)); targets=[target_rule(list(v)) for v in values]
            singleton=max((mutual_information([v[i] for v in values],targets) for i in range(order)),default=2.0)
            proper=max((mutual_information([tuple(v[i] for i in subset) for v in values],targets)
                        for size in range(1,order) for subset in itertools.combinations(range(order),size)),default=0.0)
            full=mutual_information(values,targets)
        else:
            singleton=0.0;proper=0.0;full=2.0
        rows.append({"predictive_order":order,"max_singleton_MI_bits":singleton,
                     "max_prohibited_subset_MI_bits":proper,"full_pattern_MI_bits":full,
                     "validation_mode":"exhaustive" if order<=4 else "analytic_uniform_modular_secret_sharing",
                     "passed":proper<1e-9 and abs(full-2)<1e-9 and (order==1 or singleton<1e-9)})
    result={"schema_version":"paper05.predictive_order.mi.v1","passed":all(r["passed"] for r in rows),
            "note":"For p*=6,8, every proper subset is independent of the uniform modular sum because a missing uniform variable one-time-pads the target."}
    result["artifact_hash"]=stable_hash(rows);return result,rows


def _surface_positions(order: int, span: str) -> list[int]:
    gap={"contiguous":1,"moderate":2,"large":3}[span]
    return [i*gap for i in range(order)]


def make_example(config: dict, order: int, length: int, span: str, nuisance: int,
                 index: int, split: str) -> dict:
    seed=config["dataset_seed"]+(0 if split=="train" else 20_000_000)+index*1597+order*101+length*17+nuisance*7+len(span)
    rng=random.Random(seed); values=[rng.randrange(4) for _ in range(order)]; target=target_rule(values)
    positions=_surface_positions(order,span); pattern=[NEUTRAL]*(positions[-1]+1)
    for role,(position,value) in enumerate(zip(positions,values)): pattern[position]=VALUE_ROLES[role][value]
    extras=length-order
    pattern.extend([NEUTRAL]*extras)
    nuisance_tokens=[rng.choice(sum((list(role) for role in VALUE_ROLES),[])) for _ in range(nuisance)]
    body=[START,ORDER_MARKERS[order],LENGTH_MARKERS[length],*pattern]
    padding=config["sequence_length"]-len(nuisance_tokens)-len(body)-1
    if padding < 0: raise ValueError((order,length,span,nuisance,len(body)))
    tokens=[NEUTRAL]*padding+nuisance_tokens+body+[QUERY]
    return {"tokens":tokens,"target":TARGETS[target],"predictive_order":order,"raw_length":length,
            "dependency_span":positions[-1]+1,"span_mode":span,"nuisance_count":nuisance,
            "family_id":f"p{order}:{tuple(values)}:{index%16}"}


def evaluation(config: dict, examples: int | None = None) -> list[dict]:
    count=examples or config["evaluation_examples_per_cell"]
    return [make_example(config,p,n,s,k,i,"test") for p in config["predictive_orders"]
            for n in config["surface_lengths"][str(p)] for s in config["spans"]
            for k in config["nuisance_counts"] for i in range(count)]


def training_batch(config: dict, rng: random.Random, batch_size: int, device: torch.device):
    rows=[]
    for _ in range(batch_size):
        p=rng.choice(config["predictive_orders"]);n=rng.choice(config["surface_lengths"][str(p)])
        rows.append(make_example(config,p,n,rng.choice(config["spans"]),rng.choice(config["nuisance_counts"]),rng.randrange(10_000_000),"train"))
    return (torch.tensor([r["tokens"] for r in rows],device=device),torch.tensor([r["target"] for r in rows],device=device))


def resolve_device(name: str) -> torch.device:
    if name != "auto": return torch.device(name)
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")


def parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train_model(config: dict, depth: int, width: int, budget: int, seed: int, device: torch.device, checkpoint: Path):
    torch.manual_seed(seed+depth*100+width+budget);heads=config["heads_by_width"][str(width)]
    model=TinyTransformerLM(config["vocab_size"],config["sequence_length"],width,depth,heads).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"]);rng=random.Random(seed+depth*1009+width*31+budget)
    steps=config["base_steps"]*budget;loss=[]
    for step in range(steps):
        x,y=training_batch(config,rng,config["batch_size"],device);optimizer.zero_grad(set_to_none=True)
        logits,_=model(x);value=F.cross_entropy(logits[:,-1],y);value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        if step in (0,steps-1) or (step+1)%100==0:loss.append({"step":step+1,"loss":float(value.detach().cpu())})
    checkpoint.parent.mkdir(parents=True,exist_ok=True);torch.save(model.to("cpu").state_dict(),checkpoint)
    return model.eval(),loss


@torch.no_grad()
def evaluate_model(model: TinyTransformerLM, rows: list[dict], meta: dict, batch_size: int = 128) -> tuple[list[dict],list[dict]]:
    device=next(model.parameters()).device;final_rows=[];decision_rows=[]
    for start in range(0,len(rows),batch_size):
        batch=rows[start:start+batch_size];x=torch.tensor([r["tokens"] for r in batch],device=device);y=torch.tensor([r["target"] for r in batch],device=device)
        logits,trace=model(x,capture=True);states=[trace.layers[0].pre_sa]+[layer.post_block for layer in trace.layers]
        correct=[];margins=[]
        for state in states:
            z=model.diagnostic_logits(state[:,-1]);chosen=z.gather(1,y[:,None]).squeeze(1);other=z.clone();other.scatter_(1,y[:,None],float("-inf"));margins.append((chosen-other.max(1).values).cpu());correct.append((z.argmax(1)==y).cpu())
        correct=torch.stack(correct,1);margins=torch.stack(margins,1)
        for i,r in enumerate(batch):
            c=correct[i].tolist();m=margins[i].tolist();first=next((j for j,v in enumerate(c) if v),None);stable=next((j for j in range(len(c)) if all(c[j:])),None)
            common={**meta,**{k:r[k] for k in ("predictive_order","raw_length","dependency_span","span_mode","nuisance_count","family_id")}}
            final_rows.append({**common,"top1_correct":int(c[-1]),"final_margin":m[-1]})
            decision_rows.append({**common,"first_top1_layer":first if first is not None else "","stable_top1_layer":stable if stable is not None else "",
                                  "settling_delay":stable-first if stable is not None and first is not None else "",
                                  "top1_reversals":sum(a!=b for a,b in zip(c,c[1:])),"margin_trajectory":";".join(f"{v:.6g}" for v in m)})
    return final_rows,decision_rows


def architecture_grid(config: dict, smoke: bool) -> list[tuple[int,int,int,int]]:
    if smoke:return [(2,32,1,config["model_seeds"][0])]
    grid=[]
    for seed,depth,width in itertools.product(config["model_seeds"],config["depths"],config["widths"]):
        grid.append((depth,width,1,seed))
        if depth in config["extended_budget_depths"] and width in config["extended_budget_widths"]:
            grid.extend((depth,width,b,seed) for b in config["training_budgets"] if b>1)
    return grid


def main(args) -> None:
    config=json.loads(Path(args.config).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if args.base_steps:
        config={**config,"base_steps":args.base_steps}
    if args.smoke:
        config={**config,"base_steps":4,"batch_size":8}
    audit,audit_rows=information_audit(config);atomic_write_json(out/"phase_mi_validation.json",audit);write_csv(out/"phase_mi_validation.csv",audit_rows)
    if not audit["passed"]:raise RuntimeError(audit)
    device=resolve_device(args.device or config["device"]);eval_rows=evaluation(config,args.eval_examples)
    all_results=[];all_decisions=[];models=[];losses=[]
    grid=architecture_grid(config,args.smoke)
    if args.max_models:grid=grid[:args.max_models]
    for model_index,(depth,width,budget,seed) in enumerate(grid,1):
        print(f"[{model_index}/{len(grid)}] depth={depth} width={width} budget={budget} seed={seed}",flush=True)
        checkpoint=out/f"checkpoints/d{depth}_w{width}_b{budget}_s{seed}.pt"
        model,loss=train_model(config,depth,width,budget,seed,device,checkpoint)
        model=model.to(device);meta={"model_type":"transformer","model_depth":depth,"model_width":width,"training_budget":budget,
                                    "training_steps":config["base_steps"]*budget,"model_seed":seed,"parameter_count":parameter_count(model)}
        result,decision=evaluate_model(model,eval_rows,meta);all_results.extend(result);all_decisions.extend(decision)
        models.append(meta);losses.extend({**meta,**row} for row in loss)
        write_csv(out/"phase_grid_results.csv",all_results);write_csv(out/"phase_internal_decision_depth.csv",all_decisions)
    write_csv(out/"phase_parameter_counts.csv",models);write_csv(out/"phase_training_loss.csv",losses)
    manifest={"schema_version":"paper05.predictive_order_phase.run.v1","device":str(device),"smoke":args.smoke,"models":len(models),
              "evaluation_examples":len(eval_rows),"config":config,"artifact_hash":stable_hash({"models":models,"results":all_results})}
    atomic_write_json(out/"phase_manifest.json",manifest);print(json.dumps(manifest,indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/paper05/predictive_order_phase.json");p.add_argument("--output",default="docs/papers/paper0_5/results/predictive_order_phase");p.add_argument("--device");p.add_argument("--smoke",action="store_true");p.add_argument("--eval-examples",type=int);p.add_argument("--base-steps",type=int);p.add_argument("--max-models",type=int);main(p.parse_args())
