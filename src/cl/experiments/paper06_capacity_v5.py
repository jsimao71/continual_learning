"""Stage-A resource diagnostics for the Paper 0.6 predicate capacity frontier."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from cl.common.artifacts import atomic_write_json,write_csv
from cl.experiments.paper05_predictive_order_phase import resolve_device
from cl.experiments.paper06_predicate_phase import aggregate,evaluate,params,train
from cl.semantic.predicates import evaluation_matrix,validate

def tier(row):
    D,d,b,N=row.total_depth,row.required_path,row.branching,row.distractors
    marker=int(row.example_id.rsplit(":i",1)[1])
    if marker>=20_000:
        return "easy" if N==4 else "medium" if N==16 else "hard" if N==64 else None
    if marker>=10_000:
        return "easy" if b==2 else "medium" if b==4 else "hard" if b==8 else None
    if D<=3 and d<=2:return "easy"
    if D==6 and d<=4:return "medium"
    if D in (12,16) and d<=12:return "hard"
    return None
def main(args):
    design=json.loads(Path(args.config).read_text());base=json.loads(Path(design["base_config"]).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    assert validate(base)["passed"]
    architectures=design["architectures"][:1] if args.smoke else design["architectures"]
    if args.architectures:
        selected=set(args.architectures.split(","));architectures=[a for a in architectures if a["name"] in selected]
    seeds=design["model_seeds"][:1] if args.smoke else design["model_seeds"]
    if args.seeds:
        selected_seeds={int(x) for x in args.seeds.split(",")};seeds=[s for s in seeds if s in selected_seeds]
    raw=[];curves=[];meta=[];device=resolve_device(args.device or base["device"])
    for arch in architectures:
      for seed in seeds:
        cfg={**base,"width":arch["width"],"heads":arch["heads"]};steps=4 if args.smoke else base["training_steps"]*args.budget
        started=time.time();model,loss=train(cfg,arch["layers"],seed,steps,device);elapsed=time.time()-started
        rows=[r for r in evaluation_matrix(cfg,seed,1 if args.smoke else design["stage_a_eval_examples"]) if tier(r)]
        measured=evaluate(model,rows,arch["layers"],seed)
        for r in measured:r.update({"architecture":arch["name"],"width":arch["width"],"heads":arch["heads"],"budget_multiplier":args.budget,"diagnostic_tier":tier(next(x for x in rows if x.example_id==r["example_id"] and x.tree_seed==int(r["tree_seed"])))})
        raw+=measured
        for r in loss:r.update({"architecture":arch["name"],"width":arch["width"],"heads":arch["heads"],"budget_multiplier":args.budget});curves+=loss
        meta.append({"architecture":arch["name"],"layers":arch["layers"],"width":arch["width"],"heads":arch["heads"],"head_dimension":arch["width"]//arch["heads"],"ff_dimension":2*arch["width"],"model_seed":seed,"budget_multiplier":args.budget,"training_steps":steps,"parameters":params(model),"elapsed_seconds":elapsed,"final_loss":loss[-1]["loss"]})
        torch.save(model.to("cpu").state_dict(),out/f"{arch['name']}_seed{seed}_T{args.budget}x.pt")
    groups={}
    for r in raw:groups.setdefault((r["architecture"],r["model_seed"],r["width"],r["heads"],r["model_depth"],r["budget_multiplier"],r["diagnostic_tier"],r["predicate"]),[]).append(r)
    cells=[]
    for key,v in groups.items():
        fields=("architecture","model_seed","width","heads","layers","budget_multiplier","diagnostic_tier","predicate");cells.append({**dict(zip(fields,key)),"accuracy":np.mean([int(x["top1_correct"]) for x in v]),"mean_margin":np.mean([float(x["target_margin"]) for x in v]),"n":len(v)})
    write_csv(out/"capacity_diagnostic_grid.csv",cells);write_csv(out/"capacity_learning_curves.csv",curves);write_csv(out/"capacity_architecture_metadata.csv",meta);write_csv(out/"capacity_diagnostic_raw.csv",raw)
    atomic_write_json(out/"capacity_stage_a_manifest.json",{"architectures":len(architectures),"seeds":len(seeds),"budget_multiplier":args.budget,"cells":len(cells),"full_grid_expanded":False})
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/paper06/capacity_v5.json");p.add_argument("--output",default="docs/papers/paper0_6/results/v5/stage_a_1x");p.add_argument("--device");p.add_argument("--budget",type=int,default=1);p.add_argument("--architectures");p.add_argument("--seeds",help="comma-separated subset of configured model seeds");p.add_argument("--smoke",action="store_true");main(p.parse_args())
