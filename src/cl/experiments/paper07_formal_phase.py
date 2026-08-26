"""Materialize Paper 0.7 symbolic splits and a deterministic training plan."""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
from cl.common.artifacts import atomic_write_json, stable_hash, write_csv, write_jsonl
from cl.semantic.formal_datasets import proposition_ladder, unification_ladder, validate_dataset

def build(config: dict, smoke: bool=False):
    counts=config["smoke_examples_per_family"] if smoke else config["examples_per_family"]
    all_rows=[]
    for split,seed in config["split_seeds"].items():
        count=counts[split]
        all_rows += proposition_ladder(split,count,seed,config["proposition_chain_lengths"])
        all_rows += unification_ladder(split,count,seed+1000,config["functor_depths"])
    grid=config["smoke_model_grid"] if smoke else config["model_grid"]
    plan=[]
    for family,depth,width,heads,budget,seed in itertools.product(
        ("proposition","unification"),grid["layers"],grid["widths"],grid["heads"],grid["budgets"],config["model_seeds"]):
        if width%heads: continue
        plan.append({"family":family,"layers":depth,"width":width,"heads":heads,
            "training_budget":budget,"seed":seed,"status":"planned"})
    return all_rows,plan

def main(args):
    config=json.loads(Path(args.config).read_text()); rows,plan=build(config,args.smoke)
    validation=validate_dataset(rows)
    if not validation["valid"]: raise RuntimeError(validation)
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    for split in ("train","validation","test"):
        write_jsonl(output/f"{split}.jsonl",(r.as_dict() for r in rows if r.split==split))
    write_csv(output/"phase_plan.csv",plan)
    atomic_write_json(output/"manifest.json",{"schema_version":config["schema_version"],"smoke":args.smoke,
        "validation":validation,"dataset_hash":stable_hash([r.as_dict() for r in rows]),"planned_runs":len(plan),
        "model_results":False,"note":"CPU dataset/plan artifact; no trained-model outcomes."})
    print(json.dumps({"validation":validation,"planned_runs":len(plan)},indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/paper07/formal_v1.json")
    parser.add_argument("--output",default="docs/papers/paper0_7/results/formal_v1"); parser.add_argument("--smoke",action="store_true")
    main(parser.parse_args())
