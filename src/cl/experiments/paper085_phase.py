"""CPU materialization, oracle validation, and learned-run plan for Paper 0.85."""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
from cl.common.artifacts import atomic_write_json, stable_hash, write_csv, write_jsonl
from cl.semantic.autoregressive_proofs import evaluate_trace, generate_split, validate_example

def run(config:dict,smoke:bool=False):
    distractors=config["smoke_distractors"] if smoke else config["distractors"]
    per_cell=config["smoke_examples_per_cell"] if smoke else config["examples_per_cell"]
    rows=[]
    for split,seed in config["generator_seeds"].items():
        if split=="train":
            depths=[d for d in config["train_depths"] if not smoke or d in config["smoke_depths"]]
        else:
            depths=config["smoke_depths"] if smoke else config["test_depths"]
        rows+=generate_split(split,seed,per_cell,depths,distractors)
    validations=[{"example_id":r.example_id,**validate_example(r)} for r in rows]
    oracle=[]
    for r in rows:
        for output in config["output_conditions"]:
            oracle.append({"example_id":r.example_id,"split":r.split,"stage":r.stage,"depth":r.proof_depth,
                "distractors":r.distractors,"branching":r.branching,"condition":output,
                **evaluate_trace(r,r.target(output),output),"result_type":"reference_oracle"})
    grid=config["smoke_grid"] if smoke else config["architecture_grid"]
    plan=[]
    for values in itertools.product(grid["layers"],grid["widths"],grid["heads"],grid["budgets"],
                                    config["model_seeds"],config["output_conditions"]):
        layers,width,heads,budget,seed,output=values
        if width%heads: continue
        plan.append({"layers":layers,"width":width,"heads":heads,"budget":budget,"seed":seed,
            "condition":output,"status":"planned_accelerator","model_result":False})
    return rows,validations,oracle,plan

def main(args):
    config=json.loads(Path(args.config).read_text()); rows,validations,oracle,plan=run(config,args.smoke)
    if not all(r["valid"] for r in validations): raise RuntimeError("invalid symbolic dataset")
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    for split in config["generator_seeds"]: write_jsonl(output/f"{split}.jsonl",(r.as_dict() for r in rows if r.split==split))
    write_csv(output/"dataset_validation.csv",validations); write_csv(output/"reference_oracle.csv",oracle)
    write_csv(output/"accelerator_phase_plan.csv",plan)
    atomic_write_json(output/"manifest.json",{"schema_version":config["schema_version"],"smoke":args.smoke,
        "examples":len(rows),"oracle_rows":len(oracle),"planned_accelerator_runs":len(plan),
        "train_depths":config["train_depths"],"evaluation_depths":config["test_depths"],
        "all_symbolic_rows_valid":True,"all_reference_traces_exact":all(r["exact"] for r in oracle),
        "learned_model_results":False,"dataset_hash":stable_hash([r.as_dict() for r in rows])})
    print(json.dumps({"examples":len(rows),"oracle_rows":len(oracle),"planned_runs":len(plan)},indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/paper085/phase_v1.json")
    p.add_argument("--output",default="docs/papers/paper0_85/results/phase_v1"); p.add_argument("--smoke",action="store_true")
    main(p.parse_args())
