"""CPU-only design/audit stage for Paper 0.6 relation stress experiments."""
from __future__ import annotations
import argparse,json
from collections import Counter
from pathlib import Path
from cl.common.artifacts import atomic_write_json,write_csv
from cl.semantic.predicate_stress import PAIRWISE,axis,stress_design,validate_stress

def design_rows(config,examples):
    rows=stress_design(config,config["model_seeds"][0],examples)
    counts=Counter((r.predicate,axis(r)) for r in rows)
    task=lambda p:"fixed_hop" if p in ("parent","grandparent","sameParent") else "variable_hop"
    return [{"predicate":p,"task_type":task(p),"evaluation_axis":a,"examples":n,
             "mechanism_status":"gated_pending_competence"} for (p,a),n in sorted(counts.items())]

def main(args):
    cfg=json.loads(Path(args.config).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    audit=validate_stress(cfg); assert audit["passed"]
    write_csv(out/"stress_design_cells.csv",design_rows(cfg,args.examples))
    atomic_write_json(out/"stress_generator_validation.json",audit)
    atomic_write_json(out/"stress_manifest.json",{"schema_version":"paper06.stress_v6.design.v1","training_performed":False,"baseline_preservation":"v4/v5 modules and artifacts unchanged","model_seeds":cfg["model_seeds"],"training_budgets":[1,2,4],"frontiers":["D_max","d_max","b_max","N_max"],"mechanism_gate":"newly competent seed-stable cells only"})
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/paper06/stress_v6.json");p.add_argument("--output",default="docs/papers/paper0_6/results/v6/design");p.add_argument("--examples",type=int,default=2);main(p.parse_args())
