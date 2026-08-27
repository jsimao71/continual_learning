"""Aggregate Paper 0.85 recurrence Stage-1 without extrapolating frontiers."""
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from cl.common.artifacts import atomic_write_json,write_csv

def read(path):
    with Path(path).open(newline="",encoding="utf-8") as h:return list(csv.DictReader(h))

def main(args):
    out=Path(args.output);raw=read(out/"stage1_raw_latest.csv");groups=defaultdict(list)
    for r in raw:groups[(r["condition"],int(r["seed"]),int(r["depth"]))].append(r)
    seed_rows=[]
    for (condition,seed,depth),rows in sorted(groups.items()):
        mean=lambda k:float(np.mean([float(r[k]) for r in rows]))
        seed_rows.append({"condition":condition,"seed":seed,"depth":depth,"examples":len(rows),
            "final_accuracy":mean("final_correct"),"trajectory_accuracy":mean("trajectory_exact"),
            "transition_accuracy":mean("transition_accuracy"),"termination_accuracy":mean("termination_correct")})
    curves=[]
    for condition in ("O0","O1","O2","O3"):
      for depth in sorted({r["depth"] for r in seed_rows}):
        rows=[r for r in seed_rows if r["condition"]==condition and r["depth"]==depth]
        curves.append({"condition":condition,"depth":depth,"seeds":len(rows),
            **{f"{key}_{stat}":fn([r[key] for r in rows]) for key in ("final_accuracy","trajectory_accuracy","transition_accuracy","termination_accuracy") for stat,fn in (("mean",np.mean),("worst",min),("best",max))},
            "three_seed_competent":int(min(r["final_accuracy"] for r in rows)>=.95 and min(r["termination_accuracy"] for r in rows)>=.95)})
    frontiers=[]
    for condition in ("O0","O1","O2","O3"):
        passed=[r["depth"] for r in curves if r["condition"]==condition and r["three_seed_competent"]]
        frontier=max(passed) if passed else 0
        frontiers.append({"condition":condition,"measured_frontier":frontier,"acquisition_probability_at_K4":np.mean([
            r["final_accuracy"]>=.95 and r["termination_accuracy"]>=.95 for r in seed_rows if r["condition"]==condition and r["depth"]==4]),
            "recurrence_gain_vs_O0":frontier-3,"claim":"measured_stage1_frontier_not_closure"})
    write_csv(out/"chain_depth_curves.csv",curves);write_csv(out/"step_transition_accuracy.csv",seed_rows)
    write_csv(out/"one_token_vs_multitoken.csv",frontiers)
    write_csv(out/"acquisition_probability.csv",({"condition":r["condition"],"depth":r["depth"],"seed_pass_fraction":np.mean([
        x["final_accuracy"]>=.95 and x["termination_accuracy"]>=.95 for x in seed_rows if x["condition"]==r["condition"] and x["depth"]==r["depth"]])} for r in curves))
    write_csv(out/"termination_errors.csv",({"condition":r["condition"],"seed":r["seed"],"depth":r["depth"],"termination_accuracy":r["termination_accuracy"],"termination_error":1-r["termination_accuracy"]} for r in seed_rows))
    figures=out/"figures";figures.mkdir(exist_ok=True);fig,ax=plt.subplots(figsize=(6.2,3.6))
    for condition in ("O0","O1","O2","O3"):
        rows=[r for r in curves if r["condition"]==condition]
        ax.plot([r["depth"] for r in rows],[r["final_accuracy_mean"] for r in rows],"o-",label=condition)
    ax.axvline(3,color="black",ls="--",lw=1,label="max train depth");ax.axhline(.95,color="gray",ls=":",lw=1)
    ax.set(xlabel="proof depth K",ylabel="mean final accuracy",ylim=(-.03,1.03),title="One-token versus autoregressive Stage 1")
    ax.legend(ncol=3);ax.grid(alpha=.25);fig.tight_layout();fig.savefig(figures/"one_token_vs_multitoken_depth.png",dpi=180);plt.close(fig)
    atomic_write_json(out/"analysis_manifest.json",{"schema_version":"paper085.recurrence_analysis_v1","cells":len(seed_rows),
        "raw_rows":len(raw),"frontiers":{r["condition"]:r["measured_frontier"] for r in frontiers},"recurrence_gain":0,
        "closure_signature_observed":False,"invalid_predecessor":"recurrence_stage1_v1_query_target_leak"})
    print(json.dumps({"frontiers":{r["condition"]:r["measured_frontier"] for r in frontiers},"raw_rows":len(raw)},indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",default="docs/papers/paper0_85/results/recurrence_stage1_v2");main(p.parse_args())
