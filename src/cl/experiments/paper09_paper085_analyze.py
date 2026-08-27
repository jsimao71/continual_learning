"""Reference-only resource analysis for the matched Paper 0.85 harness."""
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
from cl.common.artifacts import atomic_write_json,write_csv

def read(path):
    with Path(path).open(newline="",encoding="utf-8") as handle:return list(csv.DictReader(handle))

def main(args):
    source=Path(args.source); rows=read(source/"paper085_tool_strength.csv")
    groups=defaultdict(list)
    for row in rows: groups[(int(row["depth"]),row["tool_strength"])].append(row)
    summary=[]
    for (depth,strength),group in sorted(groups.items()):
        def numeric(value): return 1.0 if value=="True" else 0.0 if value=="False" else float(value)
        mean=lambda key:sum(numeric(r[key]) for r in group)/len(group)
        summary.append({"depth":depth,"tool_strength":strength,"examples":len(group),"result_type":"reference_oracle",
            "tool_calls_mean":mean("tool_calls"),"model_forwards_mean":mean("model_forwards"),
            "generated_action_tokens_mean":mean("generated_action_tokens"),"total_context_tokens_mean":mean("total_context_tokens"),
            "termination_accuracy":mean("termination_correct"),"final_accuracy":mean("correct")})
    write_csv(source/"paper085_context_budget.csv",summary)
    write_csv(source/"paper085_termination.csv",({k:r[k] for k in ("depth","tool_strength","examples","result_type","termination_accuracy")} for r in summary))
    figures=source/"figures";figures.mkdir(exist_ok=True)
    base=[r for r in summary if r["tool_strength"]=="T1"]
    fig,axes=plt.subplots(1,2,figsize=(8.2,3.3))
    axes[0].plot([r["depth"] for r in base],[r["tool_calls_mean"] for r in base],"o-",label="tool calls")
    axes[0].plot([r["depth"] for r in base],[r["model_forwards_mean"] for r in base],"s-",label="model invocations")
    axes[0].set(xlabel="proof depth K",ylabel="reference count",title="Explicit iteration cost");axes[0].legend();axes[0].grid(alpha=.25)
    axes[1].plot([r["depth"] for r in base],[r["total_context_tokens_mean"] for r in base],"o-",color="#6a3d9a")
    axes[1].set(xlabel="proof depth K",ylabel="serialized tokens",title="Accumulated context cost");axes[1].grid(alpha=.25)
    fig.suptitle("Oracle-controller resource geometry (not learned-model performance)",fontsize=10);fig.tight_layout()
    fig.savefig(figures/"paper085_reference_execution_cost.png",dpi=180);plt.close(fig)
    atomic_write_json(source/"analysis_manifest.json",{"schema_version":"paper09.paper085_analysis_v1",
        "result_type":"reference_oracle","learned_model_results":False,"reference_rows":len(rows),
        "depth_cells":len(summary),"figure":"figures/paper085_reference_execution_cost.png"})
    print(json.dumps({"reference_rows":len(rows),"depth_cells":len(summary)},indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source",default="docs/papers/paper0_9/results/paper085_comparison");main(p.parse_args())
