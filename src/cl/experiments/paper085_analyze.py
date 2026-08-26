"""Analyze Paper 0.85 dataset geometry without reporting model outcomes."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
from cl.common.artifacts import atomic_write_json, write_csv

def main(args):
    source=Path(args.source); rows=[]
    for split in ("train","validation","test"):
        rows += [json.loads(line) for line in (source/f"{split}.jsonl").read_text().splitlines()]
    grouped=defaultdict(list)
    for row in rows:
        if row["distractors"]==0 and row["branching"]==1 and not row["shuffled"]:
            grouped[row["proof_depth"]].append(row)
    table=[]
    for depth,group in sorted(grouped.items()):
        prompt=sum(r["prompt_tokens"] for r in group)/len(group)
        record={"proof_depth":depth,"prompt_tokens_mean":prompt}
        for output in ("O0","O1","O2","O3"):
            generated=sum(r["generated_tokens"][output] for r in group)/len(group)
            record[f"{output}_generated_tokens_mean"]=generated
            record[f"{output}_total_tokens_mean"]=prompt+generated
        table.append(record)
    out=Path(args.output); figures=out/"figures"; figures.mkdir(parents=True,exist_ok=True)
    write_csv(out/"context_budget.csv",table)
    fig,ax=plt.subplots(figsize=(6.3,3.5))
    for output,label in (("O0","one-token O0"),("O1","free trace O1"),("O2","structured O2/O3")):
        ax.plot([r["proof_depth"] for r in table],[r[f"{output}_total_tokens_mean"] for r in table],marker="o",label=label)
    ax.set(xlabel="proof depth K",ylabel="serialized prompt + target tokens",
           title="Reference serialization budget (not model competence)"); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(figures/"context_budget_frontier.png",dpi=180); plt.close(fig)
    atomic_write_json(out/"analysis_manifest.json",{"schema_version":"paper085.analysis_v1",
        "source_model_results":False,"analysis_type":"dataset_serialization_only","rows":len(rows),
        "depths":[r["proof_depth"] for r in table],"figure":"figures/context_budget_frontier.png"})
    print(json.dumps({"rows":len(rows),"depth_cells":len(table)},indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--source",default="docs/papers/paper0_85/results/phase_v1")
    p.add_argument("--output",default="docs/papers/paper0_85/results"); main(p.parse_args())
