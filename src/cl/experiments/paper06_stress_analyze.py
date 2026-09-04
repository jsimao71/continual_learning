"""Gate and summarize independently controlled Paper 0.6 stress frontiers."""
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from cl.common.artifacts import atomic_write_json,write_csv

def read(path):
    with Path(path).open(newline="",encoding="utf-8") as handle:return list(csv.DictReader(handle))

def contiguous_frontier(values,threshold):
    """Largest tested value in the initial all-competent prefix; None if first fails."""
    frontier=None
    for value,score in sorted(values):
        if score<threshold:break
        frontier=value
    return frontier

def summarize(cells,threshold):
    seed_groups=defaultdict(list)
    for row in cells:
        key=(row["architecture"],row["evaluation_axis"],row["predicate"],int(row["total_depth"]),
             int(row["required_path"]),int(row["branching"]),int(row["distractors"]))
        seed_groups[key].append(float(row["accuracy"]))
    stable=[]
    for key,accuracies in seed_groups.items():
        stable.append({**dict(zip(("architecture","evaluation_axis","predicate","total_depth","required_path","branching","distractors"),key)),
          "mean_accuracy":float(np.mean(accuracies)),"worst_seed_accuracy":min(accuracies),
          "seed_variance":float(np.var(accuracies)),"seed_count":len(accuracies),
          "competent_all_seeds":int(len(accuracies)==3 and min(accuracies)>=threshold)})
    frontiers=[]
    for architecture in sorted({r["architecture"] for r in stable}):
      selected=[r for r in stable if r["architecture"]==architecture and r["predicate"]=="isAncestor"]
      definitions={
        "D_max":("total_depth",[r for r in selected if r["evaluation_axis"]=="depth_path" and r["required_path"]==1]),
        "d_max":("required_path",[r for r in selected if r["evaluation_axis"]=="depth_path" and r["total_depth"]==16]),
        "b_max":("branching",[r for r in selected if r["evaluation_axis"]=="branching"]),
        "N_max":("distractors",[r for r in selected if r["evaluation_axis"]=="distractors"]),
      }
      for name,(field,rows) in definitions.items():
        values=[(int(r[field]),float(r["worst_seed_accuracy"])) for r in rows if int(r["seed_count"])==3]
        frontiers.append({"architecture":architecture,"predicate":"isAncestor","frontier":name,
          "control":"d=1" if name=="D_max" else "D=16" if name=="d_max" else "D=8,d=3,N=4" if name=="b_max" else "D=8,d=3,b=2",
          "largest_contiguous_competent":contiguous_frontier(values,threshold),"tested_values":";".join(map(str,sorted(v for v,_ in values))),
          "threshold":threshold,"three_seed_complete":int(bool(values))})
    return stable,frontiers

def main(args):
    output=Path(args.output);cells=read(output/"stress_frontier_cells.csv")
    manifest=json.loads((output/"stress_frontier_manifest.json").read_text())
    if manifest["completed_models"]!=manifest["planned_models"]:raise RuntimeError("frontier run is incomplete")
    stable,frontiers=summarize(cells,args.threshold);write_csv(output/"stress_frontier_three_seed.csv",stable)
    write_csv(output/"capacity_frontiers.csv",frontiers)
    raw=read(output/"stress_frontier_raw.csv");gaps=[]
    for architecture in sorted({r["architecture"] for r in raw}):
      rows=[r for r in raw if r["architecture"]==architecture and r["predicate"]=="isAncestor" and r["evaluation_axis"]=="depth_path"]
      shallow=[int(r["top1_correct"]) for r in rows if int(r["total_depth"])<=4 and int(r["required_path"])<=3]
      extrap=[int(r["top1_correct"]) for r in rows if int(r["total_depth"])>=8 or int(r["required_path"])>3]
      gaps.append({"architecture":architecture,"predicate":"isAncestor","train_regime_accuracy":float(np.mean(shallow)),
        "extrapolation_accuracy":float(np.mean(extrap)),"extrapolation_gap":float(np.mean(shallow)-np.mean(extrap)),
        "shallow_n":len(shallow),"extrapolation_n":len(extrap)})
    write_csv(output/"capacity_extrapolation_gap.csv",gaps)
    figure_rows=[r for r in stable if r["predicate"]=="isAncestor"]
    panels=((r"$D_{\max}$","total_depth","depth_path",lambda r:r["required_path"]==1,"total tree depth $D$"),
      (r"$d_{\max}$","required_path","depth_path",lambda r:r["total_depth"]==16,"required path depth $d$"),
      (r"$b_{\max}$","branching","branching",lambda r:True,"branching $b$"),
      (r"$N_{\max}$","distractors","distractors",lambda r:True,"distractors $N$"))
    fig,axes=plt.subplots(1,4,figsize=(12.2,3.2),sharey=True)
    for ax,(title,field,kind,keep,xlabel) in zip(axes,panels):
      for architecture,marker in (("baseline","o"),("depth8","s")):
        rows=sorted((r for r in figure_rows if r["architecture"]==architecture and
          r["evaluation_axis"]==kind and keep(r)),key=lambda r:r[field])
        ax.plot([r[field] for r in rows],[r["worst_seed_accuracy"] for r in rows],marker=marker,label=architecture)
      ax.axhline(args.threshold,color="black",ls="--",lw=.8);ax.set_title(title);ax.set_xlabel(xlabel);ax.grid(alpha=.2)
    axes[0].set_ylabel("worst-seed accuracy");axes[0].legend(fontsize=8);fig.tight_layout()
    fig.savefig(output/"capacity_frontiers.png",dpi=180,bbox_inches="tight");plt.close(fig)
    atomic_write_json(output/"stress_analysis_manifest.json",{"schema_version":"paper06.stress_v6.analysis.v1",
      "threshold":args.threshold,"frontiers":frontiers,"mechanism_followup_eligible":any(
        r["frontier"]=="d_max" and r["largest_contiguous_competent"] not in (None,"",3) and int(r["largest_contiguous_competent"])>3 for r in frontiers)})

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",default="docs/papers/paper0_6/results/v6/frontier_isancestor")
    p.add_argument("--threshold",type=float,default=.8);main(p.parse_args())
