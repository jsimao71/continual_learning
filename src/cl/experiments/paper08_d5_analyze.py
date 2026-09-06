"""Validate and summarize the official learned Paper 0.8 D5 first pass."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from cl.common.artifacts import atomic_write_json,write_csv

SPLITS=("seen_parameter","unseen_parameter","unseen_family")
def read(path):
    with path.open(newline="",encoding="utf-8") as h:return list(csv.DictReader(h))

def main(args):
    out=Path(args.output);manifest=json.loads((out/"manifest.json").read_text());cells=read(out/"cell_summary.csv");raw=read(out/"raw_results.csv");audit=read(out/"dataset_audit.csv")
    if manifest["smoke"] or manifest["completed_cells"]!=manifest["planned_cells"] or manifest["completed_cells"]!=12:raise RuntimeError("official run is incomplete or smoke")
    keys={(int(r["layers"]),int(r["width"]),int(r["heads"]),r["variant"],int(r["seed"]),r["split"]) for r in cells}
    if len(cells)!=36 or len(keys)!=36:raise RuntimeError("cell summary is incomplete or duplicated")
    if len(raw)!=51456 or {r["condition"] for r in raw}!={"correct","none","shuffled","wrong_rule"}:raise RuntimeError("raw result schema/count mismatch")
    if len(audit)!=4 or any(r["all_answer_entropy_zero"]!="True" or int(r["serialized_episode_overlap"]) for r in audit):raise RuntimeError("dataset validity gate failed")
    architecture=[]
    arch_keys=sorted({(int(r["layers"]),int(r["width"]),int(r["heads"]),r["variant"],r["split"]) for r in cells})
    for l,w,h,v,split in arch_keys:
        q=[r for r in cells if (int(r["layers"]),int(r["width"]),int(r["heads"]),r["variant"],r["split"])==(l,w,h,v,split)]
        architecture.append({"layers":l,"width":w,"heads":h,"variant":v,"split":split,"seeds":len(q),
          "mean_correct_accuracy":float(np.mean([float(r["correct_accuracy"]) for r in q])),"max_seed_correct_accuracy":max(float(r["correct_accuracy"]) for r in q),
          "mean_selectivity":float(np.mean([float(r["selectivity"]) for r in q])),"min_seed_selectivity":min(float(r["selectivity"]) for r in q),
          "passing_seeds":sum(int(r["competent"]) for r in q),"uniform_baseline":float(q[0]["uniform_baseline"]),"majority_baseline":float(q[0]["majority_baseline"])})
    write_csv(out/"architecture_summary.csv",architecture)
    family=[]
    for split in SPLITS:
      for fam in sorted({r["family"] for r in raw if r["split"]==split}):
       q=[r for r in raw if r["split"]==split and r["family"]==fam and r["condition"]=="correct"]
       family.append({"split":split,"family":fam,"evaluations":len(q),"mean_correct_accuracy":float(np.mean([int(r["top1"]) for r in q])),"mean_target_margin":float(np.mean([float(r["target_margin"]) for r in q]))})
    write_csv(out/"family_summary.csv",family)
    labels=[f"L{r['layers']}/W{r['width']}\n{r['variant']}" for r in architecture if r["split"]=="seen_parameter"]
    x=np.arange(len(labels));fig,ax=plt.subplots(figsize=(7.2,3.8));width=.24
    for i,split in enumerate(SPLITS):
        vals=[r["mean_correct_accuracy"] for r in architecture if r["split"]==split]
        ax.bar(x+(i-1)*width,vals,width,label=split.replace("_"," "))
    ax.axhline(np.mean([r["uniform_baseline"] for r in architecture]),color="black",ls="--",lw=.8,label="mean uniform baseline")
    ax.set(ylabel="three-seed mean accuracy",ylim=(0,1),xticks=x,xticklabels=labels);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/"d5_accuracy_by_split.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.4,3.5));ax.bar([f"{r['split']}\n{r['family']}" for r in family],[r["mean_correct_accuracy"] for r in family],color="#0072B2")
    ax.set(ylabel="mean accuracy across all cells/seeds",ylim=(0,1));ax.tick_params(axis="x",labelrotation=25,labelsize=8);fig.tight_layout();fig.savefig(out/"d5_family_accuracy.png",dpi=180);plt.close(fig)
    atomic_write_json(out/"analysis_manifest.json",{"schema_version":"paper08.d5_acquisition.analysis.v1","official_cells":12,"summary_rows":36,"raw_rows":len(raw),
      "three_seed_complete":True,"passing_architecture_split_cells":sum(r["passing_seeds"]==3 for r in architecture),"any_passing_seed":any(r["passing_seeds"] for r in architecture),
      "all_information_valid":True,"invalid_or_smoke_inputs_excluded":True})

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--output",default="docs/papers/paper0_8/results/d5_acquisition_v1");main(p.parse_args())
