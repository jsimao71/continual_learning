"""Validate and analyze authoritative Paper 0.9 v3 Stages A--E."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from cl.common.artifacts import atomic_write_json,write_csv
from cl.experiments.paper09_learned_controller_stage_a import read_csv
from cl.experiments.paper09_learned_controller_v1 import stable_sha256

STAGE_DIR={"A":"stage_a_budget","B":"stage_b_depth","C":"stage_c_width_heads","D":"stage_d_diversity","E":"stage_e_train_depth"}

def _load(root,stage):
 prefix=f"stage_{stage.lower()}";directory=Path(root)/STAGE_DIR[stage];mp=directory/f"{prefix}_manifest.json"
 if not mp.exists():raise RuntimeError(f"Stage {stage} manifest absent")
 manifest=json.loads(mp.read_text())
 if manifest.get("completed") is not True:raise RuntimeError(f"Stage {stage} is partial")
 frontiers=read_csv(directory/f"{prefix}_{'gates' if stage=='A' else 'frontiers'}.csv")
 by_depth=read_csv(directory/f"{prefix}_by_seed_depth.csv")
 raw=read_csv(directory/f"{prefix}_raw.csv")
 if not frontiers or not by_depth or not raw:raise RuntimeError(f"Stage {stage} authoritative tables incomplete")
 return directory,manifest,frontiers,by_depth,raw

def validate_chain(root,plan):
 loaded={stage:_load(root,stage) for stage in "ABCDE"};plan_hash=stable_sha256(plan)
 for stage in "ABCDE":
  if loaded[stage][1].get("config_sha256")!=plan_hash:raise RuntimeError(f"Stage {stage} config hash mismatch")
 a_hash=stable_sha256(loaded["A"][1]);b=loaded["B"][1]
 if b.get("stage_a_prerequisite",{}).get("manifest_hash")!=a_hash:raise RuntimeError("Stage B -> A hash mismatch")
 for stage,upstream in (("C","B"),("D",None),("E",None)):
  directory,manifest,*_=loaded[stage];selection_path=directory/f"stage_{stage.lower()}_selection_manifest.json"
  if not selection_path.exists():raise RuntimeError(f"Stage {stage} selection manifest absent")
  selection=json.loads(selection_path.read_text())
  if manifest.get("selection_manifest_hash")!=stable_sha256(selection):raise RuntimeError(f"Stage {stage} selection hash mismatch")
  if upstream:
   if selection.get("stage_b_manifest_hash")!=stable_sha256(loaded[upstream][1]):raise RuntimeError("Stage C -> B manifest hash mismatch")
   if selection.get("stage_b_frontiers_hash")!=stable_sha256(loaded[upstream][2]):raise RuntimeError("Stage C -> B frontier hash mismatch")
 d_selection=json.loads((loaded["D"][0]/"stage_d_selection_manifest.json").read_text())
 source={row["stage"]:row for row in d_selection.get("sources",[])}
 for stage in ("B","C"):
  if source.get(stage,{}).get("manifest_hash")!=stable_sha256(loaded[stage][1]):raise RuntimeError(f"Stage D -> {stage} hash mismatch")
  if source.get(stage,{}).get("frontier_hash")!=stable_sha256(loaded[stage][2]):raise RuntimeError(f"Stage D -> {stage} frontier hash mismatch")
 e_selection=json.loads((loaded["E"][0]/"stage_e_selection_manifest.json").read_text())
 if e_selection.get("stage_d_manifest_hash")!=stable_sha256(loaded["D"][1]):raise RuntimeError("Stage E -> D manifest hash mismatch")
 if e_selection.get("stage_d_selection_hash")!=stable_sha256(d_selection):raise RuntimeError("Stage E -> D selection hash mismatch")
 if e_selection.get("stage_d_frontiers_hash")!=stable_sha256(loaded["D"][2]):raise RuntimeError("Stage E -> D frontier hash mismatch")
 return loaded,{"valid":True,"config_sha256":plan_hash,"stage_manifest_hashes":{s:stable_sha256(loaded[s][1]) for s in loaded}}

def frontier_rows(loaded):
 out=[]
 for stage in "ABCDE":
  for row in loaded[stage][2]:
   out.append({"stage":stage,"condition":row.get("architecture",f"budget_{row.get('snapshot_updates')}") ,"machine":row["machine"],
               "contiguous_frontier":row.get("contiguous_frontier",""),"gate_complete":row["gate_complete"]})
 return out

def error_rows(loaded):
 out=[]
 for stage in "ABCDE":
  grouped={}
  for row in loaded[stage][3]:
   condition=row.get("architecture",f"budget_{row.get('snapshot_updates')}");key=(condition,row["machine"],int(row["depth"]));grouped.setdefault(key,[]).append(row)
  for (condition,machine,depth),values in sorted(grouped.items()):
   mean=lambda k:float(np.mean([float(v[k]) for v in values if v.get(k,"")!=""]))
   record={"stage":stage,"condition":condition,"machine":machine,"depth":depth,"final_error":1-mean("final_accuracy"),"invalid_call_rate":mean("invalid_call_rate")}
   if machine=="M3":record["execution_error"]=1-mean("one_call_coverage");record["termination_error"]=""
   else:record["execution_error"]=1-mean("per_transition_accuracy");record["termination_error"]=1-mean("termination_accuracy")
   out.append(record)
 return out

def plots(out,loaded,frontiers,errors):
 figures=out/"figures";figures.mkdir(parents=True,exist_ok=True)
 # One compact frontier panel per causal factor.
 for stage,title,name in (("A","Training budget","stage_a_budget_frontiers"),("B","Depth/width allocation","stage_b_allocation_frontiers"),("C","Width/head controls","stage_c_control_frontiers"),("D","Training diversity","stage_d_diversity_frontiers"),("E","Training boundary","stage_e_ktrain_frontiers")):
  rows=[r for r in frontiers if r["stage"]==stage];conditions=[]
  for r in rows:
   if r["condition"] not in conditions:conditions.append(r["condition"])
  x=np.arange(len(conditions));fig,ax=plt.subplots(figsize=(max(5,1.25*len(conditions)),3.6))
  for offset,(machine,marker) in zip((-.16,.16),(('M3','o'),('M4','s'))):
   lookup={r["condition"]:float(r["contiguous_frontier"]) for r in rows if r["machine"]==machine};ax.scatter(x+offset,[lookup[c] for c in conditions],label=machine,marker=marker,s=55)
  ax.set_xticks(x,conditions,rotation=20,ha="right");ax.set_ylabel("all-seed contiguous frontier");ax.set_title(title);ax.set_ylim(bottom=0);ax.grid(axis="y",alpha=.25);ax.legend();fig.tight_layout();fig.savefig(figures/f"{name}.png",dpi=180);plt.close(fig)
  
  # Depthwise final/execution errors retain machine semantics.
 fig,axes=plt.subplots(1,2,figsize=(9,3.8),sharey=True)
 for ax,machine in zip(axes,("M3","M4")):
  values=[r for r in errors if r["machine"]==machine]
  for stage in "ABCDE":
   subset=[r for r in values if r["stage"]==stage]
   if not subset:continue
   depths=sorted({r["depth"] for r in subset});ax.plot(depths,[np.mean([r["execution_error"] for r in subset if r["depth"]==d]) for d in depths],marker="o",label=f"Stage {stage}")
  ax.set_title(f"{machine} execution error");ax.set_xlabel("proof depth");ax.grid(alpha=.25)
 axes[0].set_ylabel("mean error across conditions/seeds");axes[1].legend(fontsize=7);fig.tight_layout();fig.savefig(figures/"machine_specific_error_profiles.png",dpi=180);plt.close(fig)

def analyze(root,out,plan):
 loaded,audit=validate_chain(root,plan);out=Path(out);(out/"tables").mkdir(parents=True,exist_ok=True)
 frontiers=frontier_rows(loaded);errors=error_rows(loaded);write_csv(out/"tables"/"staged_frontiers.csv",frontiers);write_csv(out/"tables"/"machine_specific_errors.csv",errors);plots(out,loaded,frontiers,errors)
 manifest={"schema_version":"paper09.learned_controller.staged_analysis.v1","authoritative_only":True,"source_audit":audit,"frontier_rows":len(frontiers),"error_rows":len(errors),"frontier_hash":stable_sha256(frontiers),"error_hash":stable_sha256(errors)};atomic_write_json(out/"analysis_manifest.json",manifest);return manifest

def main(args=None):
 p=argparse.ArgumentParser();p.add_argument("--config",default="configs/paper09/learned_controller_v3_staged.json");p.add_argument("--root",default="docs/papers/paper0_9/results/learned_controller_v3");p.add_argument("--output",default="docs/papers/paper0_9/results/learned_controller_v3/analysis");ns=p.parse_args(args);print(json.dumps(analyze(ns.root,ns.output,json.loads(Path(ns.config).read_text())),indent=2))
if __name__=="__main__":main()
