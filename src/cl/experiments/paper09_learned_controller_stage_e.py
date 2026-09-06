"""Paper 0.9 v3 Stage E: Kmax=3 versus Kmax=4 with a shared draw prefix."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from cl.common.artifacts import atomic_write_json
from cl.experiments.paper09_learned_controller_staged import rows,run_cells,stable_sha256

def select_from_d(plan,cells_path,stage_d):
 root=Path(stage_d);mp=root/"stage_d_manifest.json";sp=root/"stage_d_selection_manifest.json";fp=root/"stage_d_frontiers.csv"
 if not all(p.exists() for p in (mp,sp,fp)):raise RuntimeError("Stage D selection prerequisite absent")
 manifest=json.loads(mp.read_text());selection=json.loads(sp.read_text());frontiers=rows(fp)
 if manifest.get("completed") is not True:raise RuntimeError("Stage D selection prerequisite partial")
 if manifest.get("selection_manifest_hash")!=stable_sha256(selection):raise RuntimeError("Stage D selection manifest hash mismatch")
 expected={(c,m) for c in ("diversity_24k","diversity_48k","diversity_96k") for m in plan["common"]["machines"]};seen={(r["architecture"],r["machine"]) for r in frontiers if int(r.get("gate_complete",0))==1 and r.get("contiguous_frontier","")!=""}
 if seen!=expected:raise RuntimeError("Stage D frontiers incomplete")
 scores={}
 for condition in sorted({r["architecture"] for r in frontiers}):
  values=[int(r["contiguous_frontier"]) for r in frontiers if r["architecture"]==condition];scores[condition]={"min_frontier":min(values),"sum_frontier":sum(values),"pool_size":int(condition.split("_")[1][:-1])*1000}
 winner=max(sorted(scores),key=lambda c:(scores[c]["min_frontier"],scores[c]["sum_frontier"],-scores[c]["pool_size"]))
 return {"schema_version":"paper09.learned_controller.stage_e.selection.v1","selected_architecture":selection["substituted_architecture"],"selected_diversity":winner,"scores":scores,
  "rule":"maximize minimum M3/M4 frontier, then sum, then minimize pool size, then lexical name","stage_d_manifest_hash":stable_sha256(manifest),"stage_d_selection_hash":stable_sha256(selection),"stage_d_frontiers_hash":stable_sha256(frontiers)}

def main(args=None):
 p=argparse.ArgumentParser();p.add_argument("--config",default="configs/paper09/learned_controller_v3_staged.json");p.add_argument("--cells",default="configs/paper09/learned_controller_v3_cells.csv");p.add_argument("--stage-d",default="docs/papers/paper0_9/results/learned_controller_v3/stage_d_diversity")
 p.add_argument("--output");p.add_argument("--device",default="mps");p.add_argument("--resume",action="store_true");p.add_argument("--smoke",action="store_true");p.add_argument("--prepare-only",action="store_true");ns=p.parse_args(args);plan=json.loads(Path(ns.config).read_text());selection=select_from_d(plan,ns.cells,ns.stage_d);out=Path(ns.output or plan["stages"]["E"]["output"]);out.mkdir(parents=True,exist_ok=True)
 with Path(ns.cells).open(newline="",encoding="utf-8") as h:specs=[r for r in csv.DictReader(h) if r["stage"]=="E"]
 all_specs={r["condition"]:r for r in rows(ns.cells)};chosen=all_specs[selection["selected_architecture"]]
 for spec in specs:
  for key in ("layers","width","heads","parameters"):spec[key]=chosen[key]
 selection["conditions"]=[s["condition"] for s in specs];selection["config_sha256"]=stable_sha256(plan);atomic_write_json(out/"stage_e_selection_manifest.json",selection)
 if ns.prepare_only:return
 def protocol(spec,machine,seed,draws,smoke):
  shared=min(plan["stages"]["E"]["shared_prefix_draws"],draws) if smoke else plan["stages"]["E"]["shared_prefix_draws"]
  pool_size=int(selection["scores"][selection["selected_diversity"]]["pool_size"])
  if smoke:pool_size=min(pool_size,draws)
  return {"kind":"train_boundary","train_kmax":int(spec["train_kmax"]),"pool_size":pool_size,"selected_diversity":selection["selected_diversity"],"matched_draws":draws,"shared_prefix_draws":shared,"shared_prefix_depths":[1,2,3]}
 run_cells("E",plan,specs,selection,out,ns.device,ns.resume,ns.smoke,protocol)
if __name__=="__main__":main()
