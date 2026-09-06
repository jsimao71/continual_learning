"""Paper 0.9 v3 Stage D: matched-draw training-diversity controls."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from cl.common.artifacts import atomic_write_json
from cl.experiments.paper09_learned_controller_staged import run_cells,select_architecture,stable_sha256

def main(args=None):
 p=argparse.ArgumentParser();p.add_argument("--config",default="configs/paper09/learned_controller_v3_staged.json");p.add_argument("--cells",default="configs/paper09/learned_controller_v3_cells.csv")
 p.add_argument("--stage-b",default="docs/papers/paper0_9/results/learned_controller_v3/stage_b_depth");p.add_argument("--stage-c",default="docs/papers/paper0_9/results/learned_controller_v3/stage_c_width_heads");p.add_argument("--output");p.add_argument("--device",default="mps");p.add_argument("--resume",action="store_true");p.add_argument("--smoke",action="store_true");p.add_argument("--prepare-only",action="store_true");ns=p.parse_args(args)
 plan=json.loads(Path(ns.config).read_text());chosen,selection=select_architecture(plan,ns.cells,ns.stage_b,ns.stage_c);out=Path(ns.output or plan["stages"]["D"]["output"]);out.mkdir(parents=True,exist_ok=True)
 with Path(ns.cells).open(newline="",encoding="utf-8") as h:specs=[r for r in csv.DictReader(h) if r["stage"]=="D"]
 for spec in specs:
  for key in ("layers","width","heads","parameters"):spec[key]=chosen[key]
 selection.update({"schema_version":"paper09.learned_controller.stage_d.selection.v1","substituted_architecture":chosen["condition"],"conditions":[s["condition"] for s in specs],"config_sha256":stable_sha256(plan)})
 atomic_write_json(out/"stage_d_selection_manifest.json",selection)
 if ns.prepare_only:return
 def protocol(spec,machine,seed,draws,smoke):
  pool=min(int(spec["diversity_pool"]),draws) if smoke else int(spec["diversity_pool"])
  return {"kind":"diversity","pool_size":pool,"matched_draws":draws,"exact_prefix":True}
 run_cells("D",plan,specs,selection,out,ns.device,ns.resume,ns.smoke,protocol)
if __name__=="__main__":main()
