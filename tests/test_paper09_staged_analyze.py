import csv,json
from pathlib import Path
import pytest
from cl.experiments.paper09_learned_controller_staged_analyze import STAGE_DIR,analyze,validate_chain
from cl.experiments.paper09_learned_controller_v1 import stable_sha256

PLAN=json.loads(Path("configs/paper09/learned_controller_v3_staged.json").read_text())
def write_csv(path,values):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("w",newline="") as h:w=csv.DictWriter(h,fieldnames=values[0]);w.writeheader();w.writerows(values)
def build(root):
 ph=stable_sha256(PLAN);loaded={}
 for stage in "ABCDE":
  d=root/STAGE_DIR[stage];d.mkdir(parents=True);prefix=f"stage_{stage.lower()}";conditions=([2000,4000] if stage=="A" else PLAN["stages"][stage].get("architectures") or [f"diversity_{n//1000}k" for n in PLAN["stages"][stage].get("diversity_pool_sizes",[])] or [f"train_kmax{n}" for n in PLAN["stages"][stage]["train_kmax"]])
  front=[];depth=[];raw=[]
  for c in conditions:
   for m in ("M3","M4"):
    front.append({"snapshot_updates":c if stage=="A" else "","architecture":"" if stage=="A" else c,"machine":m,"gate_complete":1,"contiguous_frontier":1})
    depth.append({"snapshot_updates":c if stage=="A" else "","architecture":"" if stage=="A" else c,"machine":m,"seed":11,"depth":1,"final_accuracy":1,"invalid_call_rate":0,"one_call_coverage":1 if m=="M3" else "","per_transition_accuracy":1 if m=="M4" else "","termination_accuracy":1 if m=="M4" else ""})
    raw.append({"ok":1})
  frontier_path=d/f"{prefix}_{'gates' if stage=='A' else 'frontiers'}.csv";write_csv(frontier_path,front);write_csv(d/f"{prefix}_by_seed_depth.csv",depth);write_csv(d/f"{prefix}_raw.csv",raw)
  from cl.experiments.paper09_learned_controller_stage_a import read_csv
  loaded[stage]=(d,read_csv(frontier_path))
 a={"completed":True};(loaded["A"][0]/"stage_a_manifest.json").write_text(json.dumps(a))
 b={"completed":True,"config_sha256":ph,"stage_a_prerequisite":{"manifest_hash":stable_sha256(a)}};(loaded["B"][0]/"stage_b_manifest.json").write_text(json.dumps(b))
 cs={"stage_b_manifest_hash":stable_sha256(b),"stage_b_frontiers_hash":stable_sha256(loaded["B"][1])};(loaded["C"][0]/"stage_c_selection_manifest.json").write_text(json.dumps(cs));c={"completed":True,"config_sha256":ph,"selection_manifest_hash":stable_sha256(cs)};(loaded["C"][0]/"stage_c_manifest.json").write_text(json.dumps(c))
 ds={"sources":[{"stage":"B","manifest_hash":stable_sha256(b)},{"stage":"C","manifest_hash":stable_sha256(c)}]};(loaded["D"][0]/"stage_d_selection_manifest.json").write_text(json.dumps(ds));d={"completed":True,"config_sha256":ph,"selection_manifest_hash":stable_sha256(ds)};(loaded["D"][0]/"stage_d_manifest.json").write_text(json.dumps(d))
 es={"stage_d_manifest_hash":stable_sha256(d),"stage_d_selection_hash":stable_sha256(ds),"stage_d_frontiers_hash":stable_sha256(loaded["D"][1])};(loaded["E"][0]/"stage_e_selection_manifest.json").write_text(json.dumps(es));e={"completed":True,"config_sha256":ph,"selection_manifest_hash":stable_sha256(es)};(loaded["E"][0]/"stage_e_manifest.json").write_text(json.dumps(e))

def test_analysis_validates_chain_and_writes_separate_outputs(tmp_path):
 root=tmp_path/"stages";build(root);manifest=analyze(root,tmp_path/"analysis",PLAN)
 assert manifest["authoritative_only"] is True and manifest["frontier_rows"]==28
 assert (tmp_path/"analysis/figures/stage_e_ktrain_frontiers.png").exists()

def test_analysis_rejects_partial_or_hash_broken_stage(tmp_path):
 root=tmp_path/"stages";build(root);p=root/STAGE_DIR["D"]/"stage_d_manifest.json";value=json.loads(p.read_text());value["completed"]=False;p.write_text(json.dumps(value))
 with pytest.raises(RuntimeError,match="partial"):validate_chain(root,PLAN)
