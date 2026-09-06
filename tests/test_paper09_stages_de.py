import csv,json
from pathlib import Path
import pytest
import torch
from cl.experiments.paper09_learned_controller_stage_e import select_from_d
from cl.experiments.paper09_learned_controller_staged import protocol_index,select_architecture,train_protocol
from cl.semantic.recurrence_chains import recurrence_pair_split

PLAN=json.loads(Path("configs/paper09/learned_controller_v3_staged.json").read_text())

def _write_csv(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("w",newline="") as h:w=csv.DictWriter(h,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def _bc(root,stage,frontiers,completed=True):
 d=root/stage.lower();d.mkdir(parents=True);manifest={"completed":completed}
 if stage=="C":
  from cl.experiments.paper09_learned_controller_staged import stable_sha256
  selection={"enabled":True};(d/"stage_c_selection_manifest.json").write_text(json.dumps(selection));manifest["selection_manifest_hash"]=stable_sha256(selection)
 (d/f"stage_{stage.lower()}_manifest.json").write_text(json.dumps(manifest))
 _write_csv(d/f"stage_{stage.lower()}_frontiers.csv",[{"architecture":a,"machine":m,"gate_complete":1,"contiguous_frontier":frontiers[a][m]} for a in PLAN["stages"][stage]["architectures"] for m in PLAN["common"]["machines"]]);return d

def test_d_selection_is_deterministic_and_requires_complete_bc(tmp_path):
 b={a:{"M3":1,"M4":2} for a in PLAN["stages"]["B"]["architectures"]};c={a:{"M3":1,"M4":2} for a in PLAN["stages"]["C"]["architectures"]};c["one_head"]={"M3":2,"M4":3}
 bd=_bc(tmp_path,"B",b);cd=_bc(tmp_path,"C",c);spec,decision=select_architecture(PLAN,"configs/paper09/learned_controller_v3_cells.csv",bd,cd)
 assert spec["condition"]==decision["selected_architecture"]=="one_head"
 (cd/"stage_c_manifest.json").write_text(json.dumps({"completed":False}))
 with pytest.raises(RuntimeError,match="partial"):select_architecture(PLAN,"configs/paper09/learned_controller_v3_cells.csv",bd,cd)

def test_d_prefix_pool_and_e_shared_prefix_indices():
 d24={"kind":"diversity","pool_size":24000};d48={"kind":"diversity","pool_size":48000}
 assert [protocol_index(d24,i)[0] for i in (0,23999,24000,47999)]==[0,23999,0,23999]
 assert [protocol_index(d48,i)[0] for i in (0,23999)]==[0,23999]
 e3={"kind":"train_boundary","train_kmax":3,"pool_size":48000,"shared_prefix_draws":72000};e4={"kind":"train_boundary","train_kmax":4,"pool_size":48000,"shared_prefix_draws":72000}
 assert protocol_index(e3,71999)==protocol_index(e4,71999)
 assert protocol_index(e3,72000)[1]==[1,2,3] and protocol_index(e4,72000)[1]==[1,2,3,4]
 assert protocol_index(e3,48000)[0]==0

def test_e_requires_complete_d_and_selects_frontier(tmp_path):
 from cl.experiments.paper09_learned_controller_staged import stable_sha256
 d=tmp_path/"d";d.mkdir();selection={"substituted_architecture":"deep_matched"};(d/"stage_d_selection_manifest.json").write_text(json.dumps(selection));(d/"stage_d_manifest.json").write_text(json.dumps({"completed":True,"selection_manifest_hash":stable_sha256(selection)}))
 values={"diversity_24k":1,"diversity_48k":3,"diversity_96k":2};_write_csv(d/"stage_d_frontiers.csv",[{"architecture":a,"machine":m,"gate_complete":1,"contiguous_frontier":v} for a,v in values.items() for m in ("M3","M4")])
 assert select_from_d(PLAN,"configs/paper09/learned_controller_v3_cells.csv",d)["selected_diversity"]=="diversity_48k"
 (d/"stage_d_manifest.json").write_text(json.dumps({"completed":False}))
 with pytest.raises(RuntimeError,match="partial"):select_from_d(PLAN,"configs/paper09/learned_controller_v3_cells.csv",d)

def test_staged_protocol_resume_is_exact(tmp_path):
 _,pairs,_=recurrence_pair_split(16,8502,.2);cfg={"train_depths":[1,2,3],"max_length":48,"model":{"width":16,"layers":1,"heads":1,"mlp_ratio":2},"learning_rate":.001,"log_every":1,"checkpoint_every":2};device=torch.device("cpu");protocol={"kind":"diversity","pool_size":12,"matched_draws":24,"exact_prefix":True}
 full,loss1=train_protocol("M4",11,cfg,device,tmp_path/"full.pt",pairs,4,6,protocol);train_protocol("M4",11,cfg,device,tmp_path/"resume.pt",pairs,2,6,protocol);resumed,loss2=train_protocol("M4",11,cfg,device,tmp_path/"resume.pt",pairs,4,6,protocol)
 assert loss1==loss2 and all(torch.equal(full.state_dict()[k],resumed.state_dict()[k]) for k in full.state_dict())
