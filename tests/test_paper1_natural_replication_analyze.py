import json
from pathlib import Path
import pytest
from cl.experiments.paper1_natural_replication import validate_seed_output
from cl.experiments.paper1_natural_replication_analyze import analyze,validate_complete
from test_paper1_natural_replication import CONFIG,_seed_fixture

def jsonl(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text("".join(json.dumps(r)+"\n" for r in rows))

def fixture(root):
 (root/"preregistered_config.json").parent.mkdir(parents=True,exist_ok=True);(root/"preregistered_config.json").write_text(json.dumps(CONFIG))
 for seed in CONFIG["sampling_seeds"]:
  d=_seed_fixture(root,seed);candidates=[]
  for dataset in ("hotpotqa","qasper"):
   for split in ("validation","test"):
    for identity in range(CONFIG["per_split"]):
     for candidate in range(CONFIG["candidates"]):candidates.append({"dataset":dataset,"split":split,"identity_id":f"{dataset}-{identity}-{split}","candidate_id":candidate})
  expected=2*CONFIG["per_split"]*len(CONFIG["selectors"])*len(CONFIG["budget_chunks"]);jsonl(d/"raw/candidate_features.jsonl",candidates);jsonl(d/"raw/selection_traces.jsonl",[{"trace":i} for i in range(expected)])
  audit=validate_seed_output(d,CONFIG,seed);(d/"replication_complete.json").write_text(json.dumps(audit))

def test_complete_analysis_generates_publication_artifacts(tmp_path):
 fixture(tmp_path/"run");manifest=analyze(tmp_path/"run",tmp_path/"analysis",CONFIG)
 assert manifest["completed_seeds"]==3 and manifest["gate_decision"]["persistent_learning_gate"]==1
 assert (tmp_path/"analysis/tables/natural_replication_results.tex").exists()
 assert (tmp_path/"analysis/figures/replication_paired_frontier.png").exists()

def test_complete_analysis_rejects_raw_count_drift(tmp_path):
 fixture(tmp_path);path=tmp_path/"seeds"/f"seed-{CONFIG['sampling_seeds'][0]}"/"raw/candidate_features.jsonl";path.write_text("")
 with pytest.raises(RuntimeError,match="raw sample counts"):validate_complete(tmp_path,CONFIG)
