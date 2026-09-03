import csv,json
from pathlib import Path

from cl.experiments.paper07_p1_f1_gate import aggregate,build
from cl.semantic.paper07_gates import YES,NO,SYMBOL
from cl.semantic.paper07_p1_f1 import SYMBOL_COUNT,step_gate_examples,validate_step_gates

CONFIG=json.loads(Path("configs/paper07/p1_f1_gate_v1.json").read_text())


def test_p1_f1_gate_dataset_is_template_balanced_and_split_clean():
    rows=build(CONFIG,smoke=True);audit=validate_step_gates(rows)
    assert audit["valid"] and audit["rows"]==2*3*48
    assert audit["P1_constant_baseline_accuracy"]==.5
    assert audit["F1_symbol_baseline_accuracy"]==1/12


def test_p1_each_template_has_balanced_entailed_controls():
    rows=step_gate_examples("P1","test",48,17)
    for template in {r.template for r in rows}:
        subset=[r for r in rows if r.template==template]
        assert sum(r.target==YES for r in subset)==sum(r.target==NO for r in subset)
        assert {len(r.tokens) for r in subset}=={20}


def test_f1_targets_are_balanced_and_present_under_all_templates():
    rows=step_gate_examples("F1","validation",96,19)
    assert {r.template for r in rows}=={"canonical","reversed","constant_distractor","predicate_distractor"}
    assert {r.target for r in rows}==set(range(SYMBOL,SYMBOL+SYMBOL_COUNT))
    assert len({sum(r.target==target for r in rows) for target in range(SYMBOL,SYMBOL+SYMBOL_COUNT)})==1
    assert all(r.target in r.tokens for r in rows)


def test_template_gate_exposes_weakest_stratum():
    raw=[]
    for template,correct in (("canonical",1),("reversed",0)):
      for index in range(4):raw.append({"stage":"F1","model_seed":11,"split":"test","template":template,
          "correct":correct,"target_margin":1 if correct else -1})
    rows=aggregate(raw,.95)
    assert {r["template"]:r["competent"] for r in rows}=={"canonical":1,"reversed":0}


def test_resume_preserves_runtime_aggregation(tmp_path,monkeypatch):
    from cl.experiments import paper07_p1_f1_gate as runner
    args=type("Args",(),{"config":"configs/paper07/p1_f1_gate_v1.json","output":str(tmp_path),
         "device":"cpu","smoke":True,"resume":False,"max_models":1,"only_stage":"P1"})()
    runner.main(args)
    args.resume=True;runner.main(args)
    with (tmp_path/"gate_runtime.csv").open(newline="") as handle:
        rows=list(csv.DictReader(handle))
    assert len(rows)==1 and rows[0]["stage"]=="P1"
    manifest=json.loads((tmp_path/"gate_manifest.json").read_text())
    assert manifest["dataset_hash"] and manifest["config_hash"]
