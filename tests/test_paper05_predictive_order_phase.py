import json
from pathlib import Path

from cl.experiments.paper05_predictive_order_phase import evaluation, information_audit, make_example

CONFIG=json.loads(Path("configs/paper05/predictive_order_phase.json").read_text())


def test_phase_information_audit():
    result,rows=information_audit(CONFIG)
    assert result["passed"] and len(rows)==6
    assert all(r["max_singleton_MI_bits"]<1e-9 for r in rows if r["predictive_order"]>1)
    assert next(r for r in rows if r["predictive_order"]==1)["max_singleton_MI_bits"]>1.9
    assert all(abs(r["full_pattern_MI_bits"]-2)<1e-9 for r in rows)


def test_phase_matrix_and_replay():
    rows=evaluation(CONFIG,2)
    expected=sum(len(CONFIG["surface_lengths"][str(p)]) for p in CONFIG["predictive_orders"])*3*4*2
    assert len(rows)==expected
    a=make_example(CONFIG,4,8,"large",8,17,"test")
    b=make_example(CONFIG,4,8,"large",8,17,"test")
    assert a==b and len(a["tokens"])==CONFIG["sequence_length"]
    assert a["predictive_order"]==4 and a["raw_length"]==8
