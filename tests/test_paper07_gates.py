import json
from pathlib import Path

from cl.experiments.paper07_gate_run import build
from cl.semantic.paper07_gates import NO, YES, gate_examples, validate_gates


CONFIG = json.loads(Path("configs/paper07/gates_v1.json").read_text())


def test_gates_are_balanced_and_namespaces_disjoint():
    rows = build(CONFIG, smoke=True); audit = validate_gates(rows)
    assert audit["valid"] and audit["constant_baseline_accuracy"] == .5
    assert all(value == {"positive": 8, "negative": 8} for value in audit["balance"].values())


def test_p0_matched_negative_changes_query_not_length():
    rows = sorted(gate_examples("P0", "test", 6, 4), key=lambda row: row.example_id)
    assert {row.target for row in rows} == {YES, NO}
    assert {len(row.tokens) for row in rows} == {20}
    assert {row.namespace for row in rows} == {"te"}


def test_f0_covers_constant_and_predicate_mismatch():
    rows = gate_examples("F0", "validation", 8, 8)
    assert {row.template for row in rows} >= {"identity", "constant_mismatch", "predicate_mismatch"}
    assert sum(row.label == "positive" for row in rows) == 4


def test_f0_budget_is_an_explicit_acquisition_gate():
    assert CONFIG["steps_by_stage"]["P0"] == 400
    assert CONFIG["steps_by_stage"]["F0"] == 3200
