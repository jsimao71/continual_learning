import json
from pathlib import Path
from cl.experiments.paper05_nested_length_depth import evaluation,validation

CONFIG=json.loads(Path("configs/paper05/nested_length_depth.json").read_text())

def test_nested_length_information_gate():
    result,rows=validation(CONFIG)
    assert result["passed"] and len(rows)==15
    assert all(r["max_singleton_MI_bits"]<1e-9 and abs(r["full_pattern_MI_bits"]-2)<1e-9 for r in rows)

def test_nested_length_matrix_is_complete_and_fixed_length():
    rows=evaluation(CONFIG);expected=3*5*4*2*2*CONFIG["evaluation_examples_per_cell"]
    assert len(rows)==expected
    assert all(len(r["tokens"])==CONFIG["sequence_length"] for r in rows)
    assert {r["pattern_length"] for r in rows}=={2,3,4,6,8}
    assert {r["nuisance_count"] for r in rows}=={0,2,4,8}
