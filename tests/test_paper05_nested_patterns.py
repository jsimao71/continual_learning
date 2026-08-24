import json,itertools
from pathlib import Path
from cl.experiments.paper05_nested_patterns import evaluation_pairs,validate

CONFIG=json.loads(Path("configs/paper05/nested_patterns.json").read_text())

def test_nested_no_shortcut_validation_passes():
    result,checks=validate(CONFIG)
    assert result["passed"]
    assert all(r["max_singleton_MI_bits"]<1e-9 and r["max_proper_subset_MI_bits"]<1e-9 and abs(r["full_pattern_MI_bits"]-2)<1e-9 for r in checks)

def test_nested_matrix_and_controls_are_complete():
    rows=evaluation_pairs(CONFIG);expected=5*3*3*2*CONFIG["evaluation_families_per_cell"]
    assert len(rows)==expected
    assert {(r["relation_type"],r["base_length"],r["extension_length"],r["position_mode"]) for r in rows}==set(itertools.product(CONFIG["relation_types"],CONFIG["base_lengths"],CONFIG["extension_lengths"],CONFIG["positions"]))
    assert all(len(r[name])==CONFIG["sequence_length"] for r in rows for name in ("short_tokens","long_tokens","same_target_tokens","nonequivalent_tokens"))
    assert all(r["short_target"]==r["long_target"] for r in rows if r["relation_type"] in {"irrelevant_extension","supportive_extension"})
    assert all(r["short_target"]!=r["long_target"] for r in rows if r["relation_type"]=="override_extension")
