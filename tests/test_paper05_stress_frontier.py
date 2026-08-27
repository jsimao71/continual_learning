import json
from pathlib import Path

from cl.experiments.paper05_stress_frontier import (
    architecture_plan, designed_cells, evaluation, make_example,
    measured_frontiers, parameter_matched_pairs, transformer_parameter_count, validate,
)
from cl.common.model_adapter import TinyTransformerLM
from cl.experiments.paper05_stress_run import PILOT_ARCHITECTURES, data_seed, initialization_seed, tranche

CONFIG = json.loads(Path("configs/paper05/stress_frontier.json").read_text())


def test_stress_generator_replay_and_independent_axes():
    a = make_example(CONFIG, order=4, raw_length=16, nuisance=8, span=16,
                     family="necessary_pattern", index=19, split="test")
    b = make_example(CONFIG, order=4, raw_length=16, nuisance=8, span=16,
                     family="necessary_pattern", index=19, split="test")
    assert a == b and len(a["tokens"]) == CONFIG["sequence_length"]
    assert (a["predictive_order"], a["raw_length"], a["nuisance_count"], a["dependency_span"]) == (4, 16, 8, 16)
    control = make_example(CONFIG, order=4, raw_length=24, nuisance=0, span=16,
                           family="necessary_pattern", index=19, split="test")
    assert a["target"] == control["target"] and a["family_id"] == control["family_id"]


def test_designed_matrix_records_infeasible_cells_and_balances_evaluation():
    cells, excluded = designed_cells(CONFIG)
    assert cells and excluded
    assert {r["axis"] for r in cells} == {"predictive_order", "raw_length", "nuisance", "dependency_span", "generator_family"}
    assert any(r["status"] == "span_below_predictive_order" for r in excluded)
    baseline = CONFIG["design"]
    assert any(r["axis"] == "raw_length" and r["raw_length"] == baseline["baseline_raw_length"] for r in cells)
    assert any(r["axis"] == "nuisance" and r["nuisance_count"] == baseline["baseline_nuisance"] for r in cells)
    assert any(r["axis"] == "generator_family" and r["generator_family"] == baseline["baseline_family"] for r in cells)
    rows = evaluation(CONFIG, 2)
    assert len(rows) == 2 * len(cells)
    assert validate(CONFIG)["passed"]


def test_architecture_metadata_and_parameter_matching():
    rows = architecture_plan(CONFIG)
    expected = 3 * 7 * 4 * 4 * 3
    assert len(rows) == expected
    assert all(r["model_width"] % r["head_count"] == 0 and r["parameter_count"] > 0 for r in rows)
    pairs = parameter_matched_pairs(CONFIG, rows)
    assert pairs and all(r["relative_parameter_gap"] <= CONFIG["parameter_match_relative_tolerance"] for r in pairs)
    assert all((r["depth_a"] > r["depth_b"]) != (r["width_a"] > r["width_b"]) for r in pairs)
    model = TinyTransformerLM(CONFIG["vocab_size"], CONFIG["sequence_length"], 32, 2, 2)
    assert transformer_parameter_count(CONFIG["vocab_size"], CONFIG["sequence_length"], 32, 2) == sum(p.numel() for p in model.parameters())


def test_frontier_reducer_distinguishes_failure_from_unmeasured():
    base = {"model_depth": 4, "model_width": 64, "head_count": 4,
            "training_budget": 1, "model_seed": 11}
    rows = [{**base, "axis": "predictive_order", "predictive_order": p, "accuracy": a}
            for p, a in [(1, .99), (2, .85), (3, .70)]]
    frontiers = measured_frontiers(rows)
    pstar = next(r for r in frontiers if r["frontier"] == "pstar_max")
    nmax = next(r for r in frontiers if r["frontier"] == "n_max")
    assert pstar["maximum_competent_value"] == 2 and pstar["status"] == "estimated"
    assert nmax["maximum_competent_value"] == "" and nmax["status"] == "not_measured"


def test_initial_tranche_is_three_seed_parameter_matched_contrast():
    rows = tranche(CONFIG, "pilot")
    assert len(rows) == 6
    assert {(d, w, h, b) for d, w, h, b, _ in rows} == set(PILOT_ARCHITECTURES)
    assert {seed for *_, seed in rows} == set(CONFIG["model_seeds"])
    shallow = transformer_parameter_count(CONFIG["vocab_size"], CONFIG["sequence_length"], 64, 2)
    deep = transformer_parameter_count(CONFIG["vocab_size"], CONFIG["sequence_length"], 32, 8)
    assert abs(shallow - deep) / max(shallow, deep) < CONFIG["parameter_match_relative_tolerance"]
    rescue = tranche(CONFIG, "rescue_t2")
    assert len(rescue) == 6 and {budget for _, _, _, budget, _ in rescue} == {2}
    rescue_t4 = tranche(CONFIG, "rescue_t4")
    assert len(rescue_t4) == 6 and {budget for _, _, _, budget, _ in rescue_t4} == {4}
    assert initialization_seed(2, 64, 4, 11) == 11 + 2 * 1009 + 64 * 31 + 4 * 7 + 1
    assert data_seed(2, 64, 4, 11) == 11 + 2 * 10007 + 64 * 101 + 4 * 11 + 1
