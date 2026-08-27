import numpy as np
import csv

from cl.manual_transformers.m1_root import build_root_variants, evaluate as eval_root
from cl.manual_transformers.m1_implication import build_implication_variants, evaluate as eval_implication
from cl.manual_transformers.m1_recurrence import generate
from cl.manual_transformers.run_milestone2 import main as generate_artifacts


def test_root_recurrence_exhausts_all_permutations_starts_and_depths():
    variants = build_root_variants()
    rows, steps = eval_root(*variants["sa_only"])
    assert len(rows) == 24 and {r["depth_to_root"] for r in rows} == {0, 1, 2, 3}
    assert all(r["correct"] for r in rows)
    assert min(s["attention_probability_margin"] for s in steps) > .999998
    assert not all(r["correct"] for r in eval_root(*variants["ff_only"])[0])
    assert all(r["correct"] for r in eval_root(*variants["sa_ff"])[0])


def test_unary_chain_recurrence_exhausts_all_chains_starts_and_lengths():
    variants = build_implication_variants()
    rows, steps = eval_implication(*variants["sa_only"])
    assert len(rows) == 96 and {r["chain_steps"] for r in rows} == {1, 2, 3, 4}
    assert all(r["correct"] for r in rows)
    assert min(s["signed_target_margin"] for s in steps) > .999998
    assert min(s["winner_runner_up_margin"] for s in steps) > .999998
    assert not all(r["correct"] for r in eval_implication(*variants["ff_only"])[0])
    assert all(r["correct"] for r in eval_implication(*variants["sa_ff"])[0])


def test_canonical_root_has_a_complete_trace_at_every_generated_step():
    model, vocab = build_root_variants()["sa_only"]
    generated, steps = generate(model, vocab, {"R":"STOP", "A":"R", "B":"A", "C":"B"}, "C", "STOP")
    assert generated == ["B", "A", "R", "STOP"] and len(steps) == 4
    required = {"embeddings", "layer1_Q", "layer1_K", "layer1_QK_T_raw", "layer1_scores", "layer1_probabilities", "layer1_V", "layer1_head_output", "layer1_post_sa_residual", "layer1_ff_preactivation", "layer1_ff_activation", "layer1_post_ff_residual", "final_logits"}
    assert all(required <= set(step["trace"]) for step in steps)
    assert all(np.isfinite(step["trace"]["final_logits"]).all() for step in steps)


def test_context_order_controls_and_signed_failure_margin():
    model, vocab = build_root_variants()["sa_only"]
    mapping = {"R":"STOP", "A":"R", "B":"A", "C":"B"}
    expected = ["B", "A", "R", "STOP"]
    for order in (tuple(reversed(tuple(mapping))), ("A", "C", "R", "B")):
        generated, steps = generate(model, vocab, mapping, "C", "STOP", context_order=order)
        assert generated == expected and min(s["signed_target_margin"] for s in steps) > 0
    failed_model, failed_vocab = build_root_variants()["ff_only"]
    _, failed_steps = generate(failed_model, failed_vocab, mapping, "C", "STOP")
    assert failed_steps[0]["winner_runner_up_margin"] == 0
    assert failed_steps[0]["signed_target_margin"] == 0


def test_artifact_raw_summary_and_canonical_chain_trace_agree(tmp_path):
    # Seed the cumulative M0 files exactly as the documented regeneration order does.
    from cl.manual_transformers.run_milestone1 import main as generate_m0
    generate_m0(tmp_path); generate_artifacts(tmp_path)
    summary = list(csv.DictReader((tmp_path / "manual_autoregressive_summary.csv").open()))
    primary = next(r for r in summary if r["task"] == "unary_chain_recurrence" and r["topology"] == "sa_only")
    raw = list(csv.DictReader((tmp_path / "unary_chain_recurrence/sa_only/legal_domain_results.csv").open()))
    steps = list(csv.DictReader((tmp_path / "unary_chain_recurrence/sa_only/generation_steps.csv").open()))
    assert int(primary["legal_cases"]) == len(raw) == 96
    assert int(primary["correct_cases"]) == sum(r["correct"] == "True" for r in raw) == 96
    assert np.isclose(float(primary["minimum_signed_target_margin"]), min(float(r["signed_target_margin"]) for r in steps))
    trace_root = tmp_path / "unary_chain_recurrence/sa_only/canonical_steps"
    for step in range(1, 5):
        names = {p.stem for p in (trace_root / f"step_{step:02d}/canonical_trace").glob("*.csv")}
        assert {"embeddings", "layer1_Q", "layer1_K", "layer1_QK_T_raw", "layer1_scores", "layer1_probabilities", "layer1_V", "layer1_head_output", "layer1_post_sa_residual", "layer1_ff_preactivation", "layer1_ff_activation", "layer1_post_ff_residual", "final_logits"} <= names
    controls = list(csv.DictReader((tmp_path / "unary_chain_recurrence/serialization_controls.csv").open()))
    assert len(controls) == 192 and all(r["correct"] == "True" for r in controls)
