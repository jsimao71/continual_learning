import numpy as np

from cl.manual_transformers.m1_root import build_root_variants, evaluate as eval_root
from cl.manual_transformers.m1_implication import build_implication_variants, evaluate as eval_implication
from cl.manual_transformers.m1_recurrence import generate


def test_root_recurrence_exhausts_all_permutations_starts_and_depths():
    variants = build_root_variants()
    rows, steps = eval_root(*variants["sa_only"])
    assert len(rows) == 24 and {r["depth_to_root"] for r in rows} == {0, 1, 2, 3}
    assert all(r["correct"] for r in rows)
    assert min(s["attention_probability_margin"] for s in steps) > .999998
    assert not all(r["correct"] for r in eval_root(*variants["ff_only"])[0])
    assert all(r["correct"] for r in eval_root(*variants["sa_ff"])[0])


def test_implication_recurrence_exhausts_all_chains_starts_and_lengths():
    variants = build_implication_variants()
    rows, steps = eval_implication(*variants["sa_only"])
    assert len(rows) == 96 and {r["chain_steps"] for r in rows} == {1, 2, 3, 4}
    assert all(r["correct"] for r in rows)
    assert min(s["logit_margin"] for s in steps) > .999998
    assert not all(r["correct"] for r in eval_implication(*variants["ff_only"])[0])
    assert all(r["correct"] for r in eval_implication(*variants["sa_ff"])[0])


def test_canonical_root_has_a_complete_trace_at_every_generated_step():
    model, vocab = build_root_variants()["sa_only"]
    generated, steps = generate(model, vocab, {"R":"STOP", "A":"R", "B":"A", "C":"B"}, "C", "STOP")
    assert generated == ["B", "A", "R", "STOP"] and len(steps) == 4
    required = {"embeddings", "layer1_Q", "layer1_K", "layer1_QK_T_raw", "layer1_scores", "layer1_probabilities", "layer1_V", "layer1_head_output", "layer1_post_sa_residual", "layer1_ff_preactivation", "layer1_ff_activation", "layer1_post_ff_residual", "final_logits"}
    assert all(required <= set(step["trace"]) for step in steps)
    assert all(np.isfinite(step["trace"]["final_logits"]).all() for step in steps)
