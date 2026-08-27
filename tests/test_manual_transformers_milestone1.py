import numpy as np

from cl.manual_transformers.m0_successor import build_successor_variants, evaluate as eval_successor
from cl.manual_transformers.m0_lookup import build_lookup_variants, evaluate as eval_lookup
from cl.manual_transformers.m0_grandparent import build_grandparent_variants, evaluate as eval_grandparent


def outcomes(builder, evaluator):
    return {name: all(row["correct"] for row in evaluator(model)) for name, model in builder().items()}


def test_successor_exhaustive_component_pattern():
    assert outcomes(build_successor_variants, eval_successor) == {"sa_only": False, "ff_only": True, "sa_ff": True}
    rows = eval_successor(build_successor_variants()["ff_only"])
    assert len(rows) == 3 and min(row["winner_runner_up_margin"] for row in rows) == 1.0
    assert min(row["signed_target_margin"] for row in rows) == 1.0


def test_associative_lookup_all_bijections_and_queries():
    assert outcomes(build_lookup_variants, eval_lookup) == {"sa_only": True, "ff_only": False, "sa_ff": True}
    rows = eval_lookup(build_lookup_variants()["sa_only"])
    assert len(rows) == 18
    assert min(row["attention_probability_margin"] for row in rows) > .999
    assert min(row["winner_runner_up_margin"] for row in rows) > .999
    assert min(row["signed_target_margin"] for row in rows) > .999


def test_grandparent_complete_legal_domain_and_depth_control():
    assert outcomes(build_grandparent_variants, eval_grandparent) == {
        "sa_only": True, "ff_only": False, "sa_ff": True, "sa_only_1layer": False,
    }
    rows = eval_grandparent(build_grandparent_variants()["sa_only"])
    assert len(rows) == 2
    assert min(row["layer1_parent_probability"] for row in rows) > .999
    assert min(row["layer2_grandparent_probability"] for row in rows) > .999
    assert min(row["winner_runner_up_margin"] for row in rows) > .999
    assert min(row["signed_target_margin"] for row in rows) > .999


def test_canonical_traces_are_finite_and_complete():
    cases = [
        (build_successor_variants()["ff_only"], [1]),
        (build_lookup_variants()["sa_only"], [1, 5, 6, 10]),
        (build_grandparent_variants()["sa_only"], [0, 1, 2, 4]),
    ]
    required = ("Q", "K", "QK_T_raw", "scores", "probabilities", "V", "head_output", "post_sa_residual", "ff_preactivation", "ff_activation", "post_ff_residual")
    for model, ids in cases:
        _, trace = model.forward(ids)
        assert np.isfinite(trace["embeddings"]).all()
        assert np.isfinite(trace["final_logits"]).all()
        for layer in range(1, len(model.layers)+1):
            for field in required:
                value = trace[f"layer{layer}_{field}"]
                if field == "scores":
                    assert np.isfinite(value[np.isfinite(value)]).all()
                else:
                    assert np.isfinite(value).all()
