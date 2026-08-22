import numpy as np

from cl.common.metrics import (
    bootstrap_ci,
    cosine,
    entropy,
    invariant_score,
    linear_cka,
    onset_persistence,
    residual_geometry,
    spearman,
)


def test_metric_formulas_on_toy_values():
    assert entropy([1, 0]) == 0
    assert abs(entropy([0.5, 0.5]) - np.log(2)) < 1e-12
    assert cosine([1, 0], [0, 1]) == 0
    geometry = residual_geometry([1, 0], [1, 0])
    assert geometry["candidate_cosine"] == 1
    assert geometry["candidate_update_ratio"] == 1
    assert spearman([1, 2, 3], [10, 20, 30]) == 1
    assert abs(linear_cka(np.eye(3), np.eye(3)) - 1) < 1e-12
    assert invariant_score(np.ones((2, 3)), np.ones((2, 3))) == 1


def test_bootstrap_and_persistence_are_deterministic():
    assert bootstrap_ci([1, 2, 3], samples=50, seed=4) == bootstrap_ci([1, 2, 3], samples=50, seed=4)
    assert onset_persistence([False, True, True, False, True]) == {
        "onset": 1,
        "persistence": 2,
        "reentries": [4],
    }

