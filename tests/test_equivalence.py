import numpy as np

from cl.analysis.equivalence import common_component_metrics, distribution_metrics, jensen_shannon_bits, topk_overlap


def test_distribution_metrics_use_bits_and_rank():
    row = distribution_metrics([0.5, 0.25, 0.25, 0.0], 1)
    assert row["entropy_bits"] == 1.5
    assert row["target_surprisal_bits"] == 2.0
    assert row["target_rank"] == 2


def test_equivalence_distribution_distances():
    assert jensen_shannon_bits([1, 0], [1, 0]) == 0
    assert jensen_shannon_bits([1, 0], [0, 1]) == 1
    assert topk_overlap([3, 2, 1], [3, 1, 2], 2) == 0.5


def test_common_component_separates_orthogonal_families():
    rows = [
        {"family": "a", "vector": [1, 0]}, {"family": "a", "vector": [2, 0]},
        {"family": "b", "vector": [0, 1]}, {"family": "b", "vector": [0, 2]},
    ]
    result = common_component_metrics(rows, ("family",))
    assert all(row["angular_concentration"] == 1 for row in result)
    assert all(np.isclose(row["between_cosine_distance"], 1) for row in result)
