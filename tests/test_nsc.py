import numpy as np

from cl.nsc.config import NSCConfig, SelectorConfig
from cl.nsc.features import candidate_features
from cl.nsc.graph import bridge_scores, canonical_components
from cl.nsc.selectors import power_sharpen, route_with_config, select_candidates
from cl.nsc.synthetic import build_bridge_suite
from cl.nsc.natural import NaturalCandidate, NaturalExample, identity_disjoint
from cl.nsc.qwen_natural import compact_candidates, lexical_score


def toy_features():
    base = np.array([0.9, 0.7, 0.3, 0.2])
    scores = np.stack([[base, base + np.array([0.0, 0.1, -0.1, 0.0])]])
    graph = np.zeros((1, 2, 4, 4))
    graph[:, :, 0, 1] = graph[:, :, 1, 0] = 0.8
    graph[:, :, 1, 2] = graph[:, :, 2, 1] = 0.4
    graph[:, :, 2, 3] = graph[:, :, 3, 2] = 0.8
    return candidate_features(base, base, base, scores, graph, np.array([0, 0, 1, 1]))


def test_power_sharpen_normalizes_and_gamma_one_is_identity_up_to_shift():
    scores = np.array([0.2, 0.4, 0.8])
    sharpened = power_sharpen(scores, 1.0)
    expected = (scores - scores.min() + 1e-9)
    expected /= expected.sum()
    assert np.allclose(sharpened, expected)
    assert np.isclose(sharpened.sum(), 1.0)
    assert np.all(np.diff(sharpened) > 0)


def test_components_and_bridge_score_on_toy_graph():
    adjacency = np.array([[0, 1, 0, 0], [1, 0, .4, 0], [0, .4, 0, 1], [0, 0, 1, 0.]])
    labels = canonical_components(adjacency, threshold=.5)
    assert labels.tolist() == [0, 0, 1, 1]
    scores = bridge_scores(adjacency, labels)
    assert scores[1] > scores[0]
    assert scores[2] > scores[3]


def test_budget_and_determinism_for_all_non_oracle_selectors():
    features = toy_features()
    weights = np.zeros(features.shape[1] + 1)
    weights[1] = 1.0
    for mode in (
        "base_topk", "budget_tuned_topk", "power_sharpen",
        "entropy_adaptive_sharpen", "persistence", "agreement",
        "community", "bridge_preserving", "combined_structural",
    ):
        kwargs = {"combined_weights": weights} if mode == "combined_structural" else {}
        first = select_candidates(features, 2, mode, **kwargs)
        second = select_candidates(features, 2, mode, **kwargs)
        assert first == second
        assert len(first) == 2


def test_disabled_nsc_exactly_matches_base_selection():
    features = toy_features()
    disabled = NSCConfig(enabled=False, selector=SelectorConfig(mode="bridge_preserving"))
    assert route_with_config(features, 2, disabled) == select_candidates(features, 2, "base_topk")


def test_bridge_suite_is_split_deterministic_and_has_known_utility():
    left = build_bridge_suite(11, validation=2, test=2)
    right = build_bridge_suite(11, validation=2, test=2)
    assert [row.example_id for row in left] == [row.example_id for row in right]
    assert np.allclose(left[0].features, right[0].features)
    utility = left[0].utility()
    assert np.count_nonzero(utility) == left[0].hop_count
    assert all(utility[index] > 0 for index in left[0].evidence)


def test_natural_identity_split_and_candidate_compaction():
    candidates = tuple(
        NaturalCandidate(str(index), "title", f"candidate text {index}", index in {7, 8})
        for index in range(10)
    )
    validation = NaturalExample("toy", "a", "paper-a", "validation", "which text", "answer", candidates)
    test = NaturalExample("toy", "b", "paper-b", "test", "which text", "answer", candidates)
    assert identity_disjoint((validation, test))
    compact = compact_candidates(validation, maximum=4)
    assert len(compact) == 4
    assert {value.candidate_id for value in compact if value.is_evidence} == {"7", "8"}
    assert lexical_score("which text", "candidate text") > lexical_score("which text", "unrelated words")
