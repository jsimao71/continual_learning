"""Compact candidate features and dependency-free residualization."""

from __future__ import annotations

import numpy as np

from .graph import bridge_scores, community_centrality


FEATURE_NAMES = (
    "base_score",
    "lexical_score",
    "semantic_score",
    "entropy_contribution",
    "persistence",
    "agreement",
    "community_centrality",
    "bridge_score",
)


def _ranks_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def candidate_features(base, lexical, semantic, layer_head_scores, adjacency, communities):
    """Create O(LHN + N^2) compact features without retaining token attention."""
    base = np.asarray(base, dtype=np.float64)
    lexical = np.asarray(lexical, dtype=np.float64)
    semantic = np.asarray(semantic, dtype=np.float64)
    scores = np.asarray(layer_head_scores, dtype=np.float64)
    if scores.ndim != 3 or scores.shape[-1] != len(base):
        raise ValueError("layer_head_scores must have shape [layers, heads, candidates]")
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    mean_probability = probabilities.mean(axis=(0, 1))
    entropy_contribution = -mean_probability * np.log(np.clip(mean_probability, 1e-12, None))
    threshold = np.quantile(probabilities, 0.75, axis=-1, keepdims=True)
    persistence = (probabilities >= threshold).mean(axis=(0, 1))
    ranks = np.stack([_ranks_desc(row) for row in scores.reshape(-1, scores.shape[-1])])
    agreement = 1.0 - ranks.std(axis=0) / max(scores.shape[-1] - 1, 1)
    aggregated = np.asarray(adjacency, dtype=np.float64).mean(axis=(0, 1))
    matrix = np.column_stack(
        [
            base,
            lexical,
            semantic,
            entropy_contribution,
            persistence,
            agreement,
            community_centrality(aggregated, communities),
            bridge_scores(aggregated, communities),
        ]
    )
    return matrix


def standardize(train, other=None):
    train = np.asarray(train, dtype=np.float64)
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-9] = 1.0
    transformed = (train - mean) / scale
    return transformed if other is None else (transformed, (np.asarray(other) - mean) / scale)


def ridge_fit(features, target, alpha: float = 1.0) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)


def ridge_predict(features, coefficients) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    return np.column_stack([np.ones(len(x)), x]) @ np.asarray(coefficients)


def residualize(values, controls, alpha: float = 1e-6) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    controls = np.asarray(controls, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    residuals = np.empty_like(values)
    for column in range(values.shape[1]):
        fit = ridge_fit(controls, values[:, column], alpha=alpha)
        residuals[:, column] = values[:, column] - ridge_predict(controls, fit)
    return residuals
