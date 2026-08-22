"""Matched-budget frozen selectors for Paper 1."""

from __future__ import annotations

import numpy as np

from .features import FEATURE_NAMES
from .config import NSCConfig


SELECTORS = (
    "base_topk",
    "budget_tuned_topk",
    "power_sharpen",
    "entropy_adaptive_sharpen",
    "persistence",
    "agreement",
    "community",
    "bridge_preserving",
    "combined_structural",
    "oracle_evidence",
)


def power_sharpen(scores, gamma: float) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    shifted = values - values.min() + 1e-9
    sharpened = shifted**gamma
    return sharpened / sharpened.sum()


def _topk(scores, budget: int) -> tuple[int, ...]:
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    return tuple(sorted(int(value) for value in order[:budget]))


def select_candidates(
    features,
    budget: int,
    mode: str,
    *,
    combined_weights=None,
    evidence=None,
    gamma: float = 2.0,
    bridge_fraction: float = 0.25,
) -> tuple[int, ...]:
    matrix = np.asarray(features, dtype=np.float64)
    if budget < 1 or budget > len(matrix):
        raise ValueError("budget must be within the candidate count")
    columns = {name: index for index, name in enumerate(FEATURE_NAMES)}
    base = matrix[:, columns["base_score"]]
    if mode in {"base_topk", "budget_tuned_topk"}:
        return _topk(base, budget)
    if mode == "power_sharpen":
        return _topk(power_sharpen(base, gamma), budget)
    if mode == "entropy_adaptive_sharpen":
        probabilities = power_sharpen(base, 1.0)
        normalized_entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, None))) / np.log(len(base))
        return _topk(power_sharpen(base, 1.0 + 2.0 * normalized_entropy), budget)
    if mode in {"persistence", "agreement", "community"}:
        feature = {"persistence": "persistence", "agreement": "agreement", "community": "community_centrality"}[mode]
        return _topk(base + matrix[:, columns[feature]], budget)
    if mode == "bridge_preserving":
        reserve = min(budget, max(1, int(round(budget * bridge_fraction))))
        bridge = _topk(matrix[:, columns["bridge_score"]], reserve)
        remaining = [index for index in np.argsort(-base, kind="mergesort") if int(index) not in bridge]
        return tuple(sorted((*bridge, *(int(value) for value in remaining[: budget - reserve]))))
    if mode == "combined_structural":
        if combined_weights is None:
            raise ValueError("combined_structural requires fitted weights")
        return _topk(np.column_stack([np.ones(len(matrix)), matrix]) @ combined_weights, budget)
    if mode == "oracle_evidence":
        if evidence is None:
            raise ValueError("oracle_evidence requires labels")
        label = np.asarray(evidence, dtype=np.float64)
        return _topk(label * 1e6 + base, budget)
    raise ValueError(f"unknown selector: {mode}")


def route_with_config(features, budget: int, config: NSCConfig, **kwargs) -> tuple[int, ...]:
    """Opt-in adapter boundary; disabled NSC is exactly base top-k."""
    mode = config.selector.mode if config.enabled else "base_topk"
    return select_candidates(
        features,
        budget,
        mode,
        gamma=config.selector.gamma,
        bridge_fraction=config.selector.bridge_budget_fraction,
        **kwargs,
    )
