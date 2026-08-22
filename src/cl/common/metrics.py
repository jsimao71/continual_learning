"""Metric ontology shared by the n-gram and abstraction papers."""

from __future__ import annotations

import math
from typing import Callable, Iterable

import numpy as np


def entropy(probabilities: Iterable[float], eps: float = 1e-12) -> float:
    p = np.asarray(list(probabilities), dtype=np.float64)
    if p.ndim != 1 or np.any(p < 0):
        raise ValueError("probabilities must be a non-negative vector")
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p / total
    return float(-(p * np.log(np.clip(p, eps, None))).sum())


def cosine(left, right, eps: float = 1e-12) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("vectors must have identical shapes")
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), eps))


def residual_geometry(state, candidate_update, effective_update=None) -> dict[str, float]:
    x = np.asarray(state, dtype=np.float64)
    delta = np.asarray(candidate_update, dtype=np.float64)
    effective = delta if effective_update is None else np.asarray(effective_update, dtype=np.float64)
    state_norm = float(np.linalg.norm(x))
    candidate_norm = float(np.linalg.norm(delta))
    effective_norm = float(np.linalg.norm(effective))
    return {
        "residual_norm": state_norm,
        "candidate_update_norm": candidate_norm,
        "effective_update_norm": effective_norm,
        "candidate_update_ratio": candidate_norm / max(state_norm, 1e-12),
        "effective_update_ratio": effective_norm / max(state_norm, 1e-12),
        "candidate_cosine": cosine(x, delta),
        "effective_cosine": cosine(x, effective),
    }


def bootstrap_ci(
    values: Iterable[float],
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    samples: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("bootstrap requires at least one value")
    rng = np.random.default_rng(seed)
    draws = np.asarray([
        statistic(rng.choice(array, size=array.size, replace=True)) for _ in range(samples)
    ])
    alpha = (1.0 - confidence) / 2.0
    return float(statistic(array)), float(np.quantile(draws, alpha)), float(np.quantile(draws, 1 - alpha))


def pairwise_cosine_distance(matrix) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    normalized = x / np.clip(norms, 1e-12, None)
    return 1.0 - normalized @ normalized.T


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(left, right) -> float:
    a = _rankdata(np.asarray(left, dtype=np.float64).reshape(-1))
    b = _rankdata(np.asarray(right, dtype=np.float64).reshape(-1))
    if a.size != b.size or a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def linear_cka(left, right) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    numerator = np.linalg.norm(y.T @ x, ord="fro") ** 2
    denominator = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    return float(numerator / max(denominator, 1e-12))


def effective_rank(matrix) -> float:
    x = np.asarray(matrix, dtype=np.float64)
    covariance = np.cov(x, rowvar=False)
    eigenvalues = np.clip(np.linalg.eigvalsh(np.atleast_2d(covariance)), 0.0, None)
    return float(eigenvalues.sum() ** 2 / max(np.square(eigenvalues).sum(), 1e-12))


def invariant_score(original, transformed, normalization: float | None = None) -> float:
    distances = np.linalg.norm(np.asarray(original) - np.asarray(transformed), axis=-1)
    z = normalization if normalization is not None else float(np.mean(np.linalg.norm(original, axis=-1)))
    return float(1.0 - np.mean(distances) / max(z, 1e-12))


def onset_persistence(mask: Iterable[bool]) -> dict[str, int | list[int] | None]:
    values = [bool(value) for value in mask]
    onset = next((index for index, value in enumerate(values) if value), None)
    longest = 0
    current = 0
    entries: list[int] = []
    previous = False
    for index, value in enumerate(values):
        current = current + 1 if value else 0
        longest = max(longest, current)
        if value and index > 0 and not previous:
            entries.append(index)
        previous = value
    return {"onset": onset, "persistence": longest, "reentries": entries[1:] if onset is not None else []}


def classify_update_relation(cosine_value: float, threshold: float = 0.25) -> str:
    if cosine_value <= -threshold:
        return "opposing_candidate"
    if cosine_value >= threshold:
        return "aligned_candidate"
    return "orthogonal_candidate"
