"""Distributional and residual metrics for predictive/semantic equivalence tests."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from cl.common.metrics import cosine, effective_rank, pairwise_cosine_distance


def distribution_metrics(probabilities, target: int) -> dict[str, float | int]:
    """Return the shared entropy/confidence ontology using base-2 units."""
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if p.size < 2 or np.any(p < 0) or p.sum() <= 0:
        raise ValueError("probabilities must be a non-negative vector with at least two entries")
    if not 0 <= target < p.size:
        raise ValueError("target outside vocabulary")
    p = p / p.sum()
    entropy_bits = float(-(p * np.log2(np.clip(p, 1e-300, None))).sum())
    ordered = np.argsort(-p, kind="stable")
    return {
        "entropy_bits": entropy_bits,
        "normalized_entropy": entropy_bits / math.log2(p.size),
        "effective_vocab_fraction": float(2.0**entropy_bits / p.size),
        "target_surprisal_bits": float(-math.log2(max(p[target], 1e-300))),
        "target_probability": float(p[target]),
        "target_rank": int(np.flatnonzero(ordered == target)[0] + 1),
        "top1_margin": float(p[ordered[0]] - p[ordered[1]]),
    }


def jensen_shannon_bits(left, right) -> float:
    """Jensen--Shannon divergence in bits (bounded by one)."""
    p = np.asarray(left, dtype=np.float64).reshape(-1)
    q = np.asarray(right, dtype=np.float64).reshape(-1)
    if p.shape != q.shape or np.any(p < 0) or np.any(q < 0) or p.sum() <= 0 or q.sum() <= 0:
        raise ValueError("distributions must be non-negative vectors of identical shape")
    p, q = p / p.sum(), q / q.sum()
    midpoint = (p + q) / 2
    def kl(a):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / midpoint[mask])))
    return (kl(p) + kl(q)) / 2


def topk_overlap(left, right, k: int = 5) -> float:
    p, q = np.asarray(left).reshape(-1), np.asarray(right).reshape(-1)
    if p.shape != q.shape:
        raise ValueError("vectors must have identical shape")
    k = min(max(int(k), 1), p.size)
    return len(set(np.argsort(-p)[:k]) & set(np.argsort(-q)[:k])) / k


def common_component_metrics(rows: list[dict], group_fields: tuple[str, ...]) -> list[dict]:
    """Measure centroid energy, alignment, rank, and within/between geometry."""
    groups: dict[tuple, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(np.asarray(row["vector"], dtype=np.float64))
    centroids = {key: np.mean(vectors, axis=0) for key, vectors in groups.items()}
    output = []
    for key, vectors in sorted(groups.items(), key=lambda item: str(item[0])):
        matrix = np.asarray(vectors)
        centroid = centroids[key]
        total = float(np.square(matrix).sum())
        residual = float(np.square(matrix - centroid).sum())
        within = pairwise_cosine_distance(matrix)
        between = [1.0 - cosine(vector, other) for other_key, other in centroids.items()
                   if other_key != key and np.asarray(other).shape == centroid.shape for vector in matrix]
        output.append({
            **dict(zip(group_fields, key)),
            "n": len(matrix),
            "common_energy_fraction": 1.0 - residual / max(total, 1e-12),
            "angular_concentration": float(np.mean([cosine(vector, centroid) for vector in matrix])),
            "effective_rank": effective_rank(matrix) if len(matrix) > 1 else 1.0,
            "within_cosine_distance": float(within[np.triu_indices(len(matrix), 1)].mean()) if len(matrix) > 1 else 0.0,
            "between_cosine_distance": float(np.mean(between)) if between else 0.0,
        })
    return output
