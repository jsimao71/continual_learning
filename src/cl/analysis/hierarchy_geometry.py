"""Hierarchy geometry metrics grounded in the shared residual ontology."""

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from cl.common.metrics import cosine, effective_rank, pairwise_cosine_distance, spearman
from cl.semantic.hierarchy import Hierarchy


def _upper_triangle(matrix: np.ndarray) -> np.ndarray:
    indices = np.triu_indices(matrix.shape[0], k=1)
    return matrix[indices]


def hierarchy_metrics(
    hierarchy: Hierarchy,
    representations: list[dict],
    *,
    seed: int = 0,
) -> list[dict]:
    """Aggregate at semantic item before pairwise inference."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in representations:
        groups[(row["model_setting"], row["seed"], row["layer"], row["location"], row["family"], row["natural"], row["abstraction_level"])].append(row)
    output = []
    for key, rows in sorted(groups.items()):
        by_item: dict[str, list[np.ndarray]] = defaultdict(list)
        for row in rows:
            by_item[row["item_id"]].append(np.asarray(row["vector"], dtype=np.float64))
        item_ids = sorted(by_item)
        means = np.asarray([np.mean(by_item[item_id], axis=0) for item_id in item_ids])
        activation_distance = pairwise_cosine_distance(means)
        tree_distance = np.asarray([[hierarchy.distance(a, b) for b in item_ids] for a in item_ids], dtype=np.float64)
        within, between = [], []
        correct_neighbors = 0
        for i, left in enumerate(item_ids):
            parent = hierarchy.items[left].parent_id
            nearest = min((j for j in range(len(item_ids)) if j != i), key=lambda j: activation_distance[i, j])
            correct_neighbors += int(hierarchy.items[item_ids[nearest]].parent_id == parent)
            for j in range(i + 1, len(item_ids)):
                same_parent = hierarchy.items[item_ids[j]].parent_id == parent
                (within if same_parent else between).append(activation_distance[i, j])
        within_mean, between_mean = float(np.mean(within)), float(np.mean(between))
        template_cosines = []
        for item_id, vectors in by_item.items():
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    template_cosines.append(cosine(vectors[i], vectors[j]))

        # Negative control: permute parent identities within the matched family/domain.
        rng = random.Random(seed + int(key[2]))
        parent_labels = [hierarchy.items[item_id].parent_id for item_id in item_ids]
        rng.shuffle(parent_labels)
        permuted_hits = 0
        for i in range(len(item_ids)):
            nearest = min((j for j in range(len(item_ids)) if j != i), key=lambda j: activation_distance[i, j])
            permuted_hits += int(parent_labels[nearest] == parent_labels[i])

        output.append(
            {
                "model_setting": key[0],
                "seed": key[1],
                "layer": key[2],
                "location": key[3],
                "family": key[4],
                "natural": key[5],
                "abstraction_level": key[6],
                "n_items": len(item_ids),
                "within_parent_dispersion": within_mean,
                "between_parent_dispersion": between_mean,
                "normalized_separation": (between_mean - within_mean) / max(between_mean, 1e-12),
                "hierarchy_rsa_spearman": spearman(_upper_triangle(activation_distance), _upper_triangle(tree_distance)),
                "tree_neighbor_recovery": correct_neighbors / len(item_ids),
                "permuted_neighbor_recovery": permuted_hits / len(item_ids),
                "cross_template_cosine": float(np.mean(template_cosines)),
                "effective_rank": effective_rank(means),
            }
        )
    return output


def shared_update_metrics(hierarchy: Hierarchy, update_rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in update_rows:
        parent = hierarchy.items[row["item_id"]].parent_id
        groups[(row["model_setting"], row["seed"], row["layer"], row["component"], row["family"], row["natural"], parent)].append(np.asarray(row["vector"], dtype=np.float64))
    output = []
    for key, vectors in sorted(groups.items()):
        matrix = np.asarray(vectors)
        centroid = matrix.mean(axis=0)
        total = np.square(matrix).sum()
        residual = np.square(matrix - centroid).sum()
        output.append(
            {
                "model_setting": key[0],
                "seed": key[1],
                "layer": key[2],
                "component": key[3],
                "family": key[4],
                "natural": key[5],
                "parent_id": key[6],
                "shared_update_explained_variance": 1.0 - residual / max(total, 1e-12),
                "shared_update_rank": effective_rank(matrix),
                "mean_update_cosine": float(np.mean([cosine(vector, centroid) for vector in matrix])),
            }
        )
    return output
