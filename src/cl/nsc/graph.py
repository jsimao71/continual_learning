"""Small deterministic graph operations used outside model hot paths."""

from __future__ import annotations

import numpy as np


def canonical_components(adjacency, threshold: float = 0.5) -> np.ndarray:
    """Return deterministic connected-component labels in first-node order."""
    graph = np.asarray(adjacency, dtype=np.float64)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("adjacency must be square")
    links = np.maximum(graph, graph.T) >= threshold
    np.fill_diagonal(links, True)
    labels = np.full(graph.shape[0], -1, dtype=np.int64)
    label = 0
    for root in range(graph.shape[0]):
        if labels[root] >= 0:
            continue
        stack = [root]
        labels[root] = label
        while stack:
            node = stack.pop()
            for neighbor in np.flatnonzero(links[node]):
                if labels[neighbor] < 0:
                    labels[neighbor] = label
                    stack.append(int(neighbor))
        label += 1
    return labels


def bridge_scores(adjacency, communities) -> np.ndarray:
    """Cross-community incident mass, normalized by total incident mass."""
    graph = np.asarray(adjacency, dtype=np.float64)
    labels = np.asarray(communities)
    if graph.shape != (len(labels), len(labels)):
        raise ValueError("community labels must match adjacency")
    graph = np.maximum(graph, graph.T).copy()
    np.fill_diagonal(graph, 0.0)
    cross = labels[:, None] != labels[None, :]
    return (graph * cross).sum(axis=1) / np.clip(graph.sum(axis=1), 1e-12, None)


def community_centrality(adjacency, communities) -> np.ndarray:
    graph = np.asarray(adjacency, dtype=np.float64)
    labels = np.asarray(communities)
    same = labels[:, None] == labels[None, :]
    np.fill_diagonal(same, False)
    counts = same.sum(axis=1)
    return (graph * same).sum(axis=1) / np.clip(counts, 1, None)
