"""Controlled multi-hop bridge graphs with known candidate utility."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import candidate_features


@dataclass(frozen=True)
class BridgeExample:
    example_id: str
    split: str
    seed: int
    hop_count: int
    lexical_overlap: str
    bridge_strength: float
    candidate_tokens: int
    evidence: tuple[int, ...]
    bridge_candidate: int
    features: np.ndarray

    def quality(self, selected) -> tuple[float, float, float]:
        chosen = set(selected)
        recall = len(chosen.intersection(self.evidence)) / len(self.evidence)
        complete = float(set(self.evidence).issubset(chosen))
        return 0.5 * recall + 0.5 * complete, recall, complete

    def utility(self) -> np.ndarray:
        full = tuple(range(len(self.features)))
        full_quality = self.quality(full)[0]
        return np.asarray(
            [full_quality - self.quality(tuple(j for j in full if j != index))[0] for index in full]
        )


def _example(seed: int, index: int, split: str, candidates: int, layers: int, heads: int) -> BridgeExample:
    rng = np.random.default_rng(seed * 100_003 + index)
    hop_count = 2 + (index % 2)
    communities = np.repeat(np.arange(3), int(np.ceil(candidates / 3)))[:candidates]
    evidence = tuple(int(group * (candidates // 3) + rng.integers(max(candidates // 3 - 1, 1))) for group in range(hop_count))
    evidence = tuple(dict.fromkeys(evidence))
    while len(evidence) < hop_count:
        candidate = int(rng.integers(candidates))
        if candidate not in evidence:
            evidence += (candidate,)
    bridge_candidate = evidence[1]
    lexical_overlap = "high" if index % 3 else "low"
    bridge_strength = (0.16, 0.28, 0.40)[index % 3]
    semantic = rng.normal(0.0, 0.45, candidates)
    lexical = rng.normal(0.0, 0.45, candidates)
    for candidate in evidence:
        semantic[candidate] += 1.0
        lexical[candidate] += 0.75 if lexical_overlap == "high" else 0.15
    distractor = int(np.argmax(lexical + np.isin(np.arange(candidates), evidence) * -10.0))
    lexical[distractor] += 1.1
    base = 0.62 * semantic + 0.38 * lexical + rng.normal(0.0, 0.18, candidates)
    base[bridge_candidate] -= 0.85
    scores = np.empty((layers, heads, candidates), dtype=np.float64)
    adjacency = np.empty((layers, heads, candidates, candidates), dtype=np.float64)
    for layer in range(layers):
        for head in range(heads):
            scores[layer, head] = base + rng.normal(0.0, 0.22, candidates)
            scores[layer, head, list(evidence)] += 0.30 + 0.06 * layer
            scores[layer, head, bridge_candidate] += 0.18 * (head % 2)
            graph = rng.uniform(0.0, 0.08, (candidates, candidates))
            graph += (communities[:, None] == communities[None, :]) * rng.uniform(0.25, 0.45)
            for left, right in zip(evidence[:-1], evidence[1:]):
                graph[left, right] = graph[right, left] = bridge_strength + 0.025 * layer
            graph = (graph + graph.T) / 2.0
            np.fill_diagonal(graph, 0.0)
            adjacency[layer, head] = graph
    features = candidate_features(base, lexical, semantic, scores, adjacency, communities)
    return BridgeExample(
        example_id=f"bridge-{seed}-{index:04d}",
        split=split,
        seed=seed,
        hop_count=hop_count,
        lexical_overlap=lexical_overlap,
        bridge_strength=bridge_strength,
        candidate_tokens=32,
        evidence=evidence,
        bridge_candidate=bridge_candidate,
        features=features,
    )


def build_bridge_suite(seed: int, validation: int = 80, test: int = 120, candidates: int = 24) -> tuple[BridgeExample, ...]:
    return tuple(
        _example(seed, index, "validation" if index < validation else "test", candidates, 6, 4)
        for index in range(validation + test)
    )
