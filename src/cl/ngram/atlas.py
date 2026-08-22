"""Token n-gram counting, continuation statistics, and deterministic strata."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import math
import random
from typing import Iterable, Sequence

from cl.common.artifacts import stable_hash
from cl.common.metrics import entropy


@dataclass(frozen=True)
class AtlasEntry:
    n: int
    prefix_token_ids: tuple[int, ...]
    decoded_prefix: str
    reference_corpus_frequency: int
    continuation_counts: dict[int, int]
    continuation_probabilities: dict[int, float]
    continuation_entropy: float
    top_continuation: int
    top_continuation_probability: float
    association_log_ratio: float
    distinct_continuations: int
    source_corpus: str
    split: str
    atlas_hash: str

    def as_dict(self) -> dict:
        value = asdict(self)
        value["prefix_token_ids"] = list(self.prefix_token_ids)
        return value


def build_atlas(
    sequences: Iterable[Sequence[int]],
    *,
    n_values: Sequence[int] = (1, 2, 3, 4, 5, 8),
    source_corpus: str = "controlled-ngram-v1",
    split: str = "train",
) -> list[AtlasEntry]:
    materialized = [tuple(int(token) for token in sequence) for sequence in sequences]
    continuations: dict[tuple[int, tuple[int, ...]], Counter[int]] = defaultdict(Counter)
    marginal = Counter()
    total_targets = 0
    for sequence in materialized:
        for token in sequence[1:]:
            marginal[token] += 1
            total_targets += 1
        for n in n_values:
            for end in range(n, len(sequence)):
                continuations[(n, sequence[end - n : end])][sequence[end]] += 1
    version = stable_hash({"sequences": materialized, "n_values": list(n_values), "source": source_corpus})
    entries = []
    for (n, prefix), counts in sorted(continuations.items()):
        frequency = sum(counts.values())
        probabilities = {token: count / frequency for token, count in sorted(counts.items())}
        top_token, top_count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        top_probability = top_count / frequency
        marginal_probability = marginal[top_token] / max(total_targets, 1)
        entries.append(
            AtlasEntry(
                n=n,
                prefix_token_ids=prefix,
                decoded_prefix=" ".join(f"t{token}" for token in prefix),
                reference_corpus_frequency=frequency,
                continuation_counts=dict(sorted(counts.items())),
                continuation_probabilities=probabilities,
                continuation_entropy=entropy(probabilities.values()),
                top_continuation=top_token,
                top_continuation_probability=top_probability,
                association_log_ratio=math.log(max(top_probability, 1e-12) / max(marginal_probability, 1e-12)),
                distinct_continuations=len(counts),
                source_corpus=source_corpus,
                split=split,
                atlas_hash=version,
            )
        )
    return entries


def sample_strata(entries: Sequence[AtlasEntry], per_stratum: int = 16, seed: int = 0) -> dict[str, list[AtlasEntry]]:
    if not entries:
        return {}
    frequencies = sorted(entry.reference_corpus_frequency for entry in entries)
    entropies = sorted(entry.continuation_entropy for entry in entries)
    median_frequency = frequencies[len(frequencies) // 2]
    median_entropy = entropies[len(entropies) // 2]
    groups = {
        "high_frequency_low_entropy": [],
        "high_frequency_high_entropy": [],
        "low_frequency_low_entropy": [],
        "low_frequency_high_entropy": [],
    }
    for entry in entries:
        frequency = "high_frequency" if entry.reference_corpus_frequency >= median_frequency else "low_frequency"
        uncertainty = "high_entropy" if entry.continuation_entropy >= median_entropy else "low_entropy"
        groups[f"{frequency}_{uncertainty}"].append(entry)
    rng = random.Random(seed)
    sampled = {}
    for name, values in groups.items():
        ordered = sorted(values, key=lambda entry: (entry.n, entry.prefix_token_ids))
        rng.shuffle(ordered)
        sampled[name] = ordered[:per_stratum]
    return sampled


def occurrence_positions(sequence: Sequence[int], prefix: Sequence[int]) -> list[int]:
    n = len(prefix)
    return [end for end in range(n, len(sequence)) if tuple(sequence[end - n : end]) == tuple(prefix)]

