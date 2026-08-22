"""Controlled corpus with frequency, entropy, induction, and override factors."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

from cl.common.artifacts import stable_hash


PAD = 0
VOCAB_SIZE = 40
SEQUENCE_LENGTH = 12


@dataclass(frozen=True)
class ProbeExample:
    example_id: str
    tokens: tuple[int, ...]
    target: int
    habitual_target: int
    stratum: str
    relation_id: str
    control_tokens: tuple[int, ...]


@dataclass(frozen=True)
class SyntheticCorpus:
    train_sequences: tuple[tuple[int, ...], ...]
    probes: tuple[ProbeExample, ...]
    vocab_size: int
    sequence_length: int
    corpus_hash: str


def _noise(rng: random.Random, length: int) -> list[int]:
    return [rng.randrange(20, VOCAB_SIZE) for _ in range(length)]


def _embed_relation(rng: random.Random, prefix: Sequence[int], target: int) -> tuple[int, ...]:
    sequence = _noise(rng, SEQUENCE_LENGTH)
    start = rng.randrange(1, SEQUENCE_LENGTH - len(prefix))
    sequence[start : start + len(prefix)] = prefix
    sequence[start + len(prefix)] = target
    return tuple(sequence)


def build_corpus(seed: int = 0, train_size: int = 960) -> SyntheticCorpus:
    rng = random.Random(seed)
    sequences: list[tuple[int, ...]] = []
    # Independently controlled relation families.
    for index in range(train_size):
        draw = index % 12
        if draw < 5:  # frequent, deterministic
            sequences.append(_embed_relation(rng, (1, 2), 3))
        elif draw < 8:  # frequent, high entropy
            sequences.append(_embed_relation(rng, (4, 5), (6, 7, 8)[index % 3]))
        elif draw == 8:  # rare, deterministic
            sequences.append(_embed_relation(rng, (9, 10), 11))
        elif draw < 11:  # contextual override: marker 12 changes the habitual target 3 to 13
            sequence = list(_embed_relation(rng, (1, 2), 13))
            sequence[0] = 12
            sequences.append(tuple(sequence))
        else:  # in-context arbitrary mapping: 14 A B ... A -> B
            a = 15 + (index % 3) * 2
            b = a + 1
            sequence = _noise(rng, SEQUENCE_LENGTH)
            sequence[:3] = [14, a, b]
            sequence[-2:] = [a, b]
            sequences.append(tuple(sequence))

    probes: list[ProbeExample] = []
    for index in range(24):
        context = tuple(_noise(rng, 6) + [1, 2])
        control = tuple(list(context[:-2]) + [4, 2])
        probes.append(ProbeExample(f"familiar-{index}", context, 3, 3, "familiar_low_entropy", "1-2-to-3", control))
        context = tuple(_noise(rng, 6) + [4, 5])
        control = tuple(list(context[:-2]) + [1, 5])
        probes.append(ProbeExample(f"ambiguous-{index}", context, (6, 7, 8)[index % 3], 6, "familiar_high_entropy", "4-5-to-many", control))
        a = 15 + (index % 3) * 2
        b = a + 1
        context = tuple([14, a, b] + _noise(rng, 3) + [a])
        control = tuple([14, a, b] + _noise(rng, 3) + [a + 2 if a < 19 else 15])
        probes.append(ProbeExample(f"context-{index}", context, b, b, "context_introduced", f"map-{a}-{b}", control))
        context = tuple([12] + _noise(rng, 4) + [1, 2])
        control = tuple([20] + list(context[1:]))
        probes.append(ProbeExample(f"override-{index}", context, 13, 3, "override_repair", "override-1-2", control))
    corpus_hash = stable_hash({"seed": seed, "train": sequences, "probes": [probe.tokens for probe in probes]})
    return SyntheticCorpus(tuple(sequences), tuple(probes), VOCAB_SIZE, SEQUENCE_LENGTH, corpus_hash)
