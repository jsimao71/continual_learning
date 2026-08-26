"""Leakage-audited contextual-copy tasks for Paper 0.8.

The prediction is always the value paired with the repeated query key.  Literal
key--value associations are balanced across episodes, so weights alone cannot
solve a held-out prompt.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import math
import random

PAD, SEP, QUERY = 0, 1, 2


@dataclass(frozen=True)
class CopyExample:
    tokens: tuple[int, ...]
    target: int
    query_key: int
    pairs: tuple[tuple[int, int], ...]
    target_pair: int
    family: int
    condition: str = "correct"

    @property
    def roles(self) -> tuple[str, ...]:
        roles: list[str] = []
        for index in range(len(self.pairs)):
            prefix = "demo" if index == self.target_pair else "distractor"
            roles.extend((f"{prefix}_key", f"{prefix}_value", "separator"))
        return tuple(roles + ["query_marker", "query_key"])


def _derangement(values: list[int], rng: random.Random) -> list[int]:
    if len(values) < 2:
        return values[:]
    candidate = values[:]
    for _ in range(100):
        rng.shuffle(candidate)
        if all(a != b for a, b in zip(candidate, values)):
            return candidate
    return values[1:] + values[:1]


def generate_copy_examples(
    regime: str,
    vocabulary: int,
    episodes: int,
    *,
    pairs_per_prompt: int = 1,
    mappings: int | str = "fresh-random",
    seed: int = 0,
    family: int = 0,
    symbol_offset: int = 3,
) -> list[CopyExample]:
    """Generate C0/C1/C2/C3-copy episodes with fixed-length prompts."""
    if regime not in {"C0", "C1", "C2", "C3-copy"}:
        raise ValueError(f"unknown copy regime {regime}")
    if vocabulary < 2:
        raise ValueError("copy vocabulary requires at least two symbols")
    if episodes < vocabulary * (vocabulary - 1):
        raise ValueError("episodes must cover every non-identity key/value pair")
    if pairs_per_prompt < 1 or pairs_per_prompt > vocabulary:
        raise ValueError("pairs_per_prompt must be in [1, vocabulary]")
    rng = random.Random(seed)
    symbols = list(range(symbol_offset, symbol_offset + vocabulary))
    # A complete directed non-identity cycle is the balancing primitive.  Its
    # order changes with seed, while each X maps to every Y equally often.
    balanced = [(x, y) for x in symbols for y in symbols if y != x]
    rng.shuffle(balanced)
    if mappings != "fresh-random":
        mapping_count = int(mappings)
        if mapping_count < 1:
            raise ValueError("mappings must be positive")
        balanced = balanced[: min(mapping_count, len(balanced))]
    rows: list[CopyExample] = []
    for episode in range(episodes):
        query, target = balanced[episode % len(balanced)]
        other_keys = [x for x in symbols if x != query]
        rng.shuffle(other_keys)
        keys = [query] + other_keys[: pairs_per_prompt - 1]
        values = [target]
        available = [x for x in symbols if x != target]
        rng.shuffle(available)
        for key in keys[1:]:
            choice_index = next((i for i, value in enumerate(available) if value != key), 0)
            values.append(available.pop(choice_index))
        pairs = list(zip(keys, values))
        rng.shuffle(pairs)
        target_pair = next(i for i, pair in enumerate(pairs) if pair[0] == query)
        tokens = tuple(x for pair in pairs for x in (*pair, SEP)) + (QUERY, query)
        rows.append(CopyExample(tokens, target, query, tuple(pairs), target_pair, family))
    return rows


def controlled_copy(row: CopyExample, condition: str, seed: int = 0) -> CopyExample:
    """Controls preserve length/query position while disrupting key association."""
    if condition == "correct":
        return row
    rng = random.Random(seed)
    pairs = list(row.pairs)
    if condition == "shuffled_pairings":
        values = [v for _, v in pairs]
        shifted = _derangement(values, rng)
        pairs = [(pair[0], value) for pair, value in zip(pairs, shifted)]
    elif condition == "shuffled_order":
        rng.shuffle(pairs)
    elif condition == "wrong_query":
        alternatives = [key for key, _ in pairs if key != row.query_key]
        query = alternatives[0] if alternatives else row.target
        tokens = tuple(x for pair in pairs for x in (*pair, SEP)) + (QUERY, query)
        return replace(row, tokens=tokens, query_key=query, condition=condition)
    elif condition == "none":
        return replace(row, tokens=(QUERY, row.query_key), condition=condition)
    else:
        raise ValueError(condition)
    target_pair = next((i for i, pair in enumerate(pairs) if pair[0] == row.query_key), -1)
    tokens = tuple(x for pair in pairs for x in (*pair, SEP)) + (QUERY, row.query_key)
    return replace(row, tokens=tokens, pairs=tuple(pairs), target_pair=target_pair, condition=condition)


def validate_copy(rows: list[CopyExample]) -> dict[str, int | float | str]:
    """Quantify shortcuts relevant to the copy competence claim."""
    if not rows:
        raise ValueError("cannot validate an empty dataset")
    conditional: dict[int, Counter[int]] = defaultdict(Counter)
    target_positions: Counter[int] = Counter()
    query_positions: Counter[int] = Counter()
    pair_positions: dict[int, Counter[int]] = defaultdict(Counter)
    for row in rows:
        conditional[row.query_key][row.target] += 1
        target_positions[row.tokens.index(row.target)] += 1
        query_positions[len(row.tokens) - 1] += 1
        pair_positions[row.target_pair][row.target] += 1
    max_conditional = max(max(counts.values()) / sum(counts.values()) for counts in conditional.values())
    expected = 1 / max(1, len(next(iter(conditional.values()))))
    max_pair_leak = max(max(counts.values()) / sum(counts.values()) for counts in pair_positions.values())
    entropy = sum(
        -sum((count / sum(counts.values())) * math.log2(count / sum(counts.values())) for count in counts.values())
        for counts in conditional.values()
    ) / len(conditional)
    v = len(conditional)
    return {
        "examples": len(rows), "vocabulary": v,
        "max_p_y_given_x": max_conditional, "ideal_p_y_given_x": 1 / max(1, v - 1),
        "conditional_entropy_bits": entropy,
        "target_position_count": len(target_positions), "query_position_count": len(query_positions),
        "max_target_given_pair_position": max_pair_leak,
        "v2_degenerate": int(v == 2),
        "passed": int(v > 2 and max_conditional <= 1 / (v - 1) + 1e-9 and max_pair_leak < 0.8),
    }
