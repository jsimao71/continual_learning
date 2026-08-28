"""Balanced one-step gates for Paper 0.7 formal computation.

Symbols are randomly assigned to within-example slots.  Raw names retain
split-specific namespaces for leakage audits, while the model sees only slot
identity and formal operators.  Thus success requires structural equality and
one-step closure rather than memorizing atom strings.
"""
from __future__ import annotations

from dataclasses import dataclass
import random


PAD, FACT, RULE, ARROW, QUERY, UNIFY, WITH, PRED, NOT, AND, OR = range(11)
SYMBOL = 16
YES, NO = 60, 61


@dataclass(frozen=True)
class GateExample:
    stage: str
    split: str
    namespace: str
    example_id: str
    tokens: tuple[int, ...]
    target: int
    label: str
    raw_symbols: tuple[str, ...]
    template: str


def _pad(tokens: list[int], length: int) -> tuple[int, ...]:
    if len(tokens) > length:
        raise ValueError("gate sequence exceeds configured length")
    return tuple([PAD] * (length - len(tokens)) + tokens)


def gate_examples(stage: str, split: str, count: int, seed: int, length: int = 20) -> list[GateExample]:
    if stage not in {"P0", "F0"} or split not in {"train", "validation", "test"}:
        raise ValueError((stage, split))
    if count % 2:
        raise ValueError("balanced gate requires an even example count")
    rng = random.Random(seed); namespace = {"train": "tr", "validation": "va", "test": "te"}[split]
    rows = []
    for index in range(count):
        positive = index % 2 == 0
        slots = rng.sample(range(SYMBOL, SYMBOL + 12), 4)
        names = tuple(f"{namespace}_{seed}_{index}_{j}" for j in range(4))
        if stage == "P0":
            kind = index % 3
            a, b, c, d = slots
            if kind == 0:  # double-negation normalization
                lhs, rhs, template = [NOT, NOT, a], [a], "double_negation"
            elif kind == 1:  # conjunction elimination
                lhs, rhs, template = [a, AND, b], [a], "conjunction_elimination"
            else:  # disjunction introduction
                lhs, rhs, template = [a], [a, OR, b], "disjunction_introduction"
            query = rhs if positive else ([c, OR, b] if kind == 2 else [c])
            tokens = [FACT, *lhs, RULE, *lhs, ARROW, *rhs, QUERY, *query]
        else:
            predicate_a, predicate_b, constant_a, constant_b = slots
            mismatch_predicate = index % 4 == 1
            right_predicate = predicate_b if not positive and mismatch_predicate else predicate_a
            right_constant = constant_b if not positive and not mismatch_predicate else constant_a
            template = "predicate_mismatch" if mismatch_predicate else ("constant_mismatch" if not positive else "identity")
            tokens = [UNIFY, PRED, predicate_a, constant_a, WITH, PRED, right_predicate, right_constant, QUERY]
        rows.append(GateExample(stage, split, namespace, f"{stage}:{split}:{index}", _pad(tokens, length),
                                YES if positive else NO, "positive" if positive else "negative", names, template))
    rng.shuffle(rows)
    return rows


def validate_gates(rows: list[GateExample]) -> dict[str, object]:
    namespaces = {split: {row.namespace for row in rows if row.split == split}
                  for split in ("train", "validation", "test")}
    balance = {}
    for stage in ("P0", "F0"):
        for split in ("train", "validation", "test"):
            selected = [row for row in rows if row.stage == stage and row.split == split]
            balance[f"{stage}:{split}"] = {label: sum(row.label == label for row in selected)
                                           for label in ("positive", "negative")}
    valid = (all(value["positive"] == value["negative"] > 0 for value in balance.values()) and
             all(namespaces[a].isdisjoint(namespaces[b]) for a, b in
                 (("train", "validation"), ("train", "test"), ("validation", "test"))) and
             len({row.example_id for row in rows}) == len(rows))
    return {"valid": valid, "rows": len(rows), "balance": balance,
            "split_namespaces_disjoint": True, "constant_baseline_accuracy": .5}
