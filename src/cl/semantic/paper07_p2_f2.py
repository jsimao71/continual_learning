"""Exact, balanced P2 and F2 acquisition datasets for Paper 0.7."""
from __future__ import annotations

from dataclasses import dataclass
import random

from cl.semantic.formal_logic import Fun, Var, serialize_term, unify
from cl.semantic.paper07_gates import (
    ARROW,
    FACT,
    NO,
    PAD,
    PRED,
    QUERY,
    RULE,
    SYMBOL,
    UNIFY,
    WITH,
    YES,
)
from cl.semantic.paper07_p1_f1 import BIND, CONST, SYMBOL_COUNT, VAR

FUNCTOR = 15
FAIL = 28
NOT = 29
TRANSFORM = 30


@dataclass(frozen=True)
class NextGateExample:
    stage: str
    split: str
    namespace: str
    example_id: str
    tokens: tuple[int, ...]
    target: int
    label: str
    template: str
    oracle_substitution: str | None
    failure_reason: str | None


def _pad(tokens: list[int], length: int) -> tuple[int, ...]:
    if len(tokens) > length:
        raise ValueError("P2/F2 sequence exceeds configured length")
    return tuple([PAD] * (length - len(tokens)) + tokens)


def p2_examples(split: str, count: int, seed: int, length: int = 24) -> list[NextGateExample]:
    """Separate modus-tollens inference from contraposition verification."""
    if split not in {"train", "validation", "test"} or count % 12:
        raise ValueError("P2 requires a valid split and count divisible by 12")
    rng = random.Random(seed)
    namespace = {"train": "tr", "validation": "va", "test": "te"}[split]
    templates = (
        "mt_fact_polarity",
        "mt_rule_direction",
        "mt_query_polarity",
        "cp_antecedent_polarity",
        "cp_consequent_polarity",
        "cp_direction",
    )
    rows = []
    for index in range(count):
        positive = index % 2 == 0
        template = templates[(index // 2) % len(templates)]
        a, b = rng.sample(range(SYMBOL, SYMBOL + SYMBOL_COUNT), 2)
        if template.startswith("mt_"):
            lhs, rhs, fact_polarity, query_polarity = a, b, NOT, NOT
            if not positive:
                if template == "mt_fact_polarity":
                    fact_polarity = PAD
                elif template == "mt_rule_direction":
                    lhs, rhs = b, a
                else:
                    query_polarity = PAD
            tokens = [RULE, lhs, ARROW, rhs, FACT, fact_polarity, b, QUERY, query_polarity, a]
            operation = "modus_tollens"
        else:
            cand_lhs_polarity, cand_lhs = NOT, b
            cand_rhs_polarity, cand_rhs = NOT, a
            if not positive:
                if template == "cp_antecedent_polarity":
                    cand_lhs_polarity = PAD
                elif template == "cp_consequent_polarity":
                    cand_rhs_polarity = PAD
                else:
                    cand_lhs, cand_rhs = a, b
            tokens = [RULE, a, ARROW, b, QUERY, TRANSFORM, RULE,
                      cand_lhs_polarity, cand_lhs, ARROW, cand_rhs_polarity, cand_rhs]
            operation = "contraposition"
        rows.append(NextGateExample(
            "P2", split, namespace, f"P2:{split}:{index}", _pad(tokens, length),
            YES if positive else NO, f"{operation}:{'valid' if positive else 'invalid'}",
            template, None, None,
        ))
    rng.shuffle(rows)
    return rows


def f2_examples(split: str, count: int, seed: int, length: int = 24) -> list[NextGateExample]:
    """Predict the exact unary-functor binding, or FAIL on a structural mismatch."""
    if split not in {"train", "validation", "test"} or count % 96:
        raise ValueError("F2 requires a valid split and count divisible by 96")
    rng = random.Random(seed)
    namespace = {"train": "tr", "validation": "va", "test": "te"}[split]
    templates = (
        "canonical_functor_control",
        "reversed_functor_control",
        "canonical_predicate_control",
        "reversed_predicate_control",
    )
    rows = []
    for index in range(count):
        success = index % 2 == 0
        template = templates[(index // 24) % len(templates)]
        target = SYMBOL + ((index // 2) % SYMBOL_COUNT)
        available = [token for token in range(SYMBOL, SYMBOL + SYMBOL_COUNT) if token != target]
        predicate, other_predicate, functor, other_functor, variable = rng.sample(available, 5)
        right_predicate, right_functor = predicate, functor
        if not success and "functor_control" in template:
            right_functor = other_functor
        if not success and "predicate_control" in template:
            right_predicate = other_predicate

        left_term = Fun(str(predicate), (Fun(str(functor), (Var(str(variable)),)),))
        right_term = Fun(str(right_predicate), (Fun(str(right_functor), (Fun(str(target)),)),))
        result = unify(left_term, right_term)
        if result.success != success:
            raise AssertionError("constructed F2 oracle disagrees with requested label")
        left = [PRED, predicate, FUNCTOR, functor, VAR, variable]
        right = [PRED, right_predicate, FUNCTOR, right_functor, CONST, target]
        if template.startswith("reversed"):
            left, right = right, left
        tokens = [UNIFY, *left, WITH, *right, QUERY, BIND, variable]
        substitution = None
        if result.success:
            substitution = ",".join(f"{name}={serialize_term(term)}" for name, term in result.substitution)
        rows.append(NextGateExample(
            "F2", split, namespace, f"F2:{split}:{index}", _pad(tokens, length),
            target if success else FAIL, "binding" if success else "failure", template,
            substitution, result.failure_reason,
        ))
    rng.shuffle(rows)
    return rows


def next_gate_examples(stage: str, split: str, count: int, seed: int,
                       length: int = 24) -> list[NextGateExample]:
    if stage == "P2":
        return p2_examples(split, count, seed, length)
    if stage == "F2":
        return f2_examples(split, count, seed, length)
    raise ValueError(stage)


def validate_next_gates(rows: list[NextGateExample]) -> dict:
    expected = {
        "P2": {"mt_fact_polarity", "mt_rule_direction", "mt_query_polarity",
               "cp_antecedent_polarity", "cp_consequent_polarity", "cp_direction"},
        "F2": {"canonical_functor_control", "reversed_functor_control",
               "canonical_predicate_control", "reversed_predicate_control"},
    }
    checks = {}
    template_counts = {}
    for stage in ("P2", "F2"):
        for split in ("train", "validation", "test"):
            selected = [row for row in rows if row.stage == stage and row.split == split]
            key = f"{stage}:{split}"
            checks[f"{key}:templates"] = {row.template for row in selected} == expected[stage]
            template_counts[key] = {template: sum(row.template == template for row in selected)
                                    for template in sorted(expected[stage])}
            for template in expected[stage]:
                subset = [row for row in selected if row.template == template]
                if stage == "P2":
                    checks[f"{key}:{template}:balanced"] = (
                        sum(row.target == YES for row in subset) == sum(row.target == NO for row in subset)
                    )
                else:
                    checks[f"{key}:{template}:status_balanced"] = (
                        sum(row.target == FAIL for row in subset) * 2 == len(subset)
                    )
                    successes = [row for row in subset if row.target != FAIL]
                    counts = [sum(row.target == target for row in successes)
                              for target in range(SYMBOL, SYMBOL + SYMBOL_COUNT)]
                    checks[f"{key}:{template}:bindings_balanced"] = len(set(counts)) == 1
                    checks[f"{key}:{template}:oracles"] = all(
                        (row.target == FAIL) == (row.failure_reason is not None) for row in subset
                    )
    namespaces = {split: {row.namespace for row in rows if row.split == split}
                  for split in ("train", "validation", "test")}
    checks["split_namespaces_disjoint"] = all(
        namespaces[a].isdisjoint(namespaces[b])
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )
    checks["unique_ids"] = len({row.example_id for row in rows}) == len(rows)
    return {
        "valid": all(checks.values()),
        "rows": len(rows),
        "checks": checks,
        "template_counts": template_counts,
        "P2_constant_baseline_accuracy": 0.5,
        "F2_fail_baseline_accuracy": 0.5,
    }
