"""Information-theoretic taxonomy and finite-hypothesis D5 episodes.

This module is intentionally independent of model training.  It makes the
declared prior family explicit and rejects episodes whose demonstrations do
not determine a unique answer, even when several parameter hypotheses remain.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Callable, Iterable


@dataclass(frozen=True)
class FunctionHypothesis:
    family: str
    parameters: tuple[int, ...]
    modulus: int

    def __call__(self, x: int) -> int:
        k = self.modulus
        if self.family == "successor":
            return (x + 1) % k
        if self.family == "add_n":
            return (x + self.parameters[0]) % k
        if self.family == "multiply_a":
            return (self.parameters[0] * x) % k
        if self.family == "affine":
            a, b = self.parameters
            return (a * x + b) % k
        if self.family == "square":
            return x * x % k
        raise ValueError(self.family)


@dataclass(frozen=True)
class RuleEpisode:
    family: str
    modulus: int
    parameters: tuple[int, ...]
    demonstrations: tuple[tuple[int, int], ...]
    query: int
    target: int
    consistent_hypotheses: int
    answer_entropy_bits: float
    minimum_examples_needed: int


def hypotheses(family: str, modulus: int) -> tuple[FunctionHypothesis, ...]:
    """Enumerate the finite declared prior for a unary D5 family."""
    if modulus < 2:
        raise ValueError("modulus must be at least two")
    if family in {"successor", "square"}:
        params = [()]
    elif family in {"add_n", "multiply_a"}:
        params = [(n,) for n in range(modulus)]
    elif family == "affine":
        params = [(a, b) for a in range(modulus) for b in range(modulus)]
    else:
        raise ValueError(f"unknown family {family}")
    return tuple(FunctionHypothesis(family, p, modulus) for p in params)


def posterior(prior: Iterable[FunctionHypothesis], demonstrations: Iterable[tuple[int, int]]) -> tuple[FunctionHypothesis, ...]:
    demos = tuple(demonstrations)
    return tuple(h for h in prior if all(h(x) == y for x, y in demos))


def answer_distribution(post: Iterable[FunctionHypothesis], query: int) -> dict[int, float]:
    members = tuple(post)
    if not members:
        raise ValueError("demonstrations are inconsistent with the declared prior")
    counts: dict[int, int] = {}
    for h in members:
        counts[h(query)] = counts.get(h(query), 0) + 1
    return {answer: count / len(members) for answer, count in counts.items()}


def entropy(distribution: dict[int, float]) -> float:
    return -sum(p * math.log2(p) for p in distribution.values() if p)


def minimum_identifying_prefix(true_h: FunctionHypothesis, ordered_inputs: Iterable[int], query: int) -> int:
    prior = hypotheses(true_h.family, true_h.modulus)
    demonstrations: list[tuple[int, int]] = []
    for count, x in enumerate(ordered_inputs, 1):
        demonstrations.append((x, true_h(x)))
        if entropy(answer_distribution(posterior(prior, demonstrations), query)) == 0.0:
            return count
    raise ValueError("available demonstrations cannot identify the query answer")


def make_identifiable_episode(family: str, modulus: int, parameters: tuple[int, ...], seed: int) -> RuleEpisode:
    true_h = FunctionHypothesis(family, parameters, modulus)
    prior = hypotheses(family, modulus)
    if true_h not in prior:
        raise ValueError("parameters fall outside the declared prior")
    rng = random.Random(seed)
    query = rng.randrange(modulus)
    inputs = [x for x in range(modulus) if x != query]
    rng.shuffle(inputs)
    needed = minimum_identifying_prefix(true_h, inputs, query)
    demos = tuple((x, true_h(x)) for x in inputs[:needed])
    post = posterior(prior, demos)
    distribution = answer_distribution(post, query)
    uncertainty = entropy(distribution)
    if uncertainty != 0.0:
        raise ValueError("non-identifiable deterministic-rule episode")
    return RuleEpisode(family, modulus, parameters, demos, query, true_h(query), len(post), uncertainty, needed)


def dataset_taxonomy(chain_length: int = 3) -> list[dict[str, object]]:
    """Return explicit computational labels for the existing ladder and D5."""
    mapping_space = math.factorial(chain_length)
    rows = [
        ("D0", "T0", "random/no aligned task", False, False, False, False, False),
        ("D1", "T0", "local unary relation exposure", False, False, False, False, False),
        ("D2", "T0", "isomorphic local relation families", False, False, False, False, False),
        ("D3-A", "T3", "partial fixed alignment", False, True, True, False, False),
        ("D3-B", "T4", "finite mapping selection", False, True, True, False, False),
        ("D3-C", "T4", "low-entropy transformation selection", False, True, True, False, False),
        ("D4", "T3", "all-but-one permutation completion", False, True, False, False, False),
        ("D5-unary", "T5", "parameterized modular rule induction", False, True, False, True, False),
        ("D5-compositional", "T6", "composed structured rules", False, True, False, True, True),
        ("D5-procedural", "T7", "recursive/iterative execution", False, True, False, True, True),
    ]
    out = []
    for dataset, category, description, explicit, unique, selected, inferred, iterative in rows:
        # For D4, X alone leaves a uniform value permutation; all-but-one
        # demonstrations determine the complement exactly.
        h_y_x = math.log2(chain_length) if dataset == "D4" else None
        h_y_context = 0.0 if unique else None
        out.append({
            "dataset": dataset, "taxonomy": category, "description": description,
            "h_y_given_x_bits": h_y_x,
            "h_y_given_x_context_bits": h_y_context,
            "i_y_context_given_x_bits": h_y_x if dataset == "D4" else None,
            "bayes_oracle_ceiling": 1.0 if unique else None,
            "target_explicitly_present": explicit,
            "uniquely_determined_by_constraints": unique,
            "finite_known_task_selection": selected,
            "new_structured_function_inferred": inferred,
            "iterative_execution_required": iterative,
            "mapping_space_size": mapping_space if dataset == "D4" else None,
            "demonstrations_per_episode": chain_length - 1 if dataset == "D4" else None,
            "admissible_as_rule_induction": dataset.startswith("D5") and unique,
        })
    return out


def episode_record(row: RuleEpisode) -> dict[str, object]:
    record = asdict(row)
    record["parameters"] = ",".join(map(str, row.parameters))
    record["demonstrations"] = ";".join(f"{x}->{y}" for x, y in row.demonstrations)
    record["bayes_oracle_ceiling"] = 1.0
    record["admissible"] = int(row.answer_entropy_bits == 0.0)
    return record
