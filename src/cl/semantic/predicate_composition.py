"""Cue-free two-chain ancestor composition dataset for Paper 0.6."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import random

from cl.common.artifacts import stable_hash
from cl.semantic.predicate_stress import (
    BOOL_FALSE,
    BOOL_TRUE,
    EDGE,
    HOP,
    PAD,
    PREDICATE,
    QUERY,
    TEMPLATE,
)

TRAIN_NODES = tuple(range(32, 272))
VALIDATION_NODES = tuple(range(272, 392))
TEST_NODES = tuple(range(392, 512))
DISTRACTOR = 8
DISTRACTOR_TOKENS = tuple(range(512, 1024))


@dataclass(frozen=True)
class CompositionExample:
    tokens: tuple[int, ...]
    target: int
    example_id: str
    split: str
    required_path: int
    total_depth: int
    template_id: int
    position_mode: str
    tree_seed: int
    positive: int
    candidate: int
    query_leaf: int
    paths: tuple[tuple[int, ...], tuple[int, ...]]
    serialized_edges: tuple[tuple[int, int], ...]
    candidate_chain_first: int
    candidate_role_count: tuple[int, int]


def _reachable_at_depth(edges: list[tuple[int, int]], leaf: int, candidate: int, depth: int) -> bool:
    parents = {child: parent for child, parent in edges}
    current = leaf
    for _ in range(depth):
        if current not in parents:
            return False
        current = parents[current]
    return current == candidate


def composition_example(config: dict, *, split: str, required_path: int, template: int,
                        position_mode: str, tree_seed: int, index: int,
                        positive: int | None = None, candidate_chain_first: int | None = None,
                        model_seed: int = 0) -> CompositionExample:
    if split not in {"train", "validation", "test"} or required_path < 1:
        raise ValueError((split, required_path))
    total_depth = required_path + 2
    pool = {"train": TRAIN_NODES, "validation": VALIDATION_NODES, "test": TEST_NODES}[split]
    seed = (config["ontology_seed"] + tree_seed * 1_000_003 + index * 101 +
            required_path * 37 + model_seed * 17)
    rng = random.Random(seed)
    main = tuple(rng.sample(pool, total_depth + 1))
    remaining = [token for token in pool if token not in main]
    decoy = tuple(rng.sample(remaining, total_depth + 1))
    positive = index % 2 if positive is None else int(positive)
    candidate_chain_first = (index // 2) % 2 if candidate_chain_first is None else int(candidate_chain_first)
    candidate = (main if positive else decoy)[required_path]
    paths = (main, decoy) if (positive == candidate_chain_first) else (decoy, main)
    edges = [(child, parent) for path in paths for child, parent in zip(path, path[1:])]
    if _reachable_at_depth(edges, main[0], candidate, required_path) != bool(positive):
        raise AssertionError("cue-free composition oracle disagrees with label")
    serialized = []
    edge_order = list(edges)
    if template in (2, 3):
        rng.shuffle(edge_order)
    for child, parent in edge_order:
        serialized.extend((EDGE, child, parent))
    noise = [DISTRACTOR, *[DISTRACTOR_TOKENS[(seed + offset) % len(DISTRACTOR_TOKENS)]
                             for offset in range(4)]]
    body = [TEMPLATE[template], *(noise + serialized if template in (1, 3) else serialized + noise)]
    padding = rng.randrange(9) if position_mode == "randomized" else 0
    tokens = tuple([PAD] * padding + body + [QUERY, PREDICATE["isAncestor"],
                                             *([HOP] * required_path), main[0], candidate])
    if len(tokens) > config["max_length"]:
        raise ValueError((len(tokens), config["max_length"]))
    child_count = sum(child == candidate for child, _ in edges)
    parent_count = sum(parent == candidate for _, parent in edges)
    return CompositionExample(
        tokens, BOOL_TRUE if positive else BOOL_FALSE,
        f"{split}:d{required_path}:t{template}:p{position_mode}:tree{tree_seed}:i{index}",
        split, required_path, total_depth, template, position_mode, tree_seed, positive,
        candidate, main[0], paths, tuple(edge_order), candidate_chain_first,
        (child_count, parent_count),
    )


def evaluation_rows(config: dict, model_seed: int, examples_per_cell: int) -> list[CompositionExample]:
    if examples_per_cell % 4:
        raise ValueError("examples_per_cell must be divisible by four to cross label and chain order")
    rows = []
    for split in ("validation", "test"):
        for depth in config["test_depths"]:
            for template in config["templates"]:
                for position in config["position_modes"]:
                    for tree_seed in config["tree_seeds"]:
                        for index in range(examples_per_cell):
                            rows.append(composition_example(
                                config, split=split, required_path=depth, template=template,
                                position_mode=position, tree_seed=tree_seed,
                                index=index + depth * 10_000, model_seed=model_seed,
                                positive=index % 2, candidate_chain_first=(index // 2) % 2,
                            ))
    return rows


def training_batch(config: dict, rng: random.Random, model_seed: int,
                   batch_size: int) -> list[CompositionExample]:
    rows = []
    base = rng.randrange(1_000_000)
    for offset in range(batch_size):
        rows.append(composition_example(
            config, split="train", required_path=rng.choice(config["train_depths"]),
            template=rng.choice(config["templates"]),
            position_mode=rng.choice(config["position_modes"]),
            tree_seed=rng.choice(config["tree_seeds"]), index=base + offset,
            model_seed=model_seed, positive=offset % 2,
            candidate_chain_first=(offset // 2) % 2,
        ))
    return rows


def validate_composition(config: dict) -> dict:
    rows = evaluation_rows(config, 11, config["smoke_examples_per_cell"])
    checks = {}
    for split in ("validation", "test"):
        selected_split = [row for row in rows if row.split == split]
        for depth in config["test_depths"]:
            for template in config["templates"]:
                for position in config["position_modes"]:
                    subset = [row for row in selected_split if row.required_path == depth and
                              row.template_id == template and row.position_mode == position]
                    key = f"{split}:d{depth}:t{template}:{position}"
                    checks[f"{key}:label_balance"] = Counter(row.positive for row in subset) == {
                        0: len(subset) // 2, 1: len(subset) // 2,
                    }
                    checks[f"{key}:order_balance"] = Counter(row.candidate_chain_first for row in subset) == {
                        0: len(subset) // 2, 1: len(subset) // 2,
                    }
    checks["candidate_always_exposed"] = all(row.candidate in row.tokens for row in rows)
    checks["candidate_role_matched"] = all(row.candidate_role_count == (1, 1) for row in rows)
    checks["paths_disjoint"] = all(set(row.paths[0]).isdisjoint(row.paths[1]) for row in rows)
    checks["split_symbol_namespaces_disjoint"] = (
        set(TRAIN_NODES).isdisjoint(VALIDATION_NODES) and
        set(TRAIN_NODES).isdisjoint(TEST_NODES) and
        set(VALIDATION_NODES).isdisjoint(TEST_NODES)
    )
    validation_tokens = {row.tokens for row in rows if row.split == "validation"}
    test_tokens = {row.tokens for row in rows if row.split == "test"}
    checks["validation_test_examples_disjoint"] = validation_tokens.isdisjoint(test_tokens)
    checks["membership_baseline_accuracy"] = 0.5
    checks["maximum_length_valid"] = max(len(row.tokens) for row in rows) <= config["max_length"]
    checks["vocabulary_valid"] = max(max(row.tokens) for row in rows) < config["vocab_size"]
    return {
        "schema_version": "paper06.compositional_frontier_v7.validation.v1",
        "valid": all(value is True or key == "membership_baseline_accuracy"
                     for key, value in checks.items()),
        "rows": len(rows), "checks": checks,
        "example_hash": stable_hash([asdict(row) for row in rows]),
        "oracle": "candidate reached from query leaf after exactly d parent edges",
        "shortcut_control": "positive and negative candidates are both exposed with matched edge roles",
    }
