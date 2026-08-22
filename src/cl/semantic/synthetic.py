"""Balanced natural-label and arbitrary-label semantic hierarchy corpus."""

from __future__ import annotations

from dataclasses import dataclass
import random

from cl.common.artifacts import stable_hash
from .hierarchy import Hierarchy, HierarchyItem


VOCAB_SIZE = 96
MAX_LENGTH = 14
Q_PARENT, Q_ROOT, DEFINE, IS_A, SEP = 1, 2, 3, 4, 5


@dataclass(frozen=True)
class SemanticProbe:
    example_id: str
    item_id: str
    tokens: tuple[int, ...]
    target: int
    control_tokens: tuple[int, ...]
    relation_id: str
    stratum: str
    family: str
    abstraction_level: str
    template_id: str
    natural: bool


@dataclass(frozen=True)
class SemanticCorpus:
    hierarchy: Hierarchy
    train_sequences: tuple[tuple[int, ...], ...]
    probes: tuple[SemanticProbe, ...]
    corpus_hash: str
    vocab_size: int = VOCAB_SIZE
    max_length: int = MAX_LENGTH


def _items() -> list[HierarchyItem]:
    values: list[HierarchyItem] = []
    token = 10
    natural_specs = {
        "noun": ("entity", {"bird": ("falcon", "penguin"), "mammal": ("dog", "cat")}),
        "action": ("act", {"movement": ("walk", "crawl"), "communication": ("speak", "signal")}),
        "relation": ("relation", {"spatial": ("above", "below"), "social": ("friend", "rival")}),
    }
    for family, (root_label, branches) in natural_specs.items():
        root_id = f"natural:{family}:{root_label}"
        values.append(HierarchyItem(root_id, root_label, token, family, "natural", None, 0, True)); token += 1
        for parent_label, leaves in branches.items():
            parent_id = f"natural:{family}:{parent_label}"
            values.append(HierarchyItem(parent_id, parent_label, token, family, "natural", root_id, 1, True)); token += 1
            for leaf_label in leaves:
                leaf_id = f"natural:{family}:{leaf_label}"
                values.append(HierarchyItem(leaf_id, leaf_label, token, family, "natural", parent_id, 2, True)); token += 1
    for family in ("noun", "action", "relation"):
        root_id = f"synthetic:{family}:root"
        values.append(HierarchyItem(root_id, f"z{family}0", token, family, "synthetic", None, 0, False)); token += 1
        for branch in range(2):
            parent_id = f"synthetic:{family}:p{branch}"
            values.append(HierarchyItem(parent_id, f"z{family}p{branch}", token, family, "synthetic", root_id, 1, False)); token += 1
            for leaf in range(2):
                leaf_id = f"synthetic:{family}:l{branch}{leaf}"
                values.append(HierarchyItem(leaf_id, f"z{family}l{branch}{leaf}", token, family, "synthetic", parent_id, 2, False)); token += 1
    return values


def _training_sequence(hierarchy: Hierarchy, item: HierarchyItem, level: str, template: int, rng: random.Random) -> tuple[int, ...]:
    path = hierarchy.path(item.item_id)
    parent = hierarchy.items[path[-2]]
    root = hierarchy.items[path[0]]
    query = Q_PARENT if level == "parent" else Q_ROOT
    target = parent.token_id if level == "parent" else root.token_id
    filler = [70 + template]
    # Definitions make arbitrary-label hierarchies learnable from the same interface.
    return tuple([DEFINE, item.token_id, IS_A, parent.token_id, SEP, parent.token_id, IS_A, root.token_id, *filler, query, item.token_id, target])


def _probe(hierarchy: Hierarchy, item: HierarchyItem, level: str, template: int, index: int) -> SemanticProbe:
    path = hierarchy.path(item.item_id)
    parent = hierarchy.items[path[-2]]
    root = hierarchy.items[path[0]]
    query = Q_PARENT if level == "parent" else Q_ROOT
    target = parent.token_id if level == "parent" else root.token_id
    template_token = 70 + template
    tokens = (template_token, query, item.token_id)
    cross_parent = next(
        candidate for candidate in hierarchy.leaves(family=item.semantic_family, natural=item.natural)
        if candidate.parent_id != item.parent_id
    )
    control_tokens = (template_token, query, cross_parent.token_id)
    return SemanticProbe(
        example_id=f"{item.item_id}:{level}:t{template}:{index}",
        item_id=item.item_id,
        tokens=tokens,
        target=target,
        control_tokens=control_tokens,
        relation_id=f"{item.semantic_family}:{level}:{'natural' if item.natural else 'synthetic'}",
        stratum=f"{'natural' if item.natural else 'synthetic'}_{item.semantic_family}_{level}",
        family=item.semantic_family,
        abstraction_level=level,
        template_id=f"template-{template}",
        natural=item.natural,
    )


def build_semantic_corpus(seed: int = 0, repeats: int = 24) -> SemanticCorpus:
    rng = random.Random(seed)
    hierarchy = Hierarchy(_items())
    sequences = []
    leaves = hierarchy.leaves()
    for repeat in range(repeats):
        for item in leaves:
            for level in ("parent", "root"):
                sequences.append(_training_sequence(hierarchy, item, level, repeat % 3, rng))
    rng.shuffle(sequences)
    probes = []
    for item in leaves:
        for level in ("parent", "root"):
            for template in range(3):
                probes.append(_probe(hierarchy, item, level, template, 0))
    corpus_hash = stable_hash({"seed": seed, "hierarchy": hierarchy.version_hash, "sequences": sequences})
    return SemanticCorpus(hierarchy, tuple(sequences), tuple(probes), corpus_hash)
