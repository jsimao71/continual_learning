import json
import random
from collections import Counter
from pathlib import Path

from cl.experiments.paper06_compositional_frontier import aggregate
from cl.semantic.predicate_composition import (
    TRAIN_NODES,
    VALIDATION_NODES,
    TEST_NODES,
    composition_example,
    evaluation_rows,
    training_batch,
    validate_composition,
)


CONFIG = json.loads(Path("configs/paper06/compositional_frontier_v7.json").read_text())


def test_composition_generator_is_oracle_valid_and_leakage_audited():
    audit = validate_composition(CONFIG)
    assert audit["valid"] and audit["checks"]["candidate_always_exposed"]
    assert audit["checks"]["candidate_role_matched"]
    assert audit["checks"]["membership_baseline_accuracy"] == 0.5
    assert set(TRAIN_NODES).isdisjoint(VALIDATION_NODES)
    assert set(TRAIN_NODES).isdisjoint(TEST_NODES)
    assert set(VALIDATION_NODES).isdisjoint(TEST_NODES)
    assert audit["checks"]["validation_test_examples_disjoint"]


def test_positive_and_negative_candidates_have_identical_membership_roles():
    rows = [composition_example(
        CONFIG, split="test", required_path=6, template=template,
        position_mode="aligned", tree_seed=101, index=index,
        positive=index % 2, candidate_chain_first=(index // 2) % 2,
    ) for template in CONFIG["templates"] for index in range(4)]
    assert {row.candidate_role_count for row in rows} == {(1, 1)}
    assert all(row.candidate in row.tokens for row in rows)
    assert Counter((row.positive, row.candidate_chain_first) for row in rows) == {
        (0, 0): 4, (0, 1): 4, (1, 0): 4, (1, 1): 4,
    }


def test_shuffled_template_breaks_contiguous_chain_serialization():
    canonical = composition_example(
        CONFIG, split="test", required_path=6, template=0, position_mode="aligned",
        tree_seed=101, index=8, positive=0, candidate_chain_first=0,
    )
    shuffled = composition_example(
        CONFIG, split="test", required_path=6, template=2, position_mode="aligned",
        tree_seed=101, index=8, positive=0, candidate_chain_first=0,
    )
    assert canonical.paths == shuffled.paths
    canonical_edges = tuple((child, parent) for path in canonical.paths
                            for child, parent in zip(path, path[1:]))
    assert canonical.serialized_edges == canonical_edges
    assert shuffled.serialized_edges not in {canonical_edges, tuple(reversed(canonical_edges))}


def test_evaluation_crosses_depth_template_position_tree_label_and_order():
    rows = evaluation_rows(CONFIG, 11, 4)
    assert len(rows) == 2 * 8 * 4 * 2 * 3 * 4
    for split in ("validation", "test"):
        for depth in CONFIG["test_depths"]:
            subset = [row for row in rows if row.split == split and row.required_path == depth]
            assert Counter(row.positive for row in subset) == {0: len(subset) // 2, 1: len(subset) // 2}
            assert Counter(row.candidate_chain_first for row in subset) == {
                0: len(subset) // 2, 1: len(subset) // 2,
            }


def test_training_batch_is_balanced_and_shallow():
    rows = training_batch(CONFIG, random.Random(9), 11, 64)
    assert Counter(row.positive for row in rows) == {0: 32, 1: 32}
    assert set(row.required_path for row in rows) <= {1, 2, 3}


def test_cell_aggregation_preserves_depth_and_seed():
    raw = [{"architecture": "L2", "layers": 2, "width": 64, "heads": 4,
            "model_seed": 11, "split": "test", "required_path": depth,
            "top1_correct": correct, "target_margin": 1 if correct else -1}
           for depth in (1, 4) for correct in (1, 0)]
    cells = aggregate(raw)
    assert len(cells) == 2 and {row["accuracy"] for row in cells} == {0.5}
