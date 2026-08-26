import json
from pathlib import Path

import pytest

from cl.experiments.paper09_harness_phase import run
from cl.semantic.harness import (Action, NO_PARENT, TypedState,
    ancestor_reference_controller, execute, generate_tree, root_reference_controller,
    run_loop, validate_tree)


CONFIG = json.loads(Path("configs/paper09/harness_v1.json").read_text())


def test_tree_generation_is_deterministic_and_valid():
    first = generate_tree(4, 2, 11)
    second = generate_tree(4, 2, 11)
    assert first == second
    assert validate_tree(first, 4, 2)["passed"]
    assert len(first.nodes) == 9  # root + depth * local branching


def test_deep_branched_tree_remains_linear_size():
    tree = generate_tree(64, 8, 11)
    assert len(tree.nodes) == 1 + 64 * 8
    assert validate_tree(tree, 64, 8)["passed"]


def test_executor_has_only_local_parent_primitive():
    tree = generate_tree(2, 2, 11)
    child = next(iter(tree.parent_of))
    assert execute(tree, Action("CALL", "parent", child)) == tree.parent_of[child]
    assert execute(tree, Action("CALL", "parent", tree.root)) == NO_PARENT
    with pytest.raises(ValueError, match="unsupported primitive"):
        execute(tree, Action("CALL", "children", tree.root))


def test_root_and_ancestor_reference_protocols_log_all_transitions():
    tree = generate_tree(6, 1, 23)
    leaf = next(node for node in tree.nodes if node not in set(tree.parent_of.values()))
    root = run_loop(tree, TypedState("root", leaf), root_reference_controller, 8, "root")
    ancestor = run_loop(tree, TypedState("isAncestor", leaf, tree.root), ancestor_reference_controller, 8, "ancestor")
    # Root needs one final parent(root)->NO_PARENT callback to observe stopping.
    assert root.answer == tree.root and root.tool_calls == 7 and root.model_forwards == 8
    assert ancestor.answer is True and ancestor.tool_calls == 6
    assert all(t.state_before and t.action for t in root.transitions)


def test_max_steps_is_explicit_failure_not_silent_answer():
    tree = generate_tree(6, 1, 37)
    leaf = next(node for node in tree.nodes if node not in set(tree.parent_of.values()))
    result = run_loop(tree, TypedState("root", leaf), root_reference_controller, 2, "bounded")
    assert result.answer is None and result.failure == "max_steps" and result.tool_calls == 2


def test_cpu_smoke_matrix_passes_and_has_transition_accounting():
    validation, summaries, transitions = run(CONFIG, smoke=True)
    assert len(validation) == 4 and len(summaries) == 12
    assert all(row["passed"] for row in validation)
    assert all(row["correct"] for row in summaries)
    assert sum(row["model_forwards"] for row in summaries) == len(transitions)
