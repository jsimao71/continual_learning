"""Minimal, auditable tree harness for Paper 0.9.

The executor implements only the declared local primitive ``parent(node)``.  The
reference controllers below are deterministic protocol oracles used to validate
serialization, stopping, and accounting; they are not intended as model results.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Callable, Literal


NO_PARENT = "NO_PARENT"
ActionKind = Literal["CALL", "ANSWER"]


@dataclass(frozen=True)
class Tree:
    tree_id: str
    parent_of: dict[str, str]
    root: str

    @property
    def nodes(self) -> tuple[str, ...]:
        return tuple(sorted({self.root, *self.parent_of, *self.parent_of.values()}))

    def parent(self, node: str) -> str:
        if node not in self.nodes:
            raise ValueError(f"unknown node: {node}")
        return self.parent_of.get(node, NO_PARENT)


@dataclass(frozen=True)
class TypedState:
    goal: str
    current_node: str
    target_node: str | None = None
    tool_result: str | None = None
    step_count: int | None = None

    def serialize(self) -> str:
        fields = [f"GOAL={self.goal}", f"CURRENT_NODE={self.current_node}"]
        if self.target_node is not None:
            fields.append(f"TARGET_NODE={self.target_node}")
        if self.tool_result is not None:
            fields.append(f"TOOL_RESULT={self.tool_result}")
        if self.step_count is not None:
            fields.append(f"STEP_COUNT={self.step_count}")
        return " <STATE> ".join(fields)


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    op: str | None = None
    argument: str | None = None
    value: str | bool | None = None

    def proto(self) -> str:
        if self.kind == "CALL":
            return f"<CALL {self.op} {self.argument}>"
        return f"<ANSWER {str(self.value).lower() if isinstance(self.value, bool) else self.value}>"


@dataclass(frozen=True)
class Transition:
    run_id: str
    iteration: int
    state_before: str
    action: str
    result: str | None
    state_after: str
    terminated: bool
    error_type: str | None


@dataclass(frozen=True)
class RunResult:
    answer: str | bool | None
    terminated: bool
    transitions: tuple[Transition, ...]
    tool_calls: int
    model_forwards: int
    failure: str | None


def generate_tree(depth: int, branching: int, seed: int) -> Tree:
    """Generate a deterministic depth spine with controlled off-path branches.

    A full depth-64, branching-8 tree is intractably large.  This construction
    varies the local branching factor without coupling it exponentially to
    traversal depth: one child continues the spine and the others are leaves.
    """
    if depth < 1 or branching < 1:
        raise ValueError("depth and branching must be positive")
    rng = random.Random(seed)
    root = f"n{seed}_0"
    parent_of: dict[str, str] = {}
    serial = 1
    spine_parent = root
    for _ in range(depth):
        children = []
        for _ in range(branching):
            child = f"n{seed}_{serial}_{rng.randrange(1_000_000):06d}"
            serial += 1
            parent_of[child] = spine_parent
            children.append(child)
        spine_parent = children[0]
    return Tree(f"tree_D{depth}_b{branching}_s{seed}", parent_of, root)


def execute(tree: Tree, action: Action) -> str:
    """Execute a declared primitive without planning or recursive search."""
    if action.kind != "CALL":
        raise ValueError("executor accepts CALL actions only")
    if action.op != "parent" or action.argument is None:
        raise ValueError(f"unsupported primitive: {action.op}")
    return tree.parent(action.argument)


Controller = Callable[[TypedState], Action]


def run_loop(tree: Tree, initial: TypedState, controller: Controller, max_steps: int, run_id: str) -> RunResult:
    """Run a controller with a frozen protocol and log every transition."""
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    state = initial
    trace: list[Transition] = []
    for iteration in range(max_steps + 1):
        before = state.serialize()
        try:
            action = controller(state)
        except Exception as exc:  # controller failures are data, not silent crashes
            trace.append(Transition(run_id, iteration, before, "CONTROLLER_ERROR", None, before, True, type(exc).__name__))
            return RunResult(None, True, tuple(trace), sum(t.result is not None for t in trace), len(trace), "controller_error")
        if action.kind == "ANSWER":
            trace.append(Transition(run_id, iteration, before, action.proto(), None, before, True, None))
            return RunResult(action.value, True, tuple(trace), sum(t.result is not None for t in trace), len(trace), None)
        if iteration == max_steps:
            trace.append(Transition(run_id, iteration, before, action.proto(), None, before, True, "max_steps"))
            return RunResult(None, True, tuple(trace), sum(t.result is not None for t in trace), len(trace), "max_steps")
        try:
            result = execute(tree, action)
        except ValueError as exc:
            error = "wrong_primitive" if "primitive" in str(exc) else "wrong_argument"
            trace.append(Transition(run_id, iteration, before, action.proto(), None, before, True, error))
            return RunResult(None, True, tuple(trace), sum(t.result is not None for t in trace), len(trace), error)
        state = TypedState(state.goal, result if result != NO_PARENT else state.current_node,
                           state.target_node, result,
                           None if state.step_count is None else state.step_count + 1)
        trace.append(Transition(run_id, iteration, before, action.proto(), result, state.serialize(), False, None))
    raise AssertionError("unreachable")


def root_reference_controller(state: TypedState) -> Action:
    """Protocol oracle: repeat the local parent call and stop at NO_PARENT."""
    if state.tool_result == NO_PARENT:
        return Action("ANSWER", value=state.current_node)
    return Action("CALL", op="parent", argument=state.current_node)


def ancestor_reference_controller(state: TypedState) -> Action:
    """Protocol oracle for ancestry; comparison is controller-visible state."""
    if state.current_node == state.target_node:
        return Action("ANSWER", value=True)
    if state.tool_result == NO_PARENT:
        return Action("ANSWER", value=False)
    return Action("CALL", op="parent", argument=state.current_node)


def transition_rows(result: RunResult) -> list[dict]:
    return [asdict(t) for t in result.transitions]


def validate_tree(tree: Tree, expected_depth: int, expected_branching: int) -> dict:
    indegree = {node: 0 for node in tree.nodes}
    for child, parent in tree.parent_of.items():
        if child == parent or parent not in indegree:
            return {"passed": False, "reason": "invalid_edge"}
        indegree[child] += 1
    depths = []
    for node in tree.nodes:
        seen = set()
        current = node
        depth = 0
        while current != tree.root:
            if current in seen or current not in tree.parent_of:
                return {"passed": False, "reason": "cycle_or_disconnected"}
            seen.add(current)
            current = tree.parent_of[current]
            depth += 1
        depths.append(depth)
    children = {node: 0 for node in tree.nodes}
    for parent in tree.parent_of.values():
        children[parent] += 1
    internal = [n for n, count in children.items() if count]
    passed = max(depths) == expected_depth and all(children[n] == expected_branching for n in internal)
    return {"passed": passed, "reason": None if passed else "shape_mismatch", "nodes": len(tree.nodes),
            "edges": len(tree.parent_of), "depth": max(depths), "branching": expected_branching}
