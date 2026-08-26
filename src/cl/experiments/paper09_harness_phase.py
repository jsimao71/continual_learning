"""CPU validation/smoke phase for the Paper 0.9 harness protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cl.common.artifacts import atomic_write_json, write_csv
from cl.semantic.harness import (TypedState, ancestor_reference_controller,
    generate_tree, root_reference_controller, run_loop, transition_rows, validate_tree)


def deepest_leaf(tree):
    return max(tree.nodes, key=lambda node: sum(1 for _ in ancestor_chain(tree, node)))


def ancestor_chain(tree, node):
    while node in tree.parent_of:
        node = tree.parent_of[node]
        yield node


def run(config: dict, smoke: bool = False) -> tuple[list[dict], list[dict], list[dict]]:
    depths = config["smoke_depths"] if smoke else config["tree_depths"]
    branchings = config["smoke_branching"] if smoke else config["branching_factors"]
    seeds = config["tree_seeds"][:1] if smoke else config["tree_seeds"]
    validations, summaries, transitions = [], [], []
    for depth in depths:
        for branching in branchings:
            for seed in seeds:
                tree = generate_tree(depth, branching, seed)
                validations.append({"tree_id": tree.tree_id, **validate_tree(tree, depth, branching)})
                leaf = deepest_leaf(tree)
                cases = [
                    ("root", root_reference_controller, TypedState("root", leaf), tree.root),
                    ("isAncestor_true", ancestor_reference_controller, TypedState("isAncestor", leaf, tree.root), True),
                    ("isAncestor_false", ancestor_reference_controller, TypedState("isAncestor", tree.root, leaf), False),
                ]
                for task, controller, state, expected in cases:
                    run_id = f"{tree.tree_id}:{task}"
                    result = run_loop(tree, state, controller, config["max_steps"], run_id)
                    summaries.append({"run_id": run_id, "task": task, "depth": depth, "branching": branching,
                        "tree_seed": seed, "expected": expected, "answer": result.answer,
                        "correct": result.answer == expected, "terminated": result.terminated,
                        "tool_calls": result.tool_calls, "model_forwards": result.model_forwards,
                        "failure": result.failure})
                    transitions.extend(transition_rows(result))
    return validations, summaries, transitions


def main(args):
    config = json.loads(Path(args.config).read_text())
    validation, summary, transitions = run(config, args.smoke)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "harness_tree_validation.csv", validation)
    write_csv(output / "harness_tree_phase.csv", summary)
    write_csv(output / "harness_transition_log.csv", transitions)
    atomic_write_json(output / "harness_manifest.json", {
        "schema_version": config["schema_version"], "reference_controller": True,
        "reference_results_are_model_results": False, "primitive_grammar": ["parent(node)"],
        "smoke": args.smoke, "trees": len(validation), "runs": len(summary),
        "transitions": len(transitions), "all_valid": all(r["passed"] for r in validation),
        "all_reference_runs_correct": all(r["correct"] for r in summary),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper09/harness_v1.json")
    parser.add_argument("--output", default="docs/papers/paper0_9/results/harness_v1")
    parser.add_argument("--smoke", action="store_true")
    main(parser.parse_args())
