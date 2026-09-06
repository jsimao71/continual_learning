"""Paper 0.9 v3 Stage B: depth and parameter-matched allocation controls."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from cl.common.artifacts import atomic_write_json, write_csv
from cl.experiments.paper05_predictive_order_phase import resolve_device
from cl.experiments.paper09_learned_controller_stage_a import read_csv
from cl.experiments.paper09_learned_controller_v1 import (
    dataset_sha256,
    evaluate_m3,
    evaluate_m4,
    stable_sha256,
    train,
)
from cl.semantic.recurrence_chains import generate_chains, recurrence_pair_split


def stage_b_cells(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["stage"] == "B"]
    if [row["condition"] for row in rows] != ["deep_full", "deep_matched", "shallow_matched"]:
        raise RuntimeError("Stage B architecture specification drift")
    return rows


def validate_stage_a(stage_a: Path, plan: dict) -> dict:
    manifest_path = stage_a / "stage_a_manifest.json"
    gates_path = stage_a / "stage_a_gates.csv"
    if not manifest_path.exists() or not gates_path.exists():
        raise RuntimeError("Stage A prerequisite is absent")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("completed") is not True:
        raise RuntimeError("Stage A prerequisite is partial")
    gates = read_csv(gates_path)
    expected = {
        (int(snapshot), machine)
        for snapshot in plan["stages"]["A"]["snapshot_updates"]
        for machine in plan["common"]["machines"]
    }
    observed = {
        (int(row["snapshot_updates"]), row["machine"])
        for row in gates if int(row.get("gate_complete", 0)) == 1
    }
    if observed != expected:
        raise RuntimeError(f"Stage A prerequisite lacks complete gates: expected={expected}, observed={observed}")
    return {"manifest": str(manifest_path), "manifest_hash": stable_sha256(manifest),
            "complete_gate_cells": len(observed)}


def summarize(rows: list[dict], threshold: float = .95) -> tuple[list[dict], list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["architecture"], row["machine"], int(row["seed"]), int(row["depth"]))].append(row)
    by_seed_depth = []
    for (architecture, machine, seed, depth), values in sorted(grouped.items()):
        mean = lambda key: float(np.mean([float(value[key]) for value in values]))
        record = {"architecture": architecture, "machine": machine, "seed": seed,
                  "depth": depth, "n": len(values), "final_accuracy": mean("final_correct"),
                  "invalid_call_rate": mean("invalid_call")}
        if machine == "M3":
            record.update(one_call_coverage=mean("one_call_edge_coverage"),
                          selected_edge_validity=mean("selected_edge_valid"),
                          post_tool_answer_accuracy=mean("post_tool_answer_correct"))
            record["seed_pass"] = int(record["final_accuracy"] >= threshold and
                                      record["one_call_coverage"] >= threshold and
                                      record["post_tool_answer_accuracy"] >= threshold)
        else:
            stopped = np.asarray([float(value["stop_emitted"]) for value in values])
            correct = np.asarray([float(value["termination_correct"]) for value in values])
            record.update(per_transition_accuracy=mean("per_transition_accuracy"),
                          exact_trajectory_accuracy=mean("exact_trajectory_correct"),
                          termination_precision=float(correct.sum() / max(stopped.sum(), 1.0)),
                          termination_recall=float(correct.mean()),
                          termination_accuracy=float(correct.mean()),
                          nontermination_rate=mean("nontermination"),
                          mean_tool_calls=mean("tool_calls"), mean_model_forwards=mean("model_forwards"))
            record["seed_pass"] = int(record["final_accuracy"] >= threshold and
                                      record["per_transition_accuracy"] >= threshold and
                                      record["exact_trajectory_accuracy"] >= threshold and
                                      record["termination_accuracy"] >= threshold)
        by_seed_depth.append(record)
    frontiers = []
    for architecture in sorted({row["architecture"] for row in by_seed_depth}):
        for machine in ("M3", "M4"):
            values = [row for row in by_seed_depth
                      if row["architecture"] == architecture and row["machine"] == machine]
            if not values:
                continue
            seeds = sorted({row["seed"] for row in values})
            depths = sorted({row["depth"] for row in values})
            complete = len(seeds) == 3 and len(depths) == 6
            frontier = 0
            if complete:
                for depth in depths:
                    cells = [row for row in values if row["depth"] == depth]
                    if len(cells) == 3 and all(row["seed_pass"] for row in cells):
                        frontier = depth
                    else:
                        break
            frontiers.append({"architecture": architecture, "machine": machine,
                              "gate_complete": int(complete), "seeds_observed": len(seeds),
                              "depths_observed": len(depths),
                              "contiguous_frontier": frontier if complete else ""})
    return by_seed_depth, frontiers


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper09/learned_controller_v3_staged.json")
    parser.add_argument("--cells", default="configs/paper09/learned_controller_v3_cells.csv")
    parser.add_argument("--stage-a", default="docs/papers/paper0_9/results/learned_controller_v3/stage_a_budget")
    parser.add_argument("--output")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    ns = parser.parse_args(args)
    plan = json.loads(Path(ns.config).read_text())
    prerequisite = validate_stage_a(Path(ns.stage_a), plan)
    specifications = stage_b_cells(Path(ns.cells))
    common, dataset = plan["common"], plan["dataset"]
    output = Path(ns.output or plan["stages"]["B"]["output"])
    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(ns.device)
    _, train_pairs, test_pairs = recurrence_pair_split(
        dataset["symbol_count"], dataset["pair_split_seed"], dataset["test_pair_fraction"])
    machines = common["machines"][:1] if ns.smoke else common["machines"]
    seeds = common["model_seeds"][:1] if ns.smoke else common["model_seeds"]
    specifications = specifications[:1] if ns.smoke else specifications
    depths = dataset["test_depths"][:2] if ns.smoke else dataset["test_depths"]
    eval_per = 4 if ns.smoke else dataset["eval_per_depth_per_seed"]
    updates = 2 if ns.smoke else plan["stages"]["B"]["updates"]
    batch_size = 4 if ns.smoke else common["batch_size"]
    raw = read_csv(output / "stage_b_raw.csv") if ns.resume else []
    losses = read_csv(output / "stage_b_loss.csv") if ns.resume else []
    required_rows = len(depths) * eval_per
    counts = defaultdict(int)
    for row in raw:
        counts[(row["architecture"], row["machine"], int(row["seed"]))] += 1
    for specification in specifications:
        architecture = specification["condition"]
        model_cfg = {"layers": int(specification["layers"]), "width": int(specification["width"]),
                     "heads": int(specification["heads"]), "mlp_ratio": common["mlp_ratio"]}
        cfg = {"train_depths": dataset["train_depth_default"], "max_length": common["max_length"],
               "model": model_cfg, "learning_rate": common["learning_rate"], "log_every": 50,
               "checkpoint_every": common["checkpoint_every"]}
        for machine in machines:
            for seed in seeds:
                key = (architecture, machine, seed)
                if counts[key] == required_rows:
                    print(f"skip complete {architecture} {machine} seed={seed}", flush=True)
                    continue
                raw = [row for row in raw if (row["architecture"], row["machine"], int(row["seed"])) != key]
                checkpoint = output / "checkpoints" / f"{architecture}_{machine}_seed{seed}.pt"
                model, train_losses = train(machine, seed, cfg, device, checkpoint,
                                            train_pairs, updates, batch_size)
                cell = []
                for depth in depths:
                    examples = generate_chains(test_pairs, depth, eval_per, seed * 1000 + depth, "test")
                    metrics = (evaluate_m3(model, examples, device) if machine == "M3" else
                               evaluate_m4(model, examples, device, common["max_extra_steps"]))
                    cell.extend({"architecture": architecture, "machine": machine, "seed": seed,
                                 "depth": depth, "example_id": index, **row}
                                for index, row in enumerate(metrics))
                raw.extend(cell)
                losses = [row for row in losses if not (
                    row["architecture"] == architecture and row["machine"] == machine and
                    int(row["seed"]) == seed)]
                losses.extend({"architecture": architecture, "machine": machine, "seed": seed, **row}
                              for row in train_losses)
                write_csv(output / "stage_b_raw.csv", raw)
                write_csv(output / "stage_b_loss.csv", losses)
                by_seed_depth, frontiers = summarize(raw, plan["gates"]["competence_threshold"])
                write_csv(output / "stage_b_by_seed_depth.csv", by_seed_depth)
                write_csv(output / "stage_b_frontiers.csv", frontiers)
                print(f"complete {architecture} {machine} seed={seed}", flush=True)
    by_seed_depth, frontiers = summarize(raw, plan["gates"]["competence_threshold"])
    write_csv(output / "stage_b_by_seed_depth.csv", by_seed_depth)
    write_csv(output / "stage_b_frontiers.csv", frontiers)
    required_cells = len(specifications) * len(machines) * len(seeds)
    complete_cells = sum(count == required_rows for count in (
        sum(1 for row in raw if (row["architecture"], row["machine"], int(row["seed"])) ==
            (specification["condition"], machine, seed))
        for specification in specifications for machine in machines for seed in seeds))
    atomic_write_json(output / "stage_b_manifest.json", {
        "schema_version": "paper09.learned_controller.stage_b.v1", "device": str(device),
        "smoke": ns.smoke, "completed": complete_cells == required_cells,
        "complete_cells": complete_cells, "required_cells": required_cells,
        "exact_resume": True, "stage_a_prerequisite": prerequisite,
        "config_sha256": stable_sha256(plan), "cell_spec_sha256": stable_sha256(specifications),
        "dataset_sha256": dataset_sha256(train_pairs), "updates": updates,
        "architectures": [row["condition"] for row in specifications],
        "machines": machines, "seeds": seeds, "raw_rows": len(raw), "frontiers": frontiers,
    })


if __name__ == "__main__":
    main()
