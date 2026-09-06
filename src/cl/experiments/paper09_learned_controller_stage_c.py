"""Paper 0.9 v3 Stage C: width and head-count controls."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from cl.common.artifacts import atomic_write_json, write_csv
from cl.experiments.paper05_predictive_order_phase import resolve_device
from cl.experiments.paper09_learned_controller_stage_a import read_csv
from cl.experiments.paper09_learned_controller_stage_b import summarize
from cl.experiments.paper09_learned_controller_v1 import (
    dataset_sha256,
    evaluate_m3,
    evaluate_m4,
    stable_sha256,
    train,
)
from cl.semantic.recurrence_chains import generate_chains, recurrence_pair_split


def validate_stage_b(stage_b: Path, plan: dict, *, control_required: bool = False,
                     control_rationale: str = "") -> dict:
    manifest_path = stage_b / "stage_b_manifest.json"
    frontiers_path = stage_b / "stage_b_frontiers.csv"
    if not manifest_path.exists() or not frontiers_path.exists():
        raise RuntimeError("Stage B prerequisite is absent")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("completed") is not True:
        raise RuntimeError("Stage B prerequisite is partial")
    rows = read_csv(frontiers_path)
    expected = {(architecture, machine)
                for architecture in plan["stages"]["B"]["architectures"]
                for machine in plan["common"]["machines"]}
    observed = {(row["architecture"], row["machine"]) for row in rows
                if int(row.get("gate_complete", 0)) == 1 and row.get("contiguous_frontier", "") != ""}
    if observed != expected:
        raise RuntimeError(f"Stage B prerequisite lacks complete frontiers: expected={expected}, observed={observed}")
    winners = {}
    for machine in plan["common"]["machines"]:
        values = [row for row in rows if row["machine"] == machine]
        best = max(int(row["contiguous_frontier"]) for row in values)
        winners[machine] = sorted(row["architecture"] for row in values
                                  if int(row["contiguous_frontier"]) == best)
    unique = [values[0] for values in winners.values() if len(values) == 1]
    allocation_unresolved = any(len(values) != 1 for values in winners.values()) or len(set(unique)) != 1
    if control_required and not control_rationale.strip():
        raise RuntimeError("an explicit control rationale is required")
    enabled = allocation_unresolved or control_required
    if not enabled:
        raise RuntimeError("Stage C is disabled: Stage B has one shared unique allocation winner")
    return {
        "stage_b_manifest": str(manifest_path), "stage_b_manifest_hash": stable_sha256(manifest),
        "stage_b_frontiers": str(frontiers_path), "stage_b_frontiers_hash": stable_sha256(rows),
        "winner_sets": winners, "allocation_unresolved": allocation_unresolved,
        "control_required": control_required, "control_rationale": control_rationale.strip(),
        "enabled": enabled,
    }


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper09/learned_controller_v3_staged.json")
    parser.add_argument("--cells", default="configs/paper09/learned_controller_v3_cells.csv")
    parser.add_argument("--stage-b", default="docs/papers/paper0_9/results/learned_controller_v3/stage_b_depth")
    parser.add_argument("--output")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--control-required", action="store_true")
    parser.add_argument("--control-rationale", default="")
    ns = parser.parse_args(args)
    plan = json.loads(Path(ns.config).read_text())
    selection = validate_stage_b(Path(ns.stage_b), plan, control_required=ns.control_required,
                                 control_rationale=ns.control_rationale)
    output = Path(ns.output or plan["stages"]["C"]["output"])
    output.mkdir(parents=True, exist_ok=True)
    selection.update({"schema_version": "paper09.learned_controller.stage_c.selection.v1",
                      "config_sha256": stable_sha256(plan)})
    atomic_write_json(output / "stage_c_selection_manifest.json", selection)
    if ns.prepare_only:
        return
    # Use the same committed cell-table schema while selecting the four Stage C rows.
    with Path(ns.cells).open(newline="", encoding="utf-8") as handle:
        specifications = [row for row in csv.DictReader(handle) if row["stage"] == "C"]
    expected_architectures = ["narrow", "wide", "one_head", "eight_heads"]
    if [row["condition"] for row in specifications] != expected_architectures:
        raise RuntimeError("Stage C architecture specification drift")
    common, dataset = plan["common"], plan["dataset"]
    device = resolve_device(ns.device)
    _, train_pairs, test_pairs = recurrence_pair_split(
        dataset["symbol_count"], dataset["pair_split_seed"], dataset["test_pair_fraction"])
    machines = common["machines"][:1] if ns.smoke else common["machines"]
    seeds = common["model_seeds"][:1] if ns.smoke else common["model_seeds"]
    specifications = specifications[:1] if ns.smoke else specifications
    depths = dataset["test_depths"][:2] if ns.smoke else dataset["test_depths"]
    eval_per = 4 if ns.smoke else dataset["eval_per_depth_per_seed"]
    updates = 2 if ns.smoke else plan["stages"]["C"]["updates"]
    batch_size = 4 if ns.smoke else common["batch_size"]
    raw = read_csv(output / "stage_c_raw.csv") if ns.resume else []
    losses = read_csv(output / "stage_c_loss.csv") if ns.resume else []
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
                write_csv(output / "stage_c_raw.csv", raw)
                write_csv(output / "stage_c_loss.csv", losses)
                by_seed_depth, frontiers = summarize(raw, plan["gates"]["competence_threshold"])
                write_csv(output / "stage_c_by_seed_depth.csv", by_seed_depth)
                write_csv(output / "stage_c_frontiers.csv", frontiers)
                print(f"complete {architecture} {machine} seed={seed}", flush=True)
    by_seed_depth, frontiers = summarize(raw, plan["gates"]["competence_threshold"])
    write_csv(output / "stage_c_by_seed_depth.csv", by_seed_depth)
    write_csv(output / "stage_c_frontiers.csv", frontiers)
    required_cells = len(specifications) * len(machines) * len(seeds)
    complete_cells = sum(
        sum(1 for row in raw if (row["architecture"], row["machine"], int(row["seed"])) ==
            (specification["condition"], machine, seed)) == required_rows
        for specification in specifications for machine in machines for seed in seeds)
    atomic_write_json(output / "stage_c_manifest.json", {
        "schema_version": "paper09.learned_controller.stage_c.v1", "device": str(device),
        "smoke": ns.smoke, "completed": complete_cells == required_cells,
        "complete_cells": complete_cells, "required_cells": required_cells,
        "exact_resume": True, "selection_manifest_hash": stable_sha256(selection),
        "config_sha256": stable_sha256(plan), "cell_spec_sha256": stable_sha256(specifications),
        "dataset_sha256": dataset_sha256(train_pairs), "updates": updates,
        "architectures": [row["condition"] for row in specifications],
        "machines": machines, "seeds": seeds, "raw_rows": len(raw), "frontiers": frontiers,
    })


if __name__ == "__main__":
    main()
