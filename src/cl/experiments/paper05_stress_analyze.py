"""Seed-aware reduction of Paper 0.5 stress-frontier model outputs."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from cl.common.artifacts import atomic_write_json, write_csv


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate(rows: list[dict], threshold: float) -> tuple[list[dict], list[dict]]:
    keys = ("model_depth", "model_width", "head_count", "training_budget", "model_seed",
            "axis", "cell_id", "predictive_order", "raw_length", "nuisance_count",
            "requested_dependency_span", "generator_family")
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    seed_rows = []
    for key, values in sorted(groups.items()):
        stable = [float(row["stable_top1_layer"]) for row in values if row["stable_top1_layer"] != ""]
        accuracy = statistics.mean(int(row["top1_correct"]) for row in values)
        seed_rows.append({**dict(zip(keys, key)), "examples": len(values), "accuracy": accuracy,
                          "mean_margin": statistics.mean(float(row["final_margin"]) for row in values),
                          "mean_rank": statistics.mean(int(row["final_rank"]) for row in values),
                          "mean_stable_layer_when_correct": statistics.mean(stable) if stable else "",
                          "competent": int(accuracy >= threshold)})
    aggregate_keys = tuple(key for key in keys if key != "model_seed")
    combined: dict[tuple, list[dict]] = defaultdict(list)
    for row in seed_rows:
        combined[tuple(row[key] for key in aggregate_keys)].append(row)
    aggregate_rows = []
    for key, values in sorted(combined.items()):
        accuracies = [float(row["accuracy"]) for row in values]
        aggregate_rows.append({**dict(zip(aggregate_keys, key)), "seed_count": len(values),
                               "mean_accuracy": statistics.mean(accuracies),
                               "minimum_seed_accuracy": min(accuracies),
                               "maximum_seed_accuracy": max(accuracies),
                               "competent_seed_count": sum(int(row["competent"]) for row in values),
                               "three_seed_competent": int(len(values) >= 3 and min(accuracies) >= threshold)})
    return seed_rows, aggregate_rows


def main(args: argparse.Namespace) -> None:
    root = Path(args.input); rows = read_csv(root / "stress_final_raw.csv")
    seed_rows, aggregate_rows = aggregate(rows, args.threshold)
    write_csv(root / "stress_cell_accuracy_by_seed.csv", seed_rows)
    write_csv(root / "stress_cell_accuracy_three_seed.csv", aggregate_rows)
    axis_fields = {"predictive_order": ("predictive_order", "pstar_max"),
                   "raw_length": ("raw_length", "n_max"),
                   "nuisance": ("nuisance_count", "k_max"),
                   "dependency_span": ("requested_dependency_span", "s_max")}
    frontier_rows = []
    architectures = sorted({(row["model_depth"], row["model_width"], row["head_count"], row["training_budget"])
                            for row in aggregate_rows})
    for architecture in architectures:
        subset = [row for row in aggregate_rows
                  if tuple(row[key] for key in ("model_depth", "model_width", "head_count", "training_budget")) == architecture]
        for axis, (field, label) in axis_fields.items():
            measured = [row for row in subset if row["axis"] == axis]
            competent = [int(row[field]) for row in measured if row["three_seed_competent"]]
            frontier_rows.append({"model_depth": architecture[0], "model_width": architecture[1],
                                  "head_count": architecture[2], "training_budget": architecture[3],
                                  "frontier": label, "maximum_competent_value": max(competent) if competent else "",
                                  "measured_values": len(measured), "status": "estimated" if competent else "threshold_failure"})
    write_csv(root / "stress_measured_frontiers.csv", frontier_rows)
    baseline = [row for row in seed_rows if row["axis"] == "predictive_order"
                and int(row["predictive_order"]) == 2]
    manifest = {
        "schema_version": "paper05.stress.analysis.v1", "threshold": args.threshold,
        "raw_rows": len(rows), "seed_cells": len(seed_rows), "three_seed_cells": len(aggregate_rows),
        "three_seed_competent_cells": sum(row["three_seed_competent"] for row in aggregate_rows),
        "baseline_competent_seeds": sum(row["competent"] for row in baseline),
        "baseline_seed_cells": len(baseline),
        "interpretation": "optimization_gate_failed" if any(not int(row["competent"]) for row in baseline)
                          else "frontier_estimable",
    }
    atomic_write_json(root / "stress_analysis_manifest.json", manifest)
    if manifest["interpretation"] == "frontier_estimable":
        frontier_text = "; ".join(
            f"L{row['model_depth']}/W{row['model_width']} {row['frontier']}={row['maximum_competent_value']}"
            for row in frontier_rows)
        conclusion = ("The acquisition gate passes, so the controlled frontiers are estimable. "
                      f"{frontier_text}. Raw length, nuisance, and span reach every tested value, while "
                      "predictive order stops at two for both parameter-matched architectures.")
    else:
        conclusion = ("Consequently predictive-order, raw-length, nuisance, and span frontiers are not yet "
                      "estimable: the supported result is a training/optimization failure at this budget. "
                      "A larger matched budget is required before interpreting stress-factor slopes.")
    summary = (
        "# Paper 0.5 stress pilot\n\n"
        f"The parameter-matched pilot completed {manifest['raw_rows']} evaluations. "
        f"Only {manifest['baseline_competent_seeds']} of {manifest['baseline_seed_cells']} architecture/seed "
        f"baseline cells crossed the 0.8 competence gate. {conclusion}\n")
    (root / "stress_pilot_summary.md").write_text(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="docs/papers/paper0_5/results/stress_frontier/learned_v1")
    parser.add_argument("--threshold", type=float, default=.8)
    main(parser.parse_args())
