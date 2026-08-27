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
    summary = (
        "# Paper 0.5 stress pilot\n\n"
        f"The parameter-matched pilot completed {manifest['raw_rows']} evaluations. "
        f"Only {manifest['baseline_competent_seeds']} of {manifest['baseline_seed_cells']} architecture/seed "
        "baseline cells crossed the 0.8 competence gate. No dataset cell was competent in all three seeds. "
        "Consequently predictive-order, raw-length, nuisance, and span frontiers are not yet estimable: "
        "the supported result is a training/optimization failure at the nominal budget. A matched T2 budget "
        "rescue is required before interpreting stress-factor slopes.\n")
    (root / "stress_pilot_summary.md").write_text(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="docs/papers/paper0_5/results/stress_frontier/learned_v1")
    parser.add_argument("--threshold", type=float, default=.8)
    main(parser.parse_args())
