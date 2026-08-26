"""Analyze the observed Paper 0.6 v5 Stage-A 1x capacity diagnostic.

This module deliberately does not merge planned training-budget controls with
observed measurements.  The plan is emitted as a separate table.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cl.common.artifacts import atomic_write_json, write_csv


ROOT = Path("docs/papers/paper0_6/results/v5")
STAGE = ROOT / "stage_a_1x"
FIGURES = ROOT / "figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def save(name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    cells = read_csv(STAGE / "capacity_diagnostic_grid.csv")
    metadata = read_csv(STAGE / "capacity_architecture_metadata.csv")
    expected = 7 * 3 * 3 * 5
    assert len(cells) == expected
    assert len({(r["architecture"], r["model_seed"], r["diagnostic_tier"], r["predicate"]) for r in cells}) == expected

    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in cells:
        grouped[(row["architecture"], row["diagnostic_tier"], row["predicate"])].append(float(row["accuracy"]))
    summary = []
    for (architecture, tier, predicate), values in sorted(grouped.items()):
        summary.append({
            "architecture": architecture,
            "diagnostic_tier": tier,
            "predicate": predicate,
            "mean_accuracy": float(np.mean(values)),
            "worst_seed_accuracy": min(values),
            "seed_variance": float(np.var(values)),
            "seed_count": len(values),
            "passes_0_80_worst_seed": min(values) >= .8,
            "passes_0_90_worst_seed": min(values) >= .9,
            "status": "observed_1x",
        })
    write_csv(ROOT / "capacity_diagnostic_summary.csv", summary)

    competent = [r for r in summary if r["passes_0_80_worst_seed"]]
    write_csv(ROOT / "capacity_competence_cells.csv", competent)

    by_arch_pred = {(r["architecture"], r["predicate"], r["diagnostic_tier"]): r for r in summary}
    gaps = []
    for architecture in sorted({r["architecture"] for r in summary}):
        for predicate in sorted({r["predicate"] for r in summary}):
            easy = by_arch_pred[(architecture, predicate, "easy")]
            hard = by_arch_pred[(architecture, predicate, "hard")]
            gaps.append({
                "architecture": architecture,
                "predicate": predicate,
                "easy_mean_accuracy": easy["mean_accuracy"],
                "hard_mean_accuracy": hard["mean_accuracy"],
                "diagnostic_extrapolation_gap": easy["mean_accuracy"] - hard["mean_accuracy"],
                "hard_worst_seed_accuracy": hard["worst_seed_accuracy"],
                "status": "observed_1x_diagnostic_gap",
            })
    write_csv(ROOT / "capacity_extrapolation_gap.csv", gaps)

    meta_by_arch = {}
    for row in metadata:
        meta_by_arch.setdefault(row["architecture"], row)
    pairs = [
        ("baseline", "depth8", "depth allocation; parameter count not matched"),
        ("baseline", "width128", "width allocation; parameter count not matched"),
        ("heads2", "baseline", "same L/W/parameter count; 2 vs 4 heads"),
        ("baseline", "heads8", "same L/W/parameter count; 4 vs 8 heads"),
        ("deep_narrow_match", "shallow_wide_match", "approximate allocation control; parameter count differs 37%"),
    ]
    contrasts = []
    for left, right, design_note in pairs:
        for predicate in sorted({r["predicate"] for r in summary}):
            for tier in ("easy", "medium", "hard"):
                a, b = by_arch_pred[(left, predicate, tier)], by_arch_pred[(right, predicate, tier)]
                contrasts.append({
                    "left_architecture": left,
                    "right_architecture": right,
                    "design_note": design_note,
                    "predicate": predicate,
                    "diagnostic_tier": tier,
                    "left_parameters": int(meta_by_arch[left]["parameters"]),
                    "right_parameters": int(meta_by_arch[right]["parameters"]),
                    "right_minus_left_mean_accuracy": b["mean_accuracy"] - a["mean_accuracy"],
                    "right_minus_left_worst_seed_accuracy": b["worst_seed_accuracy"] - a["worst_seed_accuracy"],
                    "status": "observed_1x",
                })
    write_csv(ROOT / "capacity_parameter_match.csv", contrasts)

    plan = [
        {"wave": 1, "architecture": architecture, "budget_multiplier": 2, "seeds": "11;23;37", "status": "planned_not_run", "rationale": rationale}
        for architecture, rationale in (
            ("baseline", "training-only control for the unstable 1x reference"),
            ("depth8", "test whether the hard-tier worst seed 0.759 crosses the 0.80 gate"),
        )
    ]
    plan += [
        {"wave": 2, "architecture": architecture, "budget_multiplier": 4, "seeds": "11;23;37", "status": "conditional_not_run", "rationale": "run only if 2x approaches/crosses a gate or changes the easy-hard gap materially"}
        for architecture in ("baseline", "depth8")
    ]
    write_csv(ROOT / "capacity_training_budget_plan.csv", plan)

    architectures = ["baseline", "depth8", "width128", "heads2", "heads8", "deep_narrow_match", "shallow_wide_match"]
    predicates = ["parent", "grandparent", "ancestor_k", "root", "isAncestor"]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.1), sharey=True)
    for ax, tier in zip(axes, ("easy", "medium", "hard")):
        matrix = np.array([[by_arch_pred[(a, p, tier)]["mean_accuracy"] for a in architectures] for p in predicates])
        ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_title(tier.capitalize())
        ax.set_xticks(range(len(architectures)), [a.replace("_match", "") for a in architectures], rotation=55, ha="right", fontsize=7)
        ax.set_yticks(range(len(predicates)), predicates)
        for y in range(len(predicates)):
            for x in range(len(architectures)):
                ax.text(x, y, f"{matrix[y,x]:.2f}", ha="center", va="center", fontsize=6, color="white" if matrix[y,x] < .45 else "black")
    fig.suptitle("Three-seed mean accuracy (shared 0--1 color scale)", fontsize=10)
    save("capacity_diagnostic_accuracy.png")

    labels = ["baseline\nL4 W64", "depth8\nL8 W64", "wide\nL4 W128", "deep/narrow\nL12 W48", "shallow/wide\nL4 W96"]
    selected = ["baseline", "depth8", "width128", "deep_narrow_match", "shallow_wide_match"]
    x = np.arange(len(selected)); width = .24
    plt.figure(figsize=(8.2, 3.8))
    for offset, tier in zip((-width, 0, width), ("easy", "medium", "hard")):
        vals = [by_arch_pred[(a, "isAncestor", tier)]["worst_seed_accuracy"] for a in selected]
        plt.bar(x + offset, vals, width, label=tier)
    plt.axhline(.8, color="black", linestyle="--", linewidth=1, label="competence gate")
    plt.xticks(x, labels)
    plt.ylabel("Worst-seed accuracy")
    plt.ylim(0, 1.04)
    plt.legend(ncol=4, fontsize=8)
    save("capacity_parameter_matched_comparison.png")

    atomic_write_json(ROOT / "capacity_analysis_manifest.json", {
        "schema_version": "paper06.capacity_v5.stage_a_analysis.v1",
        "observed_budget_multiplier": 1,
        "observed_architectures": architectures,
        "observed_seeds": [11, 23, 37],
        "observed_competent_cells": competent,
        "central_result": "depth8_isAncestor_easy_medium_only",
        "hard_tier_status": "threshold_failure_worst_seed_0.759",
        "training_budget_status": "planned_not_run",
        "next_run": "baseline,depth8 at 2x across seeds 11,23,37",
    })


if __name__ == "__main__":
    main()
