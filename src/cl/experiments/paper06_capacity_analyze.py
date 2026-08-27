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
        {"wave": 1, "architecture": "baseline", "budget_multiplier": 2, "seeds": "11;23;37", "status": "observed_complete", "rationale": "training-only control for the unstable 1x reference"},
        {"wave": 1, "architecture": "depth8", "budget_multiplier": 2, "seeds": "11;23", "status": "observed_partial", "rationale": "two observed seeds; central gate unevaluable"},
        {"wave": 1, "architecture": "depth8", "budget_multiplier": 2, "seeds": "37", "status": "planned_completion", "rationale": "complete the preregistered three-seed hard-tier gate"},
    ]
    plan += [
        {"wave": 2, "architecture": architecture, "budget_multiplier": 4, "seeds": "11;23;37", "status": "conditional_not_run", "rationale": "run only if 2x approaches/crosses a gate or changes the easy-hard gap materially"}
        for architecture in ("baseline", "depth8")
    ]
    write_csv(ROOT / "capacity_training_budget_plan.csv", plan)

    selected_2x = ROOT / "stage_a_2x_selected"
    partial_path = selected_2x / "capacity_diagnostic_grid_partial.csv"
    partial_manifest_path = ROOT / "stage_a_2x_selected" / "capacity_stage_a_partial_manifest.json"
    partial_summary = []
    complete_2x_summary = []
    if partial_path.exists():
        partial = read_csv(partial_path)
        completion_path = selected_2x / "capacity_diagnostic_grid.csv"
        completion = read_csv(completion_path) if completion_path.exists() else []
        completion_is_seed37 = {
            (r["architecture"], int(r["model_seed"])) for r in completion
        } == {("depth8", 37)}
        complete_2x = partial + completion if completion_is_seed37 else []
        if complete_2x:
            keys = [(r["architecture"], r["model_seed"], r["diagnostic_tier"], r["predicate"]) for r in complete_2x]
            assert len(complete_2x) == 90 and len(set(keys)) == 90
            for row in complete_2x:
                row["status"] = "observed_complete_2x"
            write_csv(selected_2x / "capacity_diagnostic_grid_complete.csv", complete_2x)
            raw_partial = read_csv(selected_2x / "capacity_diagnostic_raw_partial.csv")
            raw_completion = read_csv(selected_2x / "capacity_diagnostic_raw.csv")
            assert len(raw_partial) == 10_800 and len(raw_completion) == 2_160
            write_csv(selected_2x / "capacity_diagnostic_raw_complete.csv", raw_partial + raw_completion)
        partial_groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for row in partial:
            partial_groups[(row["architecture"], row["diagnostic_tier"], row["predicate"])].append(float(row["accuracy"]))
        for (architecture, tier, predicate), values in sorted(partial_groups.items()):
            partial_summary.append({
                "architecture": architecture,
                "diagnostic_tier": tier,
                "predicate": predicate,
                "budget_multiplier": 2,
                "mean_accuracy": float(np.mean(values)),
                "worst_observed_seed_accuracy": min(values),
                "observed_seed_count": len(values),
                "expected_seed_count": 3,
                "gate_status": "observed_pass" if len(values) == 3 and min(values) >= .8 else "observed_fail" if len(values) == 3 else "not_evaluable_incomplete_seeds",
                "status": "observed_partial_2x",
            })
        write_csv(ROOT / "capacity_training_sweep_partial.csv", partial_summary)

        if complete_2x:
            complete_groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
            for row in complete_2x:
                complete_groups[(row["architecture"], row["diagnostic_tier"], row["predicate"])].append(float(row["accuracy"]))
            for (architecture, tier, predicate), values in sorted(complete_groups.items()):
                assert len(values) == 3
                complete_2x_summary.append({
                    "architecture": architecture,
                    "diagnostic_tier": tier,
                    "predicate": predicate,
                    "budget_multiplier": 2,
                    "mean_accuracy": float(np.mean(values)),
                    "worst_seed_accuracy": min(values),
                    "seed_variance": float(np.var(values)),
                    "seed_count": len(values),
                    "passes_0_80_worst_seed": min(values) >= .8,
                    "passes_0_90_worst_seed": min(values) >= .9,
                    "status": "observed_complete_2x",
                })
            write_csv(ROOT / "capacity_training_sweep_complete.csv", complete_2x_summary)
            completed_plan = [
                {"wave": 1, "architecture": architecture, "budget_multiplier": 2, "seeds": "11;23;37", "status": "observed_complete", "rationale": rationale}
                for architecture, rationale in (
                    ("baseline", "training-only control for the unstable 1x reference"),
                    ("depth8", "completed preregistered three-seed hard-tier gate"),
                )
            ]
            completed_plan += [
                {"wave": 2, "architecture": architecture, "budget_multiplier": 4, "seeds": "11;23;37", "status": "not_run_not_justified", "rationale": "complete 2x comparison resolves the selected budget control; prioritize independent D/d/b/N frontiers"}
                for architecture in ("baseline", "depth8")
            ]
            write_csv(ROOT / "capacity_training_budget_plan.csv", completed_plan)
            atomic_write_json(selected_2x / "capacity_stage_a_complete_manifest.json", {
                "schema_version": "paper06.capacity_v5.stage_a_complete.v1",
                "architectures": ["baseline", "depth8"],
                "model_seeds": [11, 23, 37],
                "budget_multiplier": 2,
                "checkpoints": 6,
                "aggregate_cells": 90,
                "raw_rows": 12_960,
                "status": "complete_6_of_6",
            })

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

    budget_summary = complete_2x_summary or partial_summary
    if budget_summary:
        budget_lookup = {(r["architecture"], r["diagnostic_tier"], r["predicate"]): r for r in budget_summary}
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)
        for ax, architecture in zip(axes, ("baseline", "depth8")):
            x = np.arange(3); tiers = ("easy", "medium", "hard")
            one_x = [by_arch_pred[(architecture, "isAncestor", tier)]["worst_seed_accuracy"] for tier in tiers]
            worst_field = "worst_seed_accuracy" if complete_2x_summary else "worst_observed_seed_accuracy"
            two_x = [budget_lookup[(architecture, tier, "isAncestor")][worst_field] for tier in tiers]
            ax.bar(x - .18, one_x, .36, label="1x, 3 seeds")
            count_field = "seed_count" if complete_2x_summary else "observed_seed_count"
            seed_count = budget_lookup[(architecture, "easy", "isAncestor")][count_field]
            ax.bar(x + .18, two_x, .36, label=f"2x, {seed_count} seeds")
            ax.axhline(.8, color="black", linestyle="--", linewidth=1)
            ax.set_xticks(x, tiers)
            ax.set_title(architecture)
            ax.set_ylim(0, 1.04)
            ax.set_ylabel("Worst observed seed accuracy")
            ax.legend(fontsize=8, loc="lower left")
        fig.suptitle("Training-budget recovery; complete three-seed comparison" if complete_2x_summary else "Training-budget recovery; depth-8 2x remains incomplete")
        save("capacity_training_budget_recovery.png")

    atomic_write_json(ROOT / "capacity_analysis_manifest.json", {
        "schema_version": "paper06.capacity_v5.stage_a_analysis.v1",
        "observed_budget_multiplier": 1,
        "observed_architectures": architectures,
        "observed_seeds": [11, 23, 37],
        "observed_competent_cells": competent,
        "central_result": "depth8_isAncestor_easy_medium_only",
        "hard_tier_status": "threshold_failure_worst_seed_0.759",
        "training_budget_status": "complete_2x_six_of_six" if complete_2x_summary else "partial_2x_five_of_six" if partial_summary else "planned_not_run",
        "training_manifest": str(selected_2x / "capacity_stage_a_complete_manifest.json") if complete_2x_summary else str(partial_manifest_path) if partial_summary else None,
        "next_run": "independently controlled D/d/b/N frontier; no broad 4x grid" if complete_2x_summary else "depth8 seed37 at 2x; reconstruct complete aggregate from all six checkpoints" if partial_summary else "baseline,depth8 at 2x across seeds 11,23,37",
    })


if __name__ == "__main__":
    main()
