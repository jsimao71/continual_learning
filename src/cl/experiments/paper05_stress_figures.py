"""Publication figures for the matched Paper 0.5 stress-budget experiment."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(args: argparse.Namespace) -> None:
    root = Path(args.results); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    runs = {1: "learned_v1", 2: "learned_v2_matched", 4: "learned_v4_matched"}
    baselines = {}
    for budget, directory in runs.items():
        rows = read(root / directory / "stress_cell_accuracy_by_seed.csv")
        baselines[budget] = [row for row in rows if row["axis"] == "predictive_order" and row["predictive_order"] == "2"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    colors = {("2", "64"): "#2369a1", ("8", "32"): "#d45b2c"}
    for architecture, color in colors.items():
        for seed, marker in zip(("11", "23", "37"), ("o", "s", "^")):
            values = [next(float(row["accuracy"]) for row in baselines[budget]
                           if (row["model_depth"], row["model_width"], row["model_seed"]) == (*architecture, seed))
                      for budget in runs]
            ax.plot(list(runs), values, marker=marker, color=color, alpha=.82,
                    label=f"L{architecture[0]}/W{architecture[1]}, seed {seed}")
    ax.axhline(.8, color="black", linestyle="--", linewidth=1, label="competence gate")
    ax.set(xlabel="training budget multiplier", ylabel="held-out accuracy", xticks=[1, 2, 4], ylim=(0, 1.04))
    ax.legend(ncol=2, fontsize=7, frameon=False); fig.tight_layout()
    fig.savefig(output / "stress_budget_rescue.png", dpi=220); plt.close(fig)

    rows = read(root / "learned_v4_matched" / "stress_cell_accuracy_three_seed.csv")
    axes = (("predictive_order", "predictive_order", "predictive order $p^*$"),
            ("raw_length", "raw_length", "raw length $n$"),
            ("nuisance", "nuisance_count", "nuisance count $k$"),
            ("dependency_span", "requested_dependency_span", "dependency span $s$"))
    fig, panels = plt.subplots(2, 2, figsize=(7.2, 5.2))
    for ax, (axis, field, xlabel) in zip(panels.flat, axes):
        for architecture, color in colors.items():
            selected = [row for row in rows if row["axis"] == axis and
                        (row["model_depth"], row["model_width"]) == architecture]
            selected.sort(key=lambda row: int(row[field]))
            ax.plot([int(row[field]) for row in selected], [float(row["minimum_seed_accuracy"]) for row in selected],
                    marker="o", color=color, label=f"L{architecture[0]}/W{architecture[1]}")
        ax.axhline(.8, color="black", linestyle="--", linewidth=.8)
        ax.set(xlabel=xlabel, ylabel="minimum-seed accuracy", ylim=(0, 1.04))
    panels[0, 0].legend(frameon=False, fontsize=8); fig.tight_layout()
    fig.savefig(output / "stress_factor_frontiers.png", dpi=220); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="docs/papers/paper0_5/results/stress_frontier")
    parser.add_argument("--output", default="docs/papers/paper0_5/figures/stress_frontier")
    main(parser.parse_args())
