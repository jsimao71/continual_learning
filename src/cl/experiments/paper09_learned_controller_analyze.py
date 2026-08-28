"""Re-label and aggregate Paper 0.9 learned-controller v1 evidence.

This analysis preserves the original prediction rows.  It corrects the semantic
error that called M3's one valid call divided by proof depth a transition
accuracy; that quantity is one-call edge coverage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cl.common.artifacts import atomic_write_json, write_csv


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corrected_row(row: dict) -> dict:
    depth = int(row["depth"])
    base = {
        "machine": row["machine"], "seed": int(row["seed"]), "depth": depth,
        "example_id": int(row["example_id"]), "final_correct": int(row["final_correct"]),
        "invalid_call": int(row["invalid_call"]), "nontermination": int(row["nontermination"]),
        "tool_calls": int(row["tool_calls"]), "model_forwards": int(row["model_forwards"]),
    }
    if row["machine"] == "M3":
        edge_valid = int(round(float(row["rule_selection_accuracy"])))
        base.update({
            "selected_edge_valid": edge_valid,
            "one_call_edge_coverage": edge_valid / depth,
            "post_tool_answer_correct": int(row["final_correct"]),
            "per_transition_accuracy": "", "exact_trajectory_correct": "",
            "stop_emitted": "", "termination_correct": "",
        })
    else:
        termination = int(row["termination_correct"])
        stop_emitted = int(not int(row["invalid_call"]) and not int(row["nontermination"]))
        base.update({
            "selected_edge_valid": "", "one_call_edge_coverage": "",
            "post_tool_answer_correct": "",
            "per_transition_accuracy": float(row["transition_accuracy"]),
            "exact_trajectory_correct": int(row["final_correct"]),
            "stop_emitted": stop_emitted, "termination_correct": termination,
        })
    return base


def aggregate(rows: list[dict], threshold: float = .95):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["machine"], row["seed"], row["depth"])].append(row)
    by_seed_depth = []
    for (machine, seed, depth), values in sorted(grouped.items()):
        mean = lambda key: float(np.mean([float(v[key]) for v in values]))
        record = {"machine": machine, "seed": seed, "depth": depth, "n": len(values),
                  "final_accuracy": mean("final_correct"), "invalid_call_rate": mean("invalid_call"),
                  "nontermination_rate": mean("nontermination")}
        if machine == "M3":
            record.update({"selected_edge_validity": mean("selected_edge_valid"),
                           "one_call_edge_coverage": mean("one_call_edge_coverage"),
                           "post_tool_answer_accuracy": mean("post_tool_answer_correct"),
                           "per_transition_accuracy": "", "exact_trajectory_accuracy": "",
                           "termination_precision": "", "termination_recall": "", "termination_accuracy": ""})
            record["seed_pass"] = int(record["final_accuracy"] >= threshold and record["selected_edge_validity"] >= threshold)
        else:
            stop = sum(int(v["stop_emitted"]) for v in values);correct_stop = sum(int(v["termination_correct"]) for v in values)
            record.update({"selected_edge_validity": "", "one_call_edge_coverage": "", "post_tool_answer_accuracy": "",
                           "per_transition_accuracy": mean("per_transition_accuracy"),
                           "exact_trajectory_accuracy": mean("exact_trajectory_correct"),
                           "termination_precision": correct_stop / stop if stop else 0.0,
                           "termination_recall": correct_stop / len(values), "termination_accuracy": correct_stop / len(values)})
            record["seed_pass"] = int(record["final_accuracy"] >= threshold and record["per_transition_accuracy"] >= threshold
                                      and record["termination_accuracy"] >= threshold)
        by_seed_depth.append(record)

    depth_groups = defaultdict(list)
    for row in by_seed_depth:
        depth_groups[(row["machine"], row["depth"])].append(row)
    by_depth = []
    for (machine, depth), values in sorted(depth_groups.items()):
        record = {"machine": machine, "depth": depth, "seeds": len(values),
                  "mean_final_accuracy": float(np.mean([v["final_accuracy"] for v in values])),
                  "worst_seed_final_accuracy": min(v["final_accuracy"] for v in values),
                  "acquisition_probability": float(np.mean([v["seed_pass"] for v in values])),
                  "all_seed_gate": int(all(v["seed_pass"] for v in values))}
        metric = "one_call_edge_coverage" if machine == "M3" else "per_transition_accuracy"
        record["mean_execution_metric"] = float(np.mean([v[metric] for v in values]))
        by_depth.append(record)

    frontiers = []
    for machine in sorted({r["machine"] for r in by_depth}):
        values = sorted((r for r in by_depth if r["machine"] == machine), key=lambda r: r["depth"])
        prefix = 0
        for row in values:
            if row["all_seed_gate"] and row["depth"] == prefix + 1:
                prefix = row["depth"]
            else:
                break
        frontiers.append({"machine": machine, "contiguous_frontier": prefix,
                          "acquisition_probability_K4": next(r["acquisition_probability"] for r in values if r["depth"] == 4),
                          "claim": "finite_learned_frontier_not_closure"})
    return by_seed_depth, by_depth, frontiers


def plot(by_depth: list[dict], output: Path):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.1))
    for machine, marker in (("M3", "o"), ("M4", "s")):
        values = [r for r in by_depth if r["machine"] == machine]
        axes[0].plot([r["depth"] for r in values], [r["mean_final_accuracy"] for r in values], marker=marker, label=machine)
        axes[1].plot([r["depth"] for r in values], [r["mean_execution_metric"] for r in values], marker=marker, label=machine)
    for axis in axes:
        axis.axvline(3, color="black", ls="--", lw=.8);axis.axhline(.95, color="gray", ls=":", lw=.8)
        axis.set_xlabel("proof depth K");axis.set_ylim(-.03, 1.03);axis.grid(alpha=.2);axis.legend()
    axes[0].set_ylabel("exact final-answer accuracy");axes[0].set_title("Learned controller outcome")
    axes[1].set_ylabel("execution metric");axes[1].set_title("M3 one-call coverage; M4 transitions")
    fig.tight_layout();output.parent.mkdir(parents=True, exist_ok=True);fig.savefig(output, dpi=220);plt.close(fig)


def main(args=None):
    parser = argparse.ArgumentParser();parser.add_argument("--input", default="docs/papers/paper0_9/results/learned_controller_v1")
    parser.add_argument("--output", default="docs/papers/paper0_9/results/learned_controller_v1/analysis_v2")
    ns = parser.parse_args(args);source=Path(ns.input);out=Path(ns.output);out.mkdir(parents=True, exist_ok=True)
    raw_path=source/"learned_controller_raw.csv";rows=[corrected_row(r) for r in read_csv(raw_path)]
    by_seed,by_depth,frontiers=aggregate(rows);write_csv(out/"learned_controller_raw_semantic_v2.csv",rows)
    write_csv(out/"learned_controller_by_seed_depth.csv",by_seed);write_csv(out/"learned_controller_by_depth.csv",by_depth)
    write_csv(out/"learned_controller_frontiers.csv",frontiers);plot(by_depth,out/"figures/learned_m3_m4_depth.png")
    atomic_write_json(out/"analysis_manifest.json",{"schema_version":"paper09.learned_controller.analysis_v2.v1",
        "source_raw":str(raw_path),"source_raw_sha256":sha256(raw_path),"source_predictions_preserved":True,
        "raw_rows":len(rows),"metric_correction":{"M3_legacy_transition_accuracy":"relabelled_one_call_edge_coverage",
        "M3_selected_edge_validity":"legacy_rule_selection_accuracy","M4_transition_accuracy":"retained_as_per_transition_accuracy"},
        "frontiers":{r["machine"]:r["contiguous_frontier"] for r in frontiers},"learned_model_results":True})


if __name__ == "__main__": main()
