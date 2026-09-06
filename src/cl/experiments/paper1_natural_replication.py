"""Resumable orchestration and aggregation for Paper 1 natural replication v2."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from cl.common.artifacts import atomic_write_json, stable_hash, write_csv


REQUIRED_TABLES = (
    "selector_by_example.csv",
    "selector_summary.csv",
    "paired_frontier_deltas.csv",
    "candidate_removal.csv",
    "causal_utility_prediction.csv",
)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_seed_output(seed_dir: Path, config: dict, seed: int) -> dict:
    manifest_path = seed_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"missing seed manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    child = manifest.get("config", {})
    expected = {
        "seed": seed,
        "per_split": config["per_split"],
        "candidates": config["candidates"],
        "budgets": config["budget_chunks"],
        "chunk_tokens": config["chunk_tokens"],
        "causal_examples_per_split": config["causal_examples_per_split"],
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "selectors": config["selectors"],
        "retain_full_attention": config["retain_full_attention"],
    }
    mismatches = {key: {"expected": value, "observed": child.get(key)}
                  for key, value in expected.items() if child.get(key) != value}
    if mismatches:
        raise ValueError(f"seed {seed} violates frozen protocol: {mismatches}")
    missing = [name for name in REQUIRED_TABLES if not (seed_dir / "tables" / name).exists()]
    if missing:
        raise ValueError(f"seed {seed} missing tables: {missing}")
    selector_rows = _read_csv(seed_dir / "tables" / "selector_by_example.csv")
    removal_rows = _read_csv(seed_dir / "tables" / "candidate_removal.csv")
    expected_selector_rows = 2 * config["per_split"] * len(config["selectors"]) * len(config["budget_chunks"])
    expected_removal_rows = (2 * 2 * config["causal_examples_per_split"] * config["candidates"])
    if len(selector_rows) != expected_selector_rows or len(removal_rows) != expected_removal_rows:
        raise ValueError(
            f"seed {seed} row counts differ: selectors={len(selector_rows)}/{expected_selector_rows}, "
            f"removals={len(removal_rows)}/{expected_removal_rows}"
        )
    return {
        "seed": seed,
        "manifest_hash": stable_hash(manifest),
        "selector_rows": len(selector_rows),
        "removal_rows": len(removal_rows),
        "device": child.get("device", "unknown"),
    }


def seed_command(config_path: Path, config: dict, output: Path, cache: Path,
                 seed: int, device: str) -> list[str]:
    del config_path  # Included in the parent manifest rather than passed to the v1 runner.
    return [
        sys.executable, "-u", "-m", "cl.experiments.paper1_natural_gate",
        "--repo", ".", "--output", str(output), "--cache", str(cache),
        "--device", device, "--seed", str(seed), "--per-split", str(config["per_split"]),
        "--candidates", str(config["candidates"]), "--budgets",
        *[str(value) for value in config["budget_chunks"]],
        "--causal-examples", str(config["causal_examples_per_split"]),
    ]


def _paired_identity_rows(selector_rows: list[dict], samples: int, seed: int) -> list[dict]:
    grouped = defaultdict(list)
    for row in selector_rows:
        grouped[(row["dataset"], int(row["budget_tokens"]), row["identity_id"], row["condition"])].append(
            float(row["answer_logprob"])
        )
    rng = np.random.default_rng(seed)
    output = []
    datasets = sorted({key[0] for key in grouped})
    for dataset in datasets:
        for budget in sorted({key[1] for key in grouped if key[0] == dataset}):
            identities = sorted({key[2] for key in grouped if key[0] == dataset and key[1] == budget})
            deltas = []
            for identity in identities:
                base = grouped.get((dataset, budget, identity, "base_topk"))
                bridge = grouped.get((dataset, budget, identity, "bridge_preserving"))
                if base and bridge:
                    deltas.append(float(np.mean(bridge) - np.mean(base)))
            values = np.asarray(deltas)
            bootstrap = np.asarray([rng.choice(values, len(values)).mean() for _ in range(samples)])
            output.append({
                "dataset": dataset, "budget_tokens": budget,
                "n_unique_test_identities": len(values),
                "mean_answer_logprob_delta": float(values.mean()),
                "ci_low": float(np.quantile(bootstrap, 0.025)),
                "ci_high": float(np.quantile(bootstrap, 0.975)),
                "wins": int((values > 0).sum()), "ties": int((values == 0).sum()),
            })
    return output


def aggregate(output: Path, config: dict) -> dict:
    selector_rows, prediction_rows, paired_by_seed, audits = [], [], [], []
    identities = defaultdict(lambda: defaultdict(set))
    for seed in config["sampling_seeds"]:
        seed_dir = output / "seeds" / f"seed-{seed}"
        audit = validate_seed_output(seed_dir, config, seed)
        audits.append(audit)
        for row in _read_csv(seed_dir / "tables" / "selector_by_example.csv"):
            row["sampling_seed"] = seed
            selector_rows.append(row)
            identities[row["dataset"]][seed].add(row["identity_id"])
        for row in _read_csv(seed_dir / "tables" / "causal_utility_prediction.csv"):
            row["sampling_seed"] = seed
            prediction_rows.append(row)
        for row in _read_csv(seed_dir / "tables" / "paired_frontier_deltas.csv"):
            row["sampling_seed"] = seed
            paired_by_seed.append(row)
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    write_csv(tables / "selector_by_example_all_seeds.csv", selector_rows)
    write_csv(tables / "causal_prediction_by_seed.csv", prediction_rows)
    write_csv(tables / "paired_frontier_by_seed.csv", paired_by_seed)
    paired = _paired_identity_rows(
        selector_rows, config["bootstrap_samples"], config["sampling_seeds"][0],
    )
    write_csv(tables / "paired_frontier_unique_identities.csv", paired)
    overlap_rows = []
    for dataset, by_seed in sorted(identities.items()):
        union = set().union(*by_seed.values())
        for left_index, left in enumerate(config["sampling_seeds"]):
            for right in config["sampling_seeds"][left_index + 1:]:
                overlap_rows.append({
                    "dataset": dataset, "seed_left": left, "seed_right": right,
                    "left_identities": len(by_seed[left]), "right_identities": len(by_seed[right]),
                    "overlap_identities": len(by_seed[left] & by_seed[right]),
                    "union_identities": len(by_seed[left] | by_seed[right]),
                })
        overlap_rows.append({
            "dataset": dataset, "seed_left": "all", "seed_right": "all",
            "left_identities": sum(len(values) for values in by_seed.values()),
            "right_identities": "", "overlap_identities": "",
            "union_identities": len(union),
        })
    write_csv(tables / "identity_overlap_audit.csv", overlap_rows)
    prediction_summary = []
    prediction_pass = {}
    for dataset in sorted({row["dataset"] for row in prediction_rows}):
        by_key = {(int(row["sampling_seed"]), row["model"]): float(row["r2"])
                  for row in prediction_rows if row["dataset"] == dataset}
        deltas = [by_key[(seed, "surface_plus_structure")] -
                  by_key[(seed, "surface_controls")] for seed in config["sampling_seeds"]]
        prediction_summary.append({
            "dataset": dataset, "seeds": len(deltas), "mean_incremental_r2": float(np.mean(deltas)),
            "min_incremental_r2": float(np.min(deltas)), "max_incremental_r2": float(np.max(deltas)),
            "positive_seeds": int(sum(value > 0 for value in deltas)),
        })
        prediction_pass[dataset] = bool(np.mean(deltas) > 0 and sum(value > 0 for value in deltas) >= 2)
    write_csv(tables / "causal_prediction_summary.csv", prediction_summary)
    frontier_pass = {
        dataset: any(row["dataset"] == dataset and float(row["ci_low"]) > 0 for row in paired)
        for dataset in sorted({row["dataset"] for row in paired})
    }
    decision = {
        "frontier_by_dataset": frontier_pass,
        "causal_prediction_by_dataset": prediction_pass,
        "frontier_pass": bool(frontier_pass and all(frontier_pass.values())),
        "causal_prediction_pass": bool(prediction_pass and all(prediction_pass.values())),
    }
    decision["persistent_learning_gate"] = int(
        decision["frontier_pass"] or decision["causal_prediction_pass"]
    )
    atomic_write_json(output / "gate_decision.json", decision)
    manifest = {
        "schema_version": config["schema_version"],
        "config_hash": stable_hash(config),
        "completed_seeds": len(audits),
        "seed_audits": audits,
        "pooled_selector_rows": len(selector_rows),
        "prediction_rows": len(prediction_rows),
        "paired_seed_rows": len(paired_by_seed),
        "identity_overlap_audited": True,
        "pooled_bridge_unit": "unique test identity; repeated frozen evaluations averaged",
        "paired_results_hash": stable_hash(paired),
        "gate_decision": decision,
    }
    atomic_write_json(output / "replication_manifest.json", manifest)
    return manifest


def run(args) -> None:
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    output, cache = Path(args.output), Path(args.cache)
    (output / "seeds").mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "preregistered_config.json", config)
    pending = list(config["sampling_seeds"])
    if args.max_seeds:
        pending = pending[:args.max_seeds]
    for index, seed in enumerate(pending, 1):
        seed_dir = output / "seeds" / f"seed-{seed}"
        try:
            audit = validate_seed_output(seed_dir, config, seed)
            atomic_write_json(seed_dir / "replication_complete.json", audit)
            print(f"[{index}/{len(pending)}] skip complete seed {seed}", flush=True)
            continue
        except ValueError:
            pass
        command = seed_command(config_path, config, seed_dir, cache, seed, args.device)
        print(f"[{index}/{len(pending)}] run seed {seed}: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)
        audit = validate_seed_output(seed_dir, config, seed)
        atomic_write_json(seed_dir / "replication_complete.json", audit)
    complete = all((output / "seeds" / f"seed-{seed}" / "replication_complete.json").exists()
                   for seed in config["sampling_seeds"])
    if complete:
        print(json.dumps(aggregate(output, config), indent=2))
    else:
        print("Replication remains partial; aggregation deferred.", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper1/natural_replication_v2.json")
    parser.add_argument("--output", default="docs/papers/paper1/results/natural_replication_v2")
    parser.add_argument("--cache", default="tmp/natural_replication_v2_cache")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-seeds", type=int)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    if args.aggregate_only:
        config = json.loads(Path(args.config).read_text())
        print(json.dumps(aggregate(Path(args.output), config), indent=2))
    else:
        run(args)


if __name__ == "__main__":
    parse_args()
