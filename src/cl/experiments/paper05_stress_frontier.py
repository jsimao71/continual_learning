"""Designed Paper 0.5 predictive-refinement stress frontier.

The planner varies one data axis at a time around a declared baseline.  This
keeps predictive order, raw length, nuisance count, and dependency span
separate rather than silently using an invalid Cartesian product.  Training is
intentionally delegated to a later accelerator phase; this module provides the
deterministic dataset, architecture plan, and competence/frontier reducers.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import Counter
from pathlib import Path

from cl.common.artifacts import atomic_write_json, stable_hash, write_csv

VALUE_ROLES = tuple(tuple(range(4 + 4 * i, 8 + 4 * i)) for i in range(12))
TARGETS = tuple(range(160, 164))
START, QUERY, NEUTRAL = 170, 171, 172
NUISANCE = tuple(range(100, 132))


def target_rule(values: list[int]) -> int:
    return sum(values) % 4


def feasible(order: int, raw_length: int, span: int, family: str) -> tuple[bool, str]:
    if raw_length < order:
        return False, "raw_length_below_predictive_order"
    if family == "contiguous" and span != order:
        return False, "contiguous_span_must_equal_order"
    if family in {"skip_gram", "necessary_pattern"} and span < order:
        return False, "span_below_predictive_order"
    return True, "eligible"


def core_positions(order: int, span: int, family: str) -> list[int]:
    ok, reason = feasible(order, max(order, 2), span, family)
    if not ok:
        raise ValueError(reason)
    if family == "contiguous":
        return list(range(order))
    if order == 1:
        return [0]
    # Exact endpoint-to-endpoint span with deterministic, distinct positions.
    return [(i * (span - 1)) // (order - 1) for i in range(order)]


def make_example(config: dict, *, order: int, raw_length: int, nuisance: int,
                 span: int, family: str, index: int, split: str) -> dict:
    ok, reason = feasible(order, raw_length, span, family)
    if not ok:
        raise ValueError(reason)
    seed = config["dataset_seed"] + (0 if split == "train" else 20_000_000) + index * 1597 + order * 101
    rng = random.Random(seed)
    values = [rng.randrange(4) for _ in range(order)]
    positions = core_positions(order, span, family)
    pattern = [NEUTRAL] * max(raw_length, span)
    for role, (position, value) in enumerate(zip(positions, values)):
        # Skip-grams reuse one surface alphabet and require position-sensitive
        # binding; necessary patterns make the latent roles explicit.
        surface_role = 0 if family == "skip_gram" else role
        pattern[position] = VALUE_ROLES[surface_role][value]
    nuisance_tokens = [rng.choice(NUISANCE) for _ in range(nuisance)]
    body = [START, *nuisance_tokens, *pattern, QUERY]
    padding = config["sequence_length"] - len(body)
    if padding < 0:
        raise ValueError("sequence_length_too_small")
    tokens = [NEUTRAL] * padding + body
    return {
        "tokens": tokens,
        "target": TARGETS[target_rule(values)],
        "predictive_order": order,
        "raw_length": raw_length,
        "nuisance_count": nuisance,
        "dependency_span": positions[-1] - positions[0] + 1,
        "requested_dependency_span": span,
        "generator_family": family,
        "pattern_topology": "adjacent" if family == "contiguous" else "skip",
        "all_predictors_necessary": True,
        "family_id": f"p{order}:{tuple(values)}:{index % 16}",
        "split": split,
    }


def designed_cells(config: dict) -> tuple[list[dict], list[dict]]:
    """Return unique eligible cells and an explicit exclusion ledger."""
    d = config["design"]
    candidates = []
    # One-factor contrasts; nuisance never changes raw length/order/span.
    for p in config["predictive_orders"]:
        candidates.append(("predictive_order", p, d["baseline_raw_length"], d["baseline_nuisance"], max(d["baseline_span"], p), "necessary_pattern"))
    for n in config["raw_lengths"]:
        candidates.append(("raw_length", d["baseline_order"], n, d["baseline_nuisance"], d["baseline_span"], d["baseline_family"]))
    for k in config["nuisance_counts"]:
        candidates.append(("nuisance", d["baseline_order"], d["baseline_raw_length"], k, d["baseline_span"], d["baseline_family"]))
    for s in config["dependency_spans"]:
        candidates.append(("dependency_span", d["baseline_order"], d["baseline_raw_length"], d["baseline_nuisance"], s, "skip_gram"))
    # Generator comparison is exactly matched wherever topology permits.
    for family in config["generator_families"]:
        matched_span = d["baseline_order"] if family == "contiguous" else d["baseline_span"]
        candidates.append(("generator_family", d["baseline_order"], d["baseline_raw_length"], d["baseline_nuisance"], matched_span, family))
    cells, excluded, seen = [], [], set()
    for axis, p, n, k, s, family in candidates:
        key = (p, n, k, s, family)
        ok, reason = feasible(p, n, s, family)
        row = {"axis": axis, "predictive_order": p, "raw_length": n, "nuisance_count": k,
               "requested_dependency_span": s, "generator_family": family}
        if not ok:
            excluded.append({**row, "status": reason})
        elif key not in seen:
            seen.add(key); cells.append({**row, "cell_id": stable_hash(key, 12), "status": "eligible"})
    return cells, excluded


def evaluation(config: dict, examples: int | None = None) -> list[dict]:
    count = examples or config["evaluation_examples_per_cell"]
    cells, _ = designed_cells(config)
    return [{**make_example(config, order=c["predictive_order"], raw_length=c["raw_length"],
                            nuisance=c["nuisance_count"], span=c["requested_dependency_span"],
                            family=c["generator_family"], index=i, split="test"),
             "axis": c["axis"], "cell_id": c["cell_id"]}
            for c in cells for i in range(count)]


def transformer_parameter_count(vocab: int, context: int, width: int, depth: int) -> int:
    # Exact for TinyTransformerLM's tied embedding/head, learned positions,
    # two LNs/block, QKV/O projections, ratio-2 FF, and final LayerNorm.
    return vocab * width + context * width + depth * (8 * width * width + 11 * width) + 2 * width


def architecture_plan(config: dict) -> list[dict]:
    rows = []
    for seed, depth, width, heads, budget in itertools.product(
            config["model_seeds"], config["depths"], config["widths"],
            config["heads"], config["training_budgets"]):
        if width % heads:
            continue
        params = transformer_parameter_count(config["vocab_size"], config["sequence_length"], width, depth)
        rows.append({"model_seed": seed, "model_depth": depth, "model_width": width,
                     "head_count": heads, "training_budget": budget,
                     "training_steps": config["base_steps"] * budget,
                     "parameter_count": params,
                     "architecture_id": f"L{depth}-W{width}-H{heads}-T{budget}"})
    return rows


def parameter_matched_pairs(config: dict, architectures: list[dict]) -> list[dict]:
    tolerance = config["parameter_match_relative_tolerance"]
    unique = {(r["model_depth"], r["model_width"], r["head_count"], r["parameter_count"])
              for r in architectures if r["training_budget"] == 1 and r["model_seed"] == config["model_seeds"][0]}
    pairs = []
    for a, b in itertools.combinations(sorted(unique), 2):
        if a[0] == b[0] or not ((a[0] > b[0] and a[1] < b[1]) or (b[0] > a[0] and b[1] < a[1])):
            continue
        rel = abs(a[3] - b[3]) / max(a[3], b[3])
        if rel <= tolerance:
            pairs.append({"depth_a": a[0], "width_a": a[1], "heads_a": a[2], "parameters_a": a[3],
                          "depth_b": b[0], "width_b": b[1], "heads_b": b[2], "parameters_b": b[3],
                          "relative_parameter_gap": rel})
    return pairs


def measured_frontiers(rows: list[dict], threshold: float = .8) -> list[dict]:
    """Reduce raw accuracies without assigning a frontier through missing cells."""
    axes = {"predictive_order": "pstar_max", "raw_length": "n_max", "nuisance_count": "k_max",
            "dependency_span": "s_max"}
    group_keys = ("model_depth", "model_width", "head_count", "training_budget", "model_seed")
    output = []
    for key in sorted({tuple(r[k] for k in group_keys) for r in rows}):
        subset = [r for r in rows if tuple(r[k] for k in group_keys) == key]
        for axis, name in axes.items():
            cells = [r for r in subset if r.get("axis") == axis]
            values = sorted({int(r[axis]) for r in cells if float(r["accuracy"]) >= threshold})
            output.append({**dict(zip(group_keys, key)), "frontier": name,
                           "maximum_competent_value": max(values) if values else "",
                           "competent_cell_count": len(values), "status": "estimated" if cells else "not_measured"})
    return output


def validate(config: dict) -> dict:
    cells, excluded = designed_cells(config); rows = evaluation(config, 2)
    counts = Counter(r["cell_id"] for r in rows)
    checks = {
        "unique_cells": len(cells) == len({c["cell_id"] for c in cells}),
        "balanced_examples": set(counts.values()) == {2},
        "fixed_sequence_length": all(len(r["tokens"]) == config["sequence_length"] for r in rows),
        "measured_span_matches": all(r["dependency_span"] == r["requested_dependency_span"]
                                     or (r["predictive_order"] == 1 and r["dependency_span"] == 1)
                                     for r in rows),
        "axes_present": {r["axis"] for r in rows} == {"predictive_order", "raw_length", "nuisance", "dependency_span", "generator_family"},
        "three_or_more_seeds": len(config["model_seeds"]) >= 3,
    }
    return {"schema_version": "paper05.stress_frontier.validation.v1", "passed": all(checks.values()),
            "checks": checks, "eligible_cells": len(cells), "excluded_cells": len(excluded),
            "artifact_hash": stable_hash({"cells": cells, "excluded": excluded, "checks": checks})}


def main(args) -> None:
    config = json.loads(Path(args.config).read_text()); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    cells, excluded = designed_cells(config); architectures = architecture_plan(config)
    validation = validate(config)
    write_csv(out / "stress_dataset_plan.csv", cells)
    write_csv(out / "stress_excluded_cells.csv", excluded)
    write_csv(out / "stress_architecture_plan.csv", architectures)
    write_csv(out / "stress_parameter_matched_pairs.csv", parameter_matched_pairs(config, architectures))
    write_csv(out / "stress_frontiers.csv", [])
    atomic_write_json(out / "stress_validation.json", validation)
    manifest = {"schema_version": "paper05.stress_frontier.plan.v1", "run_mode": "cpu_smoke" if args.smoke else "plan_only",
                "config": config, "eligible_dataset_cells": len(cells), "architecture_runs": len(architectures),
                "training_launched": False, "validation": validation,
                "artifact_hash": stable_hash({"cells": cells, "architectures": architectures})}
    atomic_write_json(out / "stress_manifest.json", manifest)
    if not validation["passed"]:
        raise RuntimeError(validation)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper05/stress_frontier.json")
    parser.add_argument("--output", default="docs/papers/paper0_5/results/stress_frontier")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    main(parser.parse_args())
