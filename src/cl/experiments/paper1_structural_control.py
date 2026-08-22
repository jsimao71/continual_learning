"""Run Paper 1 frozen structural-control and pretrained residualization pilots."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cl.analysis.pretrained_graph_audit import load_graph_rows, residualized_pretrained_audit
from cl.common.artifacts import RunMetadata, atomic_write_json, stable_hash, write_csv, write_jsonl
from cl.common.metrics import bootstrap_ci, spearman
from cl.nsc.features import FEATURE_NAMES, ridge_fit, ridge_predict
from cl.nsc.selectors import SELECTORS, select_candidates
from cl.nsc.synthetic import build_bridge_suite


PRIMARY_SELECTORS = SELECTORS[:-1]
SELECTOR_COLUMNS = (0, 3, 4, 5, 6, 7)


def _raw_space_ridge(features, target, alpha=2.0):
    matrix = np.asarray(features, dtype=np.float64)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-9] = 1.0
    standardized = (matrix - mean) / scale
    coefficients = ridge_fit(standardized, target, alpha=alpha)
    raw = np.empty_like(coefficients)
    raw[1:] = coefficients[1:] / scale
    raw[0] = coefficients[0] - np.dot(coefficients[1:], mean / scale)
    return raw


def _prediction_metrics(examples, combined_weights):
    rows = []
    for split in ("validation", "test"):
        selected = [example for example in examples if example.split == split]
        features = np.concatenate([example.features for example in selected])
        target = np.concatenate([example.utility() for example in selected])
        base_features = features[:, :3]
        base_fit = _raw_space_ridge(base_features, target)
        base_prediction = np.column_stack([np.ones(len(features)), base_features]) @ base_fit
        structural_prediction = np.column_stack([np.ones(len(features)), features]) @ combined_weights
        denominator = np.square(target - target.mean()).sum()
        for name, prediction in (("base_controls", base_prediction), ("base_plus_structure", structural_prediction)):
            rows.append({
                "split": split,
                "model": name,
                "n_candidates": len(target),
                "spearman": spearman(target, prediction),
                "r2": float(1.0 - np.square(target - prediction).sum() / max(denominator, 1e-12)),
                "rmse": float(np.sqrt(np.mean(np.square(target - prediction)))),
            })
    return rows


def _mean_prediction_rows(predictions, split="test"):
    groups = defaultdict(list)
    for row in predictions:
        if row["split"] == split:
            groups[row["model"]].append(row)
    return [
        {
            "model": model,
            **{
                metric: float(np.mean([row[metric] for row in values]))
                for metric in ("spearman", "r2", "rmse")
            },
        }
        for model, values in sorted(groups.items())
    ]


def _selector_rows(examples, weights, budgets):
    rows = []
    trace_rows = []
    for example in examples:
        if example.split != "test":
            continue
        labels = np.zeros(len(example.features), dtype=np.float64)
        labels[list(example.evidence)] = 1.0
        for budget in budgets:
            for mode in SELECTORS:
                started = time.perf_counter_ns()
                selected = select_candidates(
                    example.features,
                    budget,
                    mode,
                    combined_weights=weights,
                    evidence=labels,
                    gamma=2.0,
                    bridge_fraction=0.25,
                )
                selector_ns = time.perf_counter_ns() - started
                quality, recall, complete = example.quality(selected)
                bridge_selected = float(example.bridge_candidate in selected)
                rows.append({
                    "example_id": example.example_id,
                    "seed": example.seed,
                    "hop_count": example.hop_count,
                    "lexical_overlap": example.lexical_overlap,
                    "bridge_strength": example.bridge_strength,
                    "condition": mode,
                    "budget_chunks": budget,
                    "materialized_tokens": budget * example.candidate_tokens,
                    "qwen_equivalent_kv_bytes": budget * example.candidate_tokens * 4096,
                    "quality": quality,
                    "evidence_recall": recall,
                    "path_complete": complete,
                    "bridge_selected": bridge_selected,
                    "selector_time_ns": selector_ns,
                    "budget_violation": int(len(selected) != budget),
                })
                trace_rows.append({
                    "schema_version": "cl.nsc.trace.v1",
                    "example_id": example.example_id,
                    "seed": example.seed,
                    "condition": mode,
                    "budget_chunks": budget,
                    "candidate_count": len(example.features),
                    "selected_candidate_ids": list(selected),
                    "evidence_candidate_ids": list(example.evidence),
                    "bridge_candidate_id": example.bridge_candidate,
                    "retained_full_attention": False,
                })
    return rows, trace_rows


def _aggregate_selector(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["budget_chunks"])].append(row)
    summary = []
    for (condition, budget), values in sorted(groups.items()):
        record = {
            "condition": condition,
            "budget_chunks": budget,
            "materialized_tokens": values[0]["materialized_tokens"],
            "qwen_equivalent_kv_bytes": values[0]["qwen_equivalent_kv_bytes"],
            "n_examples": len(values),
        }
        for metric in ("quality", "evidence_recall", "path_complete", "bridge_selected", "selector_time_ns"):
            estimate, low, high = bootstrap_ci(
                [row[metric] for row in values], samples=1000, seed=budget + len(condition)
            )
            record[f"mean_{metric}"] = estimate
            record[f"{metric}_ci_low"] = low
            record[f"{metric}_ci_high"] = high
        summary.append(record)
    return summary


def _candidate_rows(examples):
    rows = []
    for example in examples:
        utility = example.utility()
        for index, features in enumerate(example.features):
            rows.append({
                "example_id": example.example_id,
                "split": example.split,
                "seed": example.seed,
                "candidate_id": index,
                "is_evidence": int(index in example.evidence),
                "is_bridge": int(index == example.bridge_candidate),
                "causal_utility": utility[index],
                **{name: float(value) for name, value in zip(FEATURE_NAMES, features)},
            })
    return rows


def _plots(output, aggregate, predictions, pretrained):
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    colors = {
        "base_topk": "#4c78a8", "persistence": "#f58518", "agreement": "#e45756",
        "community": "#72b7b2", "bridge_preserving": "#54a24b",
        "combined_structural": "#b279a2", "oracle_evidence": "#777777",
    }
    shown = tuple(colors)
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    for condition in shown:
        values = sorted([row for row in aggregate if row["condition"] == condition], key=lambda row: row["materialized_tokens"])
        ax.plot([row["materialized_tokens"] for row in values], [row["mean_quality"] for row in values], marker="o", label=condition.replace("_", " "), color=colors[condition])
    ax.set_xlabel("materialized tokens (matched budget)")
    ax.set_ylabel("controlled quality")
    ax.set_title("Frozen structural-control Pareto sweep")
    ax.grid(alpha=.25); ax.legend(fontsize=7, ncol=2); fig.tight_layout()
    fig.savefig(figures / "quality_budget_frontier.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    for condition in ("base_topk", "bridge_preserving", "combined_structural", "oracle_evidence"):
        values = sorted([row for row in aggregate if row["condition"] == condition], key=lambda row: row["budget_chunks"])
        ax.plot([row["budget_chunks"] for row in values], [row["mean_path_complete"] for row in values], marker="o", label=condition.replace("_", " "))
    ax.set_xlabel("materialized chunks"); ax.set_ylabel("complete multi-hop path rate")
    ax.set_title("Weak-bridge preservation"); ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(figures / "bridge_path_completion.png", dpi=180); plt.close(fig)

    test = _mean_prediction_rows(predictions)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2))
    labels = [row["model"].replace("_", "\n") for row in test]
    axes[0].bar(labels, [row["spearman"] for row in test], color=["#9ca3af", "#54a24b"])
    axes[0].set_title("Known causal utility"); axes[0].set_ylabel("held-out Spearman")
    natural = [pretrained["base"], pretrained["base_plus_structure"]]
    axes[1].bar(["surface\ncontrols", "+ structure"], [row["spearman"] for row in natural], color=["#9ca3af", "#4c78a8"])
    axes[1].set_title("Frozen-Qwen graph recovery"); axes[1].set_ylabel("identity-disjoint Spearman")
    for axis in axes: axis.axhline(0, color="black", lw=.7); axis.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(figures / "incremental_prediction.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    values = [row for row in aggregate if row["budget_chunks"] == 4 and row["condition"] != "oracle_evidence"]
    values.sort(key=lambda row: row["mean_selector_time_ns"])
    ax.barh([row["condition"].replace("_", " ") for row in values], [row["mean_selector_time_ns"] / 1e6 for row in values])
    ax.set_xlabel("selector time (ms/example)"); ax.set_title("Offline Python routing overhead, budget 4")
    ax.grid(axis="x", alpha=.25); fig.tight_layout(); fig.savefig(figures / "selector_overhead.png", dpi=180); plt.close(fig)


def _latex_table(path, aggregate):
    rows = []
    for condition in ("base_topk", "power_sharpen", "persistence", "agreement", "community", "bridge_preserving", "combined_structural", "oracle_evidence"):
        value = next(row for row in aggregate if row["condition"] == condition and row["budget_chunks"] == 4)
        rows.append((condition.replace("_", " "), value["mean_quality"], value["mean_evidence_recall"], value["mean_path_complete"], value["mean_bridge_selected"]))
    lines = [
        "\\begin{table}[H]\\centering\\small",
        "\\begin{tabular}{lrrrr}\\toprule",
        "Condition & Quality & Evidence recall & Path complete & Bridge recall \\\\\\midrule",
    ]
    lines.extend(f"{name} & {quality:.3f} & {recall:.3f} & {complete:.3f} & {bridge:.3f} \\\\" for name, quality, recall, complete, bridge in rows)
    lines.extend([
        "\\bottomrule\\end{tabular}",
        "\\caption{Controlled test performance at four 32-token chunks. The oracle is diagnostic only. All non-oracle conditions receive identical candidates and budgets.}",
        "\\label{tab:paper1-main}",
        "\\end{table}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary(output, config, aggregate, predictions, pretrained):
    at_four = {row["condition"]: row for row in aggregate if row["budget_chunks"] == 4}
    base = at_four["base_topk"]
    bridge = at_four["bridge_preserving"]
    combined = at_four["combined_structural"]
    test_prediction = {row["model"]: row for row in _mean_prediction_rows(predictions)}
    lines = [
        "# Paper 1 frozen structural-control pilot",
        "",
        "## Scope",
        "",
        f"- Controlled bridge suite: {config['test_examples_per_seed']} test examples x {len(config['seeds'])} seeds.",
        "- Conditions B0--B3 and S1--S5 use identical candidate sets and exact chunk budgets; O1 is diagnostic.",
        "- Frozen-Qwen audit: retained layerwise summaries for 84 identity-disjoint 2WikiMultiHopQA/MuSiQue examples.",
        "- No model weights were trained and no online prototype/adaptor was implemented.",
        "",
        "## Main measured results",
        "",
        f"- At four chunks, base quality/path completion is {base['mean_quality']:.4f}/{base['mean_path_complete']:.4f}.",
        f"- Bridge preservation reaches {bridge['mean_quality']:.4f}/{bridge['mean_path_complete']:.4f}; combined structural selection reaches {combined['mean_quality']:.4f}/{combined['mean_path_complete']:.4f}.",
        f"- Structural features raise held-out causal-utility Spearman from {test_prediction['base_controls']['spearman']:.4f} to {test_prediction['base_plus_structure']['spearman']:.4f} in the controlled suite.",
        f"- In the pretrained observational audit, identity-disjoint Spearman changes from {pretrained['base']['spearman']:.4f} to {pretrained['base_plus_structure']['spearman']:.4f} after structural features are added to surface controls.",
        "- Fixed-gamma and entropy-adaptive sharpening cannot change exact top-k membership because both are monotone transformations; their matched-budget rows therefore reproduce B0.",
        "",
        "## Decision gate",
        "",
        "The controlled bridge result passes the synthetic utility/frontier gate, but the natural pretrained cache lacks candidate tensors and task-generation outcomes. Paper 2 online consolidation remains blocked pending a regenerated HotpotQA/QASPER candidate cache and a natural task quality--materialization Pareto test.",
        "",
    ]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(args):
    repo = Path(args.repo).resolve()
    output = Path(args.output).resolve()
    raw = output / "raw"; tables = output / "tables"
    raw.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.seeds.split(",")]
    budgets = [int(value) for value in args.budgets]
    config = {
        "seeds": seeds, "budgets": budgets, "validation_examples_per_seed": args.validation_examples,
        "test_examples_per_seed": args.test_examples, "candidates": args.candidates,
        "candidate_tokens": 32, "qwen_kv_bytes_per_token": 4096,
        "selectors": list(SELECTORS), "full_attention_retained": False,
    }
    all_examples = []
    selector_rows = []; trace_rows = []; prediction_rows = []; metadata = []; weights_rows = []
    for seed in seeds:
        examples = build_bridge_suite(seed, args.validation_examples, args.test_examples, args.candidates)
        all_examples.extend(examples)
        validation = [example for example in examples if example.split == "validation"]
        train_features = np.concatenate([example.features for example in validation])
        train_utility = np.concatenate([example.utility() for example in validation])
        utility_weights = _raw_space_ridge(train_features, train_utility)
        selector_fit = _raw_space_ridge(train_features[:, SELECTOR_COLUMNS], train_utility)
        weights = np.zeros(train_features.shape[1] + 1, dtype=np.float64)
        weights[0] = selector_fit[0]
        weights[np.asarray(SELECTOR_COLUMNS) + 1] = selector_fit[1:]
        weights_rows.append({"seed": seed, "intercept": weights[0], **{name: weights[index + 1] for index, name in enumerate(FEATURE_NAMES)}})
        prediction_rows.extend([{"seed": seed, **row} for row in _prediction_metrics(examples, utility_weights)])
        measured, traces = _selector_rows(examples, weights, budgets)
        selector_rows.extend(measured); trace_rows.extend(traces)
        data_hash = stable_hash([example.example_id for example in examples])
        metadata.append(RunMetadata.capture(
            repo=repo, run_id=f"paper1-bridge-seed{seed}", config=config,
            model_id="frozen-controlled-routing-scores-v1", dataset_id="controlled-bridge-v1",
            seed=seed, device="cpu", dtype="float64", data_hash=data_hash,
            split="validation+test", model_revision="deterministic-v1",
        ).as_dict())
    aggregate = _aggregate_selector(selector_rows)
    candidate_rows = _candidate_rows(all_examples)

    source = Path(args.pretrained_graph_rows).resolve()
    pretrained_rows = load_graph_rows(source)
    pretrained_summary, pretrained_predictions = residualized_pretrained_audit(pretrained_rows)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    pretrained_summary["source_sha256"] = source_hash
    pretrained_summary["source_rows"] = len(pretrained_rows)

    write_jsonl(raw / "candidate_features.jsonl", candidate_rows)
    write_jsonl(raw / "selection_traces.jsonl", trace_rows)
    write_jsonl(raw / "run_metadata.jsonl", metadata)
    write_csv(raw / "pretrained_graph_rows.csv", pretrained_rows)
    write_csv(tables / "selector_by_example.csv", selector_rows)
    write_csv(tables / "selector_summary.csv", aggregate)
    write_csv(tables / "causal_utility_prediction.csv", prediction_rows)
    write_csv(tables / "combined_weights.csv", weights_rows)
    write_csv(tables / "pretrained_predictions.csv", pretrained_predictions)
    atomic_write_json(tables / "pretrained_residualization.json", pretrained_summary)
    _plots(output, aggregate, prediction_rows, pretrained_summary)
    _latex_table(tables / "main_results.tex", aggregate)
    _summary(output, config, aggregate, prediction_rows, pretrained_summary)
    manifest = {
        "schema_version": "paper1.results.v1",
        "config": config,
        "pretrained_source_sha256": source_hash,
        "pretrained_model": "Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca",
        "paper06_manifest_hash": stable_hash(json.loads((repo / "docs/papers/paper0_6/results/manifest.json").read_text())),
        "artifact_hash": stable_hash({"selector": aggregate, "prediction": prediction_rows, "pretrained": pretrained_summary}),
    }
    atomic_write_json(output / "manifest.json", manifest)
    print(json.dumps({
        "output": str(output), "controlled_examples": len(all_examples),
        "selection_rows": len(selector_rows), "pretrained_rows": len(pretrained_rows),
        "pretrained_test_examples": pretrained_summary["test_examples"],
    }, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--output", default="docs/papers/paper1/results")
    parser.add_argument("--seeds", default="11,23,37,41,53")
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 3, 4, 6, 8])
    parser.add_argument("--validation-examples", type=int, default=80)
    parser.add_argument("--test-examples", type=int, default=120)
    parser.add_argument("--candidates", type=int, default=24)
    parser.add_argument(
        "--pretrained-graph-rows",
        default=r"D:\git\rd\pdattention\docs\papers\shared\results\paper2_5_iterative_pra\layerwise_graph\layerwise_graph_rows.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
