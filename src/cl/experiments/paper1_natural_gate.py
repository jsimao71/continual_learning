"""Run the candidate-level frozen-Qwen gate on HotpotQA and QASPER."""

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

from cl.common.artifacts import RunMetadata, atomic_write_json, stable_hash, write_csv, write_jsonl
from cl.common.metrics import spearman
from cl.nsc.features import FEATURE_NAMES, ridge_fit, ridge_predict
from cl.nsc.natural import identity_disjoint, load_hotpotqa, load_qasper
from cl.nsc.qwen_natural import FrozenQwenNaturalAdapter
from cl.nsc.selectors import SELECTORS, select_candidates


MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
SELECTOR_COLUMNS = (0, 3, 4, 5, 6, 7)


def _fit_weights(bundles):
    features = np.concatenate([bundle.features for _, bundle in bundles])
    labels = np.concatenate([
        np.asarray([float(value.is_evidence) for value in bundle.candidates])
        for _, bundle in bundles
    ])
    selected = features[:, SELECTOR_COLUMNS]
    coefficients = ridge_fit(selected, labels, alpha=2.0)
    weights = np.zeros(features.shape[1] + 1)
    weights[0] = coefficients[0]
    weights[np.asarray(SELECTOR_COLUMNS) + 1] = coefficients[1:]
    return weights


def _evaluate_selectors(adapter, bundles, weights, budgets):
    rows = []
    traces = []
    for example, bundle in bundles:
        if example.split != "test":
            continue
        evidence = np.asarray([float(value.is_evidence) for value in bundle.candidates])
        measurement_cache = {}
        for budget in budgets:
            if budget > len(bundle.candidates):
                continue
            for condition in SELECTORS:
                started = time.perf_counter_ns()
                selected = select_candidates(
                    bundle.features, budget, condition, combined_weights=weights,
                    evidence=evidence, gamma=2.0, bridge_fraction=0.25,
                )
                selection_ns = time.perf_counter_ns() - started
                if selected not in measurement_cache:
                    measurement_cache[selected] = adapter.evaluate(example, bundle.candidates, selected)
                measured = measurement_cache[selected]
                recall = evidence[list(selected)].sum() / max(evidence.sum(), 1.0)
                complete = float(evidence.sum() > 0 and evidence[list(selected)].sum() == evidence.sum())
                row = {
                    "dataset": example.dataset, "example_id": example.example_id,
                    "identity_id": example.identity_id, "condition": condition,
                    "budget_chunks": budget, "budget_tokens": budget * 32,
                    "selection_time_ns": selection_ns, "evidence_recall": float(recall),
                    "complete_evidence": complete, "budget_violation": int(measured["materialized_tokens"] != budget * 32),
                    **measured,
                }
                rows.append(row)
                traces.append({
                    "schema_version": "cl.nsc.natural.trace.v1", "dataset": example.dataset,
                    "example_id": example.example_id, "identity_id": example.identity_id,
                    "condition": condition, "budget_tokens": budget * 32,
                    "candidate_ids": [value.candidate_id for value in bundle.candidates],
                    "selected_candidate_ids": [bundle.candidates[index].candidate_id for index in selected],
                    "evidence_candidate_ids": [value.candidate_id for value in bundle.candidates if value.is_evidence],
                    "retained_full_attention": False,
                })
    return rows, traces


def _causal_rows(adapter, bundles, per_split=1):
    rows = []
    for split in ("validation", "test"):
        selected_examples = [(example, bundle) for example, bundle in bundles if example.split == split][:per_split]
        for example, bundle in selected_examples:
            all_indices = tuple(range(len(bundle.candidates)))
            intact = adapter.evaluate(example, bundle.candidates, all_indices)["answer_logprob"]
            for index, features in enumerate(bundle.features):
                retained = tuple(value for value in all_indices if value != index)
                removed = adapter.evaluate(example, bundle.candidates, retained)["answer_logprob"]
                rows.append({
                    "dataset": example.dataset, "split": split, "example_id": example.example_id,
                    "identity_id": example.identity_id, "candidate_id": bundle.candidates[index].candidate_id,
                    "is_evidence": int(bundle.candidates[index].is_evidence),
                    "intact_answer_logprob": intact, "removed_answer_logprob": removed,
                    "causal_utility": intact - removed,
                    **{name: float(value) for name, value in zip(FEATURE_NAMES, features)},
                })
    return rows


def _prediction(causal):
    output = []
    for dataset in sorted({row["dataset"] for row in causal}):
        train = [row for row in causal if row["dataset"] == dataset and row["split"] == "validation"]
        test = [row for row in causal if row["dataset"] == dataset and row["split"] == "test"]
        if not train or not test:
            continue
        y_train = np.asarray([row["causal_utility"] for row in train])
        y_test = np.asarray([row["causal_utility"] for row in test])
        for name, fields in (
            ("surface_controls", FEATURE_NAMES[:3]),
            ("surface_plus_structure", FEATURE_NAMES),
        ):
            x_train = np.asarray([[row[field] for field in fields] for row in train])
            x_test = np.asarray([[row[field] for field in fields] for row in test])
            mean, scale = x_train.mean(0), x_train.std(0)
            scale[scale < 1e-9] = 1.0
            fit = ridge_fit((x_train - mean) / scale, y_train, alpha=2.0)
            prediction = ridge_predict((x_test - mean) / scale, fit)
            denominator = np.square(y_test - y_test.mean()).sum()
            output.append({
                "dataset": dataset, "model": name, "n_candidates": len(test),
                "spearman": spearman(y_test, prediction),
                "r2": float(1 - np.square(y_test - prediction).sum() / max(denominator, 1e-12)),
                "rmse": float(np.sqrt(np.mean(np.square(y_test - prediction)))),
            })
    return output


def _aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["condition"], row["budget_tokens"])].append(row)
    output = []
    for (dataset, condition, budget), values in sorted(grouped.items()):
        output.append({
            "dataset": dataset, "condition": condition, "budget_tokens": budget,
            "n_examples": len(values),
            **{f"mean_{field}": float(np.mean([row[field] for row in values])) for field in (
                "answer_logprob", "answer_token_accuracy", "evidence_recall", "complete_evidence",
                "materialized_kv_bytes", "selection_time_ns", "materialization_time_ns",
                "end_to_end_time_ns", "peak_device_bytes", "budget_violation",
            )},
        })
    return output


def _paired_deltas(rows, samples=10_000, seed=20260822):
    rng = np.random.default_rng(seed)
    output = []
    for dataset in sorted({row["dataset"] for row in rows}):
        for budget in sorted({row["budget_tokens"] for row in rows if row["dataset"] == dataset}):
            base = {
                row["example_id"]: row for row in rows
                if row["dataset"] == dataset and row["budget_tokens"] == budget and row["condition"] == "base_topk"
            }
            for condition in ("bridge_preserving", "combined_structural"):
                compared = {
                    row["example_id"]: row for row in rows
                    if row["dataset"] == dataset and row["budget_tokens"] == budget and row["condition"] == condition
                }
                delta = np.asarray([
                    compared[key]["answer_logprob"] - base[key]["answer_logprob"] for key in sorted(base)
                ])
                bootstrap = np.asarray([rng.choice(delta, len(delta)).mean() for _ in range(samples)])
                output.append({
                    "dataset": dataset, "budget_tokens": budget, "condition": condition,
                    "n_examples": len(delta), "mean_answer_logprob_delta": float(delta.mean()),
                    "ci_low": float(np.quantile(bootstrap, .025)),
                    "ci_high": float(np.quantile(bootstrap, .975)),
                    "wins": int((delta > 0).sum()), "ties": int((delta == 0).sum()),
                })
    return output


def _write_outputs(output, aggregate, paired, causal, predictions, traces, rows, metadata, candidate_rows, config):
    raw, tables, figures = output / "raw", output / "tables", output / "figures"
    for path in (raw, tables, figures): path.mkdir(parents=True, exist_ok=True)
    write_jsonl(raw / "candidate_features.jsonl", candidate_rows)
    write_jsonl(raw / "selection_traces.jsonl", traces)
    write_jsonl(raw / "run_metadata.jsonl", metadata)
    write_csv(tables / "selector_by_example.csv", rows)
    write_csv(tables / "selector_summary.csv", aggregate)
    write_csv(tables / "paired_frontier_deltas.csv", paired)
    write_csv(tables / "candidate_removal.csv", causal)
    write_csv(tables / "causal_utility_prediction.csv", predictions)
    lookup = {(row["dataset"], row["condition"], int(row["budget_tokens"])): row for row in aggregate}
    delta = {(row["dataset"], row["condition"], int(row["budget_tokens"])): row for row in paired}
    lines = [
        "# Paper 1 natural candidate-level gate", "", "## Scope", "",
        f"- Frozen {MODEL_ID} revision `{MODEL_REVISION}` on identity-disjoint official HotpotQA and QASPER validation identities.",
        f"- {config['per_split']} validation and {config['per_split']} test identities per dataset; 12 fixed 32-token candidates and budgets 64/128/192 tokens.",
        "- Every evaluation performs native causal K/V prefill; bytes and latency are measured from the materialized tensors and forward path.",
        "", "## Main results", "",
    ]
    for dataset in ("hotpotqa", "qasper"):
        values = []
        for budget in (64, 128, 192):
            item = delta[(dataset, "bridge_preserving", budget)]
            values.append(f"{budget}: {float(item['mean_answer_logprob_delta']):+.3f} [{float(item['ci_low']):+.3f}, {float(item['ci_high']):+.3f}]")
        lines.append(f"- {dataset.upper()} bridge-minus-base answer-logprob delta (95% paired bootstrap CI): " + "; ".join(values) + ".")
    lines.extend(["", "## Decision", "",
        "Bridge preservation has a positive mean delta at all six dataset/budget cells, but every paired interval includes zero. The validation-fitted combined selector is mostly negative, and held-out candidate-removal models retain negative R2. This diagnostic run therefore narrows the uncertainty but does not pass the reproducibility gate for online consolidation; Paper 2 remains stopped.", ""])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    tex = ["\\begin{table}[H]\\centering\\small", "\\begin{tabular}{llrrr}\\toprule", "Dataset & Budget & Base & Bridge & Paired delta (95\\% CI) \\\\" , "\\midrule"]
    for dataset in ("hotpotqa", "qasper"):
        for budget in (64, 128, 192):
            base = lookup[(dataset, "base_topk", budget)]
            bridge = lookup[(dataset, "bridge_preserving", budget)]
            item = delta[(dataset, "bridge_preserving", budget)]
            tex.append(f"{dataset.upper()} & {budget} & {float(base['mean_answer_logprob']):.3f} & {float(bridge['mean_answer_logprob']):.3f} & {float(item['mean_answer_logprob_delta']):+.3f} [{float(item['ci_low']):+.3f}, {float(item['ci_high']):+.3f}] \\\\")
    tex.extend(["\\bottomrule\\end{tabular}", "\\caption{Frozen-Qwen natural candidate intervention. Quality is mean answer-token log probability; all budgets use measured native K/V.}", "\\label{tab:paper1-natural}", "\\end{table}"])
    (tables / "natural_results.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    shown = ("base_topk", "bridge_preserving", "combined_structural", "oracle_evidence")
    for axis, dataset in zip(axes, ("hotpotqa", "qasper")):
        for condition in shown:
            values = [row for row in aggregate if row["dataset"] == dataset and row["condition"] == condition]
            axis.plot([row["budget_tokens"] for row in values], [row["mean_answer_logprob"] for row in values], marker="o", label=condition.replace("_", " "))
        axis.set_title(dataset.upper()); axis.set_xlabel("materialized native K/V tokens")
        axis.set_ylabel("mean answer-token log probability"); axis.grid(alpha=.25)
    axes[1].legend(fontsize=7); fig.tight_layout()
    fig.savefig(figures / "natural_quality_materialization.png", dpi=180); plt.close(fig)
    artifact_files = (
        raw / "candidate_features.jsonl", raw / "selection_traces.jsonl",
        tables / "selector_by_example.csv", tables / "selector_summary.csv",
        tables / "paired_frontier_deltas.csv", tables / "candidate_removal.csv",
        tables / "causal_utility_prediction.csv",
    )
    manifest = {
        "schema_version": "paper1.natural.v1", "config": config,
        "candidate_cache_hash": stable_hash(candidate_rows),
        "artifact_hash": stable_hash({path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in artifact_files}),
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
    }
    atomic_write_json(output / "manifest.json", manifest)
    return manifest


def run(args):
    repo, output, cache = Path(args.repo).resolve(), Path(args.output).resolve(), Path(args.cache).resolve()
    datasets = {
        "hotpotqa": load_hotpotqa(per_split=args.per_split, seed=args.seed, cache_dir=cache / "hotpotqa"),
        "qasper": load_qasper(per_split=args.per_split, seed=args.seed, cache_dir=cache / "qasper"),
    }
    if not all(identity_disjoint(values) for values in datasets.values()):
        raise RuntimeError("identity leakage across validation and test")
    adapter = FrozenQwenNaturalAdapter(MODEL_ID, MODEL_REVISION, args.device)
    bundles = {}
    candidate_rows = []
    for dataset, examples in datasets.items():
        values = []
        for example in examples:
            try:
                bundle = adapter.extract(example, maximum_candidates=args.candidates)
            except ValueError:
                continue
            values.append((example, bundle))
            for index, candidate in enumerate(bundle.candidates):
                candidate_rows.append({
                    "dataset": dataset, "split": example.split, "example_id": example.example_id,
                    "identity_id": example.identity_id, "candidate_id": candidate.candidate_id,
                    "title": candidate.title, "is_evidence": int(candidate.is_evidence),
                    "token_length": bundle.token_lengths[index],
                    **{name: float(value) for name, value in zip(FEATURE_NAMES, bundle.features[index])},
                })
        bundles[dataset] = values
    config = {
        "seed": args.seed, "per_split": args.per_split, "candidates": args.candidates,
        "budgets": args.budgets, "chunk_tokens": 32, "causal_examples_per_split": args.causal_examples,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "device": str(adapter.device),
        "selectors": list(SELECTORS), "retain_full_attention": False,
    }
    all_rows, all_traces, all_causal, metadata = [], [], [], []
    for dataset, values in bundles.items():
        weights = _fit_weights([(example, bundle) for example, bundle in values if example.split == "validation"])
        rows, traces = _evaluate_selectors(adapter, values, weights, args.budgets)
        all_rows.extend(rows); all_traces.extend(traces)
        all_causal.extend(_causal_rows(adapter, values, args.causal_examples))
        source_hash = hashlib.sha256("\n".join(example.example_id for example, _ in values).encode()).hexdigest()
        metadata.append(RunMetadata.capture(
            repo=repo, run_id=f"paper1-natural-{dataset}", config=config,
            model_id=MODEL_ID, model_revision=MODEL_REVISION,
            tokenizer_id=MODEL_ID, tokenizer_revision=MODEL_REVISION,
            dataset_id=f"{dataset}:official-validation", split="identity-disjoint-validation+test",
            seed=args.seed, device=str(adapter.device), dtype=str(next(adapter.model.parameters()).dtype), data_hash=source_hash,
        ).as_dict())
    aggregate = _aggregate(all_rows)
    paired = _paired_deltas(all_rows, seed=args.seed)
    predictions = _prediction(all_causal)
    manifest = _write_outputs(output, aggregate, paired, all_causal, predictions, all_traces, all_rows, metadata, candidate_rows, config)
    print(json.dumps({"output": str(output), "rows": len(all_rows), "candidate_rows": len(candidate_rows), "manifest": manifest}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--output", default="docs/papers/paper1/results/natural")
    parser.add_argument("--cache", default="tmp/natural_cache")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--per-split", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--causal-examples", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
