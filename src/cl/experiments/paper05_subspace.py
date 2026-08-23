"""Common-subspace and on-manifold controls for Paper 0.5 milestone 3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cl.common.artifacts import RunMetadata, atomic_write_json, stable_hash, write_csv, write_jsonl
from cl.common.metrics import bootstrap_ci
from cl.experiments.paper05_next_iter import (
    PINNED_MODELS, capture_component_updates, donor_families, expanded_family_probes,
    intervention_logprobs,
)
from cl.experiments.paper05_pretrained import Adapter


def fit_raw_subspace(vectors: list[torch.Tensor], rank: int) -> tuple[torch.Tensor, float]:
    """Fit an uncentered residual-update subspace and report energy explained."""
    matrix = torch.stack(vectors).float()
    _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    kept = min(rank, len(singular))
    explained = float(singular[:kept].square().sum() / singular.square().sum().clamp_min(1e-12))
    return vh[:kept], explained


def project(vector: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    basis = basis.to(vector)
    return (vector @ basis.T) @ basis


def family_bootstrap(rows: list[dict]) -> list[dict]:
    output = []
    models = sorted({(row["model_id"], row["revision"]) for row in rows})
    for model, revision in models:
        selected = [row for row in rows if row["model_id"] == model and row["revision"] == revision]
        for mode in sorted({row["mode"] for row in selected}):
            units = {}
            for row in selected:
                if row["mode"] == mode:
                    units.setdefault(row["family"], []).append(row["target_logprob_change"])
            values = [float(np.mean(value)) for value in units.values()]
            estimate, low, high = bootstrap_ci(values, samples=5000, seed=79)
            output.append({"model_id": model, "revision": revision, "mode": mode,
                           "n_families": len(values), "estimate": estimate,
                           "ci_low": low, "ci_high": high,
                           "negative_family_fraction": float(np.mean(np.asarray(values) < 0))})
    return output


def plot_results(output: Path, ranks: list[dict], inference: list[dict]) -> None:
    figures = output / "figures"; figures.mkdir(parents=True, exist_ok=True)
    models = sorted({row["model_id"] for row in inference})
    shown = ("remove_rank1", "remove_rank2", "add_rank1", "add_rank2",
             "nearest_nonequivalent", "norm_matched_nonequivalent")
    fig, axes = plt.subplots(1, len(models), figsize=(12, 4.7), sharey=True)
    if len(models) == 1: axes = [axes]
    for ax, model in zip(axes, models):
        lookup = {row["mode"]: row for row in inference if row["model_id"] == model}
        values = [lookup[mode]["estimate"] for mode in shown]
        errors = [[values[i] - lookup[m]["ci_low"] for i, m in enumerate(shown)],
                  [lookup[m]["ci_high"] - values[i] for i, m in enumerate(shown)]]
        ax.bar(range(len(shown)), values, yerr=errors, capsize=3, color="#4c78a8")
        ax.axhline(0, color="black", lw=.8); ax.set_title(model.split("/")[-1])
        ax.set_xticks(range(len(shown)), [x.replace("_", " ") for x in shown], rotation=32, ha="right")
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("target log-probability change (nats)")
    fig.tight_layout(); fig.savefig(figures / "subspace_and_manifold_controls.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    width = .35; x = np.arange(len(models))
    for offset, rank in ((-.5, 1), (.5, 2)):
        values = [np.median([row["explained_energy"] for row in ranks if row["model_id"] == model and row["rank"] == rank]) for model in models]
        ax.bar(x + offset * width, values, width, label=f"rank {rank}")
    ax.set_xticks(x, [model.split("/")[-1] for model in models]); ax.set_ylim(0, 1.05)
    ax.set_ylabel("median fitted update energy explained"); ax.legend(); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(figures / "common_subspace_rank.png", dpi=180); plt.close(fig)


def run(args) -> None:
    repo, output = Path(args.repo).resolve(), Path(args.output).resolve()
    raw, tables = output / "raw", output / "tables"
    raw.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)
    probes = expanded_family_probes(); evaluation = [row for row in probes if row["identity"] >= 2]
    families = {row["family"] for row in probes}; device = torch.device(args.device)
    models = PINNED_MODELS if not args.model else tuple(x for x in PINNED_MODELS if x[0].split("/")[-1] in args.model)
    rows, ranks, metadata = [], [], []
    for model_id, revision in models:
        adapter = Adapter(model_id, revision, device)
        captures = capture_component_updates(adapter, probes)
        baseline = intervention_logprobs(adapter, evaluation)
        for layer in adapter.selected:
            for component in ("sa", "ff"):
                bases, means = {}, {}
                for family in families:
                    fit = [captures[(family, identity, layer, component)] for identity in (0, 1)]
                    means[family] = torch.stack(fit).mean(0)
                    for rank in (1, 2):
                        basis, explained = fit_raw_subspace(fit, rank)
                        bases[(family, rank)] = basis
                        ranks.append({"model_id": model_id, "revision": revision, "family": family,
                                      "layer": layer, "component": component, "rank": rank,
                                      "explained_energy": explained})
                conditions = {name: [] for name in (
                    "remove_rank1", "remove_rank2", "add_rank1", "add_rank2",
                    "nearest_nonequivalent", "norm_matched_nonequivalent",
                    "equivalent_interp_025", "equivalent_interp_050", "equivalent_interp_075",
                )}
                for probe in evaluation:
                    current = captures[(probe["family"], probe["identity"], layer, component)]
                    for rank in (1, 2):
                        common = project(current, bases[(probe["family"], rank)])
                        conditions[f"remove_rank{rank}"].append(current - common)
                        conditions[f"add_rank{rank}"].append(current + common)
                    candidates = [(family, mean) for family, mean in means.items() if family != probe["family"]]
                    nearest = min(candidates, key=lambda item: float(torch.linalg.vector_norm(item[1] - current)))[1]
                    norm_match = min(candidates, key=lambda item: abs(float(torch.linalg.vector_norm(item[1]) - torch.linalg.vector_norm(current))))[1]
                    conditions["nearest_nonequivalent"].append(nearest)
                    conditions["norm_matched_nonequivalent"].append(norm_match)
                    equivalent = means[probe["family"]]
                    for label, alpha in (("025", .25), ("050", .5), ("075", .75)):
                        conditions[f"equivalent_interp_{label}"].append(current + alpha * (equivalent - current))
                for mode, vectors in conditions.items():
                    changed = intervention_logprobs(adapter, evaluation, layer, component, vectors)
                    for probe, base, value in zip(evaluation, baseline, changed):
                        rows.append({"model_id": model_id, "revision": revision,
                                     **{key: probe[key] for key in ("family", "syntax", "semantic", "identity")},
                                     "layer": layer, "component": component, "mode": mode,
                                     "baseline_target_logprob": base, "target_logprob_change": value - base})
        metadata.append(RunMetadata.capture(repo=repo, run_id=f"paper05-subspace-{model_id.split('/')[-1]}",
            config={"phase": "subspace", "schema": "paper05.next.subspace.v1"},
            model_id=f"{model_id}@{revision}", dataset_id="controlled-text-family-matrix-v2",
            seed=0, device=str(device), dtype="float32", data_hash=stable_hash(probes)).as_dict())
        del adapter.model
        if device.type == "mps": torch.mps.empty_cache()
    inference = family_bootstrap(rows)
    write_jsonl(raw / "subspace_controls.jsonl", rows); write_jsonl(raw / "subspace_metadata.jsonl", metadata)
    write_csv(tables / "subspace_controls.csv", rows); write_csv(tables / "subspace_rank.csv", ranks)
    write_csv(tables / "subspace_inference.csv", inference)
    plot_results(output, ranks, inference)
    manifest = {"schema_version": "paper05.next.subspace.v1", "models": models,
                "identity_disjoint": True, "rows": len(rows), "rank_rows": len(ranks),
                "family_clustered_bootstrap_samples": 5000,
                "artifact_hash": stable_hash({"rows": rows, "ranks": ranks, "inference": inference})}
    atomic_write_json(output / "subspace_manifest.json", manifest); print(json.dumps(manifest, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(); parser.add_argument("--repo", default=".")
    parser.add_argument("--output", default="docs/papers/paper0_5/results/next_iter")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--model", action="append"); return parser.parse_args()


if __name__ == "__main__": run(parse_args())
