"""Broader pretrained mechanism experiments for the Paper 0.5 next iteration.

The module is deliberately phase-oriented: every phase writes an independently
auditable artifact set and can be committed before the next intervention family
is run.  Phase 1 establishes the expanded behavioral family matrix.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cl.common.metrics import bootstrap_ci
from cl.common.artifacts import RunMetadata, atomic_write_json, stable_hash, write_csv, write_jsonl
from cl.experiments.paper05_pretrained import Adapter


PINNED_MODELS = (
    ("EleutherAI/pythia-70m-deduped", "step143000"),
    ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
)

# Each tuple is an identity. Fit/evaluation code must split this axis rather
# than treating the four rows as independent relation families.
DOMAINS = {
    "number": (("one", "two"), ("two", "three"), ("three", "four"), ("four", "one")),
    "color": (("red", "blue"), ("blue", "green"), ("green", "black"), ("black", "red")),
    "animal": (("cat", "dog"), ("dog", "bird"), ("bird", "fish"), ("fish", "cat")),
    "temperature": (("hot", "cold"), ("cold", "hot"), ("high", "low"), ("low", "high")),
    "magnitude": (("high", "low"), ("low", "high"), ("one", "four"), ("four", "one")),
    "geography": (("Paris", "France"), ("Rome", "Italy"), ("France", "Paris"), ("Italy", "Rome")),
    "tense": (("walk", "walked"), ("play", "played"), ("walked", "walk"), ("played", "play")),
    "direction": (("high", "low"), ("hot", "cold"), ("one", "two"), ("red", "blue")),
}
SYNTAXES = ("alternation", "mapping", "arrow", "question")


@dataclass(frozen=True)
class FamilyProbe:
    family: str
    syntax: str
    semantic: str
    identity: int
    source: str
    target: str
    prompt: str
    target_text: str

    def as_dict(self):
        return self.__dict__.copy()


def render_prompt(syntax: str, source: str, target: str, identity: int) -> str:
    """Render two demonstrations followed by a query in a fixed template."""
    if syntax == "alternation":
        return f"{source} {target} {source} {target} {source}"
    if syntax == "mapping":
        return f"{source} maps to {target}; {source} maps to {target}; {source} maps to"
    if syntax == "arrow":
        return f"{source} -> {target}; {source} -> {target}; {source} ->"
    if syntax == "question":
        return f"Q: {source}? A: {target}. Q: {source}? A: {target}. Q: {source}? A:"
    raise ValueError(f"unknown syntax: {syntax}")


def expanded_family_probes() -> list[dict]:
    rows = []
    for semantic, identities in DOMAINS.items():
        for syntax in SYNTAXES:
            family = f"{syntax}:{semantic}"
            for identity, (source, target) in enumerate(identities):
                rows.append(FamilyProbe(
                    family=family, syntax=syntax, semantic=semantic,
                    identity=identity, source=source, target=target,
                    prompt=render_prompt(syntax, source, target, identity),
                    target_text=" " + target,
                ).as_dict())
    return rows


@torch.no_grad()
def score_family_matrix(adapter: Adapter, probes: list[dict]) -> list[dict]:
    """Score probes in small padded batches, retaining selected-depth metrics."""
    rows = []
    tokenizer = adapter.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    for start in range(0, len(probes), 16):
        batch = probes[start:start + 16]
        encoded = tokenizer(
            [row["prompt"] for row in batch], return_tensors="pt", padding=True,
            add_special_tokens=True,
        ).to(adapter.device)
        output = adapter.model(**encoded, output_hidden_states=True, return_dict=True, use_cache=False)
        final_positions = encoded.attention_mask.sum(dim=1) - 1
        targets = torch.tensor(
            [adapter.target_id(row["target_text"]) for row in batch], device=adapter.device,
        )
        for layer in adapter.selected:
            states = output.hidden_states[layer + 1]
            final_states = states[torch.arange(len(batch), device=adapter.device), final_positions]
            logits = adapter.head(adapter.norm(final_states))
            probabilities = torch.softmax(logits, dim=-1)
            logprobabilities = torch.log_softmax(logits, dim=-1)
            target_logits = logits.gather(1, targets[:, None]).squeeze(1)
            masked = logits.clone()
            masked.scatter_(1, targets[:, None], float("-inf"))
            margins = target_logits - masked.max(dim=-1).values
            for index, probe in enumerate(batch):
                probability = probabilities[index].detach().cpu().numpy()
                rows.append({
                    "model_id": adapter.model_id, "revision": adapter.revision,
                    **{key: probe[key] for key in ("family", "syntax", "semantic", "identity", "source", "target")},
                    "split": "fit" if probe["identity"] < 2 else "evaluation",
                    "layer": layer,
                    "target_logprob": float(logprobabilities[index, targets[index]]),
                    "target_probability": float(probability[targets[index].item()]),
                    "target_margin": float(margins[index]),
                    "entropy_bits": float(-(probability * np.log2(np.clip(probability, 1e-12, 1.0))).sum()),
                    "top1_correct": int(int(probability.argmax()) == targets[index].item()),
                })
    return rows


def summarize_family_matrix(rows: list[dict]) -> list[dict]:
    grouped = {}
    keys = ("model_id", "revision", "family", "syntax", "semantic", "split", "layer")
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    for key, values in sorted(grouped.items()):
        output.append({
            **dict(zip(keys, key)), "n_identities": len(values),
            "mean_target_logprob": float(np.mean([row["target_logprob"] for row in values])),
            "mean_target_probability": float(np.mean([row["target_probability"] for row in values])),
            "mean_target_margin": float(np.mean([row["target_margin"] for row in values])),
            "mean_entropy_bits": float(np.mean([row["entropy_bits"] for row in values])),
            "top1_accuracy": float(np.mean([row["top1_correct"] for row in values])),
        })
    return output


def donor_families(probe: dict, families: set[str]) -> dict[str, str]:
    """Choose deterministic factorial donors without consulting representations."""
    ordered = sorted(families)
    same_semantic = [value for value in ordered if value != probe["family"] and value.endswith(":" + probe["semantic"])]
    same_syntax = [value for value in ordered if value != probe["family"] and value.startswith(probe["syntax"] + ":")]
    different = [value for value in ordered if not value.startswith(probe["syntax"] + ":") and not value.endswith(":" + probe["semantic"])]
    return {
        "replace_equivalent": probe["family"],
        "replace_syntax_mismatch": same_semantic[0],
        "replace_semantic_mismatch": same_syntax[0],
        "replace_nonequivalent": different[0],
    }


@torch.no_grad()
def capture_component_updates(adapter: Adapter, probes: list[dict]) -> dict[tuple, torch.Tensor]:
    """Capture last-position component updates with left padding."""
    tokenizer = adapter.tokenizer
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    captured = {}
    try:
        for start in range(0, len(probes), 16):
            batch = probes[start:start + 16]
            encoded = tokenizer([row["prompt"] for row in batch], return_tensors="pt", padding=True).to(adapter.device)
            batch_values = {}
            handles = []
            for layer in adapter.selected:
                for component in ("sa", "ff"):
                    def save(module, args, output, layer=layer, component=component):
                        batch_values[(layer, component)] = adapter._tensor(output)[:, -1].detach().cpu()
                    handles.append(adapter._module(layer, component).register_forward_hook(save))
            adapter.model(**encoded, return_dict=True, use_cache=False)
            for handle in handles:
                handle.remove()
            for index, probe in enumerate(batch):
                for key, values in batch_values.items():
                    captured[(probe["family"], probe["identity"], *key)] = values[index]
    finally:
        tokenizer.padding_side = original_side
    return captured


@torch.no_grad()
def intervention_logprobs(adapter: Adapter, probes: list[dict], layer: int | None = None,
                          component: str | None = None, vectors: list[torch.Tensor] | None = None) -> list[float]:
    tokenizer = adapter.tokenizer
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    values = []
    try:
        for start in range(0, len(probes), 16):
            batch = probes[start:start + 16]
            encoded = tokenizer([row["prompt"] for row in batch], return_tensors="pt", padding=True).to(adapter.device)
            targets = torch.tensor([adapter.target_id(row["target_text"]) for row in batch], device=adapter.device)
            intervention = None
            if vectors is not None:
                intervention = {"layer": layer, "component": component, "mode": "replace",
                                "vector": torch.stack(vectors[start:start + len(batch)]).to(adapter.device)}
            context = adapter.intervention(**intervention) if intervention else adapter.intervention(0, "sa", "none")
            with context:
                logits = adapter.model(**encoded, return_dict=True, use_cache=False).logits[:, -1]
            values.extend(torch.log_softmax(logits, -1).gather(1, targets[:, None]).squeeze(1).cpu().tolist())
    finally:
        tokenizer.padding_side = original_side
    return values


def causal_family_inference(rows: list[dict]) -> list[dict]:
    units = {}
    for row in rows:
        units.setdefault((row["model_id"], row["revision"], row["family"], row["mode"]), []).append(row["target_logprob_change"])
    output = []
    for model, revision in sorted({(row["model_id"], row["revision"]) for row in rows}):
        families = sorted({row["family"] for row in rows if row["model_id"] == model and row["revision"] == revision})
        for comparison in ("replace_syntax_mismatch", "replace_semantic_mismatch", "replace_nonequivalent"):
            deltas = [float(np.mean(units[(model, revision, family, "replace_equivalent")]) -
                            np.mean(units[(model, revision, family, comparison)])) for family in families]
            estimate, low, high = bootstrap_ci(deltas, samples=5000, seed=71)
            output.append({"model_id": model, "revision": revision,
                           "contrast": f"equivalent_minus_{comparison}", "n_families": len(families),
                           "estimate": estimate, "ci_low": low, "ci_high": high,
                           "positive_family_fraction": float(np.mean(np.asarray(deltas) > 0))})
    return output


def plot_factorial_causal(output: Path, rows: list[dict], inference: list[dict]) -> None:
    figures = output / "figures"; figures.mkdir(parents=True, exist_ok=True)
    models = sorted({row["model_id"] for row in rows})
    fig, axes = plt.subplots(1, len(models), figsize=(10, 4.5), sharey=True)
    if len(models) == 1: axes = [axes]
    labels = ("syntax mismatch", "semantic mismatch", "fully nonequivalent")
    modes = ("replace_syntax_mismatch", "replace_semantic_mismatch", "replace_nonequivalent")
    for ax, model in zip(axes, models):
        selected = [row for row in inference if row["model_id"] == model]
        lookup = {row["contrast"].replace("equivalent_minus_", ""): row for row in selected}
        values = [lookup[mode]["estimate"] for mode in modes]
        lows = [values[index] - lookup[mode]["ci_low"] for index, mode in enumerate(modes)]
        highs = [lookup[mode]["ci_high"] - values[index] for index, mode in enumerate(modes)]
        ax.bar(range(3), values, yerr=[lows, highs], capsize=4, color=("#4c78a8", "#f58518", "#54a24b"))
        ax.axhline(0, color="black", linewidth=.8); ax.set_xticks(range(3), labels, rotation=25, ha="right")
        ax.set_title(model.split("/")[-1]); ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("equivalent advantage (target log-probability, nats)")
    fig.tight_layout(); fig.savefig(figures / "factorial_causal_transfer.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, len(models), figsize=(11, 4.5), sharey=True)
    if len(models) == 1: axes = [axes]
    for ax, model in zip(axes, models):
        selected = [row for row in rows if row["model_id"] == model]
        layers = sorted({row["layer"] for row in selected})
        for component, marker in (("sa", "o"), ("ff", "s")):
            syntax_values, semantic_values = [], []
            for layer in layers:
                subset = [row for row in selected if row["layer"] == layer and row["component"] == component]
                units = {}
                for row in subset: units.setdefault((row["family"], row["mode"]), []).append(row["target_logprob_change"])
                families = sorted({row["family"] for row in subset})
                syntax_values.append(np.mean([np.mean(units[(family, "replace_equivalent")]) - np.mean(units[(family, "replace_syntax_mismatch")]) for family in families]))
                semantic_values.append(np.mean([np.mean(units[(family, "replace_equivalent")]) - np.mean(units[(family, "replace_semantic_mismatch")]) for family in families]))
            ax.plot(layers, syntax_values, marker=marker, label=f"{component.upper()} syntax")
            ax.plot(layers, semantic_values, marker=marker, linestyle="--", label=f"{component.upper()} semantics")
        ax.axhline(0, color="black", linewidth=.8); ax.set_title(model.split("/")[-1]); ax.set_xlabel("layer")
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    axes[0].set_ylabel("equivalent advantage (nats)")
    fig.tight_layout(); fig.savefig(figures / "syntax_semantics_by_component_depth.png", dpi=180); plt.close(fig)


def run_causal_phase(args) -> None:
    repo = Path(args.repo).resolve()
    output = Path(args.output).resolve()
    raw, tables = output / "raw", output / "tables"
    raw.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)
    probes = expanded_family_probes()
    evaluation = [row for row in probes if row["identity"] >= 2]
    families = {row["family"] for row in probes}
    rows, metadata = [], []
    device = torch.device(args.device)
    models = PINNED_MODELS if not args.model else tuple(model for model in PINNED_MODELS if model[0].split("/")[-1] in args.model)
    for model_id, revision in models:
        adapter = Adapter(model_id, revision, device)
        captures = capture_component_updates(adapter, probes)
        donors = {}
        for family in families:
            for layer in adapter.selected:
                for component in ("sa", "ff"):
                    donors[(family, layer, component)] = torch.stack([
                        captures[(family, identity, layer, component)] for identity in (0, 1)
                    ]).mean(0)
        baseline = intervention_logprobs(adapter, evaluation)
        for layer in adapter.selected:
            for component in ("sa", "ff"):
                for mode in ("replace_equivalent", "replace_syntax_mismatch", "replace_semantic_mismatch", "replace_nonequivalent"):
                    vectors = [donors[(donor_families(probe, families)[mode], layer, component)] for probe in evaluation]
                    changed = intervention_logprobs(adapter, evaluation, layer, component, vectors)
                    for probe, base, value in zip(evaluation, baseline, changed):
                        rows.append({"model_id": model_id, "revision": revision,
                                     **{key: probe[key] for key in ("family", "syntax", "semantic", "identity")},
                                     "layer": layer, "component": component, "mode": mode,
                                     "donor_family": donor_families(probe, families)[mode],
                                     "baseline_target_logprob": base, "target_logprob_change": value - base})
        metadata.append(RunMetadata.capture(
            repo=repo, run_id=f"paper05-next-causal-{model_id.split('/')[-1]}",
            config={"phase": "causal", "schema": "paper05.next.causal.v1"}, model_id=f"{model_id}@{revision}",
            dataset_id="controlled-text-family-matrix-v2", seed=0, device=str(device), dtype="float32",
            data_hash=stable_hash(probes),
        ).as_dict())
        del adapter.model
        if device.type == "mps": torch.mps.empty_cache()
    inference = causal_family_inference(rows)
    write_jsonl(raw / "factorial_causal.jsonl", rows)
    write_jsonl(raw / "causal_metadata.jsonl", metadata)
    write_csv(tables / "factorial_causal.csv", rows)
    write_csv(tables / "factorial_causal_inference.csv", inference)
    plot_factorial_causal(output, rows, inference)
    atomic_write_json(output / "causal_manifest.json", {
        "schema_version": "paper05.next.causal.v1", "models": models, "n_families": len(families),
        "fit_identities": [0, 1], "evaluation_identities": [2, 3], "rows": len(rows),
        "family_clustered_bootstrap_samples": 5000, "artifact_hash": stable_hash({"rows": rows, "inference": inference}),
    })
    print(json.dumps({"rows": len(rows), "inference": inference}, indent=2))


def run_family_phase(args) -> None:
    repo = Path(args.repo).resolve()
    output = Path(args.output).resolve()
    raw = output / "raw"
    tables = output / "tables"
    raw.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    probes = expanded_family_probes()
    rows, metadata = [], []
    device = torch.device(args.device)
    models = PINNED_MODELS if not args.model else tuple(model for model in PINNED_MODELS if model[0].split("/")[-1] in args.model)
    if not models:
        raise ValueError(f"--model did not match a pinned model: {args.model}")
    for model_id, revision in models:
        adapter = Adapter(model_id, revision, device)
        for probe in probes:
            adapter.target_id(probe["target_text"])
        rows.extend(score_family_matrix(adapter, probes))
        metadata.append(RunMetadata.capture(
            repo=repo, run_id=f"paper05-next-families-{model_id.split('/')[-1]}",
            config={"phase": "families", "family_schema": "paper05.next.families.v1"},
            model_id=f"{model_id}@{revision}", dataset_id="controlled-text-family-matrix-v2",
            seed=0, device=str(device), dtype="float32", data_hash=stable_hash(probes),
        ).as_dict())
        del adapter.model
        if device.type == "mps":
            torch.mps.empty_cache()
    summary = summarize_family_matrix(rows)
    write_jsonl(raw / "family_matrix.jsonl", rows)
    write_jsonl(raw / "metadata.jsonl", metadata)
    write_csv(tables / "family_matrix.csv", summary)
    manifest = {
        "schema_version": "paper05.next.families.v1",
        "models": models,
        "n_families": len({row["family"] for row in probes}),
        "n_semantic_domains": len(DOMAINS), "n_syntax_domains": len(SYNTAXES),
        "n_identities_per_family": 4, "identity_disjoint_split": {"fit": [0, 1], "evaluation": [2, 3]},
        "rows": len(rows), "artifact_hash": stable_hash({"rows": rows, "summary": summary}),
    }
    atomic_write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("families", "causal"))
    parser.add_argument("--repo", default=".")
    parser.add_argument("--output", default="docs/papers/paper0_5/results/next_iter")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--model", action="append", help="restrict to a pinned model basename")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.phase == "families":
        run_family_phase(arguments)
    elif arguments.phase == "causal":
        run_causal_phase(arguments)
