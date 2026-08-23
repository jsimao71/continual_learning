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
    parser.add_argument("phase", choices=("families",))
    parser.add_argument("--repo", default=".")
    parser.add_argument("--output", default="docs/papers/paper0_5/results/next_iter")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--model", action="append", help="restrict to a pinned model basename")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.phase == "families":
        run_family_phase(arguments)
