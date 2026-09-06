"""Resumable narrow L_min(d) gate for the cue-free Paper 0.6 dataset."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cl.common.artifacts import atomic_write_json, stable_hash, write_csv
from cl.common.model_adapter import TinyTransformerLM
from cl.experiments.paper05_predictive_order_phase import resolve_device
from cl.semantic.predicate_composition import evaluation_rows, training_batch, validate_composition


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _tensor(rows, device):
    length = max(len(row.tokens) for row in rows)
    return torch.tensor([[0] * (length - len(row.tokens)) + list(row.tokens) for row in rows],
                        device=device)


def _model(config, architecture, device):
    return TinyTransformerLM(
        config["vocab_size"], config["max_length"], architecture["width"],
        architecture["layers"], architecture["heads"],
        position_encoding=config["position_encoding"],
    ).to(device)


def _save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(config, architecture, seed, device, directory, smoke, resume):
    steps = 4 if smoke else config["training_steps"]
    batch_size = 8 if smoke else config["batch_size"]
    torch.manual_seed(seed + architecture["layers"] * 1009)
    rng = random.Random(seed + architecture["layers"] * 701)
    model = _model(config, architecture, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    state_path = directory / "training_state.pt"
    losses, first = [], 0
    if resume and state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng.setstate(state["rng"])
        losses, first = state["losses"], state["step"]
    started = time.perf_counter()
    for step in range(first, steps):
        rows = training_batch(config, rng, seed, batch_size)
        x = _tensor(rows, device)
        y = torch.tensor([row.target for row in rows], device=device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = F.cross_entropy(logits[:, -1], y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if step == first or (step + 1) % 50 == 0 or step + 1 == steps:
            losses.append({"step": step + 1, "loss": float(loss.detach().cpu())})
        if (step + 1) % config["checkpoint_every"] == 0 or step + 1 == steps:
            _save(state_path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                               "rng": rng.getstate(), "losses": losses, "step": step + 1})
    torch.save(model.to("cpu").state_dict(), directory / "checkpoint.pt")
    return model.to(device).eval(), losses, time.perf_counter() - started


@torch.no_grad()
def evaluate(model, rows, architecture, seed):
    device = next(model.parameters()).device
    result = []
    buckets = defaultdict(list)
    for row in rows:
        buckets[len(row.tokens)].append(row)
    for length, bucket in sorted(buckets.items()):
        for start in range(0, len(bucket), 128):
            batch = bucket[start:start + 128]
            logits, _ = model(torch.tensor([row.tokens for row in batch], device=device))
            final = logits[:, -1]
            for index, row in enumerate(batch):
                target = row.target
                other = final[index].clone()
                other[target] = -torch.inf
                result.append({
                    "architecture": architecture["name"], "layers": architecture["layers"],
                    "width": architecture["width"], "heads": architecture["heads"],
                    "model_seed": seed, "split": row.split, "required_path": row.required_path,
                    "total_depth": row.total_depth, "template_id": row.template_id,
                    "position_mode": row.position_mode, "tree_seed": row.tree_seed,
                    "positive": row.positive, "candidate_chain_first": row.candidate_chain_first,
                    "example_id": row.example_id, "top1_correct": int(final[index].argmax() == target),
                    "target_margin": float(final[index, target] - other.max()),
                })
    return result


def aggregate(raw: list[dict]) -> list[dict]:
    keys = ("architecture", "layers", "width", "heads", "model_seed", "split", "required_path")
    groups = defaultdict(list)
    for row in raw:
        groups[tuple(row[key] for key in keys)].append(row)
    return [{
        **dict(zip(keys, key)), "n": len(rows),
        "accuracy": float(np.mean([int(row["top1_correct"]) for row in rows])),
        "mean_margin": float(np.mean([float(row["target_margin"]) for row in rows])),
    } for key, rows in sorted(groups.items())]


def aggregate_strata(raw: list[dict]) -> list[dict]:
    keys = ("architecture", "layers", "width", "heads", "model_seed", "split",
            "required_path", "template_id", "position_mode", "tree_seed")
    groups = defaultdict(list)
    for row in raw:
        groups[tuple(row[key] for key in keys)].append(row)
    return [{
        **dict(zip(keys, key)), "n": len(rows),
        "accuracy": float(np.mean([int(row["top1_correct"]) for row in rows])),
        "mean_margin": float(np.mean([float(row["target_margin"]) for row in rows])),
    } for key, rows in sorted(groups.items())]


def main(args) -> None:
    config = json.loads(Path(args.config).read_text())
    audit = validate_composition(config)
    if not audit["valid"]:
        raise RuntimeError(audit)
    output = Path(args.output)
    (output / "models").mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "generator_validation.json", audit)
    device = resolve_device(args.device)
    architectures = config["architectures"]
    seeds = config["model_seeds"]
    specs = [(architecture, seed) for architecture in architectures for seed in seeds]
    if args.max_models:
        specs = specs[:args.max_models]
    for index, (architecture, seed) in enumerate(specs, 1):
        directory = output / "models" / f"{architecture['name']}_seed{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        if args.resume and (directory / "complete.json").exists():
            print(f"[{index}/{len(specs)}] skip {architecture['name']} seed={seed}", flush=True)
            continue
        print(f"[{index}/{len(specs)}] train {architecture['name']} seed={seed} on {device}", flush=True)
        model, losses, seconds = train(
            config, architecture, seed, device, directory, args.smoke, args.resume,
        )
        count = config["smoke_examples_per_cell"] if args.smoke else config["evaluation_examples_per_cell"]
        raw = evaluate(model, evaluation_rows(config, seed, count), architecture, seed)
        write_csv(directory / "raw.csv", raw)
        write_csv(directory / "loss.csv", losses)
        atomic_write_json(directory / "complete.json", {
            "architecture": architecture, "model_seed": seed, "steps": 4 if args.smoke else config["training_steps"],
            "raw_rows": len(raw), "seconds": seconds, "artifact_hash": stable_hash(raw),
        })
    raw, losses, runtimes = [], [], []
    for complete in sorted((output / "models").glob("*/complete.json")):
        meta = json.loads(complete.read_text())
        raw.extend(_read(complete.parent / "raw.csv"))
        losses.extend({"architecture": meta["architecture"]["name"],
                       "model_seed": meta["model_seed"], **row}
                      for row in _read(complete.parent / "loss.csv"))
        runtimes.append({"architecture": meta["architecture"]["name"],
                         "model_seed": meta["model_seed"], "seconds": meta["seconds"]})
    cells = aggregate(raw)
    strata = aggregate_strata(raw)
    write_csv(output / "composition_raw.csv", raw)
    write_csv(output / "composition_cells.csv", cells)
    write_csv(output / "composition_strata.csv", strata)
    write_csv(output / "composition_loss.csv", losses)
    write_csv(output / "composition_runtime.csv", runtimes)
    threshold = config["competence_threshold"]
    frontiers = []
    if not args.smoke:
        for architecture in architectures:
            name = architecture["name"]
            competent_depths = []
            for depth in config["test_depths"]:
                selected = [row for row in strata if row["architecture"] == name and
                            row["split"] == "test" and int(row["required_path"]) == depth]
                expected = 3 * len(config["templates"]) * len(config["position_modes"]) * len(config["tree_seeds"])
                if len(selected) == expected and min(float(row["accuracy"]) for row in selected) >= threshold:
                    competent_depths.append(depth)
            contiguous = []
            for depth in config["test_depths"]:
                if depth not in competent_depths:
                    break
                contiguous.append(depth)
            frontiers.append({"architecture": name, "layers": architecture["layers"],
                              "competent_depths": ";".join(map(str, competent_depths)),
                              "largest_contiguous_competent": max(contiguous) if contiguous else 0,
                              "three_seed_threshold": threshold})
    write_csv(output / "composition_frontiers.csv", frontiers)
    lmin = []
    if frontiers:
        for depth in config["test_depths"]:
            eligible = [row for row in frontiers if depth in {
                int(value) for value in str(row["competent_depths"]).split(";") if value
            }]
            lmin.append({"required_path": depth,
                         "minimum_competent_layers": min((int(row["layers"]) for row in eligible), default=""),
                         "competent_architectures": ";".join(row["architecture"] for row in eligible),
                         "three_seed_worst_stratum_threshold": threshold})
    write_csv(output / "composition_lmin.csv", lmin)
    competent_extrapolation_depths = [row for row in lmin if int(row["required_path"]) > 3 and
                                      row["minimum_competent_layers"] != ""]
    atomic_write_json(output / "composition_manifest.json", {
        "schema_version": config["schema_version"], "device": str(device), "smoke": args.smoke,
        "config_hash": stable_hash(config), "generator_hash": audit["example_hash"],
        "planned_models": len(specs),
        "completed_models": len(list((output / "models").glob("*/complete.json"))),
        "frontiers": frontiers,
        "lmin_eligible": len(competent_extrapolation_depths) >= 4,
        "lmin_required_extrapolation_points": 4,
        "claim_boundary": "L_min(d) is descriptive only; depth is not parameter matched",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper06/compositional_frontier_v7.json")
    parser.add_argument("--output", default="docs/papers/paper0_6/results/v7/compositional_frontier")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-models", type=int)
    main(parser.parse_args())
