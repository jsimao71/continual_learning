"""Behavioral phase grid for the Paper 0.8 contextual-copy calibration."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from cl.common.artifacts import atomic_write_json, write_csv
from cl.common.model_adapter import TinyTransformerLM
from cl.experiments.paper05_predictive_order_phase import resolve_device
from cl.semantic.paper08_copy import controlled_copy, generate_copy_examples, validate_copy

CONDITIONS = ("correct", "none", "shuffled_pairings", "shuffled_order", "wrong_query")


def make_model(vocab_size, max_length, width, layers, heads, mlp_ratio, family, device):
    model = TinyTransformerLM(vocab_size, max_length, width, layers, heads, mlp_ratio).to(device)
    if family == "sa_only":
        for block in model.blocks:
            for parameter in block.ff.parameters():
                parameter.data.zero_()
                parameter.requires_grad_(False)
    elif family != "sa_ff":
        raise ValueError(family)
    return model


def _metrics(logits, target):
    values = logits.detach()
    competitor = torch.cat((values[:target], values[target + 1 :])).max()
    return {
        "top1": int(values.argmax() == target),
        "rank": int((values > values[target]).sum()) + 1,
        "margin": float(values[target] - competitor),
        "probability": float(values.softmax(-1)[target]),
    }


def train_cell(setting, cfg, device, updates):
    torch.manual_seed(setting["seed"])
    rng = random.Random(setting["seed"] + 917)
    train = generate_copy_examples(
        setting["regime"], setting["vocabulary"], setting["episodes"],
        pairs_per_prompt=setting["pairs"], mappings=setting["mappings"], seed=setting["seed"],
    )
    max_token = max(max(row.tokens) for row in train)
    model = make_model(max_token + 1, len(train[0].tokens), setting["width"], setting["layers"],
                       setting["heads"], setting["mlp_ratio"], setting["family"], device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg["learning_rate"])
    loss_rows = []
    model.train()
    for step in range(updates):
        batch = [train[rng.randrange(len(train))] for _ in range(cfg["batch_size"])]
        tokens = torch.tensor([row.tokens for row in batch], device=device)
        targets = torch.tensor([row.target for row in batch], device=device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(tokens)
        loss = torch.nn.functional.cross_entropy(logits[:, -1], targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0 or step == updates - 1:
            loss_rows.append({**setting, "step": step + 1, "loss": float(loss)})
    return model.eval(), train, loss_rows


@torch.no_grad()
def evaluate(model, setting):
    test = generate_copy_examples(setting["regime"], setting["vocabulary"],
                                  max(48, setting["vocabulary"] * (setting["vocabulary"] - 1)),
                                  pairs_per_prompt=setting["pairs"], mappings="fresh-random",
                                  seed=setting["seed"] + 10000, family=1)
    raw = []
    for example_id, row in enumerate(test):
        for condition in CONDITIONS:
            controlled = controlled_copy(row, condition, setting["seed"] + example_id)
            tokens = torch.tensor([controlled.tokens], device=next(model.parameters()).device)
            logits, _ = model(tokens)
            raw.append({**setting, "example_id": example_id, "condition": condition,
                        "target": row.target, **_metrics(logits[0, -1], row.target)})
    return raw, test


def aggregate(raw, cfg):
    conditions = {name: [row for row in raw if row["condition"] == name] for name in CONDITIONS}
    def mean(name, field): return float(np.mean([row[field] for row in conditions[name]]))
    correct_accuracy = mean("correct", "top1")
    control_accuracy = max(mean(name, "top1") for name in ("shuffled_pairings", "wrong_query"))
    return {
        "correct_accuracy": correct_accuracy,
        "correct_mean_rank": mean("correct", "rank"),
        "context_free_accuracy": mean("none", "top1"),
        "context_free_mean_rank": mean("none", "rank"),
        "copy_margin_gain": mean("correct", "margin") - mean("none", "margin"),
        "max_association_control_accuracy": control_accuracy,
        "order_control_accuracy": mean("shuffled_order", "top1"),
        "competent": int(correct_accuracy >= cfg["competence_accuracy"]
                         and mean("none", "rank") >= cfg["minimum_context_free_rank"]
                         and control_accuracy <= cfg["maximum_control_accuracy"]),
    }


def calibration_settings(cfg):
    # A compact bracket first; the minimality runner expands each axis only
    # after a replicated competent reference has been found.
    settings = []
    for family in cfg["families"]:
        for layers in (1, 2):
            for width in (8, 16, 32):
                for heads in cfg["heads_by_width"][str(width)]:
                    if heads > 2:
                        continue
                    for seed in cfg["model_seeds"]:
                        settings.append({"regime": "C2", "vocabulary": 4, "episodes": 256,
                                         "mappings": "fresh-random", "pairs": 3, "family": family,
                                         "layers": layers, "width": width, "heads": heads,
                                         "mlp_ratio": 2, "seed": seed})
    return settings


def run(settings, cfg, output, device, updates):
    output.mkdir(parents=True, exist_ok=True)
    raw_rows, cells, loss_rows, audits = [], [], [], []
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    for index, setting in enumerate(settings):
        model, train, losses = train_cell(setting, cfg, device, updates)
        raw, test = evaluate(model, setting)
        summary = aggregate(raw, cfg)
        cell_id = f"{setting['regime']}_{setting['family']}_L{setting['layers']}_W{setting['width']}_H{setting['heads']}_S{setting['seed']}"
        cells.append({"cell_id": cell_id, **setting, **summary})
        audits.append({"cell_id": cell_id, "split": "train", **validate_copy(train)})
        audits.append({"cell_id": cell_id, "split": "test", **validate_copy(test)})
        raw_rows.extend({"cell_id": cell_id, **row} for row in raw)
        loss_rows.extend({"cell_id": cell_id, **row} for row in losses)
        torch.save(model.state_dict(), checkpoint_dir / f"{cell_id}.pt")
        print(f"[{index + 1}/{len(settings)}] {cell_id}: acc={summary['correct_accuracy']:.3f} "
              f"rank0={summary['context_free_mean_rank']:.2f} control={summary['max_association_control_accuracy']:.3f} "
              f"competent={summary['competent']}", flush=True)
    write_csv(output / "copy_generator_validation.csv", audits)
    write_csv(output / "copy_phase_grid.csv", cells)
    write_csv(output / "copy_phase_raw.csv", raw_rows)
    write_csv(output / "copy_training_loss.csv", loss_rows)
    atomic_write_json(output / "copy_phase_manifest.json", {
        "cells": len(cells), "competent_cells": sum(row["competent"] for row in cells),
        "updates": updates, "device": str(device),
    })
    return cells


def main(args):
    cfg = json.loads(Path(args.config).read_text())
    device = resolve_device(args.device or cfg["device"])
    settings = calibration_settings(cfg)
    if args.smoke:
        settings = settings[:1]
    if args.limit:
        settings = settings[:args.limit]
    run(settings, cfg, Path(args.output), device, 30 if args.smoke else cfg["updates"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper08/copy_v1.json")
    parser.add_argument("--output", default="docs/papers/paper0_8/results/copy")
    parser.add_argument("--device")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int)
    main(parser.parse_args())
