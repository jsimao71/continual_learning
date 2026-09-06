"""Resumable, template-stratified three-seed P2/F2 acquisition gate."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from cl.common.artifacts import atomic_write_json, stable_hash, write_csv
from cl.common.model_adapter import TinyTransformerLM
from cl.experiments.paper05_predictive_order_phase import resolve_device
from cl.semantic.paper07_gates import NO, SYMBOL, YES
from cl.semantic.paper07_p1_f1 import SYMBOL_COUNT
from cl.semantic.paper07_p2_f2 import FAIL, NextGateExample, next_gate_examples, validate_next_gates


def build(config: dict, smoke: bool = False) -> list[NextGateExample]:
    rows = []
    for stage in ("P2", "F2"):
        for split, seed in config["split_seeds"].items():
            count = config["smoke_examples"] if smoke else config["examples"][split]
            rows.extend(next_gate_examples(stage, split, count, seed, config["sequence_length"]))
    return rows


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def train(config: dict, stage: str, seed: int, rows: list[NextGateExample],
          device: torch.device, output: Path, smoke: bool, resume: bool):
    arch = config["architecture"]
    steps = config["smoke_steps"] if smoke else config["steps_by_stage"][stage]
    batch_size = config["smoke_batch_size"] if smoke else config["batch_size"]
    torch.manual_seed(seed + (20000 if stage == "F2" else 0))
    model = TinyTransformerLM(
        config["vocab_size"], config["sequence_length"], arch["width"], arch["layers"],
        arch["heads"], arch["mlp_ratio"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    rng = random.Random(seed + (22000 if stage == "F2" else 12000))
    losses, first = [], 0
    state_path = output / "training_state.pt"
    if resume and state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng.setstate(state["rng"])
        losses, first = state["losses"], state["step"]
    train_rows = [row for row in rows if row.stage == stage and row.split == "train"]
    started = time.perf_counter()
    for step in range(first, steps):
        batch = [train_rows[rng.randrange(len(train_rows))] for _ in range(batch_size)]
        x = torch.tensor([row.tokens for row in batch], device=device)
        y = torch.tensor([row.target for row in batch], device=device)
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
    torch.save(model.to("cpu").state_dict(), output / "checkpoint.pt")
    return model.eval(), losses, time.perf_counter() - started


@torch.no_grad()
def evaluate(model, rows, stage, seed, device):
    model = model.to(device)
    output = []
    grammar = [YES, NO] if stage == "P2" else list(range(SYMBOL, SYMBOL + SYMBOL_COUNT)) + [FAIL]
    for split in ("validation", "test"):
        selected = [row for row in rows if row.stage == stage and row.split == split]
        for start in range(0, len(selected), 128):
            batch = selected[start:start + 128]
            x = torch.tensor([row.tokens for row in batch], device=device)
            logits, _ = model(x)
            logits = logits[:, -1]
            restricted = logits[:, grammar]
            predictions = [grammar[index] for index in restricted.argmax(-1).tolist()]
            unrestricted = logits.argmax(-1).tolist()
            for row_index, (row, prediction, raw_prediction) in enumerate(
                    zip(batch, predictions, unrestricted)):
                competitors = [token for token in grammar if token != row.target]
                competitor = max(float(logits[row_index, token]) for token in competitors)
                output.append({
                    "stage": stage, "model_seed": seed, "split": split,
                    "example_id": row.example_id, "template": row.template, "label": row.label,
                    "target": row.target, "prediction": prediction,
                    "correct": int(prediction == row.target),
                    "unrestricted_prediction": raw_prediction,
                    "unrestricted_correct": int(raw_prediction == row.target),
                    "target_margin": float(logits[row_index, row.target]) - competitor,
                    "oracle_substitution": row.oracle_substitution or "",
                    "failure_reason": row.failure_reason or "",
                })
    return output


def aggregate(raw: list[dict], threshold: float) -> list[dict]:
    rows = []
    keys = sorted({(row["stage"], int(row["model_seed"]), row["split"], row["template"])
                   for row in raw})
    for stage, seed, split, template in keys:
        selected = [row for row in raw if
                    (row["stage"], int(row["model_seed"]), row["split"], row["template"]) ==
                    (stage, seed, split, template)]
        accuracy = sum(int(row["correct"]) for row in selected) / len(selected)
        rows.append({
            "stage": stage, "model_seed": seed, "split": split, "template": template,
            "examples": len(selected), "accuracy": accuracy,
            "mean_target_margin": sum(float(row["target_margin"]) for row in selected) / len(selected),
            "competent": int(accuracy >= threshold),
        })
    return rows


def main(args) -> None:
    config = json.loads(Path(args.config).read_text())
    rows = build(config, args.smoke)
    audit = validate_next_gates(rows)
    if not audit["valid"]:
        raise RuntimeError(audit)
    output = Path(args.output)
    (output / "models").mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "gate_validation.json", audit)
    device = resolve_device(args.device)
    specs = [(stage, seed) for stage in ("P2", "F2") for seed in config["model_seeds"]]
    if args.only_stage:
        specs = [spec for spec in specs if spec[0] == args.only_stage]
    if args.max_models:
        specs = specs[:args.max_models]
    for index, (stage, seed) in enumerate(specs, 1):
        model_dir = output / "models" / f"{stage}_seed{seed}"
        model_dir.mkdir(parents=True, exist_ok=True)
        if args.resume and (model_dir / "complete.json").exists():
            print(f"[{index}/{len(specs)}] skip {stage} seed {seed}", flush=True)
            continue
        print(f"[{index}/{len(specs)}] train {stage} seed={seed} on {device}", flush=True)
        model, losses, seconds = train(
            config, stage, seed, rows, device, model_dir, args.smoke, args.resume,
        )
        raw = evaluate(model, rows, stage, seed, device)
        write_csv(model_dir / "raw.csv", raw)
        write_csv(model_dir / "loss.csv", losses)
        atomic_write_json(model_dir / "complete.json", {
            "stage": stage, "model_seed": seed, "rows": len(raw),
            "artifact_hash": stable_hash(raw), "seconds": seconds,
        })
    raw, losses, runtimes = [], [], []
    for complete in sorted((output / "models").glob("*/complete.json")):
        meta = json.loads(complete.read_text())
        raw.extend(_read(complete.parent / "raw.csv"))
        losses.extend({"stage": meta["stage"], "model_seed": meta["model_seed"], **row}
                      for row in _read(complete.parent / "loss.csv"))
        runtimes.append({"stage": meta["stage"], "model_seed": meta["model_seed"],
                         "seconds": meta["seconds"]})
    write_csv(output / "gate_raw.csv", raw)
    write_csv(output / "gate_training_loss.csv", losses)
    write_csv(output / "gate_runtime.csv", runtimes)
    by_template = aggregate(raw, config["competence_threshold"])
    write_csv(output / "gate_accuracy_by_template.csv", by_template)
    overall = []
    for stage, seed, split in sorted({(row["stage"], int(row["model_seed"]), row["split"])
                                      for row in raw}):
        selected = [row for row in raw if
                    (row["stage"], int(row["model_seed"]), row["split"]) == (stage, seed, split)]
        templates = [row for row in by_template if
                     (row["stage"], int(row["model_seed"]), row["split"]) == (stage, seed, split)]
        accuracy = sum(int(row["correct"]) for row in selected) / len(selected)
        overall.append({
            "stage": stage, "model_seed": seed, "split": split, "examples": len(selected),
            "accuracy": accuracy, "worst_template_accuracy": min(float(row["accuracy"])
                                                                   for row in templates),
            "competent": int(accuracy >= config["competence_threshold"] and
                             all(int(row["competent"]) for row in templates)),
        })
    write_csv(output / "gate_accuracy.csv", overall)
    tests = [row for row in overall if row["split"] == "test"]
    dataset_payload = [{"stage": row.stage, "split": row.split, "example_id": row.example_id,
                        "tokens": row.tokens, "target": row.target, "template": row.template,
                        "oracle_substitution": row.oracle_substitution,
                        "failure_reason": row.failure_reason} for row in rows]
    atomic_write_json(output / "gate_manifest.json", {
        "schema_version": config["schema_version"], "device": str(device), "smoke": args.smoke,
        "config_hash": stable_hash(config), "dataset_hash": stable_hash(dataset_payload),
        "planned_models": len(specs),
        "completed_models": len(list((output / "models").glob("*/complete.json"))),
        "dataset_audit": audit, "template_stratified": True,
        "decoding": {"P2": "argmax over {YES,NO}",
                     "F2": "argmax over 12 binding symbols plus FAIL",
                     "unrestricted_argmax_also_recorded": True},
        "three_seed_gate": {
            stage: int(sum(row["stage"] == stage for row in tests) == 3 and
                       all(int(row["competent"]) for row in tests if row["stage"] == stage))
            for stage in ("P2", "F2")
        },
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper07/p2_f2_gate_v1.json")
    parser.add_argument("--output", default="docs/papers/paper0_7/results/p2_f2_gate_v1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-models", type=int)
    parser.add_argument("--only-stage", choices=("P2", "F2"))
    main(parser.parse_args())
