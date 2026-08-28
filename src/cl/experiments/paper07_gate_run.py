"""Resumable three-seed learned P0/F0 gate for Paper 0.7."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from cl.common.artifacts import atomic_write_json, stable_hash, write_csv
from cl.common.model_adapter import TinyTransformerLM
from cl.experiments.paper05_predictive_order_phase import resolve_device
from cl.semantic.paper07_gates import GateExample, gate_examples, validate_gates


def build(config: dict, smoke: bool = False) -> list[GateExample]:
    rows = []
    for stage in ("P0", "F0"):
        for split, seed in config["split_seeds"].items():
            count = 16 if smoke else config["examples"][split]
            rows.extend(gate_examples(stage, split, count, seed, config["sequence_length"]))
    return rows


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary); temporary.replace(path)


def train(config: dict, stage: str, seed: int, rows: list[GateExample], device: torch.device,
          output: Path, smoke: bool, resume: bool) -> tuple[TinyTransformerLM, list[dict]]:
    architecture = config["architecture"]
    steps = 4 if smoke else config.get("steps_by_stage", {}).get(stage, config["base_steps"])
    torch.manual_seed(seed + (0 if stage == "P0" else 10000))
    model = TinyTransformerLM(config["vocab_size"], config["sequence_length"], architecture["width"],
                              architecture["layers"], architecture["heads"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    rng = random.Random(seed + (1000 if stage == "P0" else 11000)); losses = []; first = 0
    state_path = output / "training_state.pt"
    if resume and state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        rng.setstate(state["rng"]); losses = state["losses"]; first = state["step"]
    train_rows = [row for row in rows if row.stage == stage and row.split == "train"]
    for step in range(first, steps):
        batch = [train_rows[rng.randrange(len(train_rows))] for _ in range(8 if smoke else config["batch_size"])]
        x = torch.tensor([row.tokens for row in batch], device=device)
        y = torch.tensor([row.target for row in batch], device=device)
        optimizer.zero_grad(set_to_none=True); logits, _ = model(x)
        loss = F.cross_entropy(logits[:, -1], y); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == steps:
            losses.append({"step": step + 1, "loss": float(loss.detach().cpu())})
        if (step + 1) % 100 == 0 or step + 1 == steps:
            _save(state_path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                               "rng": rng.getstate(), "losses": losses, "step": step + 1})
    torch.save(model.to("cpu").state_dict(), output / "checkpoint.pt")
    return model.eval(), losses


@torch.no_grad()
def evaluate(model: TinyTransformerLM, rows: list[GateExample], stage: str, seed: int,
             device: torch.device) -> list[dict]:
    output = []; model = model.to(device)
    for split in ("validation", "test"):
        selected = [row for row in rows if row.stage == stage and row.split == split]
        for start in range(0, len(selected), 128):
            batch = selected[start:start + 128]; x = torch.tensor([row.tokens for row in batch], device=device)
            logits, _ = model(x); logits = logits[:, -1]
            for index, row in enumerate(batch):
                target_logit = float(logits[index, row.target]); other = 61 if row.target == 60 else 60
                output.append({"stage": stage, "model_seed": seed, "split": split,
                               "example_id": row.example_id, "namespace": row.namespace,
                               "label": row.label, "template": row.template,
                               "target": row.target, "prediction": int(logits[index].argmax()),
                               "correct": int(logits[index].argmax() == row.target),
                               "binary_correct": int(logits[index, row.target] > logits[index, other]),
                               "binary_margin": target_logit - float(logits[index, other])})
    return output


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle: return list(csv.DictReader(handle))


def main(args: argparse.Namespace) -> None:
    config = json.loads(Path(args.config).read_text()); rows = build(config, args.smoke)
    if args.base_steps:
        config = {**config, "base_steps": args.base_steps,
                  "steps_by_stage": {stage: args.base_steps for stage in ("P0", "F0")}}
    audit = validate_gates(rows)
    if not audit["valid"]: raise RuntimeError(audit)
    output = Path(args.output); (output / "models").mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "gate_validation.json", audit); device = resolve_device(args.device)
    specs = [(stage, seed) for stage in ("P0", "F0") for seed in config["model_seeds"]]
    if args.only_stage: specs = [spec for spec in specs if spec[0] == args.only_stage]
    if args.max_models: specs = specs[:args.max_models]
    for index, (stage, seed) in enumerate(specs, 1):
        model_dir = output / "models" / f"{stage}_seed{seed}"; model_dir.mkdir(parents=True, exist_ok=True)
        if args.resume and (model_dir / "complete.json").exists():
            print(f"[{index}/{len(specs)}] skip {stage} seed {seed}", flush=True); continue
        print(f"[{index}/{len(specs)}] train {stage} seed {seed} on {device}", flush=True)
        model, losses = train(config, stage, seed, rows, device, model_dir, args.smoke, args.resume)
        raw = evaluate(model, rows, stage, seed, device)
        write_csv(model_dir / "raw.csv", raw); write_csv(model_dir / "loss.csv", losses)
        atomic_write_json(model_dir / "complete.json", {"stage": stage, "model_seed": seed,
            "rows": len(raw), "artifact_hash": stable_hash(raw)})
    raw = []; losses = []
    for complete in sorted((output / "models").glob("*/complete.json")):
        raw.extend(_read(complete.parent / "raw.csv")); meta = json.loads(complete.read_text())
        losses.extend({"stage": meta["stage"], "model_seed": meta["model_seed"], **row}
                      for row in _read(complete.parent / "loss.csv"))
    write_csv(output / "gate_raw.csv", raw); write_csv(output / "gate_training_loss.csv", losses)
    aggregates = []
    for stage, seed, split in sorted({(row["stage"], row["model_seed"], row["split"]) for row in raw}):
        selected = [row for row in raw if (row["stage"], row["model_seed"], row["split"]) == (stage, seed, split)]
        accuracy = sum(int(row["binary_correct"]) for row in selected) / len(selected)
        aggregates.append({"stage": stage, "model_seed": seed, "split": split, "examples": len(selected),
                           "binary_accuracy": accuracy, "competent": int(accuracy >= config["competence_threshold"])})
    write_csv(output / "gate_accuracy.csv", aggregates)
    test_cells = [row for row in aggregates if row["split"] == "test"]
    atomic_write_json(output / "gate_manifest.json", {"schema_version": config["schema_version"],
        "device": str(device), "smoke": args.smoke, "planned_models": len(specs),
        "completed_models": len(list((output / "models").glob("*/complete.json"))),
        "three_seed_gate": {stage: int(all(row["competent"] for row in test_cells if row["stage"] == stage) and
                                         sum(row["stage"] == stage for row in test_cells) == 3)
                            for stage in ("P0", "F0")}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/paper07/gates_v1.json")
    parser.add_argument("--output", default="docs/papers/paper0_7/results/gates_v1")
    parser.add_argument("--device", default="auto"); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--max-models", type=int)
    parser.add_argument("--only-stage", choices=("P0", "F0"))
    parser.add_argument("--base-steps", type=int)
    main(parser.parse_args())
