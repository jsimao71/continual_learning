"""Resumable learned-model runner for the Paper 0.5 stress frontier.

The first official tranche is deliberately small: a three-seed,
parameter-matched L2/W64 versus L8/W32 contrast at H4/T1.  Each checkpoint is
evaluated on every one-factor dataset cell.  Model directories are completion
units, and an in-progress training state permits exact step-level resumption.
"""
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
from cl.experiments.paper05_stress_frontier import (
    designed_cells,
    evaluation,
    make_example,
    transformer_parameter_count,
    validate,
)


PILOT_ARCHITECTURES = ((2, 64, 4, 1), (8, 32, 4, 1))


def initialization_seed(depth: int, width: int, heads: int, seed: int) -> int:
    """Keep weights identical across T1/T2/T4 budget comparisons."""
    # The trailing +1 preserves the already-completed T1 initialization, whose
    # original formula included budget=1.
    return seed + depth * 1009 + width * 31 + heads * 7 + 1


def data_seed(depth: int, width: int, heads: int, seed: int) -> int:
    """Keep the shorter run an exact data-order prefix of longer budgets."""
    return seed + depth * 10007 + width * 101 + heads * 11 + 1


def tranche(config: dict, name: str) -> list[tuple[int, int, int, int, int]]:
    if name not in {"pilot", "rescue_t2", "rescue_t4"}:
        raise ValueError(f"unknown tranche {name}")
    requested_budget = {"pilot": 1, "rescue_t2": 2, "rescue_t4": 4}[name]
    return [(depth, width, heads, requested_budget, seed)
            for depth, width, heads, _ in PILOT_ARCHITECTURES
            for seed in config["model_seeds"]]


def training_batch(config: dict, cells: list[dict], rng: random.Random,
                   batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    physical = {}
    for cell in cells:
        key = (cell["predictive_order"], cell["raw_length"], cell["nuisance_count"],
               cell["requested_dependency_span"], cell["generator_family"])
        physical.setdefault(key, cell)
    cells = list(physical.values())
    rows = []
    for _ in range(batch_size):
        cell = rng.choice(cells)
        rows.append(make_example(
            config, order=cell["predictive_order"], raw_length=cell["raw_length"],
            nuisance=cell["nuisance_count"], span=cell["requested_dependency_span"],
            family=cell["generator_family"], index=rng.randrange(100_000_000), split="train"))
    return (torch.tensor([row["tokens"] for row in rows], device=device),
            torch.tensor([row["target"] for row in rows], device=device))


def _save_training_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(config: dict, spec: tuple[int, int, int, int, int], device: torch.device,
          model_dir: Path, resume: bool, smoke: bool) -> tuple[TinyTransformerLM, list[dict]]:
    depth, width, heads, budget, seed = spec
    steps = (4 if smoke else config["base_steps"] * budget)
    batch_size = 8 if smoke else config.get("batch_size", 64)
    torch.manual_seed(initialization_seed(depth, width, heads, seed))
    model = TinyTransformerLM(config["vocab_size"], config["sequence_length"], width, depth, heads).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate", .003))
    cells, _ = designed_cells(config)
    rng = random.Random(data_seed(depth, width, heads, seed))
    state_path = model_dir / "training_state.pt"
    losses: list[dict] = []
    first_step = 0
    if resume and state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        rng.setstate(state["python_rng"]); first_step = int(state["step"]); losses = state["losses"]
    for step in range(first_step, steps):
        x, y = training_batch(config, cells, rng, batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = F.cross_entropy(logits[:, -1], y)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == steps:
            losses.append({"step": step + 1, "loss": float(loss.detach().cpu())})
        if (step + 1) % 100 == 0 or step + 1 == steps:
            _save_training_state(state_path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                              "python_rng": rng.getstate(), "step": step + 1, "losses": losses})
    torch.save(model.to("cpu").state_dict(), model_dir / "checkpoint.pt")
    return model.eval(), losses


def _margin_rank(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = logits.gather(1, target[:, None]).squeeze(1)
    alternatives = logits.clone(); alternatives.scatter_(1, target[:, None], float("-inf"))
    margin = selected - alternatives.max(1).values
    rank = (logits > selected[:, None]).sum(1) + 1
    return margin, rank, logits.argmax(1).eq(target)


@torch.no_grad()
def evaluate_model(model: TinyTransformerLM, rows: list[dict], meta: dict,
                   device: torch.device, batch_size: int = 64) -> tuple[list[dict], list[dict]]:
    model = model.to(device).eval(); final: list[dict] = []; trajectory: list[dict] = []
    fields = ("axis", "cell_id", "predictive_order", "raw_length", "nuisance_count",
              "dependency_span", "requested_dependency_span", "generator_family", "family_id")
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        x = torch.tensor([r["tokens"] for r in batch], device=device)
        y = torch.tensor([r["target"] for r in batch], device=device)
        logits, trace = model(x, capture=True)
        final_margin, final_rank, correct = _margin_rank(logits[:, -1], y)
        states = [(0, "embedding", trace.layers[0].pre_sa)]
        for layer, layer_trace in enumerate(trace.layers, 1):
            states.extend(((layer, "post_sa", layer_trace.post_sa), (layer, "post_ff", layer_trace.post_block)))
        diagnostics = []
        for layer, location, state in states:
            margin, rank, top1 = _margin_rank(model.diagnostic_logits(state[:, -1]), y)
            diagnostics.append((layer, location, margin.cpu(), rank.cpu(), top1.cpu()))
        for index, row in enumerate(batch):
            common = {**meta, **{field: row[field] for field in fields}}
            final.append({**common, "example_index": start + index,
                          "top1_correct": int(correct[index]), "final_rank": int(final_rank[index]),
                          "final_margin": float(final_margin[index])})
            previous_margin = None
            top1_sequence = []
            for layer, location, margins, ranks, top1s in diagnostics:
                value = float(margins[index]); top1_sequence.append(bool(top1s[index]))
                trajectory.append({**common, "example_index": start + index, "layer": layer,
                                   "location": location, "target_margin": value,
                                   "target_rank": int(ranks[index]), "top1_correct": int(top1s[index]),
                                   "margin_increment": "" if previous_margin is None else value - previous_margin})
                previous_margin = value
            first = next((i for i, value in enumerate(top1_sequence) if value), None)
            stable = next((i for i in range(len(top1_sequence)) if all(top1_sequence[i:])), None)
            final[-1]["first_top1_stage"] = "" if first is None else first
            final[-1]["stable_top1_stage"] = "" if stable is None else stable
            final[-1]["first_top1_layer"] = "" if first is None else states[first][0]
            final[-1]["first_top1_location"] = "" if first is None else states[first][1]
            final[-1]["stable_top1_layer"] = "" if stable is None else states[stable][0]
            final[-1]["stable_top1_location"] = "" if stable is None else states[stable][1]
    return final, trajectory


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def consolidate(output: Path) -> tuple[list[dict], list[dict], list[dict]]:
    final: list[dict] = []; trajectory: list[dict] = []; loss: list[dict] = []
    for manifest_path in sorted((output / "models").glob("*/complete.json")):
        model_dir = manifest_path.parent; meta = json.loads(manifest_path.read_text())
        final.extend(_read_csv(model_dir / "final.csv")); trajectory.extend(_read_csv(model_dir / "trajectory.csv"))
        loss.extend({**meta["model"], **row} for row in _read_csv(model_dir / "loss.csv"))
    write_csv(output / "stress_final_raw.csv", final)
    write_csv(output / "stress_trajectory_raw.csv", trajectory)
    write_csv(output / "stress_training_loss.csv", loss)
    return final, trajectory, loss


def main(args: argparse.Namespace) -> None:
    config = json.loads(Path(args.config).read_text())
    if not validate(config)["passed"]:
        raise RuntimeError("stress dataset validation failed")
    output = Path(args.output); (output / "models").mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    specs = tranche(config, args.tranche)
    if args.max_models:
        specs = specs[:args.max_models]
    eval_rows = evaluation(config, 2 if args.smoke else args.eval_examples)
    for index, spec in enumerate(specs, 1):
        depth, width, heads, budget, seed = spec
        model_id = f"L{depth}-W{width}-H{heads}-T{budget}-S{seed}"
        model_dir = output / "models" / model_id
        complete = model_dir / "complete.json"
        if args.resume and complete.exists():
            print(f"[{index}/{len(specs)}] skip complete {model_id}", flush=True); continue
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(specs)}] train {model_id} on {device}", flush=True)
        model, losses = train(config, spec, device, model_dir, args.resume, args.smoke)
        meta = {"model_depth": depth, "model_width": width, "head_count": heads,
                "training_budget": budget, "model_seed": seed,
                "training_steps": 4 if args.smoke else config["base_steps"] * budget,
                "parameter_count": transformer_parameter_count(config["vocab_size"], config["sequence_length"], width, depth)}
        final, trajectory = evaluate_model(model, eval_rows, meta, device)
        write_csv(model_dir / "final.csv", final); write_csv(model_dir / "trajectory.csv", trajectory)
        write_csv(model_dir / "loss.csv", losses)
        atomic_write_json(complete, {"schema_version": "paper05.stress.model.v1", "model": meta,
                                     "evaluation_rows": len(eval_rows),
                                     "artifact_hash": stable_hash({"final": final, "trajectory": trajectory})})
    final, trajectory, losses = consolidate(output)
    atomic_write_json(output / "stress_run_manifest.json", {
        "schema_version": "paper05.stress.run.v1", "device": str(device), "tranche": args.tranche,
        "smoke": args.smoke, "planned_models": len(specs),
        "completed_models": len(list((output / "models").glob("*/complete.json"))),
        "final_rows": len(final), "trajectory_rows": len(trajectory), "loss_rows": len(losses)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper05/stress_frontier.json")
    parser.add_argument("--output", default="docs/papers/paper0_5/results/stress_frontier/learned_v1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tranche", default="pilot", choices=("pilot", "rescue_t2", "rescue_t4"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-models", type=int)
    parser.add_argument("--eval-examples", type=int)
    main(parser.parse_args())
