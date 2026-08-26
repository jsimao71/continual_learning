"""Aggressive one-axis-at-a-time minimality search for contextual copying."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cl.common.artifacts import write_csv
from cl.experiments.paper05_predictive_order_phase import resolve_device
from cl.experiments.paper08_copy_phase import run


def settings(seed: int):
    reference = {"regime": "C2", "vocabulary": 4, "episodes": 256, "mappings": "fresh-random",
                 "pairs": 3, "family": "sa_ff", "layers": 2, "width": 32, "heads": 2,
                 "mlp_ratio": 2, "seed": seed}
    candidates = [("vocabulary", 3), ("mappings", 2), ("episodes", 128), ("episodes", 64),
                  ("pairs", 2), ("pairs", 1), ("width", 16), ("width", 8), ("heads", 1),
                  ("layers", 1), ("family", "sa_only")]
    rows = [{**reference, "minimality_axis": "reference", "candidate": "reference"}]
    for axis, value in candidates:
        cell = {**reference, axis: value, "minimality_axis": axis, "candidate": value}
        if axis == "vocabulary":
            cell["pairs"] = min(cell["pairs"], value)
        rows.append(cell)
    return rows


def main(args):
    cfg = json.loads(Path(args.config).read_text())
    seeds = cfg["model_seeds"][:1] if args.smoke else cfg["model_seeds"]
    requested = [cell for seed in seeds for cell in settings(seed)]
    if args.limit:
        requested = requested[:args.limit]
    output = Path(args.output)
    cells = run(requested, cfg, output / "minimality_runs", resolve_device(args.device or cfg["device"]),
                30 if args.smoke else cfg["updates"])
    write_csv(output / "copy_minimality.csv", cells)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper08/copy_v1.json")
    parser.add_argument("--output", default="docs/papers/paper0_8/results/copy")
    parser.add_argument("--device")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int)
    main(parser.parse_args())
