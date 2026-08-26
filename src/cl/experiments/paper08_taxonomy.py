"""CPU-only validation artifact for the Paper 0.8 task taxonomy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cl.common.artifacts import atomic_write_json, write_csv
from cl.semantic.paper08_taxonomy import dataset_taxonomy, episode_record, hypotheses, make_identifiable_episode


def main(args: argparse.Namespace) -> None:
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.output or cfg["output"])
    rows = []
    count = 2 if args.smoke else cfg["episodes_per_cell"]
    seed = cfg["seed"]
    for modulus in cfg["moduli"]:
        for family in cfg["families"]:
            prior = hypotheses(family, modulus)
            for index in range(count):
                chosen = prior[(seed + index) % len(prior)]
                row = make_identifiable_episode(family, modulus, chosen.parameters, seed + 1000 * modulus + index)
                rows.append(episode_record(row))
    write_csv(out / "dataset_taxonomy.csv", dataset_taxonomy())
    write_csv(out / "d5_identifiability_validation.csv", rows)
    atomic_write_json(out / "taxonomy_manifest.json", {
        "device": "cpu/no-model",
        "taxonomy_rows": len(dataset_taxonomy()),
        "d5_episodes": len(rows),
        "all_d5_admissible": all(row["admissible"] for row in rows),
        "maximum_answer_entropy_bits": max(row["answer_entropy_bits"] for row in rows),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper08/taxonomy_v1.json")
    parser.add_argument("--output")
    parser.add_argument("--smoke", action="store_true")
    main(parser.parse_args())
