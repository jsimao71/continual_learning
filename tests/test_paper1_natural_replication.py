import csv
import json
from pathlib import Path

from cl.experiments.paper1_natural_replication import (
    REQUIRED_TABLES,
    aggregate,
    seed_command,
    validate_seed_output,
)


CONFIG = json.loads(Path("configs/paper1/natural_replication_v2.json").read_text())


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)


def _seed_fixture(root, seed):
    seed_dir = root / "seeds" / f"seed-{seed}"
    config = {
        "seed": seed, "per_split": CONFIG["per_split"], "candidates": 12,
        "replacement_pool_multiplier": CONFIG["replacement_pool_multiplier"],
        "budgets": [2, 4, 6], "chunk_tokens": 32,
        "causal_examples_per_split": CONFIG["causal_examples_per_split"],
        "model_id": CONFIG["model_id"], "model_revision": CONFIG["model_revision"],
        "selectors": CONFIG["selectors"], "retain_full_attention": False, "device": "cpu",
    }
    seed_dir.mkdir(parents=True)
    (seed_dir / "manifest.json").write_text(json.dumps({"config": config}))
    selector = []
    for dataset in ("hotpotqa", "qasper"):
        for identity in range(CONFIG["per_split"]):
            for budget in (64, 128, 192):
                for condition in CONFIG["selectors"]:
                    selector.append({
                        "dataset": dataset, "identity_id": f"{dataset}-{identity}",
                        "budget_tokens": budget, "condition": condition,
                        "answer_logprob": 1.0 if condition == "bridge_preserving" else 0.0,
                    })
    removal = [{"candidate_id": index} for index in range(
        2 * 2 * CONFIG["causal_examples_per_split"] * CONFIG["candidates"]
    )]
    prediction = [{"dataset": dataset, "model": model, "r2": 0.0, "rmse": 1.0, "spearman": 0.0}
                  for dataset in ("hotpotqa", "qasper")
                  for model in ("surface_controls", "surface_plus_structure")]
    paired = [{"dataset": dataset, "budget_tokens": budget, "condition": condition,
               "mean_answer_logprob_delta": 1.0, "ci_low": 0.5, "ci_high": 1.5}
              for dataset in ("hotpotqa", "qasper") for budget in (64, 128, 192)
              for condition in ("bridge_preserving", "combined_structural")]
    _write_csv(seed_dir / "tables" / "selector_by_example.csv", selector)
    _write_csv(seed_dir / "tables" / "candidate_removal.csv", removal)
    _write_csv(seed_dir / "tables" / "causal_utility_prediction.csv", prediction)
    _write_csv(seed_dir / "tables" / "paired_frontier_deltas.csv", paired)
    for name in set(REQUIRED_TABLES) - {"selector_by_example.csv", "candidate_removal.csv",
                                        "causal_utility_prediction.csv", "paired_frontier_deltas.csv"}:
        _write_csv(seed_dir / "tables" / name, [{"ok": 1}])
    return seed_dir


def test_frozen_command_preserves_protocol(tmp_path):
    command = seed_command(Path("config.json"), CONFIG, tmp_path / "out", tmp_path / "cache",
                           CONFIG["sampling_seeds"][0], "mps")
    joined = " ".join(command)
    assert "--per-split 24" in joined and "--candidates 12" in joined
    assert "--replacement-pool-multiplier 2" in joined
    assert "--budgets 2 4 6" in joined and "--causal-examples 6" in joined


def test_seed_validation_rejects_protocol_change(tmp_path):
    seed = CONFIG["sampling_seeds"][0]
    seed_dir = _seed_fixture(tmp_path, seed)
    assert validate_seed_output(seed_dir, CONFIG, seed)["device"] == "cpu"
    manifest = json.loads((seed_dir / "manifest.json").read_text())
    manifest["config"]["candidates"] = 11
    (seed_dir / "manifest.json").write_text(json.dumps(manifest))
    try:
        validate_seed_output(seed_dir, CONFIG, seed)
    except ValueError as error:
        assert "frozen protocol" in str(error)
    else:
        raise AssertionError("protocol drift was accepted")


def test_aggregate_deduplicates_identities_across_sampling_seeds(tmp_path):
    for seed in CONFIG["sampling_seeds"]:
        seed_dir = _seed_fixture(tmp_path, seed)
        audit = validate_seed_output(seed_dir, CONFIG, seed)
        (seed_dir / "replication_complete.json").write_text(json.dumps(audit))
    manifest = aggregate(tmp_path, CONFIG)
    assert manifest["completed_seeds"] == 3
    rows = list(csv.DictReader((tmp_path / "tables" / "paired_frontier_unique_identities.csv").open()))
    assert {int(row["n_unique_test_identities"]) for row in rows} == {24}
    assert {float(row["mean_answer_logprob_delta"]) for row in rows} == {1.0}
    decision = json.loads((tmp_path / "gate_decision.json").read_text())
    assert decision["frontier_pass"] and decision["persistent_learning_gate"] == 1
    assert not decision["causal_prediction_pass"]
