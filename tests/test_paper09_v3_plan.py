import csv, hashlib, json
from pathlib import Path

from cl.common.model_adapter import TinyTransformerLM

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/paper09/learned_controller_v3_staged.json"
CELLS = ROOT / "configs/paper09/learned_controller_v3_cells.csv"
MANIFEST = ROOT / "docs/papers/paper0_9/results/learned_controller_v3/plan_manifest.json"


def parameter_count(layers, width, heads):
    model = TinyTransformerLM(76, 128, width, layers, heads, 2)
    return sum(p.numel() for p in model.parameters())


def test_v3_plan_is_bounded_staged_and_quarantines_factorial():
    cfg = json.loads(CONFIG.read_text())
    assert cfg["status"] == "plan_only_no_accelerator_results"
    assert "1620-cell" in cfg["quarantine_reason"]
    assert list(cfg["stages"]) == list("ABCDE")
    assert sum(s["max_evaluation_cells"] for s in cfg["stages"].values()) == 84
    assert cfg["stages"]["C"]["requires"]
    assert cfg["gates"]["all_three_seeds_required"] is True


def test_architecture_counts_and_parameter_match_are_exact():
    rows = list(csv.DictReader(CELLS.open()))
    assert sum(int(r["evaluation_cells"]) for r in rows) == 84
    for row in rows:
        observed = parameter_count(int(row["layers"]), int(row["width"]), int(row["heads"]))
        assert observed == int(row["parameters"])
    matched = {r["condition"]: r for r in rows if r["condition"] in {"deep_matched", "shallow_matched"}}
    assert all(abs(float(r["param_delta_vs_baseline"])) <= 0.10 for r in matched.values())


def test_dataset_and_observed_v1_hashes_are_pinned():
    manifest = json.loads(MANIFEST.read_text())
    paths = {
        "recurrence_chains.py": ROOT / "src/cl/semantic/recurrence_chains.py",
        "learned_controller_v1.json": ROOT / "configs/paper09/learned_controller_v1.json",
        "v1_manifest.json": ROOT / "docs/papers/paper0_9/results/learned_controller_v1/learned_controller_manifest.json",
        "v1_cells.csv": ROOT / "docs/papers/paper0_9/results/learned_controller_v1/learned_controller_cells.csv",
    }
    for key, path in paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["source_hashes_sha256"][key]


def test_metrics_fix_m3_semantics_and_separate_m4_termination():
    cfg = json.loads(CONFIG.read_text())
    assert "one_call_coverage" in cfg["metrics"]["M3"]
    assert "transition_accuracy" not in cfg["metrics"]["M3"]
    assert {"per_transition_accuracy", "termination_precision", "termination_recall"} <= set(cfg["metrics"]["M4"])
    assert cfg["stages"]["D"]["exact_prefix"] is True
    assert cfg["stages"]["E"]["shared_prefix_draws"] == 72000
