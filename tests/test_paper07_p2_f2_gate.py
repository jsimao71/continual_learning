import csv
import json
from pathlib import Path

from cl.experiments.paper07_p2_f2_gate import aggregate, build
from cl.semantic.paper07_gates import NO, SYMBOL, YES
from cl.semantic.paper07_p1_f1 import SYMBOL_COUNT
from cl.semantic.paper07_p2_f2 import FAIL, f2_examples, p2_examples, validate_next_gates


CONFIG = json.loads(Path("configs/paper07/p2_f2_gate_v1.json").read_text())


def test_p2_f2_smoke_dataset_is_valid():
    rows = build(CONFIG, smoke=True)
    audit = validate_next_gates(rows)
    assert audit["valid"] and audit["rows"] == 2 * 3 * 96


def test_p2_separates_operations_and_balances_each_control():
    rows = p2_examples("test", 96, 17)
    assert {row.label.split(":")[0] for row in rows} == {"modus_tollens", "contraposition"}
    for template in {row.template for row in rows}:
        subset = [row for row in rows if row.template == template]
        assert sum(row.target == YES for row in subset) == sum(row.target == NO for row in subset)
        assert {len(row.tokens) for row in subset} == {24}


def test_f2_has_exact_balanced_bindings_and_explicit_failures():
    rows = f2_examples("validation", 192, 19)
    for template in {row.template for row in rows}:
        subset = [row for row in rows if row.template == template]
        assert sum(row.target == FAIL for row in subset) * 2 == len(subset)
        successes = [row for row in subset if row.target != FAIL]
        assert {row.target for row in successes} == set(range(SYMBOL, SYMBOL + SYMBOL_COUNT))
        assert all(row.oracle_substitution and not row.failure_reason for row in successes)
        assert all(row.failure_reason and not row.oracle_substitution for row in subset if row.target == FAIL)


def test_template_gate_exposes_failure():
    raw = []
    for template, correct in (("canonical", 1), ("reversed", 0)):
        for _ in range(4):
            raw.append({"stage": "F2", "model_seed": 11, "split": "test",
                        "template": template, "correct": correct,
                        "target_margin": 1 if correct else -1})
    rows = aggregate(raw, 0.95)
    assert {row["template"]: row["competent"] for row in rows} == {
        "canonical": 1, "reversed": 0,
    }


def test_resume_preserves_runtime(tmp_path):
    from cl.experiments import paper07_p2_f2_gate as runner
    args = type("Args", (), {
        "config": "configs/paper07/p2_f2_gate_v1.json", "output": str(tmp_path),
        "device": "cpu", "smoke": True, "resume": False, "max_models": 1,
        "only_stage": "P2",
    })()
    runner.main(args)
    args.resume = True
    runner.main(args)
    with (tmp_path / "gate_runtime.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1 and rows[0]["stage"] == "P2"
    manifest = json.loads((tmp_path / "gate_manifest.json").read_text())
    assert manifest["dataset_hash"] and manifest["config_hash"]
