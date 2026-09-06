import csv
import json
from pathlib import Path

import pytest

from cl.experiments.paper09_learned_controller_stage_b import summarize, validate_stage_a


PLAN = json.loads(Path("configs/paper09/learned_controller_v3_staged.json").read_text())


def _gate_fixture(root, *, completed=True, omit=None):
    root.mkdir(parents=True, exist_ok=True)
    (root / "stage_a_manifest.json").write_text(json.dumps({"completed": completed}))
    rows = [{"snapshot_updates": snapshot, "machine": machine, "gate_complete": 1}
            for snapshot in PLAN["stages"]["A"]["snapshot_updates"]
            for machine in PLAN["common"]["machines"]
            if (snapshot, machine) != omit]
    with (root / "stage_a_gates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)


def test_stage_b_requires_complete_stage_a_not_a_scientific_pass(tmp_path):
    _gate_fixture(tmp_path)
    assert validate_stage_a(tmp_path, PLAN)["complete_gate_cells"] == 4
    (tmp_path / "stage_a_manifest.json").write_text(json.dumps({"completed": False}))
    with pytest.raises(RuntimeError, match="partial"):
        validate_stage_a(tmp_path, PLAN)


def test_stage_b_rejects_partially_observed_stage_a_gate(tmp_path):
    _gate_fixture(tmp_path, omit=(4000, "M4"))
    with pytest.raises(RuntimeError, match="lacks complete gates"):
        validate_stage_a(tmp_path, PLAN)


def test_stage_b_partial_results_never_report_a_frontier():
    rows = [{"architecture": "deep_full", "machine": "M3", "seed": 11, "depth": 1,
             "final_correct": 1, "invalid_call": 0, "one_call_edge_coverage": 1,
             "selected_edge_valid": 1, "post_tool_answer_correct": 1}]
    _, frontiers = summarize(rows)
    assert frontiers == [{"architecture": "deep_full", "machine": "M3", "gate_complete": 0,
                          "seeds_observed": 1, "depths_observed": 1, "contiguous_frontier": ""}]
