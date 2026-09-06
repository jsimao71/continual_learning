import csv
import json
from pathlib import Path

import pytest

from cl.experiments.paper09_learned_controller_stage_c import validate_stage_b


PLAN = json.loads(Path("configs/paper09/learned_controller_v3_staged.json").read_text())


def _fixture(root, frontiers, completed=True):
    root.mkdir(parents=True, exist_ok=True)
    (root / "stage_b_manifest.json").write_text(json.dumps({"completed": completed}))
    rows = [{"architecture": architecture, "machine": machine, "gate_complete": 1,
             "contiguous_frontier": frontiers[machine][architecture]}
            for architecture in PLAN["stages"]["B"]["architectures"]
            for machine in PLAN["common"]["machines"]]
    with (root / "stage_b_frontiers.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)


def test_tied_stage_b_deterministically_enables_stage_c(tmp_path):
    values = {machine: {architecture: 1 for architecture in PLAN["stages"]["B"]["architectures"]}
              for machine in PLAN["common"]["machines"]}
    _fixture(tmp_path, values)
    decision = validate_stage_b(tmp_path, PLAN)
    assert decision["allocation_unresolved"] is True and decision["enabled"] is True


def test_shared_unique_winner_requires_explicit_control_rationale(tmp_path):
    values = {machine: {"deep_full": 8, "deep_matched": 3, "shallow_matched": 3}
              for machine in PLAN["common"]["machines"]}
    _fixture(tmp_path, values)
    with pytest.raises(RuntimeError, match="disabled"):
        validate_stage_b(tmp_path, PLAN)
    with pytest.raises(RuntimeError, match="rationale"):
        validate_stage_b(tmp_path, PLAN, control_required=True)
    decision = validate_stage_b(tmp_path, PLAN, control_required=True,
                                control_rationale="head-count control required for attribution")
    assert decision["control_required"] is True and decision["enabled"] is True


def test_partial_stage_b_cannot_enable_stage_c(tmp_path):
    values = {machine: {architecture: 1 for architecture in PLAN["stages"]["B"]["architectures"]}
              for machine in PLAN["common"]["machines"]}
    _fixture(tmp_path, values, completed=False)
    with pytest.raises(RuntimeError, match="partial"):
        validate_stage_b(tmp_path, PLAN)
