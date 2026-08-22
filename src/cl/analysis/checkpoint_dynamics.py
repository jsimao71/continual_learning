"""Checkpoint-level summaries of mechanistic development."""

from __future__ import annotations

from collections import defaultdict
import numpy as np


def summarize_checkpoint(rows: list[dict], step: int) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["stratum"], row["component"])].append(row)
    output = []
    for (stratum, component), values in sorted(groups.items()):
        output.append(
            {
                "step": step,
                "stratum": stratum,
                "component": component,
                "mean_causal_logprob_drop": float(np.mean([value["causal_logprob_drop"] for value in values])),
                "mean_diagnostic_progress": float(np.mean([value["diagnostic_signed_progress"] for value in values])),
                "max_contribution_layer": int(max(values, key=lambda value: abs(value["causal_logprob_drop"]))["layer"]),
            }
        )
    return output
