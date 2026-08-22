"""Aligned attention motif representations and matched controls."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from cl.common.metrics import cosine


def motif_vector(attention) -> np.ndarray:
    array = np.asarray(attention, dtype=np.float64)
    row_sums = array.sum(axis=-1, keepdims=True)
    normalized = array / np.clip(row_sums, 1e-12, None)
    return normalized.reshape(-1)


def motif_stability(records: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    controls: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    for record in records:
        key = (record["relation_id"], int(record["layer"]))
        target = controls if record.get("control", False) else groups
        target[key].append(motif_vector(record["attention"]))
    rows = []
    for key, motifs in sorted(groups.items()):
        if len(motifs) < 2:
            continue
        within = [cosine(motifs[i], motifs[j]) for i in range(len(motifs)) for j in range(i + 1, len(motifs))]
        matched = controls.get(key, [])
        cross = [cosine(motif, control) for motif in motifs for control in matched]
        rows.append(
            {
                "relation_id": key[0],
                "layer": key[1],
                "within_motif_cosine": float(np.mean(within)),
                "matched_control_cosine": float(np.mean(cross)) if cross else float("nan"),
                "motif_specificity": float(np.mean(within) - np.mean(cross)) if cross else float("nan"),
                "n_occurrences": len(motifs),
            }
        )
    return rows

