"""Artifact writers for fixed-weight witness matrices and canonical traces."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable
import numpy as np

from .core import MicroTransformer


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows or [{"status": "empty"}])


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.atleast_2d(matrix), delimiter=",", fmt="%.12g")


def write_model(root: Path, model: MicroTransformer) -> None:
    write_matrix(root / "embedding_matrix.csv", model.embeddings)
    write_matrix(root / "positional_matrix.csv", model.positional)
    write_matrix(root / "unembedding_matrix.csv", model.unembedding)
    for layer_index, layer in enumerate(model.layers, start=1):
        attn, ff = layer.attention, layer.ff
        write_matrix(root / f"W_Q_layer{layer_index}_head1.csv", attn.W_Q)
        write_matrix(root / f"W_K_layer{layer_index}_head1.csv", attn.W_K)
        write_matrix(root / f"W_V_layer{layer_index}_head1.csv", attn.W_V)
        write_matrix(root / f"W_O_layer{layer_index}_head1.csv", attn.W_O)
        write_matrix(root / f"W1_ff_layer{layer_index}.csv", ff.W1)
        write_matrix(root / f"b1_ff_layer{layer_index}.csv", ff.b1)
        write_matrix(root / f"W2_ff_layer{layer_index}.csv", ff.W2)
        write_matrix(root / f"b2_ff_layer{layer_index}.csv", ff.b2)


def write_trace(root: Path, trace: dict[str, np.ndarray]) -> None:
    trace_root = root / "canonical_trace"
    for name, value in trace.items():
        write_matrix(trace_root / f"{name}.csv", value)
