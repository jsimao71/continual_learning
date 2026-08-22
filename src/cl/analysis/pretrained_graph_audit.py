"""Identity-disjoint residualized audit of retained frozen-Qwen graph rows."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from cl.common.metrics import spearman
from cl.nsc.features import ridge_fit, ridge_predict


CONTROL_FIELDS = ("parent_count", "annotated_hops", "graph_density", "layer")
STRUCTURE_FIELDS = (
    "duplicate_neighbor_rate",
    "reciprocal_edge_rate",
    "effective_branching_factor",
    "unique_neighbor_count",
    "shortcut_rate",
    "giant_component_fraction",
    "mean_out_degree",
)


def _number(row, key):
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def load_graph_rows(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _design(rows, fields, categories):
    numeric = [[_number(row, key) for key in fields] for row in rows]
    categorical = [
        [float(row.get("dataset") == value) for value in categories["dataset"]]
        + [float(row.get("graph_type") == value) for value in categories["graph_type"]]
        for row in rows
    ]
    return np.asarray([left + right for left, right in zip(numeric, categorical)], dtype=np.float64)


def _standardize(train, test):
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-9] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def _aggregate(rows, target, base, full):
    grouped = defaultdict(lambda: {"target": [], "base": [], "full": []})
    for row, y, b, f in zip(rows, target, base, full):
        key = (row["dataset"], row["example_id"])
        grouped[key]["target"].append(float(y))
        grouped[key]["base"].append(float(b))
        grouped[key]["full"].append(float(f))
    return [
        {
            "dataset": key[0],
            "example_id": key[1],
            "target": float(np.mean(value["target"])),
            "base_prediction": float(np.mean(value["base"])),
            "structural_prediction": float(np.mean(value["full"])),
        }
        for key, value in sorted(grouped.items())
    ]


def _metrics(rows, prediction_key):
    target = np.asarray([row["target"] for row in rows])
    prediction = np.asarray([row[prediction_key] for row in rows])
    denominator = np.square(target - target.mean()).sum()
    return {
        "rmse": float(np.sqrt(np.mean(np.square(target - prediction)))),
        "r2": float(1.0 - np.square(target - prediction).sum() / max(denominator, 1e-12)),
        "spearman": spearman(target, prediction),
    }


def residualized_pretrained_audit(rows: list[dict]):
    train = [row for row in rows if row["partition"] == "validation"]
    test = [row for row in rows if row["partition"] == "test"]
    categories = {
        key: sorted({row[key] for row in train})[:-1]
        for key in ("dataset", "graph_type")
    }
    train_control = _design(train, CONTROL_FIELDS, categories)
    test_control = _design(test, CONTROL_FIELDS, categories)
    train_structure = np.asarray([[_number(row, key) for key in STRUCTURE_FIELDS] for row in train])
    test_structure = np.asarray([[_number(row, key) for key in STRUCTURE_FIELDS] for row in test])
    train_control, test_control = _standardize(train_control, test_control)
    train_structure, test_structure = _standardize(train_structure, test_structure)
    train_target = np.asarray([_number(row, "complete_recovery") for row in train])
    test_target = np.asarray([_number(row, "complete_recovery") for row in test])
    base_fit = ridge_fit(train_control, train_target, alpha=4.0)
    full_fit = ridge_fit(np.column_stack([train_control, train_structure]), train_target, alpha=4.0)
    base = ridge_predict(test_control, base_fit)
    full = ridge_predict(np.column_stack([test_control, test_structure]), full_fit)
    aggregated = _aggregate(test, test_target, base, full)
    base_metrics = _metrics(aggregated, "base_prediction")
    full_metrics = _metrics(aggregated, "structural_prediction")
    summary = {
        "model_id": "Qwen/Qwen3-0.6B",
        "validation_examples": len({row["example_id"] for row in train}),
        "test_examples": len({row["example_id"] for row in test}),
        "scientific_unit": "example identity after averaging eight layer rows",
        "target": "fraction of layers with complete native graph recovery",
        "controls": list(CONTROL_FIELDS) + ["dataset", "graph_type"],
        "structure": list(STRUCTURE_FIELDS),
        "base": base_metrics,
        "base_plus_structure": full_metrics,
        "delta_r2": full_metrics["r2"] - base_metrics["r2"],
        "delta_spearman": full_metrics["spearman"] - base_metrics["spearman"],
        "causal": False,
    }
    return summary, aggregated
