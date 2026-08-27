"""Milestone-1 FF-only successor witness and architectural controls."""
from __future__ import annotations

import numpy as np

from .core import FFWeights, LayerWeights, MicroTransformer, zero_attention


VOCAB = ("A", "B", "C", "D")
TOKEN = {token: index for index, token in enumerate(VOCAB)}
LEGAL = (("A", "B"), ("B", "C"), ("C", "D"))


def _model(topology: str) -> MicroTransformer:
    d_model = 4
    embeddings = np.eye(d_model)
    mapping = np.zeros((d_model, d_model))
    for source, target in LEGAL:
        mapping[TOKEN[source], TOKEN[target]] = 1.0
    mapping[TOKEN["D"], TOKEN["D"]] = 1.0
    ff = FFWeights(np.eye(d_model), np.zeros(d_model), mapping - np.eye(d_model), np.zeros(d_model))
    layer = LayerWeights(
        zero_attention(d_model, d_model), ff,
        use_sa=topology in ("sa_only", "sa_ff"),
        use_ff=topology in ("ff_only", "sa_ff"),
    )
    return MicroTransformer(VOCAB, embeddings, np.zeros((1, d_model)), (layer,), np.eye(d_model), VOCAB)


def build_successor_variants() -> dict[str, MicroTransformer]:
    return {name: _model(name) for name in ("sa_only", "ff_only", "sa_ff")}


def evaluate(model: MicroTransformer) -> list[dict]:
    rows = []
    for source, target in LEGAL:
        prediction, margin, _ = model.predict([TOKEN[source]])
        rows.append({"input": source, "target": target, "prediction": prediction, "correct": prediction == target, "logit_margin": margin})
    return rows
