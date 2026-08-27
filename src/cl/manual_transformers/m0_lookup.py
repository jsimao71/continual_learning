"""One-layer finite-softmax associative pair lookup witness."""
from __future__ import annotations

from itertools import permutations
import numpy as np

from .core import AttentionWeights, LayerWeights, MicroTransformer, zero_ff


SYMBOLS = ("A", "B", "C")
PAIR_TOKENS = tuple(f"PAIR_{left}_{right}" for left in SYMBOLS for right in SYMBOLS)
QUERY_TOKENS = tuple(f"QUERY_{symbol}" for symbol in SYMBOLS)
VOCAB = PAIR_TOKENS + QUERY_TOKENS
TOKEN = {token: index for index, token in enumerate(VOCAB)}
ALPHA = 16.0


def _model(topology: str) -> MicroTransformer:
    # [pair-key(3) | query(3) | transported/output value(3)]
    d_model, d_head = 9, 3
    embeddings = np.zeros((len(VOCAB), d_model))
    for left in SYMBOLS:
        for right in SYMBOLS:
            row = embeddings[TOKEN[f"PAIR_{left}_{right}"]]
            row[SYMBOLS.index(left)] = 1.0
            row[6 + SYMBOLS.index(right)] = 1.0
    for query in SYMBOLS:
        embeddings[TOKEN[f"QUERY_{query}"], 3 + SYMBOLS.index(query)] = 1.0
    W_Q = np.zeros((d_model, d_head)); W_K = np.zeros_like(W_Q); W_V = np.zeros_like(W_Q)
    W_Q[3:6] = ALPHA * np.sqrt(d_head) * np.eye(d_head)
    W_K[:3] = np.eye(d_head)
    W_V[6:9] = np.eye(d_head)
    W_O = np.zeros((d_head, d_model)); W_O[:, 6:9] = np.eye(d_head)
    layer = LayerWeights(
        AttentionWeights(W_Q, W_K, W_V, W_O), zero_ff(d_model),
        use_sa=topology in ("sa_only", "sa_ff"), use_ff=topology in ("ff_only", "sa_ff"),
    )
    unembedding = np.zeros((d_model, len(SYMBOLS))); unembedding[6:9] = np.eye(3)
    return MicroTransformer(VOCAB, embeddings, np.zeros((4, d_model)), (layer,), unembedding, SYMBOLS)


def build_lookup_variants() -> dict[str, MicroTransformer]:
    return {name: _model(name) for name in ("sa_only", "ff_only", "sa_ff")}


def legal_cases():
    for outputs in permutations(SYMBOLS):
        mapping = dict(zip(SYMBOLS, outputs))
        context = [f"PAIR_{left}_{mapping[left]}" for left in SYMBOLS]
        for query in SYMBOLS:
            yield context + [f"QUERY_{query}"], mapping[query]


def evaluate(model: MicroTransformer) -> list[dict]:
    rows = []
    for tokens, target in legal_cases():
        prediction, margin, trace = model.predict([TOKEN[token] for token in tokens])
        probabilities = trace["layer1_probabilities"][-1]
        desired = tokens.index(f"PAIR_{tokens[-1][-1]}_{target}")
        undesired = np.delete(probabilities, desired)
        rows.append({
            "sequence": " ".join(tokens), "target": target, "prediction": prediction,
            "correct": prediction == target, "attention_probability": float(probabilities[desired]),
            "attention_probability_margin": float(probabilities[desired] - undesired.max()),
            "logit_margin": margin,
        })
    return rows
