"""Two-layer SA-only grandparent witness on a complete tiny tree domain."""
from __future__ import annotations

import numpy as np

from .core import AttentionWeights, LayerWeights, MicroTransformer, zero_ff


NODES = ("R", "A", "B", "C")
PARENT = {"A": "R", "B": "A", "C": "B"}
EDGE_TOKENS = tuple(f"EDGE_{child}_{parent}" for child, parent in PARENT.items())
QUERY_TOKENS = tuple(f"QUERY_{node}" for node in ("B", "C"))
VOCAB = EDGE_TOKENS + QUERY_TOKENS
TOKEN = {token: index for index, token in enumerate(VOCAB)}
ALPHA = 16.0


def _attention(d_model: int, query_slice: slice, output_slice: slice) -> AttentionWeights:
    d_head = len(NODES)
    W_Q = np.zeros((d_model, d_head)); W_K = np.zeros_like(W_Q); W_V = np.zeros_like(W_Q)
    W_Q[query_slice] = ALPHA * np.sqrt(d_head) * np.eye(d_head)
    W_K[0:4] = np.eye(d_head)
    W_V[8:12] = np.eye(d_head)
    W_O = np.zeros((d_head, d_model)); W_O[:, output_slice] = np.eye(d_head)
    return AttentionWeights(W_Q, W_K, W_V, W_O)


def _model(topology: str, layers: int = 2) -> MicroTransformer:
    # [edge child key(4) | initial query(4) | edge parent value(4) |
    #  first-hop state(4) | second-hop output(4)]
    d_model = 20
    embeddings = np.zeros((len(VOCAB), d_model))
    for child, parent in PARENT.items():
        row = embeddings[TOKEN[f"EDGE_{child}_{parent}"]]
        row[NODES.index(child)] = 1.0
        row[8 + NODES.index(parent)] = 1.0
    for node in ("B", "C"):
        embeddings[TOKEN[f"QUERY_{node}"], 4 + NODES.index(node)] = 1.0
    use_sa = topology in ("sa_only", "sa_ff", "sa_only_1layer")
    use_ff = topology in ("ff_only", "sa_ff")
    layer1 = LayerWeights(_attention(d_model, slice(4, 8), slice(12, 16)), zero_ff(d_model), use_sa, use_ff)
    layer2 = LayerWeights(_attention(d_model, slice(12, 16), slice(16, 20)), zero_ff(d_model), use_sa, use_ff)
    active_layers = (layer1,) if layers == 1 else (layer1, layer2)
    unembedding = np.zeros((d_model, len(NODES))); unembedding[16:20] = np.eye(4)
    return MicroTransformer(VOCAB, embeddings, np.zeros((4, d_model)), active_layers, unembedding, NODES)


def build_grandparent_variants() -> dict[str, MicroTransformer]:
    return {
        "sa_only": _model("sa_only"),
        "ff_only": _model("ff_only"),
        "sa_ff": _model("sa_ff"),
        "sa_only_1layer": _model("sa_only_1layer", layers=1),
    }


def legal_cases():
    context = list(EDGE_TOKENS)
    for query in ("B", "C"):
        yield context + [f"QUERY_{query}"], PARENT[PARENT[query]]


def evaluate(model: MicroTransformer) -> list[dict]:
    rows = []
    for tokens, target in legal_cases():
        prediction, margin, trace = model.predict([TOKEN[token] for token in tokens])
        layer1_p = trace["layer1_probabilities"][-1]
        first_parent = PARENT[tokens[-1].split("_")[-1]]
        first_edge = tokens.index(f"EDGE_{tokens[-1].split('_')[-1]}_{first_parent}")
        layer2_p = trace.get("layer2_probabilities")
        second_probability = float("nan")
        if layer2_p is not None:
            second_edge = tokens.index(f"EDGE_{first_parent}_{target}")
            second_probability = float(layer2_p[-1, second_edge])
        rows.append({
            "sequence": " ".join(tokens), "target": target, "prediction": prediction,
            "correct": prediction == target, "layer1_parent_probability": float(layer1_p[first_edge]),
            "layer2_grandparent_probability": second_probability, "winner_runner_up_margin": margin,
            "signed_target_margin": model.signed_target_margin(trace, target), "logit_margin": margin,
        })
    return rows
