"""Shared one-hop transition circuit reused autoregressively by M1 witnesses."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .core import AttentionWeights, LayerWeights, MicroTransformer, zero_ff


ALPHA = 16.0


@dataclass(frozen=True)
class TransitionVocabulary:
    symbols: tuple[str, ...]
    vocabulary: tuple[str, ...]
    token: dict[str, int]

    def fact(self, left: str, right: str) -> str:
        return f"MAP_{left}_{right}"

    def state(self, symbol: str) -> str:
        return f"STATE_{symbol}"


def make_vocabulary(symbols: tuple[str, ...]) -> TransitionVocabulary:
    facts = tuple(f"MAP_{a}_{b}" for a in symbols for b in symbols)
    states = tuple(f"STATE_{a}" for a in symbols)
    vocabulary = facts + states
    return TransitionVocabulary(symbols, vocabulary, {name: i for i, name in enumerate(vocabulary)})


def build_transition_model(symbols: tuple[str, ...], topology: str, max_positions: int = 16) -> tuple[MicroTransformer, TransitionVocabulary]:
    """Build one finite-softmax key/value transition, with zero-FF controls."""
    vocab = make_vocabulary(symbols)
    n = len(symbols)
    # [fact key(n) | current-state query(n) | fact value/output(n)]
    d_model, d_head = 3 * n, n
    embeddings = np.zeros((len(vocab.vocabulary), d_model))
    for left in symbols:
        for right in symbols:
            row = embeddings[vocab.token[vocab.fact(left, right)]]
            row[symbols.index(left)] = 1.0
            row[2 * n + symbols.index(right)] = 1.0
    for symbol in symbols:
        embeddings[vocab.token[vocab.state(symbol)], n + symbols.index(symbol)] = 1.0

    W_Q = np.zeros((d_model, d_head)); W_K = np.zeros_like(W_Q); W_V = np.zeros_like(W_Q)
    W_Q[n:2*n] = ALPHA * np.sqrt(d_head) * np.eye(n)
    W_K[:n] = np.eye(n)
    W_V[2*n:3*n] = np.eye(n)
    W_O = np.zeros((d_head, d_model)); W_O[:, 2*n:3*n] = np.eye(n)
    layer = LayerWeights(
        AttentionWeights(W_Q, W_K, W_V, W_O), zero_ff(d_model),
        use_sa=topology in ("sa_only", "sa_ff"),
        use_ff=topology in ("ff_only", "sa_ff"),
    )
    unembedding = np.zeros((d_model, n)); unembedding[2*n:3*n] = np.eye(n)
    model = MicroTransformer(vocab.vocabulary, embeddings, np.zeros((max_positions, d_model)), (layer,), unembedding, symbols)
    return model, vocab


def generate(model: MicroTransformer, vocab: TransitionVocabulary, mapping: dict[str, str], start: str, stop: str, max_steps: int = 12):
    context = [vocab.fact(left, right) for left, right in mapping.items()]
    sequence = context + [vocab.state(start)]
    generated, steps = [], []
    for step in range(1, max_steps + 1):
        prediction, margin, trace = model.predict([vocab.token[token] for token in sequence])
        desired_position = context.index(vocab.fact(generated[-1] if generated else start, mapping[generated[-1] if generated else start]))
        probabilities = trace["layer1_probabilities"][-1]
        undesired = np.delete(probabilities, desired_position)
        steps.append({
            "step": step, "current": generated[-1] if generated else start,
            "target": mapping[generated[-1] if generated else start], "prediction": prediction,
            "correct": prediction == mapping[generated[-1] if generated else start],
            "logit_margin": margin, "attention_probability": float(probabilities[desired_position]),
            "attention_probability_margin": float(probabilities[desired_position] - undesired.max()),
            "sequence": " ".join(sequence), "trace": trace,
        })
        generated.append(prediction)
        if prediction == stop:
            break
        if prediction not in mapping:
            break
        sequence.append(vocab.state(prediction))
    return generated, steps
