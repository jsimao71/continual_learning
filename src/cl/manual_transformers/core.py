"""Small deterministic Transformer core with exhaustive trace support.

Vectors multiply matrices on the left: ``Y = X @ W``.  Attention uses finite
softmax scores ``Q K^T / sqrt(d_head)`` and an optional causal mask.  LayerNorm
and dropout are intentionally absent; every layer uses identity residuals.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class AttentionWeights:
    W_Q: Array
    W_K: Array
    W_V: Array
    W_O: Array


@dataclass(frozen=True)
class FFWeights:
    W1: Array
    b1: Array
    W2: Array
    b2: Array


@dataclass(frozen=True)
class LayerWeights:
    attention: AttentionWeights
    ff: FFWeights
    use_sa: bool = True
    use_ff: bool = True


@dataclass(frozen=True)
class MicroTransformer:
    vocabulary: tuple[str, ...]
    embeddings: Array
    positional: Array
    layers: tuple[LayerWeights, ...]
    unembedding: Array
    output_labels: tuple[str, ...]
    causal: bool = True
    layernorm: str = "omitted"
    residual: str = "identity"

    def encode(self, token_ids: list[int]) -> Array:
        length = len(token_ids)
        return self.embeddings[token_ids] + self.positional[:length]

    def forward(self, token_ids: list[int]) -> tuple[Array, dict[str, Array]]:
        residual = self.encode(token_ids)
        trace: dict[str, Array] = {"embeddings": residual.copy()}
        for index, layer in enumerate(self.layers, start=1):
            attn = layer.attention
            q, k, v = residual @ attn.W_Q, residual @ attn.W_K, residual @ attn.W_V
            qk_raw = q @ k.T
            scores = qk_raw / np.sqrt(q.shape[-1])
            if self.causal:
                scores = scores.copy()
                scores[np.triu_indices(len(token_ids), 1)] = -np.inf
            shifted = scores - np.max(scores, axis=-1, keepdims=True)
            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum(axis=-1, keepdims=True)
            head = probabilities @ v
            sa_update = head @ attn.W_O if layer.use_sa else np.zeros_like(residual)
            post_sa = residual + sa_update
            ff_pre = post_sa @ layer.ff.W1 + layer.ff.b1
            ff_activation = np.maximum(ff_pre, 0.0)
            ff_update = ff_activation @ layer.ff.W2 + layer.ff.b2 if layer.use_ff else np.zeros_like(post_sa)
            residual = post_sa + ff_update
            prefix = f"layer{index}"
            trace.update({
                f"{prefix}_Q": q,
                f"{prefix}_K": k,
                f"{prefix}_QK_T_raw": qk_raw,
                f"{prefix}_scores": scores,
                f"{prefix}_probabilities": probabilities,
                f"{prefix}_V": v,
                f"{prefix}_head_output": head,
                f"{prefix}_post_sa_residual": post_sa,
                f"{prefix}_ff_preactivation": ff_pre,
                f"{prefix}_ff_activation": ff_activation,
                f"{prefix}_post_ff_residual": residual,
            })
        logits = residual @ self.unembedding
        trace["final_logits"] = logits
        return logits, trace

    def predict(self, token_ids: list[int]) -> tuple[str, float, dict[str, Array]]:
        logits, trace = self.forward(token_ids)
        final = logits[-1]
        order = np.argsort(final)
        margin = float(final[order[-1]] - final[order[-2]])
        return self.output_labels[int(order[-1])], margin, trace

    def signed_target_margin(self, trace: dict[str, Array], target: str) -> float:
        """Target logit minus the best non-target logit (negative when target loses)."""
        logits = trace["final_logits"][-1]
        target_index = self.output_labels.index(target)
        other = np.delete(logits, target_index)
        return float(logits[target_index] - other.max())


def zero_attention(d_model: int, d_head: int) -> AttentionWeights:
    return AttentionWeights(
        np.zeros((d_model, d_head)), np.zeros((d_model, d_head)),
        np.zeros((d_model, d_head)), np.zeros((d_head, d_model)),
    )


def zero_ff(d_model: int, d_ff: int | None = None) -> FFWeights:
    width = d_ff or d_model
    return FFWeights(
        np.zeros((d_model, width)), np.zeros(width),
        np.zeros((width, d_model)), np.zeros(d_model),
    )
