"""Shared token-level tracing and causal forward pass for Copy and D4."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from cl.common.model_adapter import ModelTrace, TinyTransformerLM


@dataclass(frozen=True)
class Patch:
    layer: int
    kind: str  # q, k, v, key_mask, value_mask, head_zero
    source: torch.Tensor | None = None
    positions: tuple[int, ...] = ()
    head: int | None = None


def forward_patched(model: TinyTransformerLM, input_ids: torch.Tensor, patches: list[Patch]):
    """Forward with head/token QKV patches; unpatched execution is exact."""
    batch, length = input_ids.shape
    positions = torch.arange(length, device=input_ids.device)
    positional = (model.position_embedding(positions) if model.position_embedding is not None
                  else model.sinusoidal_positions[:length])
    state = model.token_embedding(input_ids) + positional[None]
    causal = model.causal_mask(length, input_ids.device)
    for layer_index, block in enumerate(model.blocks):
        normalized = block.norm_sa(state)
        b, t, width = normalized.shape
        heads, head_width = block.attention.num_heads, width // block.attention.num_heads
        projected = F.linear(normalized, block.attention.in_proj_weight, block.attention.in_proj_bias)
        query, key, value = [part.view(b, t, heads, head_width).transpose(1, 2)
                             for part in projected.chunk(3, -1)]
        # Projected tensors are sibling views returned by chunk; causal edits
        # require independent storage and must not mutate autograd views.
        query, key, value = query.clone(), key.clone(), value.clone()
        selected = [patch for patch in patches if patch.layer == layer_index]
        for patch in selected:
            head_slice = slice(None) if patch.head is None else slice(patch.head, patch.head + 1)
            token_positions = patch.positions or tuple(range(t))
            if patch.kind in {"q", "k", "v"}:
                tensor = {"q": query, "k": key, "v": value}[patch.kind]
                if patch.source is None:
                    raise ValueError(f"{patch.kind} patch requires source")
                source = patch.source.to(tensor)
                tensor[:, head_slice, token_positions] = source[:, head_slice, token_positions]
            elif patch.kind == "value_mask":
                value[:, head_slice, token_positions] = 0
        scores = query @ key.transpose(-2, -1) / math.sqrt(head_width)
        scores = scores + causal[None, None]
        for patch in selected:
            if patch.kind == "key_mask":
                head_slice = slice(None) if patch.head is None else slice(patch.head, patch.head + 1)
                # Mask the selected sources only for the prediction token.  A
                # global source mask can leave early causal rows with no valid
                # key and manufacture NaNs unrelated to the intervention.
                scores[:, head_slice, -1:, patch.positions] = float("-inf")
        probability = torch.softmax(scores, -1)
        head_output = probability @ value
        for patch in selected:
            if patch.kind == "head_zero":
                if patch.head is None:
                    raise ValueError("head_zero requires head")
                head_output[:, patch.head] = 0
        joined = head_output.transpose(1, 2).reshape(b, t, width)
        delta_sa = F.linear(joined, block.attention.out_proj.weight, block.attention.out_proj.bias)
        post_sa = state + delta_sa
        state = post_sa + block.ff(block.norm_ff(post_sa))
    return model.lm_head(model.final_norm(state))


def output_metrics(logits: torch.Tensor, target: int) -> dict[str, float | int]:
    competitor = torch.cat((logits[:target], logits[target + 1:])).max()
    return {"top1": int(logits.argmax() == target), "rank": int((logits > logits[target]).sum()) + 1,
            "margin": float(logits[target] - competitor), "probability": float(logits.softmax(-1)[target])}


def js_divergence(first: torch.Tensor, second: torch.Tensor) -> float:
    p, q = first.softmax(-1), second.softmax(-1)
    mean = (p + q) / 2
    return float((F.kl_div(mean.log(), p, reduction="sum") + F.kl_div(mean.log(), q, reduction="sum")) / 2)


def effective_rank(matrix: torch.Tensor) -> float:
    singular = torch.linalg.svdvals(matrix.float())
    probability = singular / singular.sum().clamp_min(1e-12)
    return float(torch.exp(-(probability * probability.clamp_min(1e-12).log()).sum()))


def qk_rows(trace: ModelTrace, roles: tuple[str, ...], example_id: int):
    rows, role_rows = [], []
    query_position = len(roles) - 1
    demo_key = roles.index("demo_key")
    demo_value = roles.index("demo_value")
    for layer, layer_trace in enumerate(trace.layers):
        for head in range(layer_trace.attention.shape[1]):
            attention = layer_trace.attention[0, head]
            scores = layer_trace.qk_scores[0, head]
            allowed = torch.tril(torch.ones_like(attention, dtype=torch.bool))
            entropy = -(attention * attention.clamp_min(1e-12).log()).sum(-1).mean()
            rows.append({"example_id": example_id, "layer": layer, "head": head,
                         "diagonal_mass": float(attention.diagonal().mean()),
                         "previous_token_mass": float(attention.diagonal(-1).mean()),
                         "query_to_repeated_key": float(attention[query_position, demo_key]),
                         "query_to_associated_value": float(attention[query_position, demo_value]),
                         "row_entropy": float(entropy), "qk_frobenius": float(scores[allowed].norm()),
                         "qk_effective_rank": effective_rank(scores)})
            for destination_role in sorted(set(roles)):
                destinations = [i for i, role in enumerate(roles) if role == destination_role]
                for source_role in sorted(set(roles)):
                    sources = [i for i, role in enumerate(roles) if role == source_role]
                    values = [attention[d, s] for d in destinations for s in sources if s <= d]
                    if values:
                        role_rows.append({"example_id": example_id, "layer": layer, "head": head,
                                          "destination_role": destination_role, "source_role": source_role,
                                          "attention_mass": float(torch.stack(values).mean())})
    return rows, role_rows


def support_rows(model: TinyTransformerLM, trace: ModelTrace, roles: tuple[str, ...], target: int, example_id: int):
    rows = []
    destination = len(roles) - 1
    unembedding = model.lm_head.weight[target]
    for layer, layer_trace in enumerate(trace.layers):
        attention, values = layer_trace.attention[0], layer_trace.values[0]
        heads, head_width = values.shape[0], values.shape[-1]
        for head in range(heads):
            weight = model.blocks[layer].attention.out_proj.weight[:, head * head_width:(head + 1) * head_width]
            for source, role in enumerate(roles):
                transported = attention[head, destination, source] * values[head, source]
                projected = F.linear(transported, weight)
                rows.append({"example_id": example_id, "layer": layer, "head": head,
                             "source_position": source, "source_role": role,
                             "target_support": float(unembedding @ projected)})
    return rows


def residual_and_neuron_rows(model: TinyTransformerLM, trace: ModelTrace, target: int, example_id: int):
    residual, neurons = [], []
    w_target = model.lm_head.weight[target]
    for layer, layer_trace in enumerate(trace.layers):
        position = -1
        for boundary, state in (("pre_sa", layer_trace.pre_sa), ("post_sa", layer_trace.post_sa),
                                ("post_ff", layer_trace.post_block)):
            logits = model.diagnostic_logits(state[:, position])[0]
            residual.append({"example_id": example_id, "layer": layer, "boundary": boundary,
                             **output_metrics(logits, target), "target_logit": float(logits[target])})
        block = model.blocks[layer]
        activation = layer_trace.ff_activations[0, position]
        for neuron in range(len(activation)):
            contribution = activation[neuron] * (w_target @ block.ff[2].weight[:, neuron])
            neurons.append({"example_id": example_id, "layer": layer, "neuron": neuron,
                            "activation": float(activation[neuron]), "target_contribution": float(contribution)})
    return residual, neurons
