"""Trace validation helpers shared by controlled model adapters."""

from __future__ import annotations

import torch

from .model_adapter import ModelTrace


def validate_residual_identities(trace: ModelTrace, atol: float = 1e-6) -> None:
    for layer in trace.layers:
        if not torch.allclose(layer.post_sa, layer.pre_sa + layer.delta_sa, atol=atol):
            raise AssertionError("post-SA residual identity failed")
        if not torch.allclose(layer.post_block, layer.post_sa + layer.delta_ff, atol=atol):
            raise AssertionError("post-block residual identity failed")


def final_position_trace(trace: ModelTrace) -> list[dict[str, torch.Tensor]]:
    """Detach only the final position plus its attention row for compact artifacts."""
    rows = []
    for layer in trace.layers:
        rows.append(
            {
                "pre_sa": layer.pre_sa[:, -1].detach().cpu(),
                "delta_sa": layer.delta_sa[:, -1].detach().cpu(),
                "post_sa": layer.post_sa[:, -1].detach().cpu(),
                "delta_ff": layer.delta_ff[:, -1].detach().cpu(),
                "post_block": layer.post_block[:, -1].detach().cpu(),
                "attention": layer.attention[:, :, -1, :].detach().cpu(),
            }
        )
    return rows
