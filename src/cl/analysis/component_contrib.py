"""Signed diagnostic and causal SA/FFN contribution measurements."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cl.common.hooks import final_position_trace
from cl.common.metrics import residual_geometry
from cl.common.model_adapter import Intervention, TinyTransformerLM


@torch.no_grad()
def measure_components(
    model: TinyTransformerLM,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    example_ids: list[str],
    strata: list[str],
    relation_ids: list[str],
    control_ids: torch.Tensor | None = None,
) -> list[dict]:
    model.eval()
    logits, raw_trace = model(input_ids, capture=True)
    if raw_trace is None:
        raise RuntimeError("capture did not produce a trace")
    compact = final_position_trace(raw_trace)
    control_compact = None
    if control_ids is not None:
        _, control_trace = model(control_ids, capture=True)
        if control_trace is None:
            raise RuntimeError("control capture did not produce a trace")
        control_compact = final_position_trace(control_trace)
    intact_logprob = F.log_softmax(logits[:, -1], dim=-1).gather(1, targets[:, None]).squeeze(1)
    rows = []
    for layer_index, layer in enumerate(compact):
        states = [layer["pre_sa"], layer["post_sa"], layer["post_block"]]
        diagnostic = [
            F.log_softmax(model.diagnostic_logits(state.to(next(model.parameters()).device)), dim=-1)
            .gather(1, targets[:, None])
            .squeeze(1)
            .cpu()
            for state in states
        ]
        for component in ("sa", "ff"):
            zero_logprob = model.target_logprob(
                input_ids,
                targets,
                Intervention(layer=layer_index, component=component, mode="zero"),
            )
            mean_logprob = model.target_logprob(
                input_ids,
                targets,
                Intervention(layer=layer_index, component=component, mode="mean"),
            )
            replacement_drop = torch.full_like(intact_logprob, float("nan"))
            if control_compact is not None:
                replacement_name = "delta_sa" if component == "sa" else "delta_ff"
                replacement_logprob = model.target_logprob(
                    input_ids,
                    targets,
                    Intervention(
                        layer=layer_index,
                        component=component,
                        mode="replace",
                        replacement=control_compact[layer_index][replacement_name].to(input_ids.device)[:, None, :],
                    ),
                )
                replacement_drop = intact_logprob - replacement_logprob
            head_drop = torch.full_like(intact_logprob, float("nan"))
            if component == "sa":
                head_logprob = model.target_logprob(
                    input_ids,
                    targets,
                    Intervention(layer=layer_index, component="sa", mode="head_zero", head=0),
                )
                head_drop = intact_logprob - head_logprob
            zero_drop = (intact_logprob - zero_logprob).cpu()
            mean_drop = (intact_logprob - mean_logprob).cpu()
            update = layer["delta_sa"] if component == "sa" else layer["delta_ff"]
            base = layer["pre_sa"] if component == "sa" else layer["post_sa"]
            before = diagnostic[0] if component == "sa" else diagnostic[1]
            after = diagnostic[1] if component == "sa" else diagnostic[2]
            for index, example_id in enumerate(example_ids):
                geometry = residual_geometry(base[index].numpy(), update[index].numpy())
                rows.append(
                    {
                        "example_id": example_id,
                        "stratum": strata[index],
                        "relation_id": relation_ids[index],
                        "layer": layer_index,
                        "component": component,
                        "target": int(targets[index]),
                        "target_logprob": float(intact_logprob[index]),
                        "diagnostic_signed_progress": float(after[index] - before[index]),
                        "causal_logprob_drop": float(zero_drop[index]),
                        "zero_ablation_logprob_drop": float(zero_drop[index]),
                        "mean_ablation_logprob_drop": float(mean_drop[index]),
                        "matched_replacement_logprob_drop": float(replacement_drop[index]),
                        "head0_ablation_logprob_drop": float(head_drop[index]),
                        **geometry,
                    }
                )
    return rows


@torch.no_grad()
def probability_trajectory(model: TinyTransformerLM, input_ids: torch.Tensor, targets: torch.Tensor) -> list[list[float]]:
    _, trace = model(input_ids, capture=True)
    if trace is None:
        raise RuntimeError("capture did not produce a trace")
    positions = final_position_trace(trace)
    output = []
    for state_name in ("pre_sa", "post_sa", "post_block"):
        values = []
        for layer in positions:
            log_probs = F.log_softmax(model.diagnostic_logits(layer[state_name].to(next(model.parameters()).device)), dim=-1)
            values.append(log_probs.gather(1, targets[:, None]).exp().cpu().tolist())
        output.append(values)
    return output
