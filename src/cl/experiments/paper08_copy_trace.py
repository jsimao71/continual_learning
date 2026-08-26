"""Trace and causally validate a competent Paper 0.8 copy checkpoint."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from cl.common.artifacts import atomic_write_json, write_csv
from cl.common.model_adapter import TinyTransformerLM
from cl.experiments.paper05_predictive_order_phase import resolve_device
from cl.experiments.paper08_trace import (Patch, forward_patched, js_divergence, output_metrics,
                                           qk_rows, residual_and_neuron_rows, support_rows)
from cl.semantic.paper08_copy import controlled_copy, generate_copy_examples


def load_model(checkpoint: Path, device):
    model = TinyTransformerLM(7, 11, width=64, layers=2, heads=2, mlp_ratio=2).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return model.eval()


def mean_rows(rows, keys, values):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for group, members in sorted(grouped.items()):
        result = dict(zip(keys, group))
        result.update({value: float(np.mean([member[value] for member in members])) for value in values})
        result["examples"] = len(members)
        output.append(result)
    return output


@torch.no_grad()
def run(model, output: Path, examples: int, seed: int, checkpoint: str = ""):
    output.mkdir(parents=True, exist_ok=True)
    rows = generate_copy_examples("C2", 4, max(48, examples), pairs_per_prompt=3, seed=seed)[:examples]
    qk, role_mass, support, residual, neurons, causal = [], [], [], [], [], []
    matrices = {}
    for example_id, row in enumerate(rows):
        tokens = torch.tensor([row.tokens], device=next(model.parameters()).device)
        control_row = controlled_copy(row, "shuffled_pairings", seed + example_id)
        control_tokens = torch.tensor([control_row.tokens], device=tokens.device)
        logits, trace = model(tokens, capture=True)
        control_logits, control_trace = model(control_tokens, capture=True)
        base = logits[0, -1]
        base_metrics = output_metrics(base, row.target)
        these_qk, these_roles = qk_rows(trace, row.roles, example_id)
        qk.extend(these_qk); role_mass.extend(these_roles)
        support.extend(support_rows(model, trace, row.roles, row.target, example_id))
        residual_rows, neuron_rows = residual_and_neuron_rows(model, trace, row.target, example_id)
        residual.extend(residual_rows); neurons.extend(neuron_rows)
        if example_id < 8:
            for layer, layer_trace in enumerate(trace.layers):
                matrices[f"example{example_id}_layer{layer}_q"] = layer_trace.queries.cpu().numpy()
                matrices[f"example{example_id}_layer{layer}_k"] = layer_trace.keys.cpu().numpy()
                matrices[f"example{example_id}_layer{layer}_v"] = layer_trace.values.cpu().numpy()
                matrices[f"example{example_id}_layer{layer}_qk"] = layer_trace.qk_scores.cpu().numpy()
                matrices[f"example{example_id}_layer{layer}_attention"] = layer_trace.attention.cpu().numpy()
        demo_key = row.roles.index("demo_key")
        demo_value = row.roles.index("demo_value")
        query_position = len(row.tokens) - 1
        interventions = []
        for layer in range(len(model.blocks)):
            interventions.extend([
                (layer, "key_mask_demo_key", [Patch(layer, "key_mask", positions=(demo_key,))]),
                (layer, "value_mask_demo_value", [Patch(layer, "value_mask", positions=(demo_value,))]),
            ])
            for component in ("q", "k", "v"):
                source = getattr(control_trace.layers[layer], {"q": "queries", "k": "keys", "v": "values"}[component])
                interventions.append((layer, f"{component}_patch_query", [Patch(layer, component, source, (query_position,))]))
                interventions.append((layer, f"{component}_patch_all", [Patch(layer, component, source)]))
            for head in range(model.blocks[layer].attention.num_heads):
                interventions.append((layer, f"head_{head}_zero", [Patch(layer, "head_zero", head=head)]))
        for layer, intervention, patches in interventions:
            changed = forward_patched(model, tokens, patches)[0, -1]
            metrics = output_metrics(changed, row.target)
            causal.append({"example_id": example_id, "layer": layer, "intervention": intervention,
                           **metrics, "margin_damage": base_metrics["margin"] - metrics["margin"],
                           "rank_damage": metrics["rank"] - base_metrics["rank"],
                           "top1_flip": int(base_metrics["top1"] and not metrics["top1"]),
                           "js": js_divergence(base, changed)})
    np.savez_compressed(output / "copy_qkv_matrices.npz", **matrices)
    write_csv(output / "copy_qk_structure.csv", mean_rows(qk, ("layer", "head"),
              ("diagonal_mass", "previous_token_mass", "query_to_repeated_key", "query_to_associated_value",
               "row_entropy", "qk_frobenius", "qk_effective_rank")))
    write_csv(output / "copy_attention_role_mass.csv", mean_rows(role_mass,
              ("layer", "head", "destination_role", "source_role"), ("attention_mass",)))
    write_csv(output / "copy_target_support_flow.csv", mean_rows(support,
              ("layer", "head", "source_position", "source_role"), ("target_support",)))
    write_csv(output / "copy_sa_ff_contributions.csv", mean_rows(residual,
              ("layer", "boundary"), ("rank", "margin", "probability", "target_logit", "top1")))
    # Preserve neuron identity while averaging across role-aligned episodes.
    write_csv(output / "copy_ff_neuron_contributions.csv", mean_rows(neurons,
              ("layer", "neuron"), ("activation", "target_contribution")))
    causal_summary = mean_rows(causal, ("layer", "intervention"),
                               ("top1", "rank", "margin", "probability", "margin_damage",
                                "rank_damage", "top1_flip", "js"))
    write_csv(output / "copy_qkv_patching.csv", [row for row in causal_summary if "patch" in row["intervention"]])
    write_csv(output / "copy_head_utility.csv", [row for row in causal_summary if "head" in row["intervention"]])
    write_csv(output / "copy_token_causality.csv", [row for row in causal_summary if "mask" in row["intervention"]])
    atomic_write_json(output / "copy_trace_manifest.json", {
        "examples": len(rows), "checkpoint": checkpoint,
        "qkv_matrices": len(matrices), "causal_interventions": len(causal),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="docs/papers/paper0_8/results/copy/calibration_w64/checkpoints/C2_sa_ff_L2_W64_H2_S11.pt")
    parser.add_argument("--output", default="docs/papers/paper0_8/results/copy")
    parser.add_argument("--examples", type=int, default=48)
    parser.add_argument("--seed", type=int, default=10011)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run(load_model(Path(args.checkpoint), resolve_device(args.device)), Path(args.output), args.examples, args.seed,
        args.checkpoint)
