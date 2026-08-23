"""Run the controlled Paper 0.5 n-gram experiment suite (E1--E6).

The suite trains two inspectable causal Transformer settings on a controlled
corpus, records dense checkpoint measurements, and regenerates every table and
figure from JSONL/CSV artifacts. It is a controlled local-model experiment, not
evidence about opaque pretrained-model training frequency.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import random
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from cl.analysis.attention_motifs import motif_stability
from cl.analysis.checkpoint_dynamics import summarize_checkpoint
from cl.analysis.component_contrib import measure_components
from cl.analysis.equivalence import common_component_metrics, distribution_metrics, jensen_shannon_bits, topk_overlap
from cl.common.artifacts import RunMetadata, atomic_write_json, stable_hash, write_csv, write_jsonl
from cl.common.hooks import final_position_trace
from cl.common.metrics import bootstrap_ci, classify_update_relation, cosine, spearman
from cl.common.model_adapter import Intervention, TinyTransformerLM, train_step
from cl.ngram.atlas import build_atlas, sample_strata
from cl.ngram.synthetic import ProbeExample, build_corpus


MODEL_SETTINGS = (
    {"name": "tiny-w32-l2-h2", "width": 32, "layers": 2, "heads": 2},
    {"name": "tiny-w48-l3-h3", "width": 48, "layers": 3, "heads": 3},
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def grouped_probes(probes: list[ProbeExample]) -> list[list[ProbeExample]]:
    groups: dict[int, list[ProbeExample]] = defaultdict(list)
    for probe in probes:
        groups[len(probe.tokens)].append(probe)
    return [groups[length] for length in sorted(groups)]


def component_rows(model, probes, device) -> list[dict]:
    rows = []
    for group in grouped_probes(list(probes)):
        inputs = torch.tensor([probe.tokens for probe in group], dtype=torch.long, device=device)
        controls = torch.tensor([probe.control_tokens for probe in group], dtype=torch.long, device=device)
        targets = torch.tensor([probe.target for probe in group], dtype=torch.long, device=device)
        rows.extend(
            measure_components(
                model,
                inputs,
                targets,
                [probe.example_id for probe in group],
                [probe.stratum for probe in group],
                [probe.relation_id for probe in group],
                controls,
            )
        )
    return rows


@torch.no_grad()
def collect_motifs(model, probes, device) -> list[dict]:
    records = []
    for probe in probes:
        for control, tokens in ((False, probe.tokens), (True, probe.control_tokens)):
            inputs = torch.tensor([tokens], dtype=torch.long, device=device)
            _, trace = model(inputs, capture=True)
            if trace is None:
                raise RuntimeError("missing trace")
            for layer_index, layer in enumerate(final_position_trace(trace)):
                records.append(
                    {
                        "example_id": probe.example_id,
                        "relation_id": probe.relation_id,
                        "stratum": probe.stratum,
                        "layer": layer_index,
                        "control": control,
                        "attention": layer["attention"][0].numpy().tolist(),
                    }
                )
    return records


@torch.no_grad()
def collect_override(model, probes, device) -> list[dict]:
    rows = []
    for probe in [probe for probe in probes if probe.stratum == "override_repair"]:
        inputs = torch.tensor([probe.tokens], dtype=torch.long, device=device)
        _, trace = model(inputs, capture=True)
        if trace is None:
            raise RuntimeError("missing trace")
        compact = final_position_trace(trace)
        previous_update = None
        for layer_index, layer in enumerate(compact):
            for component, before_name, update_name, after_name in (
                ("sa", "pre_sa", "delta_sa", "post_sa"),
                ("ff", "post_sa", "delta_ff", "post_block"),
            ):
                before = F.log_softmax(model.diagnostic_logits(layer[before_name].to(device)), dim=-1)[0]
                after = F.log_softmax(model.diagnostic_logits(layer[after_name].to(device)), dim=-1)[0]
                update = layer[update_name][0].numpy()
                relation = cosine(previous_update, update) if previous_update is not None else 0.0
                rows.append(
                    {
                        "example_id": probe.example_id,
                        "layer": layer_index,
                        "component": component,
                        "correct_target": probe.target,
                        "habitual_target": probe.habitual_target,
                        "correct_signed_progress": float(after[probe.target] - before[probe.target]),
                        "habitual_signed_progress": float(after[probe.habitual_target] - before[probe.habitual_target]),
                        "update_cosine_previous": relation,
                        "repair_geometry_candidate": classify_update_relation(relation),
                    }
                )
                previous_update = update
    return rows


@torch.no_grad()
def collect_equivalence(model, probes, device, identity):
    """Collect the preregistered predictive-family, entropy, and patching tests."""
    entropy_rows, update_rows, family_rows, patch_rows = [], [], [], []
    locations = ("pre_sa", "post_sa", "post_block")
    for group in grouped_probes(list(probes)):
        inputs = torch.tensor([probe.tokens for probe in group], dtype=torch.long, device=device)
        controls = torch.tensor([probe.control_tokens for probe in group], dtype=torch.long, device=device)
        targets = torch.tensor([probe.target for probe in group], dtype=torch.long, device=device)
        logits, trace = model(inputs, capture=True)
        control_logits, control_trace = model(controls, capture=True)
        assert trace is not None and control_trace is not None
        intact = torch.softmax(logits[:, -1], -1).cpu().numpy()
        control_p = torch.softmax(control_logits[:, -1], -1).cpu().numpy()
        compact, compact_control = final_position_trace(trace), final_position_trace(control_trace)
        by_relation = defaultdict(list)
        for i, probe in enumerate(group): by_relation[probe.relation_id].append(i)
        for i, probe in enumerate(group):
            relation_members = by_relation[probe.relation_id]
            donor = relation_members[(relation_members.index(i) + 1) % len(relation_members)]
            family_rows.append({**identity, "example_id": probe.example_id, "relation_id": probe.relation_id,
                "stratum": probe.stratum, "within_family_js_bits": jensen_shannon_bits(intact[i], intact[donor]),
                "nonequivalent_js_bits": jensen_shannon_bits(intact[i], control_p[i]),
                "within_family_top5_overlap": topk_overlap(intact[i], intact[donor]),
                "nonequivalent_top5_overlap": topk_overlap(intact[i], control_p[i])})
            previous = None
            for layer, values in enumerate(compact):
                for location in locations:
                    p = torch.softmax(model.diagnostic_logits(values[location][i:i+1].to(device)), -1)[0].cpu().numpy()
                    row = distribution_metrics(p, probe.target)
                    if previous is not None:
                        for metric in ("entropy_bits", "target_surprisal_bits", "target_probability"):
                            row[f"delta_{metric}"] = float(row[metric] - previous[metric])
                    entropy_rows.append({**identity, "example_id": probe.example_id, "relation_id": probe.relation_id,
                        "stratum": probe.stratum, "layer": layer, "location": location, **row})
                    previous = row
                for component, name in (("sa", "delta_sa"), ("ff", "delta_ff")):
                    update_rows.append({**identity, "example_id": probe.example_id, "relation_id": probe.relation_id,
                        "stratum": probe.stratum, "layer": layer, "component": component,
                        "vector": values[name][i].numpy().tolist()})
                    full_update = getattr(trace.layers[layer], name)
                    full_control_update = getattr(control_trace.layers[layer], name)
                    for donor_type, replacement in (("equivalent", full_update[donor:donor+1]),
                                                     ("nonequivalent", full_control_update[i:i+1])):
                        changed, _ = model(inputs[i:i+1], intervention=Intervention(layer, component, "replace", replacement.to(device)))
                        changed_p = torch.softmax(changed[0, -1], -1).cpu().numpy()
                        patch_rows.append({**identity, "example_id": probe.example_id, "relation_id": probe.relation_id,
                            "stratum": probe.stratum, "layer": layer, "component": component, "donor_type": donor_type,
                            "output_js_bits": jensen_shannon_bits(intact[i], changed_p),
                            "target_probability_change": float(changed_p[probe.target] - intact[i, probe.target])})
    return entropy_rows, update_rows, family_rows, patch_rows


def aggregate_equivalence(families, patches, common):
    family = [{"comparison": name, "mean_js_bits": float(np.mean([r[js] for r in families])),
               "mean_top5_overlap": float(np.mean([r[top] for r in families])), "n": len(families)}
              for name, js, top in (("equivalent", "within_family_js_bits", "within_family_top5_overlap"),
                                    ("nonequivalent", "nonequivalent_js_bits", "nonequivalent_top5_overlap"))]
    groups=defaultdict(list)
    for row in patches: groups[(row["donor_type"],row["component"],row["layer"])].append(row)
    patch=[{"donor_type":k[0],"component":k[1],"layer":k[2],"mean_output_js_bits":float(np.mean([r["output_js_bits"] for r in v])),
            "mean_target_probability_change":float(np.mean([r["target_probability_change"] for r in v])),"n":len(v)} for k,v in sorted(groups.items())]
    return family, patch


def train_model(setting, seed, corpus, steps, checkpoints, device, checkpoint_dir):
    set_seed(seed)
    model = TinyTransformerLM(
        corpus.vocab_size,
        corpus.sequence_length,
        width=setting["width"],
        layers=setting["layers"],
        heads=setting["heads"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    generator = torch.Generator().manual_seed(seed)
    train_tensor = torch.tensor(corpus.train_sequences, dtype=torch.long)
    dynamics = []
    losses = []
    by_stratum = defaultdict(list)
    for probe in corpus.probes:
        by_stratum[probe.stratum].append(probe)
    diagnostic_probes = [probe for name in sorted(by_stratum) for probe in by_stratum[name][:4]]
    checkpoint_set = set(checkpoints)
    for step in range(steps + 1):
        if step in checkpoint_set:
            model.eval()
            rows = component_rows(model, diagnostic_probes, device)
            dynamics.extend(summarize_checkpoint(rows, step))
            if setting["name"] == MODEL_SETTINGS[0]["name"] and seed == 11:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"step": step, "setting": setting, "seed": seed, "state_dict": model.state_dict()},
                    checkpoint_dir / f"step-{step:04d}.pt",
                )
        if step == steps:
            break
        indices = torch.randint(len(train_tensor), (48,), generator=generator)
        batch = train_tensor[indices].to(device)
        losses.append(train_step(model, batch, optimizer))
    return model, dynamics, losses


def add_identity(rows, **identity):
    return [{**identity, **row} for row in rows]


def aggregate_components(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model_setting"], row["stratum"], row["component"], row["layer"])].append(row)
    output = []
    for key, values in sorted(groups.items()):
        causal = [value["causal_logprob_drop"] for value in values]
        diagnostic = [value["diagnostic_signed_progress"] for value in values]
        mean_causal, low, high = bootstrap_ci(causal, samples=500, seed=7)
        output.append(
            {
                "model_setting": key[0],
                "stratum": key[1],
                "component": key[2],
                "layer": key[3],
                "n_rows": len(values),
                "mean_causal_logprob_drop": mean_causal,
                "causal_ci_low": low,
                "causal_ci_high": high,
                "mean_diagnostic_signed_progress": float(np.mean(diagnostic)),
                "mean_mean_ablation_drop": float(np.mean([value["mean_ablation_logprob_drop"] for value in values])),
                "mean_matched_replacement_drop": float(np.mean([value["matched_replacement_logprob_drop"] for value in values])),
                "mean_head0_ablation_drop": float(np.nanmean([value["head0_ablation_logprob_drop"] for value in values])) if key[2] == "sa" else float("nan"),
                "mean_update_ratio": float(np.mean([value["candidate_update_ratio"] for value in values])),
            }
        )
    return output


def stored_context_summary(rows):
    selected = [row for row in rows if row["stratum"] in {"familiar_low_entropy", "context_introduced"}]
    groups = defaultdict(list)
    for row in selected:
        groups[(row["stratum"], row["component"])].append(row["causal_logprob_drop"])
    return [
        {"stratum": key[0], "component": key[1], "mean_causal_logprob_drop": float(np.mean(values)), "n": len(values)}
        for key, values in sorted(groups.items())
    ]


def regression_summary(rows):
    # Unit is relation/example aggregate, not token occurrence.
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model_setting"], row["seed"], row["example_id"], row["stratum"], row["component"])].append(row)
    unit_rows = []
    stratum_features = {
        "familiar_low_entropy": (2.0, 0.0, 0.0),
        "familiar_high_entropy": (1.5, 1.1, 0.0),
        "context_introduced": (0.0, 0.0, 1.0),
        "override_repair": (1.0, 0.0, 1.0),
    }
    for key, values in grouped.items():
        frequency, uncertainty, context = stratum_features[key[3]]
        unit_rows.append((frequency, uncertainty, context, 1.0 if key[4] == "ff" else 0.0, np.mean([v["causal_logprob_drop"] for v in values])))
    matrix = np.asarray([row[:4] for row in unit_rows], dtype=np.float64)
    target = np.asarray([row[4] for row in unit_rows], dtype=np.float64)
    design = np.column_stack([np.ones(len(matrix)), matrix])
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    prediction = design @ coefficients
    ss_total = np.square(target - target.mean()).sum()
    r2 = 1.0 - np.square(target - prediction).sum() / max(ss_total, 1e-12)
    names = ["intercept", "frequency_proxy", "continuation_entropy", "context_introduced", "ff_component"]
    return [{"predictor": name, "coefficient": float(value), "model_r2": float(r2), "n_units": len(target)} for name, value in zip(names, coefficients)]


def motif_causal_association(motifs, components):
    causal = defaultdict(list)
    for row in components:
        if row["component"] == "sa":
            causal[(row["relation_id"], row["layer"])].append(row["causal_logprob_drop"])
    pairs = [
        (row["motif_specificity"], float(np.mean(causal[(row["relation_id"], row["layer"])])))
        for row in motifs
        if (row["relation_id"], row["layer"]) in causal and np.isfinite(row["motif_specificity"])
    ]
    return [{
        "unit": "relation_x_layer",
        "spearman_motif_specificity_vs_sa_causal_drop": spearman(
            [pair[0] for pair in pairs], [pair[1] for pair in pairs]
        ),
        "n_units": len(pairs),
        "interpretation": "exploratory; relation-layer units from controlled local models",
    }]


def write_latex_tables(table_dir: Path, components):
    groups = defaultdict(list)
    for row in components:
        groups[(row["stratum"], row["component"])].append(row)
    labels = {
        "familiar_low_entropy": "Familiar, low entropy",
        "familiar_high_entropy": "Familiar, high entropy",
        "context_introduced": "Context introduced",
        "override_repair": "Override/repair",
    }
    lines = [
        "\\begin{table}[H]\\centering\\small",
        "\\begin{tabular}{llrrrr}\\toprule",
        "Stratum & Component & Zero & Mean & Replacement & Head 0 \\\\" + "\\midrule",
    ]
    for (stratum, component), values in sorted(groups.items()):
        zero = np.mean([row["zero_ablation_logprob_drop"] for row in values])
        mean = np.mean([row["mean_ablation_logprob_drop"] for row in values])
        replacement = np.mean([row["matched_replacement_logprob_drop"] for row in values])
        head = np.mean([row["head0_ablation_logprob_drop"] for row in values]) if component == "sa" else float("nan")
        head_text = f"{head:.3f}" if np.isfinite(head) else "--"
        lines.append(f"{labels[stratum]} & {component.upper()} & {zero:.3f} & {mean:.3f} & {replacement:.3f} & {head_text} " + r"\\")
    lines.extend([
        "\\bottomrule\\end{tabular}",
        "\\caption{Mean intact-minus-intervened target log probability (nats). Positive values indicate target-supporting causal contribution. Rows aggregate layers, model settings, seeds, and n-gram examples; scientific-unit uncertainty is reported in the machine-readable component table. Head 0 applies only to SA.}",
        "\\label{tab:pilot-causal}",
        "\\end{table}",
    ])
    (table_dir / "main_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(output, atlas, summary, motifs, override, dynamics, component_rows_all):
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axis("off")
    ax.text(0.05, 0.68, "prefix relation", bbox=dict(boxstyle="round", fc="#dbeafe"), fontsize=13)
    ax.text(0.40, 0.68, "SA -> FFN -> residual depth", bbox=dict(boxstyle="round", fc="#dcfce7"), fontsize=13)
    ax.text(0.78, 0.68, "continuation", bbox=dict(boxstyle="round", fc="#fef3c7"), fontsize=13)
    ax.annotate("", (0.39, 0.72), (0.23, 0.72), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", (0.77, 0.72), (0.66, 0.72), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(0.16, 0.28, "descriptive progress", ha="center")
    ax.text(0.50, 0.28, "causal intervention", ha="center")
    ax.text(0.82, 0.28, "checkpoint emergence", ha="center")
    fig.tight_layout(); fig.savefig(figure_dir / "concept.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter([max(row.reference_corpus_frequency, 1) for row in atlas], [row.continuation_entropy for row in atlas], s=12, alpha=.45)
    ax.set_xscale("log"); ax.set_xlabel("reference corpus frequency"); ax.set_ylabel("continuation entropy (nats)"); ax.set_title("Controlled n-gram atlas")
    fig.tight_layout(); fig.savefig(figure_dir / "atlas.png", dpi=170); plt.close(fig)

    def line_plot(metric, filename, title):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        settings = sorted({row["model_setting"] for row in summary})
        for axis, setting in zip(axes, settings):
            for (stratum, component), marker in zip(
                [("familiar_low_entropy", "sa"), ("familiar_low_entropy", "ff"), ("context_introduced", "sa"), ("context_introduced", "ff")],
                ["o", "s", "^", "D"],
            ):
                values = sorted([row for row in summary if row["model_setting"] == setting and row["stratum"] == stratum and row["component"] == component], key=lambda row: row["layer"])
                if values:
                    axis.plot([row["layer"] for row in values], [row[metric] for row in values], marker=marker, label=f"{stratum}/{component}")
            axis.axhline(0, color="black", lw=.7); axis.set_title(setting); axis.set_xlabel("layer"); axis.grid(alpha=.25)
        axes[0].set_ylabel(metric.replace("_", " ")); axes[-1].legend(fontsize=7); fig.suptitle(title); fig.tight_layout(); fig.savefig(figure_dir / filename, dpi=170); plt.close(fig)
    line_plot("mean_diagnostic_signed_progress", "diagnostic_contributions.png", "Signed diagnostic progress (not causal)")
    line_plot("mean_causal_logprob_drop", "causal_contributions.png", "Zero-ablation causal contribution")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = sorted({row["relation_id"] for row in motifs})
    for label in labels:
        values = sorted([row for row in motifs if row["relation_id"] == label], key=lambda row: row["layer"])
        ax.plot([row["layer"] for row in values], [row["motif_specificity"] for row in values], marker="o", label=label)
    ax.axhline(0, color="black", lw=.7); ax.set_xlabel("layer"); ax.set_ylabel("within - matched-control cosine"); ax.set_title("Attention motif specificity"); ax.legend(fontsize=7); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(figure_dir / "motif_stability.png", dpi=170); plt.close(fig)

    # Trajectory map: SVD over per-example SA/FFN causal trajectories.
    grouped = defaultdict(list)
    for row in component_rows_all:
        grouped[(row["model_setting"], row["seed"], row["example_id"], row["stratum"])].append(row)
    labels, vectors = [], []
    for key, values in sorted(grouped.items()):
        ordered = sorted(values, key=lambda row: (row["layer"], row["component"]))
        vectors.append([row["causal_logprob_drop"] for row in ordered]); labels.append(key[3])
    width = min(map(len, vectors)); matrix = np.asarray([vector[:width] for vector in vectors]); matrix -= matrix.mean(0)
    coordinates = matrix @ np.linalg.svd(matrix, full_matrices=False)[2][:2].T
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for label in sorted(set(labels)):
        index = [i for i, value in enumerate(labels) if value == label]
        ax.scatter(coordinates[index, 0], coordinates[index, 1], s=15, alpha=.55, label=label)
    ax.set_xlabel("trajectory component 1"); ax.set_ylabel("trajectory component 2"); ax.set_title("Causal trajectory map (descriptive SVD)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figure_dir / "trajectory_map.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    repair = defaultdict(list)
    for row in override:
        repair[(row["layer"], row["component"], "correct")].append(row["correct_signed_progress"])
        repair[(row["layer"], row["component"], "habitual")].append(row["habitual_signed_progress"])
    for key, values in sorted(repair.items()):
        ax.scatter(key[0] + ({"sa": -.12, "ff": .12}[key[1]]), np.mean(values), marker={"correct": "o", "habitual": "x"}[key[2]], label=f"{key[1]}/{key[2]}" if key[0] == 0 else None)
    ax.axhline(0, color="black", lw=.7); ax.set_xlabel("layer"); ax.set_ylabel("signed log-probability progress"); ax.set_title("Contextual override and repair"); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(figure_dir / "override_repair.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for (stratum, component), values in sorted(defaultdict(list, {key: [r for r in dynamics if (r["stratum"], r["component"]) == key] for key in {(r["stratum"], r["component"]) for r in dynamics}}).items()):
        by_step = defaultdict(list)
        for row in values: by_step[row["step"]].append(row["mean_causal_logprob_drop"])
        ax.plot(sorted(by_step), [np.mean(by_step[step]) for step in sorted(by_step)], marker="o", label=f"{stratum}/{component}")
    ax.axhline(0, color="black", lw=.7); ax.set_xlabel("training step"); ax.set_ylabel("mean causal log-probability drop"); ax.set_title("Checkpoint dynamics"); ax.legend(fontsize=7); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(figure_dir / "training_dynamics.png", dpi=170); plt.close(fig)


def write_summary(output, config, atlas, component_summary, motif_summary, motif_causal, stored_context, regression, losses):
    final_losses = [values[-1] for values in losses.values()]
    sa = [row["mean_causal_logprob_drop"] for row in stored_context if row["component"] == "sa"]
    ff = [row["mean_causal_logprob_drop"] for row in stored_context if row["component"] == "ff"]
    motif_values = [row["motif_specificity"] for row in motif_summary if np.isfinite(row["motif_specificity"])]
    lines = [
        "# Paper 0.5 controlled experiment summary", "",
        "## What was run", "",
        f"- Two model settings x {len(config['seeds'])} seeds on the controlled n-gram corpus.",
        f"- E1--E6: atlas, SA/FFN diagnostics, zero ablations, motif controls, stored-vs-context mapping, override/repair, and checkpoint dynamics.",
        f"- Atlas entries: {len(atlas)}; component aggregate rows: {len(component_summary)}.", "",
        "## Main pilot observations", "",
        f"- Final training loss range: {min(final_losses):.4f}--{max(final_losses):.4f}.",
        f"- Stored/context aggregate mean causal drop across SA cells: {np.mean(sa):.4f}; across FFN cells: {np.mean(ff):.4f}.",
        f"- Mean attention motif specificity over matched controls: {np.mean(motif_values):.4f}.",
        f"- Motif-specificity versus SA causal-drop Spearman correlation: {motif_causal[0]['spearman_motif_specificity_vs_sa_causal_drop']:.4f} over {motif_causal[0]['n_units']} relation-layer units.",
        f"- Exploratory unit-level regression R2: {regression[0]['model_r2']:.4f} (controlled synthetic factors; not a population estimate).", "",
        "## Interpretation and failures", "",
        "These are controlled local-model results, not evidence about pretrained-model training frequency. Corpus frequency is known here because the model is trained locally. Signed logit-lens progress is retained as diagnostic only; component results include zero, matched-mean, matched-replacement, selective-head, and FFN-layer interventions, but generic component importance remains a competing explanation. Motif similarity is compared with matched controls and shows no relation to SA causal contribution. Pretrained replication and path patching remain required before a strong mechanistic claim.", "",
        "## Exact artifacts", "",
        "- `raw/atlas.jsonl`", "- `raw/components.jsonl`", "- `raw/motifs.jsonl`", "- `raw/override.jsonl`", "- `tables/component_summary.csv`", "- `tables/stored_vs_context.csv`", "- `tables/regression.csv`", "- `tables/motif_causal_association.csv`", "- `tables/training_dynamics.csv`", "- `figures/*.png`", "",
        "## Next falsifiable question", "",
        "Do the SA/FFN causal fractions and override trajectories replicate in a small pretrained checkpoint series after matching reference-corpus frequency, entropy, tokenization, and generic layer importance?", "",
    ]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "decisions.md").write_text(
        "# Decisions\n\n- E7 online learning was not started: E1--E6 require pretrained and stronger intervention replication first.\n- Controlled local training is used so training frequency is known exactly.\n- Zero ablation supports causal pilot measurements; all logit-lens values remain explicitly diagnostic.\n",
        encoding="utf-8",
    )


def run(args):
    repo = Path(args.repo).resolve()
    output = Path(args.output).resolve()
    raw_dir, table_dir = output / "raw", output / "tables"
    raw_dir.mkdir(parents=True, exist_ok=True); table_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    seeds = [int(value) for value in args.seeds.split(",")]
    config = {"seeds": seeds, "steps": args.steps, "checkpoints": args.checkpoints, "device": str(device), "models": MODEL_SETTINGS}
    all_components, all_motif_records, all_override, all_dynamics = [], [], [], []
    all_entropy, all_updates, all_families, all_patches = [], [], [], []
    losses_by_run = {}
    corpus_for_atlas = build_corpus(seed=0, train_size=args.train_size)
    atlas = build_atlas(corpus_for_atlas.train_sequences)
    strata = sample_strata(atlas, seed=0)
    write_jsonl(raw_dir / "atlas.jsonl", [entry.as_dict() for entry in atlas])
    atomic_write_json(raw_dir / "atlas_strata.json", {name: [entry.as_dict() for entry in values] for name, values in strata.items()})
    metadata_rows = []
    for setting in MODEL_SETTINGS:
        for seed in seeds:
            corpus = build_corpus(seed=seed, train_size=args.train_size)
            run_id = f"paper05-{setting['name']}-seed{seed}"
            metadata = RunMetadata.capture(repo=repo, run_id=run_id, config=config, model_id=setting["name"], dataset_id="controlled-ngram-v1", seed=seed, device=str(device), dtype="float32", data_hash=corpus.corpus_hash)
            metadata_rows.append(metadata.as_dict())
            model, dynamics, losses = train_model(setting, seed, corpus, args.steps, args.checkpoints, device, output / "checkpoints")
            losses_by_run[run_id] = losses
            identity = {"run_id": run_id, "model_setting": setting["name"], "seed": seed}
            components = add_identity(component_rows(model, corpus.probes, device), **identity)
            motifs = add_identity(collect_motifs(model, corpus.probes, device), **identity)
            overrides = add_identity(collect_override(model, corpus.probes, device), **identity)
            all_components.extend(components); all_motif_records.extend(motifs); all_override.extend(overrides)
            all_dynamics.extend(add_identity(dynamics, **identity))
            entropy_rows, update_rows, family_rows, patch_rows = collect_equivalence(model, corpus.probes, device, identity)
            all_entropy.extend(entropy_rows); all_updates.extend(update_rows); all_families.extend(family_rows); all_patches.extend(patch_rows)
    motif_summary = []
    motif_runs = defaultdict(list)
    for row in all_motif_records:
        motif_runs[row["run_id"]].append(row)
    for run_id, records in sorted(motif_runs.items()):
        for row in motif_stability(records):
            motif_summary.append({"run_id": run_id, **row})
    # Aggregate motif statistics over runs at relation/layer granularity.
    motif_groups = defaultdict(list)
    for row in motif_summary: motif_groups[(row["relation_id"], row["layer"])].append(row)
    motif_aggregate = [{"relation_id": key[0], "layer": key[1], "within_motif_cosine": float(np.mean([r["within_motif_cosine"] for r in values])), "matched_control_cosine": float(np.mean([r["matched_control_cosine"] for r in values])), "motif_specificity": float(np.mean([r["motif_specificity"] for r in values])), "n_runs": len(values)} for key, values in sorted(motif_groups.items())]
    component_summary = aggregate_components(all_components)
    stored_context = stored_context_summary(all_components)
    regression = regression_summary(all_components)
    motif_causal = motif_causal_association(motif_aggregate, all_components)
    common = common_component_metrics(all_updates, ("model_setting", "seed", "layer", "component", "relation_id"))
    family_summary, patch_summary = aggregate_equivalence(all_families, all_patches, common)
    write_jsonl(raw_dir / "components.jsonl", all_components)
    write_jsonl(raw_dir / "motifs.jsonl", all_motif_records)
    write_jsonl(raw_dir / "override.jsonl", all_override)
    write_jsonl(raw_dir / "metadata.jsonl", metadata_rows)
    write_jsonl(raw_dir / "entropy_trajectories.jsonl", all_entropy)
    write_jsonl(raw_dir / "equivalence_families.jsonl", all_families)
    write_jsonl(raw_dir / "equivalence_patches.jsonl", all_patches)
    atomic_write_json(raw_dir / "losses.json", losses_by_run)
    write_csv(table_dir / "component_summary.csv", component_summary)
    write_csv(table_dir / "motif_summary.csv", motif_aggregate)
    write_csv(table_dir / "stored_vs_context.csv", stored_context)
    write_csv(table_dir / "regression.csv", regression)
    write_csv(table_dir / "motif_causal_association.csv", motif_causal)
    write_csv(table_dir / "training_dynamics.csv", all_dynamics)
    write_csv(table_dir / "override_summary.csv", all_override)
    write_csv(table_dir / "predictive_equivalence.csv", family_summary)
    write_csv(table_dir / "equivalence_patching.csv", patch_summary)
    write_csv(table_dir / "common_update_components.csv", common)
    write_latex_tables(table_dir, all_components)
    plot_results(output, atlas, component_summary, motif_aggregate, all_override, all_dynamics, all_components)
    write_summary(output, config, atlas, component_summary, motif_aggregate, motif_causal, stored_context, regression, losses_by_run)
    atomic_write_json(output / "manifest.json", {"schema_version": "paper05.results.v2", "config": config, "equivalence_definition": "externally grouped continuation relation", "artifact_hash": stable_hash({"components": all_components, "motifs": motif_aggregate, "dynamics": all_dynamics, "entropy": all_entropy, "patches": all_patches})})
    print(json.dumps({"output": str(output), "component_rows": len(all_components), "motif_rows": len(all_motif_records), "atlas_entries": len(atlas)}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--output", default="docs/papers/paper0_5/results")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", default="11,23")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--train-size", type=int, default=960)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[0, 40, 100, 160])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
