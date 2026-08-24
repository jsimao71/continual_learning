"""Controlled three-part experiments for Paper 0.5.

Group 1 implements the minimum confirmatory stability matrix. Later groups use
the same generator contract and artifact root.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from cl.analysis.equivalence import jensen_shannon_bits
from cl.common.artifacts import RunMetadata, atomic_write_json, stable_hash, write_csv, write_jsonl
from cl.common.metrics import bootstrap_ci
from cl.common.model_adapter import Intervention, TinyTransformerLM, train_step


@dataclass(frozen=True)
class PatternRecord:
    generator_family: str
    predictive_family_id: str
    surface_identity_id: str
    target_token: int
    pattern_tokens: tuple[int, ...]
    pattern_length: int
    dependency_span: int
    nuisance_tokens: tuple[int, ...]
    nuisance_count: int
    nuisance_type: str
    nuisance_seed: int
    answer_changing_context: bool
    continuation_entropy: str
    train_frequency: int
    split: str
    generator_seed: int

    @property
    def tokens(self):
        # Keep the prediction site at the same absolute position as training;
        # neutral left context is fixed and is not counted as varied nuisance.
        content = self.nuisance_tokens + self.pattern_tokens
        return (80,) * (11 - len(content)) + content


def _relation(generator: str, relation: int) -> tuple[tuple[int, ...], int, int]:
    base = 4 + relation * 8
    target = base + 6
    if generator == "contiguous_ngram": return (base, base + 1), target, 1
    if generator == "skip_gram": return (base, 70 + relation, base + 1), target, 2
    if generator == "binary_functor": return (60, base, base + 1), target, 2
    raise ValueError(generator)


def generate_records(config: dict, split: str, nuisance_count: int, realizations: int) -> list[PatternRecord]:
    rows = []
    for generator_index, generator in enumerate(config["generators"]):
        for relation in range(config["relations_per_generator"]):
            pattern, target, span = _relation(generator, relation)
            for realization in range(realizations):
                seed = config["seed"] + generator_index * 10000 + relation * 100 + realization + nuisance_count * 100000
                rng = random.Random(seed)
                nuisance = tuple(rng.randrange(80, 96) for _ in range(nuisance_count))
                rows.append(PatternRecord(generator, f"{generator}:r{relation}", f"{split}:{realization}", target,
                    pattern, len(pattern), span, nuisance, nuisance_count, "irrelevant_uniform", seed, False,
                    "deterministic", config["train_examples_per_relation"], split, config["seed"]))
    return rows


def training_tensor(config: dict) -> torch.Tensor:
    rows = []
    rng = random.Random(config["seed"])
    for generator in config["generators"]:
        for relation in range(config["relations_per_generator"]):
            pattern, target, _ = _relation(generator, relation)
            for _ in range(config["train_examples_per_relation"]):
                noise = tuple(rng.randrange(80, 96) for _ in range(8))
                rows.append(noise + pattern + (target,))
    maximum = max(map(len, rows)); padded = [(80,) * (maximum - len(row)) + row for row in rows]
    return torch.tensor(padded, dtype=torch.long)


def train_model(config: dict, seed: int, device: torch.device):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    spec = config["model"]
    model = TinyTransformerLM(96, spec["max_length"], spec["width"], spec["layers"], spec["heads"]).to(device)
    data = training_tensor(config).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    losses = []
    for step in range(config["train_steps"]):
        index = torch.randint(0, len(data), (config["batch_size"],), device=device)
        losses.append(train_step(model, data[index], optimizer))
        if not np.isfinite(losses[-1]):
            raise RuntimeError(f"non-finite training loss at seed={seed}, step={step}")
    model.eval(); return model, losses


@torch.no_grad()
def group1_rows(model, records: list[PatternRecord], seed: int, device: torch.device) -> list[dict]:
    output = []
    for record in records:
        inputs = torch.tensor([record.tokens], device=device)
        logits, trace = model(inputs, capture=True); assert trace is not None
        for layer, values in enumerate(trace.layers):
            state = values.post_block[:, -1]
            distribution = torch.softmax(model.diagnostic_logits(state), -1)[0].cpu().numpy()
            target = record.target_token
            ordered = np.argsort(-distribution); competitor = next(x for x in ordered if x != target)
            output.append({**asdict(record), "model_seed": seed, "layer": layer,
                "target_probability": float(distribution[target]), "target_logprob": float(np.log(max(distribution[target], 1e-30))),
                "target_margin": float(np.log(max(distribution[target], 1e-30)) - np.log(max(distribution[competitor], 1e-30))),
                "entropy_bits": float(-(distribution * np.log2(np.clip(distribution, 1e-12, 1))).sum()),
                "top1_correct": int(distribution.argmax() == target), "probabilities": distribution.tolist(),
                "residual": state[0].cpu().numpy().tolist()})
    return output


def _initial_state(model, tokens: tuple[int, ...], device: torch.device) -> torch.Tensor:
    ids = torch.tensor([tokens], device=device); positions = torch.arange(len(tokens), device=device)
    return model.token_embedding(ids) + model.position_embedding(positions)[None]


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left.flatten()[None], right.flatten()[None]).item())


def jvp_diagnostics(model, records: list[PatternRecord], seed: int, device: torch.device) -> list[dict]:
    """Test one-block linearization for nuisance and answer-changing directions."""
    selected = [row for row in records if row.nuisance_count == 8]
    by_family = defaultdict(list)
    for row in selected: by_family[row.predictive_family_id].append(row)
    rows = []
    pairs = []
    families = sorted(by_family)
    for family in families:
        pairs.append(("nuisance", by_family[family][0], by_family[family][1]))
        same_generator = [other for other in families if other != family and other.split(":")[0] == family.split(":")[0]]
        pairs.append(("signal", by_family[family][0], by_family[same_generator[0]][0]))
    for direction, left_record, right_record in pairs:
        left = _initial_state(model, left_record.tokens, device); right = _initial_state(model, right_record.tokens, device)
        for layer, block in enumerate(model.blocks):
            length = left.shape[1]; mask = torch.triu(torch.full((length, length), float("-inf"), device=device), diagonal=1)
            def mapping(state): return block(state, causal_mask=mask, capture=False, intervention=None)[0]
            tangent = right - left
            with torch.enable_grad():
                actual_left, predicted = torch.autograd.functional.jvp(mapping, left.detach().requires_grad_(True), tangent)
            actual_right = mapping(right)
            observed = actual_right[:, -1] - actual_left[:, -1]; predicted_final = predicted[:, -1]
            relative = float(torch.linalg.vector_norm(predicted_final - observed) / torch.linalg.vector_norm(observed).clamp_min(1e-12))
            base_logits = model.diagnostic_logits(actual_left[:, -1]); actual_logits = model.diagnostic_logits(actual_right[:, -1]) - base_logits
            predicted_logits = model.diagnostic_logits((actual_left + predicted)[:, -1]) - base_logits
            prediction_error = float(torch.linalg.vector_norm(predicted_logits - actual_logits) / torch.linalg.vector_norm(actual_logits).clamp_min(1e-12))
            rows.append({"model_seed":seed,"generator_family":left_record.generator_family,
                         "predictive_family_id":left_record.predictive_family_id,"direction":direction,"layer":layer,
                         "input_delta_norm":float(torch.linalg.vector_norm(tangent[:, -1])),
                         "observed_delta_norm":float(torch.linalg.vector_norm(observed)),
                         "jvp_cosine":_cosine(predicted_final,observed),"relative_norm_error":relative,
                         "prediction_space_relative_error":prediction_error})
            left, right = actual_left.detach(), actual_right.detach()
    return rows


def aggregate_group1(rows: list[dict]) -> list[dict]:
    keys = ("model_seed", "generator_family", "predictive_family_id", "nuisance_count", "layer")
    groups = defaultdict(list)
    for row in rows: groups[tuple(row[k] for k in keys)].append(row)
    preliminary = []
    for key, values in sorted(groups.items()):
        distributions = np.asarray([row["probabilities"] for row in values]); centroid = distributions.mean(0)
        residuals = np.asarray([row["residual"] for row in values]); covariance = np.cov(residuals, rowvar=False)
        if not np.isfinite(covariance).all():
            raise RuntimeError(f"non-finite residual covariance in cell {key}")
        singular = np.linalg.svd(np.atleast_2d(covariance), compute_uv=False)
        preliminary.append({**dict(zip(keys, key)), "n_realizations": len(values),
            "within_js_bits": float(np.mean([jensen_shannon_bits(p, centroid) for p in distributions])),
            "mean_target_probability": float(np.mean([row["target_probability"] for row in values])),
            "mean_target_margin": float(np.mean([row["target_margin"] for row in values])),
            "mean_entropy_bits": float(np.mean([row["entropy_bits"] for row in values])),
            "top1_accuracy": float(np.mean([row["top1_correct"] for row in values])),
            "covariance_trace": float(np.trace(covariance)),
            "covariance_effective_rank": float(singular.sum() ** 2 / max(np.square(singular).sum(), 1e-12)),
            "centroid": centroid.tolist()})
    matched = defaultdict(list)
    for row in preliminary: matched[(row["model_seed"], row["nuisance_count"], row["layer"])].append(row)
    for values in matched.values():
        for row in values:
            between = [jensen_shannon_bits(row["centroid"], other["centroid"]) for other in values if other["predictive_family_id"] != row["predictive_family_id"]]
            row["between_js_bits"] = float(np.mean(between)); row["R"] = row["between_js_bits"] / max(row["within_js_bits"], 1e-12)
    for row in preliminary: del row["centroid"]
    return preliminary


def classify(first: float, middle: list[float], last: float) -> str:
    values = [first, *middle, last]
    if last >= first * .95: return "no_contraction"
    if all(a >= b for a, b in zip(values, values[1:])): return "monotone_contraction"
    if max(middle, default=first) > first and last < first: return "expansion_then_contraction"
    if last < min(values[:-1]): return "late_contraction"
    return "oscillatory"


def group1_inference(summary: list[dict]) -> tuple[list[dict], list[dict]]:
    contrasts, trajectories = [], []
    for generator in sorted({row["generator_family"] for row in summary}):
        selected = [row for row in summary if row["generator_family"] == generator and row["nuisance_count"] == 8]
        units = defaultdict(dict)
        for row in selected: units[(row["model_seed"], row["predictive_family_id"])][row["layer"]] = row
        for metric in ("within_js_bits", "R", "covariance_trace", "mean_target_probability"):
            values = [depth[max(depth)][metric] - depth[min(depth)][metric] for depth in units.values()]
            estimate, low, high = bootstrap_ci(values, samples=5000, seed=83)
            contrasts.append({"generator_family": generator, "metric": metric, "last_minus_first": estimate,
                              "ci_low": low, "ci_high": high, "n_family_seed_units": len(values)})
        for unit, depth in units.items():
            layers = sorted(depth); values = [depth[layer]["within_js_bits"] for layer in layers]
            trajectories.append({"model_seed": unit[0], "predictive_family_id": unit[1], "generator_family": generator,
                                 "classification": classify(values[0], values[1:-1], values[-1]),
                                 "first_within_js": values[0], "last_within_js": values[-1]})
    return contrasts, trajectories


def plot_group1(output: Path, summary: list[dict], trajectories: list[dict], jvp: list[dict]) -> None:
    figures = output / "figures"; figures.mkdir(parents=True, exist_ok=True)
    def curves(metric, filename, ylabel):
        fig, ax = plt.subplots(figsize=(7.4, 4.5))
        for generator in sorted({row["generator_family"] for row in summary}):
            selected = [row for row in summary if row["generator_family"] == generator and row["nuisance_count"] == 8]
            layers = sorted({row["layer"] for row in selected})
            ax.plot(layers, [np.median([row[metric] for row in selected if row["layer"] == layer]) for layer in layers], marker="o", label=generator)
        ax.set_xlabel("layer"); ax.set_ylabel(ylabel); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(figures / filename, dpi=180); plt.close(fig)
    curves("within_js_bits", "g1_within_between_vs_depth.png", "within-family JS (bits)")
    curves("R", "g1_R_vs_depth.png", "between / within JS")
    curves("covariance_trace", "g1_covariance_trace_vs_depth.png", "residual covariance trace")
    curves("mean_entropy_bits", "g1_entropy_vs_across_realization_JS.png", "within-example entropy (bits)")
    # Required smoke figures use available orthogonal axes; captions in the paper state their scope.
    fig, ax = plt.subplots(figsize=(7.4,4.5))
    for direction in ("nuisance","signal"):
        layers=sorted({row["layer"] for row in jvp}); ax.plot(layers,[np.median([row["observed_delta_norm"] for row in jvp if row["direction"]==direction and row["layer"]==layer]) for layer in layers],marker="o",label=direction)
    ax.set_xlabel("layer"); ax.set_ylabel("observed perturbation norm"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figures/"g1_signal_vs_nuisance_direction.png",dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.4,4.5))
    for direction in ("nuisance","signal"):
        layers=sorted({row["layer"] for row in jvp}); ax.plot(layers,[np.median([row["jvp_cosine"] for row in jvp if row["direction"]==direction and row["layer"]==layer]) for layer in layers],marker="o",label=direction)
    ax.set_xlabel("layer"); ax.set_ylabel("JVP / observed delta cosine"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figures/"g1_jvp_prediction_quality.png",dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.4, 4.5)); counts = defaultdict(int)
    for row in trajectories: counts[row["classification"]] += 1
    ax.bar(list(counts), list(counts.values())); ax.set_ylabel("family-seed trajectories"); ax.tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(figures / "g1_nonmonotonic_examples.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.4, 4.5)); matrix=[]
    for noise in sorted({row["nuisance_count"] for row in summary}):
        matrix.append([np.median([row["R"] for row in summary if row["nuisance_count"] == noise and row["layer"] == layer]) for layer in sorted({r["layer"] for r in summary})])
    image=ax.imshow(matrix,aspect="auto"); ax.set_yticks(range(len(matrix)),sorted({r["nuisance_count"] for r in summary})); ax.set_xlabel("layer"); ax.set_ylabel("nuisance count"); fig.colorbar(image,ax=ax,label="R")
    fig.tight_layout(); fig.savefig(figures / "g1_depth_noise_phase_map.png", dpi=180); plt.close(fig)


def reproduce_pilot(repo: Path, aggregate: Path) -> list[dict]:
    source = repo / "docs/papers/paper0_5/results/tables/variance_by_depth_prefix_noise.csv"
    with source.open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["training_stage"] == "trained" and row["control_type"] == "irrelevant_nuisance"
                and row["target_control"] == "observed" and row["prefix_evidence"] == "pattern"
                and row["noise_level"] == "4" and row["location"] == "post_block"]
    output=[]
    for layer in sorted({int(row["layer"]) for row in selected}):
        values=[row for row in selected if int(row["layer"])==layer]
        output.append({"layer":layer,"n_cells":len(values),
            "mean_target_probability_variance":float(np.mean([float(row["target_probability_variance"]) for row in values])),
            "mean_js_dispersion_bits":float(np.mean([float(row["mean_js_to_pattern_centroid"]) for row in values])),
            "mean_entropy_bits":float(np.mean([float(row["mean_entropy_bits"]) for row in values]))})
    write_csv(aggregate / "group1_reproduction.csv", output); return output


def run_group1(config: dict, repo: Path, output: Path, device: torch.device) -> None:
    rows, jvp, losses, metadata = [], [], {}, []
    records = sum((generate_records(config, "test", noise, config["probe_realizations"]) for noise in config["nuisance_counts"]), [])
    for seed in config["model_seeds"]:
        model, curve = train_model(config, seed, device); losses[str(seed)] = curve
        rows.extend(group1_rows(model, records, seed, device))
        jvp.extend(jvp_diagnostics(model, records, seed, device))
        metadata.append(RunMetadata.capture(repo=repo, run_id=f"paper05-g1-seed{seed}", config=config,
            model_id=f"tiny-controlled-seed{seed}", dataset_id="paper05-three-family-v1", seed=seed,
            device=str(device), dtype="float32", data_hash=stable_hash([asdict(row) for row in records])).as_dict())
    summary = aggregate_group1(rows); inference, trajectories = group1_inference(summary)
    raw, aggregate = output / "raw/group1", output / "aggregates/group1"; raw.mkdir(parents=True, exist_ok=True); aggregate.mkdir(parents=True, exist_ok=True)
    compact = [{k:v for k,v in row.items() if k not in {"probabilities","residual"}} for row in rows]
    write_jsonl(raw / "group1_observations.jsonl", compact); write_jsonl(raw / "group1_jvp.jsonl", jvp); write_jsonl(raw / "metadata.jsonl", metadata)
    reproduction = reproduce_pilot(repo, aggregate)
    atomic_write_json(raw / "losses.json", losses); write_csv(aggregate / "group1_metrics.csv", summary)
    write_csv(aggregate / "group1_inference.csv", inference); write_csv(aggregate / "group1_trajectories.csv", trajectories)
    write_csv(aggregate / "group1_jvp.csv", jvp); plot_group1(output, summary, trajectories, jvp)
    manifest={"schema_version":"paper05.group1.v1","config":config,"rows":len(rows),"families":9,"seeds":3,
              "jvp_rows":len(jvp),"pilot_reproduction":reproduction,
              "artifact_hash":stable_hash({"summary":summary,"inference":inference,"trajectories":trajectories,"jvp":jvp,"reproduction":reproduction})}
    atomic_write_json(output / "manifests/group1.json", manifest); print(json.dumps(manifest,indent=2))


def reachable(span: int, window: int | None, depth: int) -> bool:
    return window is None or depth * window >= span


def _transport_batch(span: int, count: int, seed: int, local: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    rng=random.Random(seed); inputs=[]; targets=[]
    for index in range(count):
        family=index%4; source=4+family; target=20+family
        fillers=[rng.randrange(40,60) for _ in range(max(span-1,0))]
        final=(30+family) if local else 30
        inputs.append([source,*fillers,final]); targets.append((30+family) if local else target)
    return torch.tensor(inputs,dtype=torch.long),torch.tensor(targets,dtype=torch.long)


def _train_transport(cell: dict, config: dict, device: torch.device):
    torch.manual_seed(config["seed"]+cell["span"]*100+cell["depth"]*10+(cell["window"] or 99))
    model=TinyTransformerLM(96,40,config["width"],cell["depth"],config["heads"],attention_window=cell["window"]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"])
    for step in range(config["train_steps"]):
        half=config["batch_size"]//2
        long_inputs,long_targets=_transport_batch(cell["span"],half,config["seed"]+step)
        local_inputs,local_targets=_transport_batch(cell["span"],config["batch_size"]-half,config["seed"]+step,True)
        inputs=torch.cat((long_inputs,local_inputs)); targets=torch.cat((long_targets,local_targets))
        inputs,targets=inputs.to(device),targets.to(device); model.train(); optimizer.zero_grad(set_to_none=True)
        logits,_=model(inputs); loss=F.cross_entropy(logits[:,-1],targets); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step()
    model.eval(); return model


@torch.no_grad()
def _transport_metrics(model, cell: dict, config: dict, device: torch.device, local: bool=False) -> list[dict]:
    inputs,targets=_transport_batch(cell["span"],config["probe_examples"],config["seed"]+999,local); inputs,targets=inputs.to(device),targets.to(device)
    _,trace=model(inputs,capture=True); assert trace is not None; rows=[]
    for layer,values in enumerate(trace.layers):
        p=torch.softmax(model.diagnostic_logits(values.post_block[:,-1]),-1)
        for index in range(len(inputs)):
            rows.append({"span":cell["span"],"window":cell["window"] if cell["window"] is not None else "full","depth":cell["depth"],
                "layer":layer,"example":index,"family":index%4,"local_control":int(local),
                "reachable":int(reachable(cell["span"],cell["window"],layer+1)),"target_probability":float(p[index,targets[index]]),
                "target_logprob":float(torch.log(p[index,targets[index]].clamp_min(1e-30))),"top1_correct":int(p[index].argmax()==targets[index]),
                "probabilities":p[index].cpu().numpy().tolist()})
    return rows


def run_group2(config: dict, repo: Path, output: Path, device: torch.device) -> None:
    rows=[]; metadata=[]
    for cell_index,cell in enumerate(config["cells"]):
        model=_train_transport(cell,config,device); rows.extend(_transport_metrics(model,cell,config,device)); rows.extend(_transport_metrics(model,cell,config,device,True))
        checkpoint_hash=stable_hash({name:tensor.detach().cpu().numpy().astype(np.float16).tobytes().hex() for name,tensor in model.state_dict().items()})
        metadata.append({**RunMetadata.capture(repo=repo,run_id=f"paper05-g2-cell{cell_index}",config={**config,"cell":cell},
            model_id=f"controlled-transport-cell{cell_index}",dataset_id="paper05-window-transport-v1",seed=config["seed"],device=str(device),dtype="float32",data_hash=stable_hash(cell)).as_dict(),
            "checkpoint_hash":checkpoint_hash})
    aggregate=[]; groups=defaultdict(list)
    for row in rows: groups[(row["span"],row["window"],row["depth"],row["layer"],row["local_control"],row["reachable"])].append(row)
    for key,values in sorted(groups.items(),key=str):
        family_centroids={family:np.mean([x["probabilities"] for x in values if x["family"]==family],axis=0) for family in range(4)}
        within=np.mean([jensen_shannon_bits(x["probabilities"],family_centroids[x["family"]]) for x in values])
        between=np.mean([jensen_shannon_bits(family_centroids[a],family_centroids[b]) for a in range(4) for b in range(a+1,4)])
        aggregate.append({**dict(zip(("span","window","depth","layer","local_control","reachable"),key)),
            "n":len(values),"accuracy":float(np.mean([x["top1_correct"] for x in values])),"mean_target_logprob":float(np.mean([x["target_logprob"] for x in values])),
            "within_js_bits":float(within),"between_js_bits":float(between),"R":float(between/max(within,1e-12))})
    raw,agg=output/"raw/group2",output/"aggregates/group2";raw.mkdir(parents=True,exist_ok=True);agg.mkdir(parents=True,exist_ok=True)
    write_jsonl(raw/"group2_observations.jsonl",rows);write_jsonl(raw/"metadata.jsonl",metadata);write_csv(agg/"group2_transport.csv",aggregate)
    figures=output/"figures";figures.mkdir(parents=True,exist_ok=True);final=[x for x in aggregate if x["layer"]==x["depth"]-1 and not x["local_control"]]
    for filename,metric in (("g2_span_window_accuracy_phase.png","accuracy"),("g2_span_window_R_phase.png","R")):
        fig,ax=plt.subplots(figsize=(7,4.5)); labels=[f"s{x['span']}/w{x['window']}/L{x['depth']}" for x in final];ax.bar(range(len(final)),[x[metric] for x in final]);ax.set_xticks(range(len(final)),labels,rotation=35,ha="right");ax.set_ylabel(metric);fig.tight_layout();fig.savefig(figures/filename,dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5))
    for cell in config["cells"]:
        window=cell["window"] if cell["window"] is not None else "full";v=[x for x in aggregate if x["span"]==cell["span"] and str(x["window"])==str(window) and x["depth"]==cell["depth"] and not x["local_control"]]
        ax.plot([x["layer"]+1 for x in v],[x["accuracy"] for x in v],marker="o",label=f"s{cell['span']}/w{window}")
    ax.set_xlabel("depth");ax.set_ylabel("accuracy");ax.legend(fontsize=7);ax.grid(alpha=.25);fig.tight_layout();fig.savefig(figures/"g2_transport_delay.png",dpi=180);plt.close(fig)
    for filename,local in (("g2_local_pattern_control.png",1),("g2_nested_override_trajectory.png",0)):
        fig,ax=plt.subplots(figsize=(7,4.5));v=[x for x in aggregate if x["local_control"]==local];ax.scatter([x["layer"]+1 for x in v],[x["accuracy"] for x in v],c=[x["span"] for x in v]);ax.set_xlabel("depth");ax.set_ylabel("accuracy");fig.tight_layout();fig.savefig(figures/filename,dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5));ratios=[x["span"]/(x["window"] if isinstance(x["window"],int) else x["span"]) for x in final];ax.scatter(ratios,[x["depth"] for x in final],c=[x["accuracy"] for x in final]);ax.set_xlabel("span/window");ax.set_ylabel("tested depth");fig.tight_layout();fig.savefig(figures/"g2_Lstar_vs_span_over_window.png",dpi=180);plt.close(fig)
    manifest={"schema_version":"paper05.group2.v1","config":config,"rows":len(rows),"regimes":len(config["cells"]),"mask_reachability_verified":True,"artifact_hash":stable_hash(aggregate)}
    atomic_write_json(output/"manifests/group2.json",manifest);print(json.dumps(manifest,indent=2))


def _distribution_stats(probabilities: np.ndarray, families: np.ndarray) -> dict:
    centroids={family:probabilities[families==family].mean(0) for family in sorted(set(families))}
    within=float(np.mean([jensen_shannon_bits(p,centroids[int(f)]) for p,f in zip(probabilities,families)]))
    keys=sorted(centroids);between=float(np.mean([jensen_shannon_bits(centroids[a],centroids[b]) for i,a in enumerate(keys) for b in keys[i+1:]]))
    return {"within_js_bits":within,"between_js_bits":between,"R":between/max(within,1e-12)}


def run_group3(config: dict, repo: Path, output: Path, device: torch.device) -> None:
    cell={"span":config["span"],"window":config["window"],"depth":config["depth"]};model=_train_transport(cell,config,device)
    inputs,targets=_transport_batch(config["span"],config["probe_examples"],config["seed"]+999);inputs,targets=inputs.to(device),targets.to(device)
    with torch.no_grad(): logits,trace=model(inputs,capture=True)
    assert trace is not None; baseline=torch.softmax(logits[:,-1],-1);families=np.asarray([i%4 for i in range(len(inputs))])
    base_stats=_distribution_stats(baseline.cpu().numpy(),families);rows=[];pair_rows=[];covariance=[];motifs=[]
    equivalent=torch.tensor([next(j for j in range(len(inputs)) if j!=i and j%4==i%4) for i in range(len(inputs))],device=device)
    mismatch=torch.tensor([next(j for j in range(len(inputs)) if j%4!=(i%4)) for i in range(len(inputs))],device=device)
    for layer,layer_trace in enumerate(trace.layers):
        heads=layer_trace.head_outputs.shape[1]
        for head in range(heads):
            actual=layer_trace.head_outputs[:,head]
            replacements={"mean":actual.mean(0,keepdim=True).expand_as(actual),"equivalent":actual[equivalent],"mismatched":actual[mismatch]}
            interventions={"zero":Intervention(layer,"sa","head_zero",head=head)}
            interventions.update({mode:Intervention(layer,"sa","head_replace",replacement=value,head=head) for mode,value in replacements.items()})
            for mode,intervention in interventions.items():
                changed,_=model(inputs,intervention=intervention);p=torch.softmax(changed[:,-1],-1);stats=_distribution_stats(p.detach().cpu().numpy(),families)
                logdrop=(torch.log(baseline[range(len(inputs)),targets])-torch.log(p[range(len(inputs)),targets])).mean()
                rows.append({"layer":layer,"head":head,"mode":mode,**stats,"stability_utility":stats["within_js_bits"]-base_stats["within_js_bits"],
                    "discrimination_utility":base_stats["between_js_bits"]-stats["between_js_bits"],"target_logprob_drop":float(logdrop)})
            norms=torch.linalg.vector_norm(actual[:,-1],dim=-1).cpu().numpy()
            for other in range(head+1,heads):
                other_norm=torch.linalg.vector_norm(layer_trace.head_outputs[:,other,-1],dim=-1).cpu().numpy()
                covariance.append({"layer":layer,"head_a":head,"head_b":other,"norm_correlation":float(np.corrcoef(norms,other_norm)[0,1])})
                replacement=-layer_trace.head_outputs[:,other]
                changed,_=model(inputs,intervention=Intervention(layer,"sa","head_replace",replacement=replacement,head=head));p=torch.softmax(changed[:,-1],-1)
                pair_drop=float((torch.log(baseline[range(len(inputs)),targets])-torch.log(p[range(len(inputs)),targets])).mean())
                pair_rows.append({"layer":layer,"head_a":head,"head_b":other,"pair_target_logprob_drop":pair_drop})
            attention=layer_trace.attention[:,head,-1].cpu().numpy();within=[];between=[]
            for i in range(len(attention)):
                within.append(float(np.dot(attention[i],attention[int(equivalent[i])])/(np.linalg.norm(attention[i])*np.linalg.norm(attention[int(equivalent[i])])+1e-12)))
                between.append(float(np.dot(attention[i],attention[int(mismatch[i])])/(np.linalg.norm(attention[i])*np.linalg.norm(attention[int(mismatch[i])])+1e-12)))
            motifs.append({"layer":layer,"head":head,"motif_specificity":float(np.mean(within)-np.mean(between))})
    zero={(x["layer"],x["head"]):x["target_logprob_drop"] for x in rows if x["mode"]=="zero"}
    for pair in pair_rows:
        individual=zero[(pair["layer"],pair["head_a"])]+zero[(pair["layer"],pair["head_b"])]
        pair["sum_individual_drop"]=individual;pair["gamma"]=pair["pair_target_logprob_drop"]-individual
    motif_values=[x["motif_specificity"] for x in motifs];utility=[zero[(x["layer"],x["head"])] for x in motifs]
    from cl.common.metrics import spearman
    association={"spearman_motif_vs_causal":spearman(motif_values,utility),"n_heads":len(motifs),"prior_pilot_spearman":-0.009}
    raw,agg=output/"raw/group3",output/"aggregates/group3";raw.mkdir(parents=True,exist_ok=True);agg.mkdir(parents=True,exist_ok=True)
    write_jsonl(raw/"head_interventions.jsonl",rows);write_csv(agg/"head_utility.csv",rows);write_csv(agg/"head_pair_interactions.csv",pair_rows);write_csv(agg/"head_covariance.csv",covariance);write_csv(agg/"head_motifs.csv",motifs);atomic_write_json(agg/"motif_causal_association.json",association)
    figures=output/"figures";figures.mkdir(parents=True,exist_ok=True)
    for filename,metric in (("g3_head_stability_utility.png","stability_utility"),("g3_head_discrimination_utility.png","discrimination_utility"),("g3_head_by_pattern_length.png","target_logprob_drop")):
        fig,ax=plt.subplots(figsize=(8,4.5));v=[x for x in rows if x["mode"]=="zero"];ax.bar(range(len(v)),[x[metric] for x in v]);ax.set_xlabel("layer/head");ax.set_ylabel(metric);fig.tight_layout();fig.savefig(figures/filename,dpi=180);plt.close(fig)
    for filename,data,xkey,ykey in (("g3_head_pair_interactions.png",pair_rows,"sum_individual_drop","pair_target_logprob_drop"),("g3_head_output_covariance.png",covariance,"norm_correlation","layer"),("g3_motif_vs_causal_utility.png",[{**x,"utility":zero[(x["layer"],x["head"])]} for x in motifs],"motif_specificity","utility")):
        fig,ax=plt.subplots(figsize=(6,4.5));ax.scatter([x[xkey] for x in data],[x[ykey] for x in data]);ax.set_xlabel(xkey);ax.set_ylabel(ykey);fig.tight_layout();fig.savefig(figures/filename,dpi=180);plt.close(fig)
    # One fixed-capacity head-count point is complete; scaling remains explicitly pending.
    fig,ax=plt.subplots(figsize=(6,4.5));ax.scatter([config["heads"]],[base_stats["R"]]);ax.set_xlabel("head count");ax.set_ylabel("final R");fig.tight_layout();fig.savefig(figures/"g3_head_count_scaling.png",dpi=180);plt.close(fig)
    checkpoint_hash=stable_hash({name:tensor.detach().cpu().numpy().astype(np.float16).tobytes().hex() for name,tensor in model.state_dict().items()})
    manifest={"schema_version":"paper05.group3.v1","config":config,"head_cells":len(motifs),"intervention_rows":len(rows),"pair_rows":len(pair_rows),"base":base_stats,"association":association,"checkpoint_hash":checkpoint_hash,"artifact_hash":stable_hash({"rows":rows,"pairs":pair_rows,"covariance":covariance,"motifs":motifs})}
    atomic_write_json(output/"manifests/group3.json",manifest);print(json.dumps(manifest,indent=2))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--group",type=int,choices=(1,2,3),required=True)
    parser.add_argument("--config",required=True); parser.add_argument("--repo",default=".")
    parser.add_argument("--output",default="docs/papers/paper0_5/results/three_parts")
    parser.add_argument("--device",default="mps" if torch.backends.mps.is_available() else "cpu"); args=parser.parse_args()
    config=json.loads(Path(args.config).read_text()); output=Path(args.output); (output/"manifests").mkdir(parents=True,exist_ok=True)
    if args.group == 1: run_group1(config,Path(args.repo).resolve(),output,torch.device(args.device))
    elif args.group == 2: run_group2(config,Path(args.repo).resolve(),output,torch.device(args.device))
    else: run_group3(config,Path(args.repo).resolve(),output,torch.device(args.device))


if __name__ == "__main__": main()
