"""Run Paper 0.6 controlled abstraction-layer mapping experiments (E1--E8)."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cl.analysis.attention_motifs import motif_stability
from cl.analysis.hierarchy_geometry import hierarchy_metrics, shared_update_metrics
from cl.analysis.equivalence import common_component_metrics, distribution_metrics, jensen_shannon_bits
from cl.common.artifacts import RunMetadata, atomic_write_json, stable_hash, write_csv, write_jsonl
from cl.common.hooks import final_position_trace
from cl.common.metrics import bootstrap_ci, onset_persistence
from cl.common.model_adapter import Intervention, TinyTransformerLM, train_step
from cl.experiments.paper05_ngram import MODEL_SETTINGS, collect_motifs, component_rows, set_seed
from cl.ngram.atlas import build_atlas
from cl.semantic.synthetic import SemanticProbe, build_semantic_corpus


@torch.no_grad()
def collect_representations(model, probes: tuple[SemanticProbe, ...], device, identity):
    model.eval()
    inputs = torch.tensor([probe.tokens for probe in probes], dtype=torch.long, device=device)
    _, trace = model(inputs, capture=True)
    if trace is None:
        raise RuntimeError("missing semantic trace")
    compact = final_position_trace(trace)
    representations, updates = [], []
    for layer_index, layer in enumerate(compact):
        for index, probe in enumerate(probes):
            labels = {
                **identity,
                "example_id": probe.example_id,
                "item_id": probe.item_id,
                "family": probe.family,
                "natural": probe.natural,
                "abstraction_level": probe.abstraction_level,
                "template_id": probe.template_id,
                "layer": layer_index,
            }
            for location in ("pre_sa", "post_sa", "post_block"):
                representations.append({**labels, "location": location, "vector": layer[location][index].numpy().tolist()})
            for component, name in (("sa", "delta_sa"), ("ff", "delta_ff")):
                updates.append({**labels, "component": component, "vector": layer[name][index].numpy().tolist()})
    return representations, updates


@torch.no_grad()
def competence_rows(model, probes, device, identity, step):
    inputs=torch.tensor([p.tokens for p in probes],dtype=torch.long,device=device)
    targets=torch.tensor([p.target for p in probes],dtype=torch.long,device=device)
    logits,_=model(inputs); logp=torch.log_softmax(logits[:,-1],-1)
    rows=[]
    for i,p in enumerate(probes):
        rows.append({**identity,"step":step,"example_id":p.example_id,"family":p.family,"natural":p.natural,
                     "abstraction_level":p.abstraction_level,"template_id":p.template_id,
                     "correct":int(logp[i].argmax()==targets[i]),"target_logprob":float(logp[i,targets[i]])})
    return rows


@torch.no_grad()
def collect_semantic_equivalence(model, corpus, device, identity):
    probes=corpus.probes; inputs=torch.tensor([p.tokens for p in probes],dtype=torch.long,device=device)
    logits,trace=model(inputs,capture=True); assert trace is not None
    probabilities=torch.softmax(logits[:,-1],-1).cpu().numpy(); compact=final_position_trace(trace)
    entropy_rows=[]; patch_rows=[]; updates=[]; family_rows=[]
    for i,p in enumerate(probes):
        item=corpus.hierarchy.items[p.item_id]
        sibling=next(j for j,q in enumerate(probes) if q.abstraction_level==p.abstraction_level and q.template_id==p.template_id
                     and q.item_id!=p.item_id and corpus.hierarchy.items[q.item_id].parent_id==item.parent_id)
        cross=next(j for j,q in enumerate(probes) if q.abstraction_level==p.abstraction_level and q.template_id==p.template_id
                   and q.family!=p.family)
        family_rows.append({**identity,"example_id":p.example_id,"family":p.family,"natural":p.natural,
                            "abstraction_level":p.abstraction_level,"sibling_js_bits":jensen_shannon_bits(probabilities[i],probabilities[sibling]),
                            "cross_category_js_bits":jensen_shannon_bits(probabilities[i],probabilities[cross])})
        previous=None
        for layer,values in enumerate(compact):
            for location in ("pre_sa","post_sa","post_block"):
                decoded=torch.softmax(model.diagnostic_logits(values[location][i:i+1].to(device)),-1)[0].cpu().numpy()
                row=distribution_metrics(decoded,p.target)
                if previous is not None:
                    row["delta_entropy_bits"]=float(row["entropy_bits"]-previous["entropy_bits"])
                    row["delta_target_probability"]=float(row["target_probability"]-previous["target_probability"])
                entropy_rows.append({**identity,"example_id":p.example_id,"family":p.family,"natural":p.natural,
                                     "abstraction_level":p.abstraction_level,"template_id":p.template_id,"layer":layer,"location":location,**row})
                previous=row
            for component,name in (("sa","delta_sa"),("ff","delta_ff")):
                updates.append({**identity,"example_id":p.example_id,"parent_id":item.parent_id,"family":p.family,
                                "natural":p.natural,"abstraction_level":p.abstraction_level,"layer":layer,"component":component,
                                "vector":values[name][i].numpy().tolist()})
                full=getattr(trace.layers[layer],name)
                for donor_type,donor in (("sibling",sibling),("cross_category",cross)):
                    changed,_=model(inputs[i:i+1],intervention=Intervention(layer,component,"replace",full[donor:donor+1]))
                    changed_p=torch.softmax(changed[0,-1],-1).cpu().numpy()
                    patch_rows.append({**identity,"example_id":p.example_id,"family":p.family,"natural":p.natural,
                                       "abstraction_level":p.abstraction_level,"layer":layer,"component":component,"donor_type":donor_type,
                                       "output_js_bits":jensen_shannon_bits(probabilities[i],changed_p),
                                       "target_probability_change":float(changed_p[p.target]-probabilities[i,p.target])})
    return entropy_rows,patch_rows,updates,family_rows


def train_semantic(setting, seed, corpus, steps, checkpoints, device, checkpoint_dir):
    set_seed(seed)
    model = TinyTransformerLM(
        corpus.vocab_size,
        corpus.max_length,
        width=setting["width"],
        layers=setting["layers"],
        heads=setting["heads"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    data = torch.tensor(corpus.train_sequences, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    checkpoint_rows, competence, losses = [], [], []
    for step in range(steps + 1):
        if step in checkpoints:
            identity = {"run_id": f"paper06-{setting['name']}-seed{seed}", "model_setting": setting["name"], "seed": seed}
            reps, _ = collect_representations(model, corpus.probes, device, identity)
            geometry = hierarchy_metrics(corpus.hierarchy, reps, seed=seed)
            for row in geometry:
                checkpoint_rows.append({"step": step, **row})
            competence.extend(competence_rows(model, corpus.probes, device, identity, step))
            if setting["name"] == MODEL_SETTINGS[0]["name"] and seed == 11:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"step": step, "setting": setting, "seed": seed, "state_dict": model.state_dict()}, checkpoint_dir / f"step-{step:04d}.pt")
        if step == steps:
            break
        indices = torch.randint(len(data), (48,), generator=generator)
        losses.append(train_step(model, data[indices].to(device), optimizer))
    return model, checkpoint_rows, competence, losses


def attach_semantic_labels(rows, probes):
    lookup = {probe.example_id: probe for probe in probes}
    output = []
    for row in rows:
        probe = lookup[row["example_id"]]
        output.append({
            **row,
            "item_id": probe.item_id,
            "family": probe.family,
            "natural": probe.natural,
            "abstraction_level": probe.abstraction_level,
            "template_id": probe.template_id,
        })
    return output


def aggregate_geometry(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["layer"], row["location"], row["family"], row["natural"], row["abstraction_level"])].append(row)
    output = []
    metrics = ["within_parent_dispersion", "between_parent_dispersion", "normalized_separation", "hierarchy_rsa_spearman", "tree_neighbor_recovery", "permuted_neighbor_recovery", "cross_template_cosine", "effective_rank"]
    for key, values in sorted(groups.items()):
        row = {"layer": key[0], "location": key[1], "family": key[2], "natural": key[3], "abstraction_level": key[4], "n_runs": len(values)}
        for metric in metrics:
            estimate, low, high = bootstrap_ci([value[metric] for value in values], samples=500, seed=13)
            row[f"mean_{metric}"] = estimate; row[f"{metric}_ci_low"] = low; row[f"{metric}_ci_high"] = high
        output.append(row)
    return output


def aggregate_components(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["family"], row["natural"], row["abstraction_level"], row["component"], row["layer"])].append(row)
    return [
        {
            "family": key[0], "natural": key[1], "abstraction_level": key[2], "component": key[3], "layer": key[4], "n": len(values),
            "mean_zero_ablation_drop": float(np.mean([row["zero_ablation_logprob_drop"] for row in values])),
            "mean_replacement_drop": float(np.mean([row["matched_replacement_logprob_drop"] for row in values])),
            "mean_diagnostic_progress": float(np.mean([row["diagnostic_signed_progress"] for row in values])),
        }
        for key, values in sorted(groups.items())
    ]


def semantic_atlas(corpus):
    unigram = build_atlas(corpus.train_sequences, n_values=(1,), source_corpus="controlled-semantic-v1")
    token_stats = {entry.prefix_token_ids[0]: entry for entry in unigram}
    rows = []
    for item in sorted(corpus.hierarchy.items.values(), key=lambda value: value.item_id):
        stats = token_stats.get(item.token_id)
        rows.append({
            "item_id": item.item_id,
            "decoded_label": item.label,
            "token_ids": [item.token_id],
            "token_count": 1,
            "semantic_family": item.semantic_family,
            "domain": item.domain,
            "hierarchy_path": list(corpus.hierarchy.path(item.item_id)),
            "parent_ids": list(corpus.hierarchy.ancestors(item.item_id)),
            "depth": item.depth,
            "siblings": list(corpus.hierarchy.siblings(item.item_id)),
            "natural": item.natural,
            "training_frequency": stats.reference_corpus_frequency if stats else 0,
            "reference_corpus_frequency": stats.reference_corpus_frequency if stats else 0,
            "continuation_entropy": stats.continuation_entropy if stats else 0.0,
            "top_continuation_probability": stats.top_continuation_probability if stats else 0.0,
            "hierarchy_hash": corpus.hierarchy.version_hash,
            "paper05_schema": "ngram.atlas.v1-compatible",
        })
    return rows


def onset_rows(geometry_summary):
    output = []
    for natural in (False, True):
        values = sorted(
            [row for row in geometry_summary if row["location"] == "post_block" and row["natural"] == natural],
            key=lambda row: row["layer"],
        )
        by_layer = defaultdict(list)
        for row in values:
            by_layer[row["layer"]].append(row["mean_tree_neighbor_recovery"] - row["mean_permuted_neighbor_recovery"])
        layers = sorted(by_layer)
        mask = [np.mean(by_layer[layer]) > 0 for layer in layers]
        output.append({"natural": natural, **onset_persistence(mask), "layers": layers, "criterion": "neighbor recovery above permuted control"})
    return output


def plots(output, geometry, components, motifs, updates, checkpoints, onset):
    figure_dir = output / "figures"; figure_dir.mkdir(parents=True, exist_ok=True)
    # 1 concept
    fig, ax = plt.subplots(figsize=(8, 4)); ax.axis("off")
    for x, text, color in ((.08, "instance", "#dbeafe"), (.39, "category", "#dcfce7"), (.70, "superclass", "#fef3c7")):
        ax.text(x, .62, text, fontsize=14, bbox=dict(boxstyle="round", fc=color))
    ax.annotate("", (.37,.66),(.22,.66),arrowprops=dict(arrowstyle="->",lw=2)); ax.annotate("",(.68,.66),(.54,.66),arrowprops=dict(arrowstyle="->",lw=2))
    ax.text(.5,.25,"SA / FFN / residual trajectory + causal intervention",ha="center"); fig.tight_layout(); fig.savefig(figure_dir/"concept.png",dpi=170); plt.close(fig)
    # 2 hierarchy design
    fig, ax = plt.subplots(figsize=(8,4.5)); ax.axis("off")
    ax.text(.5,.88,"entity",ha="center",bbox=dict(boxstyle="round",fc="#fef3c7")); ax.text(.27,.58,"bird",ha="center",bbox=dict(boxstyle="round",fc="#dcfce7")); ax.text(.73,.58,"mammal",ha="center",bbox=dict(boxstyle="round",fc="#dcfce7"))
    for x,label in ((.16,"falcon"),(.38,"penguin"),(.62,"dog"),(.84,"cat")): ax.text(x,.25,label,ha="center",bbox=dict(boxstyle="round",fc="#dbeafe"))
    for a,b in [((.5,.84),(.27,.64)),((.5,.84),(.73,.64)),((.27,.54),(.16,.31)),((.27,.54),(.38,.31)),((.73,.54),(.62,.31)),((.73,.54),(.84,.31))]: ax.annotate("",b,a,arrowprops=dict(arrowstyle="->"))
    fig.tight_layout(); fig.savefig(figure_dir/"hierarchy_design.png",dpi=170); plt.close(fig)

    def mean_by(filters, metric):
        selected=[row for row in geometry if all(row[k]==v for k,v in filters.items())]; by=defaultdict(list)
        for row in selected: by[row["layer"]].append(row[metric])
        return sorted(by), [np.mean(by[layer]) for layer in sorted(by)]
    # 3 invariance heatmap
    locations=["pre_sa","post_sa","post_block"]; layers=sorted({row["layer"] for row in geometry}); matrix=np.full((len(locations),len(layers)),np.nan)
    for i,loc in enumerate(locations):
        for j,layer in enumerate(layers): matrix[i,j]=np.mean([r["mean_normalized_separation"] for r in geometry if r["location"]==loc and r["layer"]==layer])
    fig,ax=plt.subplots(figsize=(7,4)); image=ax.imshow(matrix,aspect="auto",cmap="viridis"); ax.set_yticks(range(len(locations)),locations); ax.set_xticks(range(len(layers)),layers); ax.set_xlabel("layer"); ax.set_title("Layer x location hierarchy separation"); fig.colorbar(image,ax=ax); fig.tight_layout(); fig.savefig(figure_dir/"invariance_heatmap.png",dpi=170); plt.close(fig)
    # 4 RSA, 5 within-between, 8 onset, 10 natural synthetic, 11 controls
    for metric, filename, title in [
        ("mean_hierarchy_rsa_spearman","hierarchy_rsa.png","Hierarchy distance / activation distance RSA"),
        ("mean_cross_template_cosine","template_invariance.png","Cross-template invariance"),
    ]:
        fig,ax=plt.subplots(figsize=(7.5,4.5))
        for natural,label in ((True,"natural"),(False,"synthetic")):
            x,y=mean_by({"location":"post_block","natural":natural},metric); ax.plot(x,y,marker="o",label=label)
        ax.axhline(0,color="black",lw=.7); ax.set_xlabel("layer"); ax.set_ylabel(metric.replace("mean_","")); ax.set_title(title); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figure_dir/filename,dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4.5))
    for metric,label in (("mean_within_parent_dispersion","within"),("mean_between_parent_dispersion","between")):
        x,y=mean_by({"location":"post_block"},metric); ax.plot(x,y,marker="o",label=label)
    ax.set_xlabel("layer"); ax.set_ylabel("cosine distance"); ax.set_title("Within vs between dispersion"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figure_dir/"within_between.png",dpi=170); plt.close(fig)
    # 6 components
    fig,ax=plt.subplots(figsize=(8,4.8))
    for level in ("parent","root"):
        for component in ("sa","ff"):
            selected=[r for r in components if r["abstraction_level"]==level and r["component"]==component]; by=defaultdict(list)
            for row in selected: by[row["layer"]].append(row["mean_zero_ablation_drop"])
            ax.plot(sorted(by),[np.mean(by[x]) for x in sorted(by)],marker="o",label=f"{level}/{component}")
    ax.axhline(0,color="black",lw=.7); ax.set_xlabel("layer"); ax.set_ylabel("causal log-probability drop"); ax.set_title("SA/FFN hierarchy contribution"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figure_dir/"component_contributions.png",dpi=170); plt.close(fig)
    # 7 motifs
    fig,ax=plt.subplots(figsize=(8,4.8))
    for relation in sorted({r["relation_id"] for r in motifs}):
        selected=sorted([r for r in motifs if r["relation_id"]==relation],key=lambda r:r["layer"]); ax.plot([r["layer"] for r in selected],[r["motif_specificity"] for r in selected],marker="o",label=relation)
    ax.axhline(0,color="black",lw=.7); ax.set_xlabel("layer"); ax.set_ylabel("motif specificity"); ax.set_title("Semantic motif invariance vs cross-category control"); ax.legend(fontsize=6,ncol=2); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figure_dir/"semantic_motifs.png",dpi=170); plt.close(fig)
    # 8 onset/persistence
    fig,ax=plt.subplots(figsize=(6,4)); labels=["synthetic","natural"]; persistence=[next(r["persistence"] for r in onset if r["natural"]==v) for v in (False,True)]; ax.bar(labels,persistence,color=["#60a5fa","#34d399"]); ax.set_ylabel("consecutive layers above permuted control"); ax.set_title("Invariant persistence"); fig.tight_layout(); fig.savefig(figure_dir/"onset_persistence.png",dpi=170); plt.close(fig)
    # 9 contextual resolution
    fig,ax=plt.subplots(figsize=(7.5,4.5))
    for level in ("parent","root"):
        selected=[r for r in components if r["abstraction_level"]==level]; by=defaultdict(list)
        for row in selected: by[row["layer"]].append(row["mean_zero_ablation_drop"])
        ax.plot(sorted(by),[np.mean(by[x]) for x in sorted(by)],marker="o",label=level)
    ax.axhline(0,color="black",lw=.7); ax.set_xlabel("layer"); ax.set_ylabel("causal contribution"); ax.set_title("Fine vs coarse contextual resolution"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figure_dir/"contextual_resolution.png",dpi=170); plt.close(fig)
    # 10 natural vs synthetic / 11 permuted control
    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    for natural,label in ((True,"natural"),(False,"synthetic")):
        x,y=mean_by({"location":"post_block","natural":natural},"mean_normalized_separation"); axes[0].plot(x,y,marker="o",label=label)
    axes[0].set_title("Natural vs synthetic separation"); axes[0].set_xlabel("layer"); axes[0].legend(); axes[0].grid(alpha=.25)
    x,y=mean_by({"location":"post_block"},"mean_tree_neighbor_recovery"); _,p=mean_by({"location":"post_block"},"mean_permuted_neighbor_recovery"); axes[1].plot(x,y,marker="o",label="hierarchy"); axes[1].plot(x,p,marker="o",label="permuted"); axes[1].set_title("N-gram/template-matched control"); axes[1].set_xlabel("layer"); axes[1].legend(); axes[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(figure_dir/"natural_synthetic_controls.png",dpi=170); plt.close(fig)
    # 12 training dynamics
    fig,ax=plt.subplots(figsize=(8,4.8)); by=defaultdict(list)
    for row in checkpoints:
        if row["location"]=="post_block": by[row["step"]].append(row["normalized_separation"])
    ax.plot(sorted(by),[np.mean(by[x]) for x in sorted(by)],marker="o"); ax.axhline(0,color="black",lw=.7); ax.set_xlabel("training step"); ax.set_ylabel("normalized separation"); ax.set_title("Hierarchy geometry over training"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figure_dir/"training_dynamics.png",dpi=170); plt.close(fig)
    # 13 update commonality (extra)
    fig,ax=plt.subplots(figsize=(7.5,4.5)); by=defaultdict(list)
    for row in updates: by[(row["layer"],row["component"])].append(row["shared_update_explained_variance"])
    for component in ("sa","ff"):
        layers=sorted({key[0] for key in by if key[1]==component}); ax.plot(layers,[np.mean(by[(layer,component)]) for layer in layers],marker="o",label=component)
    ax.set_xlabel("layer"); ax.set_ylabel("shared-update explained variance"); ax.set_title("Category-conditioned residual commonality"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figure_dir/"update_commonality.png",dpi=170); plt.close(fig)


def write_latex_table(table_dir, geometry, components):
    rows=[]
    for natural,label in ((True,"Natural"),(False,"Synthetic")):
        selected=[r for r in geometry if r["location"]=="post_block" and r["natural"]==natural]
        rows.append((label,np.mean([r["mean_normalized_separation"] for r in selected]),np.mean([r["mean_hierarchy_rsa_spearman"] for r in selected]),np.mean([r["mean_tree_neighbor_recovery"] for r in selected]),np.mean([r["mean_permuted_neighbor_recovery"] for r in selected]),np.mean([r["mean_cross_template_cosine"] for r in selected])))
    lines=["\\begin{table}[H]\\centering\\small","\\begin{tabular}{lrrrrr}\\toprule","Hierarchy & Separation & RSA & Neighbor & Permuted & Template inv. \\\\"+"\\midrule"]
    for row in rows: lines.append(f"{row[0]} & {row[1]:.3f} & {row[2]:.3f} & {row[3]:.3f} & {row[4]:.3f} & {row[5]:.3f} "+r"\\")
    lines += ["\\bottomrule\\end{tabular}","\\caption{Post-block hierarchy geometry averaged across layers, families, model settings, and seeds. Neighbor is nearest-neighbor parent recovery; Permuted shuffles parent identities under matched representations.}","\\label{tab:paper06-main}","\\end{table}"]
    (table_dir/"main_results.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")


def write_summary(output, config, geometry, components, motifs, checkpoints, losses):
    post=[r for r in geometry if r["location"]=="post_block"]
    motif_values=[r["motif_specificity"] for r in motifs if np.isfinite(r["motif_specificity"])]
    lines=["# Paper 0.6 controlled experiment summary","","## What was run","",f"- E1--E8 over two model settings x {len(config['seeds'])} seeds.","- Balanced noun, action, and relation hierarchies; natural labels, arbitrary synthetic labels, and permuted controls.",f"- Geometry aggregate rows: {len(geometry)}; component aggregate rows: {len(components)}.","","## Main pilot observations","",f"- Mean post-block normalized hierarchy separation: {np.mean([r['mean_normalized_separation'] for r in post]):.4f}.",f"- Mean post-block hierarchy RSA: {np.mean([r['mean_hierarchy_rsa_spearman'] for r in post]):.4f}.",f"- Tree-neighbor recovery versus permuted control: {np.mean([r['mean_tree_neighbor_recovery'] for r in post]):.4f} versus {np.mean([r['mean_permuted_neighbor_recovery'] for r in post]):.4f}.",f"- Mean cross-template cosine: {np.mean([r['mean_cross_template_cosine'] for r in post]):.4f}.",f"- Mean semantic motif specificity: {np.mean(motif_values):.4f}.",f"- Final training loss range: {min(v[-1] for v in losses.values()):.4f}--{max(v[-1] for v in losses.values()):.4f}.","","## Interpretation and failures","","This is a controlled local-model abstraction pilot, not evidence that pretrained Transformers encode a literal taxonomy. Templates, token length, training frequency, and continuation statistics are balanced by construction. Geometry is compared with permuted hierarchy and causal component interventions, but generic layer utility and off-manifold interventions remain limitations. Flat or non-monotonic profiles are retained.","","## Next falsifiable question","","Do hierarchy geometry and causal component effects survive residualization against natural-corpus n-gram statistics in a pretrained checkpoint series and transfer across unseen paraphrases?",""]
    (output/"summary.md").write_text("\n".join(lines),encoding="utf-8")


def run(args):
    repo=Path(args.repo).resolve(); output=Path(args.output).resolve(); raw=output/"raw"; tables=output/"tables"; raw.mkdir(parents=True,exist_ok=True); tables.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device); seeds=[int(v) for v in args.seeds.split(",")]; config={"seeds":seeds,"steps":args.steps,"checkpoints":args.checkpoints,"device":str(device),"models":MODEL_SETTINGS}
    all_reps=[]; all_updates=[]; all_components=[]; all_motif_records=[]; all_checkpoints=[]; metadata=[]; losses={}
    all_competence=[]; all_entropy=[]; all_patches=[]; all_equiv_updates=[]; all_families=[]
    atlas_corpus=build_semantic_corpus(seed=0,repeats=args.repeats); atlas=semantic_atlas(atlas_corpus); write_jsonl(raw/"semantic_atlas.jsonl",atlas)
    for setting in MODEL_SETTINGS:
        for seed in seeds:
            corpus=build_semantic_corpus(seed=seed,repeats=args.repeats); run_id=f"paper06-{setting['name']}-seed{seed}"; identity={"run_id":run_id,"model_setting":setting["name"],"seed":seed}
            metadata.append(RunMetadata.capture(repo=repo,run_id=run_id,config=config,model_id=setting["name"],dataset_id="controlled-semantic-v1",seed=seed,device=str(device),dtype="float32",data_hash=corpus.corpus_hash).as_dict())
            model,checkpoint_rows,competence,run_losses=train_semantic(setting,seed,corpus,args.steps,set(args.checkpoints),device,output/"checkpoints"); losses[run_id]=run_losses; all_checkpoints.extend(checkpoint_rows); all_competence.extend(competence)
            reps,updates=collect_representations(model,corpus.probes,device,identity); all_reps.extend(reps); all_updates.extend(updates)
            measured=component_rows(model,corpus.probes,device); all_components.extend([{**identity,**row} for row in attach_semantic_labels(measured,corpus.probes)])
            all_motif_records.extend([{**identity,**row} for row in collect_motifs(model,corpus.probes,device)])
            entropy,patches,equiv_updates,families=collect_semantic_equivalence(model,corpus,device,identity)
            all_entropy.extend(entropy); all_patches.extend(patches); all_equiv_updates.extend(equiv_updates); all_families.extend(families)
    geometry_raw=hierarchy_metrics(atlas_corpus.hierarchy,all_reps,seed=0); geometry=aggregate_geometry(geometry_raw); update_metrics=shared_update_metrics(atlas_corpus.hierarchy,all_updates); components=aggregate_components(all_components)
    motif_runs=defaultdict(list)
    for row in all_motif_records: motif_runs[row["run_id"]].append(row)
    motif_rows=[]
    for run_id,records in motif_runs.items(): motif_rows.extend([{"run_id":run_id,**row} for row in motif_stability(records)])
    motif_groups=defaultdict(list)
    for row in motif_rows: motif_groups[(row["relation_id"],row["layer"])].append(row)
    motifs=[{"relation_id":key[0],"layer":key[1],"motif_specificity":float(np.mean([r["motif_specificity"] for r in values])),"within_motif_cosine":float(np.mean([r["within_motif_cosine"] for r in values])),"matched_control_cosine":float(np.mean([r["matched_control_cosine"] for r in values])),"n_runs":len(values)} for key,values in sorted(motif_groups.items())]
    onset=onset_rows(geometry)
    final_comp=[r for r in all_competence if r["step"]==args.steps]
    competence_summary=[]
    for run_id in sorted({r["run_id"] for r in final_comp}):
        values=[r for r in final_comp if r["run_id"]==run_id]; accuracy=float(np.mean([r["correct"] for r in values]))
        competence_summary.append({"run_id":run_id,"heldout_accuracy":accuracy,"mean_target_logprob":float(np.mean([r["target_logprob"] for r in values])),"threshold":0.80,"competent":accuracy>=0.80,"n":len(values)})
    patch_groups=defaultdict(list)
    for row in all_patches: patch_groups[(row["donor_type"],row["component"],row["layer"])].append(row)
    patch_summary=[{"donor_type":k[0],"component":k[1],"layer":k[2],"mean_output_js_bits":float(np.mean([r["output_js_bits"] for r in v])),"mean_target_probability_change":float(np.mean([r["target_probability_change"] for r in v])),"n":len(v)} for k,v in sorted(patch_groups.items())]
    family_summary=[{"comparison":name,"mean_js_bits":float(np.mean([r[field] for r in all_families])),"n":len(all_families)} for name,field in (("sibling","sibling_js_bits"),("cross_category","cross_category_js_bits"))]
    equiv_common=common_component_metrics(all_equiv_updates,("model_setting","seed","layer","component","family","natural","abstraction_level","parent_id"))
    write_jsonl(raw/"representations.jsonl",all_reps); write_jsonl(raw/"updates.jsonl",all_updates); write_jsonl(raw/"components.jsonl",all_components); write_jsonl(raw/"motifs.jsonl",all_motif_records); write_jsonl(raw/"metadata.jsonl",metadata); atomic_write_json(raw/"losses.json",losses)
    write_jsonl(raw/"competence.jsonl",all_competence); write_jsonl(raw/"entropy_trajectories.jsonl",all_entropy); write_jsonl(raw/"semantic_patches.jsonl",all_patches); write_jsonl(raw/"semantic_equivalence.jsonl",all_families)
    write_csv(tables/"geometry.csv",geometry); write_csv(tables/"geometry_by_run.csv",geometry_raw); write_csv(tables/"component_summary.csv",components); write_csv(tables/"motif_summary.csv",motifs); write_csv(tables/"update_commonality.csv",update_metrics); write_csv(tables/"checkpoint_dynamics.csv",all_checkpoints); atomic_write_json(tables/"onset_persistence.json",onset)
    write_csv(tables/"competence_gate.csv",competence_summary); write_csv(tables/"semantic_equivalence.csv",family_summary); write_csv(tables/"semantic_patching.csv",patch_summary); write_csv(tables/"equivalence_common_updates.csv",equiv_common)
    plots(output,geometry,components,motifs,update_metrics,all_checkpoints,onset); write_latex_table(tables,geometry,components); write_summary(output,config,geometry,components,motifs,all_checkpoints,losses)
    atomic_write_json(output/"manifest.json",{"schema_version":"paper06.results.v2","config":config,"competence_threshold":0.80,"hierarchy_hash":atlas_corpus.hierarchy.version_hash,"paper05_manifest_hash":stable_hash(json.loads((repo/"docs/papers/paper0_5/results/manifest.json").read_text())),"artifact_hash":stable_hash({"geometry":geometry,"components":components,"motifs":motifs,"competence":competence_summary,"patches":patch_summary})})
    print(json.dumps({"output":str(output),"geometry_rows":len(geometry),"component_rows":len(all_components),"representation_rows":len(all_reps)},indent=2))


def parse_args():
    parser=argparse.ArgumentParser(); parser.add_argument("--repo",default="."); parser.add_argument("--output",default="docs/papers/paper0_6/results"); parser.add_argument("--device",default="cpu"); parser.add_argument("--seeds",default="11,23"); parser.add_argument("--steps",type=int,default=160); parser.add_argument("--repeats",type=int,default=24); parser.add_argument("--checkpoints",type=int,nargs="+",default=[0,40,100,160]); return parser.parse_args()


if __name__=="__main__": run(parse_args())
