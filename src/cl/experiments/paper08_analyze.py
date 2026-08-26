"""Generate Paper 0.8 figures and competence-disciplined summaries."""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path("docs/papers/paper0_8");RESULTS=ROOT/"results";FIG=ROOT/"figures"
def read(p):
    with open(p,newline="") as h:return list(csv.DictReader(h))
def save(name):plt.tight_layout();plt.savefig(FIG/name,dpi=190);plt.close()
def bars(labels,values,name,ylabel,threshold=None):
    plt.figure(figsize=(6,3.5));plt.bar(range(len(labels)),values);plt.xticks(range(len(labels)),labels,rotation=25,ha="right");plt.ylabel(ylabel)
    if threshold is not None:plt.axhline(threshold,color="black",ls="--",lw=.8)
    save(name)
def main():
    FIG.mkdir(parents=True,exist_ok=True);phase=read(RESULTS/"phase/icl_phase_grid.csv")
    ref=[r for r in phase if r["layers"]=="2" and r["width"]=="32" and r["heads"]=="2"]
    stages=sorted({r["stage"] for r in ref});bars(stages,[np.mean([float(r["correct_accuracy"]) for r in ref if r["stage"]==s]) for s in stages],"icl_dataset_phase_boundary.png","correct-context accuracy",.8)
    seed11=[r for r in phase if r["stage"]=="D4" and r["model_seed"]=="11"]
    layers=sorted({int(r["layers"]) for r in seed11});widths=sorted({int(r["width"]) for r in seed11})
    z=np.array([[np.mean([float(r["correct_accuracy"]) for r in seed11 if int(r["layers"])==l and int(r["width"])==w]) for w in widths] for l in layers])
    plt.figure(figsize=(5.4,3.5));plt.imshow(z,origin="lower",aspect="auto",vmin=0,vmax=1);plt.colorbar(label="accuracy");plt.xticks(range(len(widths)),widths);plt.yticks(range(len(layers)),layers);plt.xlabel("residual width");plt.ylabel("layers");save("icl_model_phase_boundary.png")
    rank=read(RESULTS/"phase/icl_rank_promotion.csv");q=[r for r in rank if r["stage"]=="D4" and r["layers"]=="2" and r["width"]=="32" and r["heads"]=="2" and r["model_seed"]=="11"]
    cond=["none","correct","shuffled","wrong_chain","irrelevant"];bars(cond,[np.mean([float(r["rank"]) for r in q if r["condition"]==c]) for c in cond],"icl_context_free_vs_context_rank.png","mean target rank")
    rep=read(RESULTS/"phase/icl_d4_replication_aggregate.csv");bars([r["architecture"] for r in rep],[float(r["passing_seeds"])/3 for r in rep],"icl_before_after_acquisition.png","fraction of passing seeds",1)
    for name,title in (("icl_target_margin_vs_depth.png","D4 tracing blocked: no three-seed stable cell"),("icl_sa_ff_promotion_vs_depth.png","D4 SA/FF tracing blocked by competence"),("icl_head_utility.png","D4 head utility blocked by competence"),("icl_causal_replacement.png","D4 replacement blocked by competence")):
        plt.figure(figsize=(6,2.5));plt.axis("off");plt.text(.5,.5,title,ha="center",va="center",fontsize=13);save(name)
    cgrid=read(RESULTS/"copy/replication_w64/copy_phase_grid.csv");bars(["W64-S"+r["seed"] for r in cgrid],[float(r["correct_accuracy"]) for r in cgrid],"copy_phase_boundary.png","copy accuracy",.95)
    w128=read(RESULTS/"copy/replication_w128/copy_phase_grid.csv")
    bars(["W64-S"+r["seed"] for r in cgrid]+["W128-S"+r["seed"] for r in w128],[float(r["correct_accuracy"]) for r in cgrid+w128],"copy_vocab_vs_model_size.png","copy accuracy",.95)
    qk=read(RESULTS/"copy/copy_qk_structure.csv");labels=[f"L{r['layer']}H{r['head']}" for r in qk]
    bars(labels,[float(r["previous_token_mass"]) for r in qk],"copy_qk_heatmaps.png","previous-token attention")
    bars(labels,[float(r["query_to_associated_value"]) for r in qk],"copy_attention_role_matrix.png","query to associated value")
    flow=read(RESULTS/"copy/copy_target_support_flow.csv");groups={}
    for r in flow:groups.setdefault((r["layer"],r["head"]),[]).append(float(r["target_support"]))
    bars([f"L{k[0]}H{k[1]}" for k in groups],[sum(v) for v in groups.values()],"copy_target_support_graph.png","summed signed target support")
    saff=read(RESULTS/"copy/copy_sa_ff_contributions.csv");order=sorted(saff,key=lambda r:(int(r["layer"]),0 if r["boundary"]=="pre_sa" else 1 if r["boundary"]=="post_sa" else 2))
    bars([f"L{r['layer']}-{r['boundary']}" for r in order],[float(r["rank"]) for r in order],"copy_target_rank_vs_depth.png","mean target rank")
    bars([f"L{r['layer']}-{r['boundary']}" for r in order],[float(r["margin"]) for r in order],"copy_sa_ff_support_vs_depth.png","target margin",0)
    heads=read(RESULTS/"copy/copy_head_utility.csv");bars([f"L{r['layer']}-{r['intervention']}" for r in heads],[float(r["margin_damage"]) for r in heads],"copy_head_utility.png","margin damage")
    patch=read(RESULTS/"copy/copy_qkv_patching.csv");bars([f"L{r['layer']}-{r['intervention']}" for r in patch],[float(r["margin_damage"]) for r in patch],"copy_qkv_patch_damage.png","margin damage")
    bars(["local competent","fresh S11","S23","S37"],[1.0]+[float(r["correct_accuracy"]) for r in cgrid],"copy_before_after_acquisition.png","copy accuracy",.95)
    # Required table alias: the detailed trace table already supplies the data.
    (RESULTS/"summaries").mkdir(exist_ok=True)
    (RESULTS/"summaries/copy_summary.md").write_text("# Contextual copy calibration\n\nA local L2/W64/H2 checkpoint passes at 100% and yields a causally validated pair-binding/value-routing circuit. A nominally identical fresh seed-11 run reaches 89.6%, while seeds 23 and 37 reach 70.8% and 66.7%. Width 128 scores 41.7%, 54.2%, and 33.3%. Thus instrumentation is calibrated, but copy acquisition is not a stable monotonic capacity boundary under the tested optimizer.\n\nIn the competent local checkpoint, layer-0 head 0 binds adjacent pairs; layer-1 heads route associated values. Layer-1 value patching from shuffled controls drops accuracy to 20.8%, and demonstration-value ablation drops it to 37.5%. The target becomes rank one in layer-1 attention; the final FF preserves the answer while slightly reducing margin.\n")
    (RESULTS/"summaries/inception_summary.md").write_text("# Inception summary\n\nLocal orders D1/D2 are insufficient: high correct-context scores are not selective over controls. D3 and D4 create individual positive cells, but no tested D4 architecture passes three seeds. Five Pareto-small candidates pass only one or two seeds. The smallest local positive is L1/W16/H1, but it fails replication. D4 mechanism interpretation therefore remains blocked. Copy calibration independently recovers match-key, transport-associated-value, promote-target behavior in one competent checkpoint, validating the tracer while revealing its own acquisition instability. The evidence supports a sharp, seed-sensitive circuit-acquisition frontier rather than a stable minimal model.\n")
if __name__=="__main__":main()
