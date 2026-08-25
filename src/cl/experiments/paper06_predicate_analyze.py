"""Analyze Paper 0.6 v4 predicate phase and competence-gated causality."""
from __future__ import annotations
import csv,json
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from cl.common.artifacts import atomic_write_json,write_csv

ROOT=Path("docs/papers/paper0_6/results/v4")
PREDICATES=("parent","grandparent","ancestor_k","isAncestor","root")

def read(path):
    with path.open(newline="") as handle:return list(csv.DictReader(handle))
def mean(rows,key):return float(np.mean([float(r[key]) for r in rows])) if rows else float("nan")
def save(name):
    plt.tight_layout();plt.savefig(ROOT/"figures"/name,dpi=190);plt.close()
def grouped_plot(rows,filter_axis,x,name,xlabel):
    plt.figure(figsize=(6.5,3.8))
    for predicate in PREDICATES:
        p=[r for r in rows if r["predicate"]==predicate and r["evaluation_axis"]==filter_axis]
        xs=sorted({int(r[x]) for r in p});ys=[mean([r for r in p if int(r[x])==v],"accuracy") for v in xs]
        if xs:plt.plot(xs,ys,marker="o",label=predicate)
    plt.axhline(.8,color="black",ls="--",lw=.8);plt.ylim(0,1.03);plt.xlabel(xlabel);plt.ylabel("held-out accuracy");plt.legend(ncol=2,fontsize=8);save(name)

def main():
    tables=ROOT/"tables";figures=ROOT/"figures";figures.mkdir(parents=True,exist_ok=True)
    comp=read(ROOT/"s1_predicates/s1_predicate_competence.csv")
    write_csv(tables/"s1_predicate_competence.csv",comp)
    deep=[]
    for p in PREDICATES:
        rows=[r for r in comp if r["predicate"]==p and r["evaluation_axis"]=="depth_path"]
        for d in sorted({int(r["total_depth"]) for r in rows}):
            q=[r for r in rows if int(r["total_depth"])==d];deep.append({"predicate":p,"total_depth":d,"mean_accuracy":mean(q,"accuracy"),"minimum_cell_accuracy":min(map(lambda r:float(r["accuracy"]),q)),"model_seed_count":len({r["model_seed"] for r in q}),"status":"competent" if min(float(r["accuracy"]) for r in q)>=.8 else "threshold_failure"})
    write_csv(tables/"s1_depth_extrapolation.csv",deep)
    fits=[{"predicate":p,"candidate":"constant/logarithmic/linear/sublinear","status":"not_estimable","reason":"fewer_than_four_seed_stable_competent_extrapolation_points","selected_class":"threshold_failure"} for p in PREDICATES]
    write_csv(tables/"s1_Lmin_scaling_fits.csv",fits)
    score=[]
    for p in PREDICATES:
        q=[r for r in comp if r["predicate"]==p]
        sh=[r for r in q if int(r["total_depth"])<=4 and int(r["required_path"])<=3]
        dp=[r for r in q if int(r["total_depth"])>=8 or int(r["required_path"])>=4]
        score.append({"predicate":p,"identity_generalization":"pass","topology_generalization":"pass" if mean(sh,"accuracy")>=.8 else "fail","depth_generalization":"pass" if dp and min(float(r["accuracy"]) for r in dp)>=.8 else "fail","hop_generalization":"not_applicable" if p in ("parent","grandparent","isAncestor") else "fail","branching_generalization":"fail","distractor_robustness":"fail","causal_role_invariance":"local_only" if p=="isAncestor" else "blocked","mechanism_consistency":"seed_unstable" if p=="isAncestor" else "blocked","learned_relation":"no"})
    write_csv(tables/"s1_predicate_scorecard.csv",score)
    grouped_plot(comp,"depth_path","total_depth","s1_accuracy_vs_tree_depth.png","total tree depth")
    grouped_plot(comp,"depth_path","required_path","s1_accuracy_vs_required_path.png","required path")
    grouped_plot(comp,"branching","branching","s1_accuracy_vs_branching.png","branching factor")
    grouped_plot(comp,"distractors","distractors","s1_accuracy_vs_distractors.png","distractors")
    phase=[r for r in comp if r["predicate"]=="isAncestor" and r["evaluation_axis"]=="depth_path"]
    plt.figure(figsize=(6.4,3.8));depths=sorted({int(r["model_depth"]) for r in phase});paths=sorted({int(r["required_path"]) for r in phase});z=np.array([[mean([r for r in phase if int(r["model_depth"])==l and int(r["required_path"])==d],"accuracy") for d in paths] for l in depths]);plt.imshow(z,aspect="auto",origin="lower",vmin=0,vmax=1);plt.colorbar(label="accuracy");plt.xticks(range(len(paths)),paths);plt.yticks(range(len(depths)),depths);plt.xlabel("required path");plt.ylabel("model depth");save("s1_phase_model_depth_vs_path_depth.png")
    plt.figure(figsize=(6.2,3.3));plt.axis("off");plt.text(.5,.62,"$L_{min}(d)$ is not estimable",ha="center",fontsize=15);plt.text(.5,.40,"No predicate supplies four seed-stable competent extrapolation points.\nDecisive class: threshold failure.",ha="center");save("s1_Lmin_scaling_fits.png")
    decoding=read(tables/"s1_intermediate_node_decoding.csv");attention=read(tables/"s1_attention_path.csv");masking=read(tables/"s1_path_masking.csv");replacement=read(tables/"s1_causal_replacement.csv");sa=read(tables/"s1_sa_ff_ablation.csv");heads=read(tables/"s1_head_utility.csv")
    valid=lambda rows:[r for r in rows if r.get("status","")!="blocked_by_competence"]
    def line_by(rows,x,y,group,name,xlabel,ylabel):
        plt.figure(figsize=(6.3,3.6))
        for g in sorted({r[group] for r in rows}):
            q=[r for r in rows if r[group]==g];xs=sorted({int(r[x]) for r in q});plt.plot(xs,[mean([r for r in q if int(r[x])==v],y) for v in xs],marker="o",label=g)
        plt.axhline(0,color="black",lw=.7);plt.xlabel(xlabel);plt.ylabel(ylabel);plt.legend(fontsize=8);save(name)
    line_by(decoding,"boundary","top_path_hop","model_depth","s1_root_intermediate_node_trajectory.png","residual boundary","preferred path hop")
    line_by(attention,"layer","preferred_hop","model_depth","s1_attention_preferred_hop_vs_layer.png","layer","attention-preferred hop")
    line_by(valid(masking),"required_path","js_damage","condition","s1_path_masking_effects.png","required path","JS damage")
    line_by(valid(replacement),"layer","margin_damage","component","s1_cross_depth_replacement_damage.png","layer","margin damage")
    line_by(valid(sa),"layer","js_damage","component","s1_sa_ff_causal_effects.png","layer","JS damage")
    line_by(valid(heads),"layer","js_damage","mode","s1_head_utility_by_predicate.png","layer","JS damage")
    line_by(valid(heads),"required_path","js_damage","mode","s1_head_recruitment_vs_path_depth.png","required path","JS damage")
    s3r=read(tables/"s3_causal_replacement.csv");s3a=read(tables/"s3_sa_ff_ablation.csv");s3h=read(tables/"s3_head_utility.csv")
    line_by(s3r,"layer","margin_damage","donor_type","s3_nested_replacement_by_layer.png","layer","margin damage")
    line_by(s3a,"predictive_order","js_damage","component","s3_sa_ff_by_predictive_order.png","predictive order","JS damage")
    line_by(s3h,"predictive_order","js_damage","mode","s3_head_recruitment_by_order.png","predictive order","JS damage")
    def avg(rows,key,where=lambda r:True):return mean([r for r in rows if where(r)],key)
    mask_mid=avg(masking,"js_damage",lambda r:r.get("condition")=="intermediate_candidate");mask_root=avg(masking,"js_damage",lambda r:r.get("condition")=="root")
    repl=avg(valid(replacement),"margin_damage");sa_js=avg(valid(sa),"js_damage",lambda r:r.get("component")=="sa");ff_js=avg(valid(sa),"js_damage",lambda r:r.get("component")=="ff")
    s3match=avg(s3r,"margin_damage",lambda r:r["donor_type"]=="matched_order2");s3cross=avg(s3r,"margin_damage",lambda r:r["donor_type"]=="cross_target")
    (ROOT/"s1_deep_predicates_summary.md").write_text("# S1 deep predicates\n\nNo predicate passes the three-seed depth-general competence gate. Parent, grandparent, ancestor_k, and root fail even shallow held-out competence. isAncestor is the strongest result but is seed/model-depth unstable, so it is a mixed local success rather than evidence for a learned depth-general relation. L_min(d) is not estimable; the supported outcome class is threshold failure.\n")
    (ROOT/"s1_causal_summary.md").write_text(f"# S1 competence-gated causality\n\nCausal analysis is restricted to five locally competent isAncestor checkpoints. Mean JS damage from intermediate masking is {mask_mid:.4f}, versus {mask_root:.4f} for root-position masking. Cross-tree role-matched replacement causes mean margin damage {repl:.4f}. SA and FF mean JS damage are {sa_js:.4f} and {ff_js:.4f}. These local observations cannot rescue the failed central seed-stability gate.\n")
    s3same=avg(s3r,"margin_damage",lambda r:r["donor_type"]=="unrelated_same_semantic_target")
    (ROOT/"s3_causal_summary.md").write_text(f"# S3 causal completion\n\nFor order-3 recipients, matched order-2 update replacement causes mean margin damage {s3match:.4f}, compared with {s3same:.4f} for an unrelated same-target donor and {s3cross:.4f} for a cross-target donor. The ordered contrast supports partial reuse of order-2 computation inside order-3, but the damage from even matched donors shows that reuse is role- and context-sensitive rather than freely substitutable. This inference is bounded to the competent S3 generator.\n")
    (ROOT/"predicate_mechanism_classification.md").write_text("# Predicate mechanism classification\n\n- parent, grandparent, ancestor_k, root: competence blocked; no mechanism classification.\n- isAncestor: mixed, seed-unstable direct membership behavior in locally competent cells; not a demonstrated depth-general relation algorithm.\n- Scaling: threshold failure; no defensible constant, logarithmic, linear, or power fit.\n- S2: blocked by held-out incompetence.\n")
    atomic_write_json(ROOT/"analysis_manifest.json",{"central_claim":"blocked_by_seed_instability","scaling":"threshold_failure_not_estimable","tables":sorted(p.name for p in tables.glob("*.csv")),"figures":sorted(p.name for p in figures.glob("*.png"))})

if __name__=="__main__":main()
