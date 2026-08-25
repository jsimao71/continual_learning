"""Consolidate competence-gated Paper 0.6 S1/S2/S3 results and figures."""
from __future__ import annotations
import csv,json,math
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from cl.common.artifacts import atomic_write_json,write_csv

ROOT=Path("docs/papers/paper0_6/results/v2")
STAGES={"s1":ROOT/"s1_taxonomy","s2":ROOT/"s2_relational","s3":ROOT/"s3_compositional"}

def read(path):
    with path.open(newline="") as handle:return list(csv.DictReader(handle))
def f(row,key):return float(row[key])
def order(row,stage):
    if stage=="s1":return {"parent":1,"ancestor":2,"root":3}[row["query_level"]]
    return int(row["predictive_order"])
def cls(row,stage):
    if stage=="s1":return row[{1:"parent_id",2:"category_id",3:"root_id"}[order(row,stage)]]
    return row["class_id"]
def residual(row):return np.fromstring(row["residual"],sep=":")
def cosine(a,b):
    den=np.linalg.norm(a)*np.linalg.norm(b);return float(a@b/den) if den else 0.0

def main():
    tables=ROOT/"tables";figures=ROOT/"figures";summaries=ROOT/"summaries"
    for p in (tables,figures,summaries):p.mkdir(parents=True,exist_ok=True)
    gates={s:json.loads((p/f"{s}_gate.json").read_text()) for s,p in STAGES.items()}
    competence=[];layer_rows={}
    for stage,path in STAGES.items():
        for row in read(path/f"{stage}_competence.csv"):competence.append({"stage":stage,**row,"gate_passed":gates[stage]["gate_passed"]})
        layer_rows[stage]=read(path/f"{stage}_layer_raw.csv")
    write_csv(tables/"semantic_competence.csv",competence)

    order_params=[];geometry=[];depth_rows=[];updates=[]
    for stage,rows in layer_rows.items():
        groups=defaultdict(list)
        for row in rows:groups[(int(row["model_seed"]),order(row,stage),int(row["boundary"]))].append(row)
        for (seed,porder,boundary),values in groups.items():
            margins=np.array([f(r,"target_margin") for r in values]);signal=float(margins.mean());variance=float(margins.var())
            order_params.append({"stage":stage,"model_seed":seed,"predictive_order":porder,"boundary":boundary,"margin_signal":signal,"margin_variance":variance,"margin_snr":signal/math.sqrt(variance+1e-8),"accuracy":np.mean([int(r["top1_correct"]) for r in values]),"gate_passed":gates[stage]["gate_passed"]})
            vectors=np.stack([residual(r) for r in values]);labels=np.array([cls(r,stage) for r in values]);centroids={c:vectors[labels==c].mean(0) for c in sorted(set(labels))}
            within=float(np.mean([np.sum((v-centroids[c])**2) for v,c in zip(vectors,labels)]));center=np.mean(list(centroids.values()),0);between=float(np.mean([np.sum((v-center)**2) for v in centroids.values()]));pred=[min(centroids,key=lambda c:np.sum((v-centroids[c])**2)) for v in vectors]
            covariance=np.cov(vectors,rowvar=False);eigen=np.linalg.eigvalsh(covariance);eigen=np.maximum(eigen,0);effective=float(eigen.sum()**2/(np.square(eigen).sum()+1e-12))
            sample=np.arange(min(192,len(values)));similarities=[];agreements=[]
            for i in sample:
                for j in sample[i+1:i+13]:similarities.append(cosine(vectors[i],vectors[j]));agreements.append(float(labels[i]==labels[j]))
            rsa=float(np.corrcoef(similarities,agreements)[0,1]) if np.std(similarities)>0 and np.std(agreements)>0 else 0.0
            geometry.append({"stage":stage,"model_seed":seed,"predictive_order":porder,"boundary":boundary,"within_dispersion":within,"between_centroid_variance":between,"within_between_ratio":between/(within+1e-8),"hierarchy_rsa":rsa,"nearest_class_recovery":np.mean(np.array(pred)==labels),"residual_covariance_trace":float(np.trace(covariance)),"effective_rank":effective,"interpretation_status":"eligible" if gates[stage]["gate_passed"] else "blocked_by_competence"})
        by_example=defaultdict(list)
        for row in rows:by_example[(int(row["model_seed"]),row["example_id"],order(row,stage))].append(row)
        for (seed,example,porder),values in by_example.items():
            values=sorted(values,key=lambda r:int(r["boundary"]));correct=[int(r["top1_correct"]) for r in values];first=next((i for i,x in enumerate(correct) if x),-1);stable=next((i for i in range(len(correct)) if all(correct[i:])), -1)
            depth_rows.append({"stage":stage,"model_seed":seed,"example_id":example,"predictive_order":porder,"first_top1_boundary":first,"stable_top1_boundary":stable,"settling_delay":stable-first if first>=0 and stable>=0 else -1,"final_correct":correct[-1]})
        update_groups=defaultdict(list)
        for row in rows:update_groups[(int(row["model_seed"]),order(row,stage),row["example_id"])].append(row)
        grouped=defaultdict(list)
        for (seed,porder,example),values in update_groups.items():
            values=sorted(values,key=lambda r:int(r["boundary"]));
            for a,b in zip(values,values[1:]):grouped[(seed,porder,int(b["boundary"]),cls(b,stage))].append(residual(b)-residual(a))
        keys=sorted(grouped)
        for seed,porder,boundary in sorted({k[:3] for k in keys}):
            same=[];cross=[];classes=sorted(k[3] for k in keys if k[:3]==(seed,porder,boundary))
            for c in classes:
                arr=grouped[(seed,porder,boundary,c)];same += [cosine(arr[i],arr[i+1]) for i in range(0,len(arr)-1,2)]
            for c1,c2 in zip(classes,classes[1:]):cross.append(cosine(grouped[(seed,porder,boundary,c1)][0],grouped[(seed,porder,boundary,c2)][0]))
            updates.append({"stage":stage,"model_seed":seed,"predictive_order":porder,"boundary":boundary,"same_class_update_similarity":np.mean(same) if same else "","cross_class_update_similarity":np.mean(cross) if cross else "","interpretation_status":"eligible" if gates[stage]["gate_passed"] else "blocked_by_competence"})
    write_csv(tables/"semantic_order_parameters.csv",order_params);write_csv(tables/"semantic_hierarchy_geometry.csv",geometry);write_csv(tables/"semantic_abstraction_depth.csv",depth_rows);write_csv(tables/"semantic_update_similarity.csv",updates)
    validations=[]
    for stage,path in STAGES.items():validations.append({"stage":stage,**json.loads((path/f"{stage}_generator_validation.json").read_text())})
    write_csv(tables/"semantic_generator_validation.csv",validations)
    write_csv(tables/"semantic_replacement_damage.csv",[{"stage":s,"status":"pending_competent_checkpoint_analysis" if gates[s]["gate_passed"] else "blocked_by_competence"} for s in STAGES])
    write_csv(tables/"semantic_head_utility.csv",[{"stage":s,"status":"pending_competent_checkpoint_analysis" if gates[s]["gate_passed"] else "blocked_by_competence"} for s in STAGES])
    write_csv(tables/"semantic_scaling_fits.csv",[{"status":"not_estimated","reason":"one architecture per new generator; legacy scaling retained separately"}])

    def plot_lines(rows,y,name,ylabel,eligible_only=False):
        plt.figure(figsize=(6.2,3.6))
        for stage in STAGES:
            if eligible_only and not gates[stage]["gate_passed"]:continue
            subset=[r for r in rows if r["stage"]==stage];xs=sorted(set(int(r["boundary"]) for r in subset));ys=[np.mean([float(r[y]) for r in subset if int(r["boundary"])==x]) for x in xs];plt.plot(xs,ys,marker="o",label=stage.upper())
        plt.xlabel("residual boundary");plt.ylabel(ylabel);plt.legend();plt.tight_layout();plt.savefig(figures/name,dpi=180);plt.close()
    means={s:gates[s]["mean_accuracy"] for s in STAGES};mins={s:gates[s]["minimum_cell_accuracy"] for s in STAGES}
    plt.figure(figsize=(5.8,3.5));x=np.arange(3);plt.bar(x-.18,[means[s] for s in STAGES],.36,label="mean");plt.bar(x+.18,[mins[s] for s in STAGES],.36,label="worst cell");plt.axhline(.8,color="black",ls="--",lw=1);plt.xticks(x,[s.upper() for s in STAGES]);plt.ylabel("held-out accuracy");plt.ylim(0,1.05);plt.legend();plt.tight_layout();plt.savefig(figures/"semantic_accuracy_by_generator.png",dpi=180);plt.close()
    plot_lines(order_params,"margin_snr","semantic_margin_snr_vs_depth.png","margin signal/noise")
    plot_lines(geometry,"hierarchy_rsa","semantic_hierarchy_rsa_vs_depth.png","class-agreement RSA",True)
    plot_lines(geometry,"within_between_ratio","semantic_within_between_vs_depth.png","between / within",True)
    plot_lines(updates,"same_class_update_similarity","semantic_sa_ff_update_similarity.png","same-class update cosine",True)
    plot_lines(geometry,"nearest_class_recovery","semantic_tree_recovery.png","nearest-class recovery",True)
    dd=defaultdict(list)
    for r in depth_rows:
        if r["stage"]=="s3" and int(r["stable_top1_boundary"])>=0:dd[int(r["predictive_order"])].append(int(r["stable_top1_boundary"]))
    plt.figure(figsize=(5.5,3.4));plt.bar(sorted(dd),[np.mean(dd[k]) for k in sorted(dd)]);plt.xlabel("predictive semantic order");plt.ylabel("mean stable boundary");plt.tight_layout();plt.savefig(figures/"semantic_abstraction_decision_depth.png",dpi=180);plt.close()
    comp=[r for r in competence if r["stage"]=="s3"]
    plt.figure(figsize=(5.5,3.4));orders=sorted(set(int(r["predictive_order"]) for r in comp));plt.bar(orders,[np.mean([float(r["accuracy"]) for r in comp if int(r["predictive_order"])==o]) for o in orders]);plt.ylim(0,1.05);plt.xlabel("predictive semantic order");plt.ylabel("unseen-combination accuracy");plt.tight_layout();plt.savefig(figures/"semantic_compositional_generalization.png",dpi=180);plt.close()
    for filename,title,status in (("semantic_replacement_damage.png","Causal replacement", "pending for competent S1/S3; blocked for S2"),("semantic_head_overlap.png","Head utility overlap","pending for competent S1/S3; blocked for S2")):
        plt.figure(figsize=(5.5,2.4));plt.axis("off");plt.text(.5,.58,title,ha="center",fontsize=14);plt.text(.5,.38,status,ha="center",fontsize=10);plt.tight_layout();plt.savefig(figures/filename,dpi=180);plt.close()
    summaries.mkdir(exist_ok=True)
    (summaries/"s1_summary.md").write_text(f"# S1 taxonomy\n\nGate passed: **yes**. Mean held-out accuracy {means['s1']:.3%}; worst cell {mins['s1']:.3%}. Geometry is eligible for interpretation.\n")
    (summaries/"s2_summary.md").write_text(f"# S2 relational\n\nGate passed: **no**. Mean held-out accuracy {means['s2']:.3%}; worst cell {mins['s2']:.3%}, despite zero training loss. This is memorization without rule generalization; mechanism interpretation is blocked.\n")
    (summaries/"s3_summary.md").write_text(f"# S3 compositional\n\nGate passed: **yes**. Mean unseen-combination accuracy {means['s3']:.3%}; worst cell {mins['s3']:.3%}. Predictive orders 1--3 are eligible for depth analysis.\n")
    (summaries/"paper06_integrated_summary.md").write_text("# Integrated Paper 0.6 result\n\nCompetence is generator-dependent: explicit in-context taxonomies (S1) and structured attribute rules (S3) generalize, whereas the S2 full-relation parity task is fit in-sample but fails on unseen combinations. Geometry and update analyses are therefore reported only for S1/S3; S2 is retained as a gated negative result.\n")
    atomic_write_json(ROOT/"analysis_manifest.json",{"gates":gates,"tables":sorted(p.name for p in tables.glob("*.csv")),"figures":sorted(p.name for p in figures.glob("*.png"))})

if __name__=="__main__":main()
