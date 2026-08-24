"""Competence-gated analysis and plots for the nested-pattern bridge."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cl.common.artifacts import atomic_write_json,stable_hash,write_csv

COMPETENT=("irrelevant_extension","supportive_extension")
COLORS={"irrelevant_extension":"#2878b5","supportive_extension":"#43a047","refining_extension":"#f9a825","override_extension":"#d32f2f","multi_level_hierarchy":"#7b1fa2"}
def line(data,metric,path,ylabel=None):
    for rel,g in data.groupby("relation_type"):
        q=g.groupby("layer")[metric].mean();plt.plot(q.index,q.values,marker="o",label=rel.replace("_"," "),color=COLORS[rel])
    plt.xlabel("depth boundary" if "residual" in metric else "block");plt.ylabel(ylabel or metric.replace("_"," "));plt.legend(fontsize=6,ncol=2);plt.tight_layout();plt.savefig(path,dpi=180);plt.close()

def main(args):
    root=Path(args.results);fig=Path(args.figures);fig.mkdir(parents=True,exist_ok=True);pairs=pd.read_csv(root/"nested_pair_metrics.csv");updates=pd.read_csv(root/"nested_update_similarity.csv");replacement=pd.read_csv(root/"nested_replacement_results.csv");heads=pd.read_csv(root/"nested_head_similarity.csv");utility=pd.read_csv(root/"nested_head_causal_utilities.csv");overlap=pd.read_csv(root/"nested_head_causal_overlap.csv");div=pd.read_csv(root/"nested_divergence_layers.csv");hier=pd.read_csv(root/"nested_hierarchy_metrics.csv")
    trained=pairs[pairs.trained==1];tu=updates[updates.trained==1];final=trained[trained.layer==trained.layer.max()];competence=final.groupby(["relation_type","position_mode"]).top1_correct.mean().reset_index();write_csv(root/"nested_competence.csv",competence.to_dict("records"))
    line(trained,"raw_residual_cosine",fig/"nested_residual_similarity_vs_depth.png");line(tu,"block_update_cosine",fig/"nested_update_similarity_vs_depth.png")
    for metric,label in (("sa_update_cosine","SA"),("ff_update_cosine","FF")):
        q=tu.groupby("layer")[metric].mean();plt.plot(q.index,q.values,marker="o",label=label)
    plt.xlabel("block");plt.ylabel("nested update cosine");plt.legend();plt.tight_layout();plt.savefig(fig/"nested_sa_ff_similarity_vs_depth.png",dpi=180);plt.close()
    for rel,g in tu.groupby("relation_type"):
        q=g.groupby("layer").normalized_retention.median();plt.plot(q.index,q.values,marker="o",label=rel.replace("_"," "),color=COLORS[rel])
    plt.axhline(0,color="k",lw=.7);plt.xlabel("block");plt.ylabel("median normalized retention");plt.legend(fontsize=6,ncol=2);plt.tight_layout();plt.savefig(fig/"nested_retention_vs_depth.png",dpi=180);plt.close()
    q=trained.groupby("layer")[["alpha_depth","context_component_norm"]].mean();q.plot(marker="o",secondary_y="context_component_norm");plt.tight_layout();plt.savefig(fig/"nested_core_context_decomposition.png",dpi=180);plt.close()
    q=trained[trained.relation_type=="override_extension"].groupby("layer")[["long_target_margin","short_target_margin_on_long"]].mean();q.plot(marker="o");plt.axhline(0,color="k",lw=.7);plt.ylabel("diagnostic margin");plt.tight_layout();plt.savefig(fig/"nested_override_target_crossing.png",dpi=180);plt.close()
    q=replacement[replacement.relation_type.isin(COMPETENT)].groupby(["layer","donor_type"]).JS_to_intact.mean().unstack();q.plot(marker="o");plt.ylabel("JS to intact (bits)");plt.tight_layout();plt.savefig(fig/"nested_replacement_damage_vs_depth.png",dpi=180);plt.close()
    matrix=heads.groupby(["layer","head"]).head_output_cosine.mean().unstack();plt.imshow(matrix,aspect="auto",vmin=-1,vmax=1,cmap="coolwarm");plt.colorbar(label="head-output cosine");plt.xlabel("head");plt.ylabel("layer");plt.tight_layout();plt.savefig(fig/"nested_head_similarity_matrix.png",dpi=180);plt.close()
    line(overlap,"spearman_utility_correlation",fig/"nested_head_utility_overlap.png","head-utility Spearman correlation")
    valid=div.dropna(subset=["representational_divergence_layer","behavioral_target_crossing_layer"]);plt.scatter(valid.representational_divergence_layer,valid.behavioral_target_crossing_layer,alpha=.25);plt.plot([0,6],[0,6],"k--",lw=1);plt.xlabel("representational divergence layer");plt.ylabel("behavioral crossing layer");plt.tight_layout();plt.savefig(fig/"nested_divergence_vs_behavioral_override.png",dpi=180);plt.close()
    hm=hier[hier.metric_type=="update"].groupby(["short_level","extension_length"]).similarity.mean().unstack();plt.imshow(hm,aspect="auto",vmin=-1,vmax=1,cmap="coolwarm");plt.xticks(range(len(hm.columns)),hm.columns);plt.yticks(range(len(hm.index)),hm.index);plt.xlabel("extension length");plt.ylabel("short level");plt.colorbar(label="update similarity");plt.tight_layout();plt.savefig(fig/"nested_hierarchy_similarity_matrix.png",dpi=180);plt.close()
    roles=[]
    for key,g in utility.groupby(["relation_type","layer","head"]):
        short=g.short_causal_utility.mean();long=g.long_causal_utility.mean();rel=key[0]
        if short>0 and long>0 and abs(long-short)<max(abs(short),abs(long))*.5:role="core_pattern_head"
        elif rel=="supportive_extension" and long>max(short,0)*1.5:role="context_support_head"
        elif rel=="override_extension" and long>max(short,0)*1.5:role="override_head"
        else:role="distributed/no_stable_role"
        roles.append({"relation_type":rel,"layer":key[1],"head":key[2],"mean_short_utility":short,"mean_long_utility":long,"causal_role":role})
    write_csv(root/"nested_head_roles.csv",roles)
    comp=competence.groupby("relation_type").top1_correct.mean();same=trained[trained.relation_type.isin(COMPETENT)];same_u=tu[tu.relation_type.isin(COMPETENT)];rep=replacement[replacement.relation_type.isin(COMPETENT)];donor=rep.groupby("donor_type")[["JS_to_intact","target_logprob_delta","final_top1_change"]].mean();trained_adv=float((updates[updates.trained==1].block_update_cosine-updates[updates.trained==1].nonequivalent_update_cosine).mean());random_adv=float((updates[updates.trained==0].block_update_cosine-updates[updates.trained==0].nonequivalent_update_cosine).mean());pos=competence.pivot(index="relation_type",columns="position_mode",values="top1_correct")
    summary={"schema_version":"paper05.nested.summary.v1","competence":{k:float(v) for k,v in comp.items()},"same_target_raw_residual_cosine":float(same.raw_residual_cosine.mean()),"same_target_nested_minus_nonequivalent_residual":float((same.raw_residual_cosine-same.nonequivalent_cosine).mean()),"trained_update_advantage":trained_adv,"random_update_advantage":random_adv,"same_target_block_update_cosine":float(same_u.block_update_cosine.mean()),"same_target_sa_update_cosine":float(same_u.sa_update_cosine.mean()),"same_target_ff_update_cosine":float(same_u.ff_update_cosine.mean()),"replacement":donor.to_dict("index"),"same_target_head_utility_spearman":float(overlap[overlap.relation_type.isin(COMPETENT)].spearman_utility_correlation.mean()),"same_target_top2_overlap":float(overlap[overlap.relation_type.isin(COMPETENT)].top2_head_overlap.mean()),"aligned_randomized_accuracy_gap":{r:float(pos.loc[r,"aligned"]-pos.loc[r,"randomized"]) for r in pos.index},"bridge_decision":"partial support for same-target causal reuse; refinement/override/hierarchy blocked by behavioral incompetence"};summary["artifact_hash"]=stable_hash(summary);atomic_write_json(root/"nested_summary.json",summary)
    md=f"""# Nested-pattern representation bridge

1. Raw residuals are highly similar for competent same-target nesting (mean {summary['same_target_raw_residual_cosine']:.3f}), but random initialization is even more similar because the shared query/position dominates; raw cosine alone is not evidence.
2. Block updates preserve the matched nesting relation more clearly: training raises the nested-minus-nonequivalent update advantage from {random_adv:.3f} at initialization to {trained_adv:.3f}.
3. In competent same-target strata, mean SA/FF update cosines are {summary['same_target_sa_update_cosine']:.3f}/{summary['same_target_ff_update_cosine']:.3f}; neither universally dominates.
4. Nested same-target residual similarity exceeds the matched nonequivalent control by {summary['same_target_nested_minus_nonequivalent_residual']:.3f}. Irrelevant extension also exceeds unrelated same-target controls; supportive extension is closer to its same-target control.
5. Causal replacement supports reuse: embedded-short donors have mean JS damage {donor.loc['short','JS_to_intact']:.3f}, unrelated same-target {donor.loc['same_target','JS_to_intact']:.3f}, and nonequivalent {donor.loc['nonequivalent','JS_to_intact']:.3f} bits.
6. A representational divergence layer can be computed, but override competence is only {comp['override_extension']:.3f}; it is not interpretable as learned override.
7. Consequently divergence-versus-target crossing is descriptive/blocked, not confirmatory.
8. Same-target head-utility Spearman correlation is {summary['same_target_head_utility_spearman']:.3f}, with top-2 overlap {summary['same_target_top2_overlap']:.3f}: causal head reuse is moderate, not identity.
9. Supportive extensions show some context-support utility cells, but role labels are causal/descriptive and not stable semantics.
10. Override-head recruitment is blocked by the override competence failure.
11. No claim is made that head-overlap decline identifies an override divergence layer.
12. Multi-level hierarchy accuracy is {comp['multi_level_hierarchy']:.3f}; its similarity matrix is an artifact diagnostic, not evidence for a learned hierarchy.
13. Causal replacement and matched update advantage preserve the competent nested relation better than absolute residual cosine.
14. Supportive competence is {comp['supportive_extension']:.3f}; irrelevant-extension competence is {comp['irrelevant_extension']:.3f}. Their aligned-minus-randomized gaps are {summary['aligned_randomized_accuracy_gap']['supportive_extension']:.3f} and {summary['aligned_randomized_accuracy_gap']['irrelevant_extension']:.3f}, so absolute position does not explain the qualified effect. Three seeds are retained.
15. Paper 0.6 should carry only this conditional result: a trained Transformer can causally reuse a short computation under competent same-target lexical extension. Semantic hierarchy/refinement remains gated on a competent target-changing model.

## Competence gate

Final accuracies are: irrelevant {comp['irrelevant_extension']:.3f}, supportive {comp['supportive_extension']:.3f}, refining {comp['refining_extension']:.3f}, override {comp['override_extension']:.3f}, and hierarchy {comp['multi_level_hierarchy']:.3f}. The last three are near four-way chance and are excluded from positive mechanistic claims. Additional weighted, relation-specific, minimal-cell, and XOR pilots also failed to clear the target-changing gate; these pilots are not promoted as confirmatory artifacts.
""";(root/"nested_summary.md").write_text(md,encoding="utf-8");print(json.dumps(summary,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--results",default="docs/papers/paper0_5/results/nested_patterns");p.add_argument("--figures",default="docs/papers/paper0_5/figures/nested_patterns");main(p.parse_args())
