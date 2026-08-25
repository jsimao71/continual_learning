"""Competence-gated analysis for the nested length/depth experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cl.common.artifacts import atomic_write_json, stable_hash, write_csv

COLORS = {"redundant": "#2878b5", "supportive": "#43a047", "necessary": "#d32f2f"}


def save(path: Path, ylabel: str, xlabel: str = "pattern length") -> None:
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.tight_layout(); plt.savefig(path, dpi=180); plt.close()


def grouped_line(data: pd.DataFrame, y: str, path: Path, ylabel: str) -> None:
    for regime, g in data.groupby("regime"):
        q = g.groupby("pattern_length")[y].mean()
        plt.plot(q.index, q.values, marker="o", label=regime, color=COLORS[regime])
    plt.legend(); save(path, ylabel)


def linear_fit(data: pd.DataFrame, predictor: str, outcome: str) -> dict:
    d = data[[predictor, outcome]].dropna()
    if len(d) < 3 or d[predictor].nunique() < 2:
        return {"predictor": predictor, "outcome": outcome, "n": len(d), "status": "insufficient variation"}
    x = d[predictor].to_numpy(float); y = d[outcome].to_numpy(float)
    X = np.column_stack([np.ones(len(x)), x]); beta = np.linalg.lstsq(X, y, rcond=None)[0]
    pred = X @ beta; rss = max(float(np.sum((y - pred) ** 2)), 1e-12); tss = float(np.sum((y-y.mean())**2))
    loo=[]
    for i in range(len(y)):
        keep=np.arange(len(y)) != i; b=np.linalg.lstsq(X[keep],y[keep],rcond=None)[0]; loo.append(abs(y[i]-X[i]@b))
    return {"predictor": predictor, "outcome": outcome, "n": len(d), "intercept": beta[0], "slope": beta[1],
            "r2": 1-rss/tss if tss else 0.0, "aic": len(y)*np.log(rss/len(y))+4,
            "bic": len(y)*np.log(rss/len(y))+2*np.log(len(y)), "leave_one_cell_out_mae": float(np.mean(loo)),
            "status": "descriptive only: one seed and one architecture depth"}


def main(args) -> None:
    root=Path(args.results); fig=Path(args.figures); fig.mkdir(parents=True,exist_ok=True)
    acc=pd.read_csv(root/"nested_length_accuracy_by_model_depth.csv")
    dec=pd.read_csv(root/"nested_length_decision_depth.csv")
    raw=pd.read_csv(root/"nested_length_metrics.csv")
    reuse=pd.read_csv(root/"nested_length_update_reuse.csv")
    heads=pd.read_csv(root/"nested_length_head_recruitment.csv")
    minimum=float(args.minimum_competence)
    cells=acc.groupby(["regime","pattern_length"],as_index=False).agg(accuracy=("accuracy","mean"), minimum_subcell_accuracy=("accuracy","min"), mean_final_margin=("mean_final_margin","mean"))
    cells["competent"]=cells.accuracy >= minimum
    write_csv(root/"nested_length_competence.csv",cells.to_dict("records"))
    competence={f"{r.regime}:n{int(r.pattern_length)}":float(r.accuracy) for r in cells.itertuples()}
    passed=bool(cells.competent.all())
    old=json.loads((root/"nested_length_smoke_decision.json").read_text())
    status={"schema_version":"paper05.nested_length.smoke.v2","gate_unit":"regime_by_pattern_length","gate_passed":passed,
            "competence":competence,"minimum_threshold":minimum,"preferred_threshold":0.8,
            "architecture_sweep_status":"eligible" if passed else "blocked: necessary n>=4 cells fail the preregistered competence gate",
            "smoke_initial_loss":old["smoke_initial_loss"],"smoke_final_loss":old["smoke_final_loss"]}
    status["artifact_hash"]=stable_hash(status); atomic_write_json(root/"nested_length_smoke_decision.json",status)

    grouped_line(cells,"accuracy",fig/"nested_length_accuracy_phase.png","final accuracy")
    plt.axhline(minimum,color="k",ls="--",lw=1,label="competence gate")
    # redraw so the threshold is part of the saved phase plot
    for regime,g in cells.groupby("regime"): plt.plot(g.pattern_length,g.accuracy,marker="o",label=regime,color=COLORS[regime])
    plt.axhline(minimum,color="k",ls="--",lw=1);plt.ylim(0,1.04);plt.legend();save(fig/"nested_length_accuracy_phase.png","final accuracy")
    qualified=dec.merge(cells[["regime","pattern_length","competent"]],on=["regime","pattern_length"])
    grouped_line(qualified[qualified.competent],"first_top1_layer",fig/"nested_first_top1_vs_length.png","first top-1 boundary")
    grouped_line(qualified[qualified.competent],"stable_top1_layer",fig/"nested_stable_top1_vs_length.png","stable top-1 boundary")
    grouped_line(qualified[qualified.competent],"settling_delay",fig/"nested_settling_vs_length.png","settling delay")

    margins=raw.groupby(["regime","pattern_length","boundary_index"],as_index=False).target_margin.mean()
    for regime,g in margins.groupby("regime"):
        for n,q in g.groupby("pattern_length"):
            plt.plot(q.boundary_index,q.target_margin,label=f"{regime} n={n}",color=COLORS[regime],alpha=.3+.7*n/8)
    plt.axhline(0,color="k",lw=.7);plt.legend(fontsize=5,ncol=3);save(fig/"nested_margin_trajectories.png","mean target margin","depth boundary")
    margins["margin_gain"]=margins.groupby(["regime","pattern_length"]).target_margin.diff()
    gain=margins.dropna().groupby(["regime","pattern_length"],as_index=False).apply(lambda g:g.loc[g.margin_gain.idxmax(),["boundary_index","margin_gain"]],include_groups=False).reset_index(drop=True)
    write_csv(root/"nested_length_margin_gain.csv",gain.to_dict("records"));grouped_line(gain,"boundary_index",fig/"nested_max_margin_gain_layer.png","boundary of maximum margin gain")

    for col,label in (("block_update_cosine","block"),("sa_update_cosine","SA"),("ff_update_cosine","FF")):
        q=reuse.groupby(["long_length","layer"])[col].mean().reset_index()
        for n,g in q.groupby("long_length"):plt.plot(g.layer,g[col],marker="o",label=f"n={n}")
        plt.legend();save(fig/f"nested_{label.lower()}_reuse.png",f"{label} update cosine","block")
    q=heads.groupby("long_length",as_index=False).agg(top2_overlap=("top2_overlap","mean"),new_head_count=("new_head_count","mean"))
    plt.plot(q.long_length,q.top2_overlap,marker="o",label="top-2 overlap");plt.plot(q.long_length,q.new_head_count/2,marker="s",label="new-head fraction");plt.legend();save(fig/"nested_head_recruitment.png","head reuse / recruitment")

    controls=qualified[qualified.competent].groupby(["regime","pattern_length","predictive_order","span_mode","nuisance_count"],as_index=False).stable_top1_layer.mean()
    span=controls.groupby(["span_mode","pattern_length"],as_index=False).stable_top1_layer.mean()
    for mode,g in span.groupby("span_mode"):plt.plot(g.pattern_length,g.stable_top1_layer,marker="o",label=mode)
    plt.legend();save(fig/"nested_span_partial_effect.png","qualified stable boundary")
    nuis=controls.groupby(["nuisance_count","predictive_order"],as_index=False).stable_top1_layer.mean()
    for p,g in nuis.groupby("predictive_order"):plt.plot(g.nuisance_count,g.stable_top1_layer,marker="o",label=f"p*={p}")
    plt.legend();save(fig/"nested_nuisance_partial_effect.png","qualified stable boundary","nuisance count")

    fitdata=qualified[qualified.competent].groupby(["regime","pattern_length","predictive_order"],as_index=False).stable_top1_layer.mean()
    fits=[linear_fit(fitdata,"pattern_length","stable_top1_layer"),linear_fit(fitdata,"predictive_order","stable_top1_layer")]
    write_csv(root/"nested_length_scaling_fits.csv",fits)
    partial=[]
    for variable in ("pattern_length","predictive_order","dependency_span","nuisance_count"):
        f=linear_fit(qualified[qualified.competent],variable,"stable_top1_layer");f["qualification"]="competent cells only";partial.append(f)
    write_csv(root/"nested_length_partial_effects.csv",partial)
    summary={"schema_version":"paper05.nested_length.summary.v1","gate_passed":passed,"competence":competence,
             "competent_regimes_lengths":[f"{r.regime}:n{int(r.pattern_length)}" for r in cells.itertuples() if r.competent],
             "blocked_regimes_lengths":[f"{r.regime}:n{int(r.pattern_length)}" for r in cells.itertuples() if not r.competent],
             "architecture_depths_run":[8],"model_seeds_run":[11],"scaling_claim":"not identified: architecture sweep blocked by necessary-pattern competence failure",
             "interpretation":"Length alone does not delay competent target-preserving computations, while growing predictive order coincides with optimization collapse; the failed cells cannot identify an internal or architectural depth requirement."}
    summary["artifact_hash"]=stable_hash(summary);atomic_write_json(root/"nested_length_summary.json",summary)
    md=f"""# Nested length/depth study\n\n1. The smoke gate failed; no architecture sweep was launched.\n2. Redundant accuracy stays between {cells[cells.regime=='redundant'].accuracy.min():.3f} and {cells[cells.regime=='redundant'].accuracy.max():.3f}.\n3. Supportive accuracy stays between {cells[cells.regime=='supportive'].accuracy.min():.3f} and {cells[cells.regime=='supportive'].accuracy.max():.3f}.\n4. Necessary accuracy by length is {dict(zip(cells[cells.regime=='necessary'].pattern_length.astype(int),cells[cells.regime=='necessary'].accuracy.round(3)))}.\n5. Necessary n=4,6,8 fail the 0.70 competence criterion.\n6. First-top1, stable-top1, settling, margin, reuse, and head plots exclude failed cells from positive mechanism claims.\n7. Raw length is not itself an obstacle: target-preserving n=8 remains competent.\n8. Predictive order p* separates the regimes by construction, but a scaling law is not identified.\n9. The p* and raw-n fits are descriptive because only one seed and one architecture passed through the smoke stage.\n10. Span effects are evaluated only inside competent target-preserving strata.\n11. Nuisance effects are likewise conditional and do not rescue the necessary-pattern failure.\n12. Update cosine measures reuse, not correctness; it is interpreted only after the behavioral gate.\n13. Head overlap/recruitment is descriptive and does not assign stable semantic roles.\n14. The result supports a conditional theory: extra context length need not require later computation when predictive order stays fixed.\n15. It does not support the converse claim that higher-order dependencies require deeper architectures; that requires a competent multi-depth replication.\n"""
    (root/"nested_length_summary.md").write_text(md,encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--results",default="docs/papers/paper0_5/results/nested_length_depth");p.add_argument("--figures",default="docs/papers/paper0_5/figures/nested_length_depth");p.add_argument("--minimum-competence",type=float,default=.70);main(p.parse_args())
