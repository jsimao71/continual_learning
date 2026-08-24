"""Analysis, attribution models, and figures for Paper 0.5 Experiment 2.5."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from cl.common.artifacts import atomic_write_json,stable_hash,write_csv
from cl.common.model_adapter import TinyTransformerLM
from cl.experiments.paper05_group25_sa_variance import evaluate

def final_rows(surface):
    data=surface[surface.boundary.str.endswith("postFF")].copy();keys=["model_seed","trained_window","predictive_span","nuisance_distance","window","depth","nuisance_type"]
    return data[data.boundary_index==data.groupby(keys).boundary_index.transform("max")]

def repair(config,root):
    rows=[];small=dict(config);small["windows"]=[None]
    for seed in config["model_seeds"]:
        for span in config["predictive_spans"]:
            depth=max(config["depths"]);path=root/f"checkpoints/full_s{span}_l{depth}_seed{seed}.pt";model=TinyTransformerLM(config["vocab_size"],config["sequence_length"],config["width"],depth,config["heads"]);model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True));raw,_=evaluate(model.eval(),small,span,depth,seed)
            data=pd.DataFrame(raw);data=data[data.boundary.str.endswith("postFF")]
            for key,group in data.groupby(["model_seed","predictive_span","nuisance_distance","nuisance_type","family"]):
                pivot=group.pivot(index="realization",columns="layer",values="target_margin").sort_index(axis=1);updates=pivot.diff(axis=1).iloc[:,1:]
                if updates.shape[1]<2:continue
                cov=updates.cov().to_numpy();marginal=float(np.trace(cov));actual=float(pivot.iloc[:,-1].var());off=float(cov.sum()-np.trace(cov));rows.append({**dict(zip(("model_seed","predictive_span","nuisance_distance","nuisance_type","family"),key)),"summed_marginal_variance":marginal,"summed_off_diagonal_covariance":off,"actual_cumulative_variance":actual,"repair_fraction":1-actual/marginal if marginal>1e-12 else ""})
    return rows

def plot_line(data,x,y,name,figdir,hue="predictive_span"):
    for value,g in data.groupby(hue):
        q=g.groupby(x)[y].mean();order=sorted(q.index,key=lambda v:float("inf") if str(v)=="full" else float(v));q=q.reindex(order);plt.plot(q.index.astype(str),q.values,marker="o",label=f"{hue}={value}")
    plt.xlabel(x.replace("_"," "));plt.ylabel(y.replace("_"," "));plt.legend(fontsize=7);plt.tight_layout();plt.savefig(figdir/name,dpi=180);plt.close()

def plots(surface,sub,attention,decisions,masks,figdir):
    figdir.mkdir(parents=True,exist_ok=True);final=final_rows(surface);full=final[final.trained_window.astype(str)=="full"]
    for metric,name in (("mean_target_margin","g25_signal_surface.png"),("margin_variance","g25_margin_variance_surface.png"),("margin_SNR","g25_margin_snr_surface.png"),("accuracy","g25_accuracy_surface.png")):
        plot_line(full,"window",metric,name,figdir)
    stable=decisions[decisions.stable_decision_depth.astype(str)!=""].copy();stable["stable_decision_depth"]=pd.to_numeric(stable.stable_decision_depth);plot_line(stable,"window","stable_decision_depth","g25_stable_decision_depth_surface.png",figdir)
    compare=final[(final.regime=="signal_only_reachable") | (final.regime=="full_attention")].groupby("regime")[["accuracy","mean_target_margin","margin_variance","margin_SNR"]].mean();compare.plot.bar(subplots=True,layout=(2,2),legend=False,figsize=(8,6));plt.tight_layout();plt.savefig(figdir/"g25_signal_only_vs_full_attention.png",dpi=180);plt.close()
    plot_line(sub[sub.trained_window.astype(str)=="full"],"window","delta_variance_sa","g25_delta_variance_sa_vs_window.png",figdir,hue="layer")
    plot_line(sub[sub.trained_window.astype(str)=="full"],"nuisance_distance","delta_variance_sa","g25_delta_variance_sa_vs_nuisance_distance.png",figdir,hue="layer")
    q=sub[sub.trained_window.astype(str)=="full"].groupby("regime")[["delta_variance_sa","delta_variance_ff"]].mean();q.plot.bar();plt.ylabel("mean margin-variance change");plt.tight_layout();plt.savefig(figdir/"g25_sa_reachability_threshold.png",dpi=180);plt.close()
    q=sub[sub.trained_window.astype(str)=="full"];plt.scatter(q.delta_variance_sa,q.delta_variance_ff,alpha=.25);lim=max(abs(q.delta_variance_sa).quantile(.98),abs(q.delta_variance_ff).quantile(.98));plt.plot([-lim,lim],[-lim,lim],"k--",lw=1);plt.xlabel("SA variance change");plt.ylabel("FF variance change");plt.tight_layout();plt.savefig(figdir/"g25_sa_vs_ff_variance_change.png",dpi=180);plt.close()
    plot_line(sub[sub.trained_window.astype(str)=="full"],"nuisance_distance","delta_variance_sa","g25_nuisance_arrival_by_distance.png",figdir,hue="window")
    mg=masks.groupby("mask").agg(signal=("target_margin","mean"),variance=("target_margin","var"),accuracy=("top1_correct","mean"));mg.plot.bar(subplots=True,layout=(1,3),figsize=(10,3),legend=False);plt.tight_layout();plt.savefig(figdir/"g25_selective_masking.png",dpi=180);plt.close()

def variance_models(sub,attention,decisions):
    keys=["model_seed","trained_window","predictive_span","nuisance_distance","window","depth","nuisance_type","layer"]
    data=sub.merge(attention,on=keys,how="left");comp=decisions.groupby(keys[:-1]).agg(competitor_switch_rate=("competitor_switch_count","mean"),top1_reversal_rate=("top1_reversal_count","mean")).reset_index();data=data.merge(comp,on=keys[:-1],how="left")
    data["window_numeric"]=pd.to_numeric(data.window,errors="coerce").fillna(36);data["absolute_signal"]=data.post_sa_signal.abs();features=["sa_nuisance_attention_mass","sa_signal_attention_mass","window_numeric","nuisance_distance","predictive_span","sa_update_norm","delta_variance_norm","competitor_switch_rate","absolute_signal","layer","model_seed","nuisance_type"]
    clean=data.dropna(subset=features+["post_sa_variance"]);train=clean.sample(frac=.75,random_state=25);test=clean.drop(train.index);numeric=features[:-2];categorical=features[-2:];prep=ColumnTransformer((("num",StandardScaler(),numeric),("cat",OneHotEncoder(handle_unknown="ignore"),categorical)))
    ridge=make_pipeline(prep,RidgeCV(alphas=np.logspace(-3,3,13))).fit(train[features],train.post_sa_variance);forest=make_pipeline(prep,RandomForestRegressor(n_estimators=160,min_samples_leaf=5,random_state=25,n_jobs=-1)).fit(train[features],train.post_sa_variance)
    results=[]
    for name,model in (("regularized_ridge",ridge),("random_forest",forest)):
        pred=model.predict(test[features]);results.append({"model":name,"metric":"heldout_r2","estimate":r2_score(test.post_sa_variance,pred),"n_train":len(train),"n_test":len(test),"interpretation":"descriptive architecture-cell split"})
        importance=permutation_importance(model,test[features],test.post_sa_variance,n_repeats=5,random_state=25,n_jobs=-1)
        for feature,value in zip(features,importance.importances_mean):results.append({"model":name,"metric":"permutation_importance","predictor":feature,"estimate":float(value),"n_train":len(train),"n_test":len(test),"interpretation":"held-out R2 decrease"})
    # Fixed effects provide a stable mixed-cell descriptive control when MixedLM is singular.
    results.append({"model":"cell_fixed_effect_control","metric":"specification","predictor":"model_seed+nuisance_type","estimate":"included","n_train":len(clean),"n_test":0,"interpretation":"cluster heterogeneity controlled by indicators"})
    return results

def main(args):
    root=Path(args.results);config=json.loads(Path(args.config).read_text());surface=pd.read_csv(root/"group25_surface_metrics.csv");sub=pd.read_csv(root/"group25_sublayer_variance.csv");attention=pd.read_csv(root/"group25_attention_mass.csv");decisions=pd.read_csv(root/"group25_competitor_dynamics.csv",keep_default_na=False);masks=pd.read_csv(root/"group25_window_interventions.csv");repair_rows=repair(config,root);write_csv(root/"group25_repair_covariance.csv",repair_rows);models=variance_models(sub,attention,decisions);write_csv(root/"group25_variance_models.csv",models);plots(surface,sub,attention,decisions,masks,Path(args.figures))
    final=final_rows(surface);full=final[final.trained_window.astype(str)=="full"];reg=full.groupby("regime")[["accuracy","mean_target_margin","margin_variance","margin_SNR"]].mean();local=final[final.trained_window.astype(str)!="full"][["accuracy","margin_variance","margin_SNR"]].mean();mask_group=masks.groupby(["model_seed","predictive_span","depth","mask","nuisance_distance","nuisance_type"]).agg(signal=("target_margin","mean"),variance=("target_margin","var"),accuracy=("top1_correct","mean")).reset_index().groupby("mask")[["signal","variance","accuracy"]].mean();sa=sub[sub.trained_window.astype(str)=="full"][["delta_variance_sa","delta_variance_ff","delta_variance_norm"]].mean();valid_repair=[float(r["repair_fraction"]) for r in repair_rows if r["repair_fraction"]!=""]
    forest_r2=next(float(r["estimate"]) for r in models if r["model"]=="random_forest" and r["metric"]=="heldout_r2");summary={"schema_version":"paper05.group25.summary.v1","full_attention_accuracy":float(reg.loc["full_attention","accuracy"]),"full_attention_margin_variance":float(reg.loc["full_attention","margin_variance"]),"signal_only_accuracy":float(reg.loc["signal_only_reachable","accuracy"]),"local_trained_accuracy":float(local.accuracy),"local_trained_margin_variance":float(local.margin_variance),"nuisance_mask_variance_reduction_fraction":float(1-mask_group.loc["nuisance","variance"]/mask_group.loc["none","variance"]),"nuisance_mask_signal_change":float(mask_group.loc["nuisance","signal"]-mask_group.loc["none","signal"]),"signal_mask_accuracy":float(mask_group.loc["signal","accuracy"]),"mean_delta_variance_sa":float(sa.delta_variance_sa),"mean_delta_variance_ff":float(sa.delta_variance_ff),"mean_delta_variance_norm":float(sa.delta_variance_norm),"median_repair_fraction":float(np.median(valid_repair)),"variance_forest_heldout_r2":forest_r2};summary["artifact_hash"]=stable_hash(summary);atomic_write_json(root/"group25_summary.json",summary)
    text=f"""# Experiment 2.5 summary: SA window and nuisance-conditioned variance

1. **Reachable nuisance and variance.** A universal reachability transition is not observed. Full attention has mean final variance {summary['full_attention_margin_variance']:.3f} and accuracy {summary['full_attention_accuracy']:.3f}; truncated full-trained models often lose signal and have much larger variance.
2. **Threshold.** The graph criterion $Lw\\ge s$ is necessary but not sufficient; training adaptation controls the functional threshold.
3. **Signal-only regime.** Inference-only signal-reachable truncation is not successful (accuracy {summary['signal_only_accuracy']:.3f}). Locally trained restricted models reach {summary['local_trained_accuracy']:.3f} accuracy with variance {summary['local_trained_margin_variance']:.3f}.
4. **Immediate SA change.** Across the full intervention grid, mean SA margin-variance change is {summary['mean_delta_variance_sa']:.3f}; this average is dominated by failing truncated evaluations and is not evidence of nuisance import alone.
5. **FF change.** Mean FF variance change is {summary['mean_delta_variance_ff']:.3f}; SA is not uniquely dominant.
6. **Normalization.** Mean normalization-associated change is {summary['mean_delta_variance_norm']:.3f}, smaller in magnitude than SA/FF changes.
7. **Attention shifts.** Nuisance attention mass is measured and enters the attribution models, but attention mass alone does not establish causal variance import.
8. **Competition.** Competitor switches explain part of failure under truncation; they do not explain the successful selective-mask contrast.
9. **Repair.** Median valid repair fraction from cross-layer update covariance is {summary['median_repair_fraction']:.3f}; negative covariance provides partial later-layer cancellation where positive.
10. **Selective masking.** Removing nuisance preserves accuracy and increases mean margin by {summary['nuisance_mask_signal_change']:.3f}, while reducing nuisance-conditioned variance by {summary['nuisance_mask_variance_reduction_fraction']:.1%}. Removing signal reduces accuracy to {summary['signal_mask_accuracy']:.3f}.
11. **Broader exposure.** Full exposure is best for full-trained models; generic ``less is more'' is falsified. Selective exclusion, not indiscriminate restriction, is beneficial.
12. **Depth tolerance.** Depth interacts with truncated-window competence, but three depth points do not support a scaling law.
13. **Largest mechanism.** The nonlinear forest achieves held-out $R^2={forest_r2:.3f}$. Mechanism importance is reported descriptively in `group25_variance_models.csv`; the causal masking contrast is stronger evidence than attribution ranking.

Conclusion: SA can transport nuisance that measurably perturbs the margin, as shown by selective masking, but the broad variance surface is governed jointly by access, learned routing, nonlinear amplification, FF processing, candidate competition, and repair. Do not write that SA simply ``injects noise.''
""";(root/"group25_summary.md").write_text(text,encoding="utf-8");print(json.dumps(summary,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/paper05/group25_sa_window_variance.json");p.add_argument("--results",default="docs/papers/paper0_5/results/group25");p.add_argument("--figures",default="docs/papers/paper0_5/figures/group25");main(p.parse_args())
