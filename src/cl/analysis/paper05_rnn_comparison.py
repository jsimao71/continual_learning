"""Analyze matched recurrent and Transformer controlled comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cl.common.artifacts import atomic_write_json,stable_hash,write_csv

COLORS={"transformer":"#2878b5","rnn":"#d32f2f","gru":"#f9a825","lstm":"#43a047"}


def main(args)->None:
    root=Path(args.results);fig=Path(args.figures);fig.mkdir(parents=True,exist_ok=True);raw=pd.read_csv(root/"rnn_transformer_raw.csv");internal=pd.read_csv(root/"rnn_transformer_internal_raw.csv");params=pd.read_csv(root/"rnn_transformer_parameter_match.csv")
    cells=raw.groupby(["model_type","axis","axis_value","predictive_order","raw_length","dependency_span","nuisance_count"],as_index=False).agg(accuracy=("top1_correct","mean"),mean_margin=("target_margin","mean"))
    write_csv(root/"rnn_transformer_results.csv",cells.to_dict("records"));fits=[]
    for (model,axis),g in cells.groupby(["model_type","axis"]):
        x=g.axis_value.to_numpy(float);y=g.accuracy.to_numpy(float);X=np.column_stack([np.ones(len(x)),x]);b=np.linalg.lstsq(X,y,rcond=None)[0];pred=X@b
        fits.append({"model_type":model,"axis":axis,"intercept":b[0],"slope":b[1],"r2":1-((y-pred)**2).sum()/max(((y-y.mean())**2).sum(),1e-12),"range_min":x.min(),"range_max":x.max()})
    write_csv(root/"rnn_transformer_model_fits.csv",fits)
    names={"length":"rnn_transformer_accuracy_vs_length.png","span":"rnn_transformer_accuracy_vs_span.png","predictive_order":"rnn_transformer_accuracy_vs_predictive_order.png","nuisance":"rnn_transformer_accuracy_vs_nuisance.png"}
    for axis,name in names.items():
        q=cells[cells.axis==axis]
        for model,g in q.groupby("model_type"):plt.plot(g.axis_value,g.accuracy,marker="o",label=model,color=COLORS[model])
        plt.axhline(.8,color="k",ls="--",lw=1);plt.xlabel(axis.replace("_"," "));plt.ylabel("held-out accuracy");plt.ylim(.2,1.03);plt.legend();plt.tight_layout();plt.savefig(fig/name,dpi=180);plt.close()
    q=internal[internal.model_type!="transformer"].groupby(["model_type","computation_index"],as_index=False).target_margin.mean()
    for model,g in q.groupby("model_type"):plt.plot(g.computation_index,g.target_margin,marker="o",label=model,color=COLORS[model])
    plt.axhline(0,color="k",lw=.7);plt.xlabel("recurrent time step");plt.ylabel("mean target margin");plt.legend();plt.tight_layout();plt.savefig(fig/"rnn_hidden_signal_vs_time.png",dpi=180);plt.close()
    q=internal[internal.model_type=="transformer"].groupby("computation_index",as_index=False).target_margin.mean();plt.plot(q.computation_index,q.target_margin,marker="o",color=COLORS["transformer"]);plt.axhline(0,color="k",lw=.7);plt.xlabel("Transformer depth boundary");plt.ylabel("mean target margin");plt.tight_layout();plt.savefig(fig/"transformer_margin_vs_depth_matched.png",dpi=180);plt.close()
    plt.bar(params.model_type,params.parameter_count,color=[COLORS[x] for x in params.model_type]);plt.ylabel("trainable parameters");plt.tight_layout();plt.savefig(fig/"rnn_transformer_parameter_matched_summary.png",dpi=180);plt.close()
    slopes=pd.DataFrame(fits).pivot(index="model_type",columns="axis",values="slope");best=cells.groupby("model_type").accuracy.mean()
    summary={"schema_version":"paper05.rnn_comparison.summary.v1","parameter_counts":dict(zip(params.model_type,map(int,params.parameter_count))),"mean_accuracy":{k:float(v) for k,v in best.items()},
             "accuracy_slopes":{m:{a:float(v) for a,v in row.dropna().items()} for m,row in slopes.iterrows()},
             "interpretation":"Sensitivity differences are controlled-benchmark evidence only; recurrent time and Transformer depth are not equated."};summary["artifact_hash"]=stable_hash(summary);atomic_write_json(root/"rnn_transformer_summary.json",summary)
    t=slopes.loc["transformer"];r=slopes.drop(index="transformer")
    slope_lines=["| model | "+" | ".join(slopes.columns)+" |","|---|"+"---|"*len(slopes.columns)]
    slope_lines += ["| "+str(model)+" | "+" | ".join(f"{value:.5f}" for value in row)+" |" for model,row in slopes.iterrows()]
    md=f"""# Matched recurrent comparison\n\nAll four architectures saw identical generated examples, optimizer family, training steps, vocabulary, target entropy, and token budget. Exact parameter counts are {summary['parameter_counts']}.\n\nLinear accuracy slopes (descriptive over the tested range) are:\n\n{chr(10).join(slope_lines)}\n\nThe fixed-order length and span hypotheses are supported only when the recurrent slopes are more negative than the Transformer slope and the relevant models are behaviorally competent. Predictive-order and nuisance curves are reported separately. RNN time is sequential transport; Transformer depth is parallel refinement, so their internal indices are plotted but never equated. This benchmark excludes state-space and recurrent-attention models.\n""";(root/"summary.md").write_text(md,encoding="utf-8");print(json.dumps(summary,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--results",default="docs/papers/paper0_5/results/rnn_comparison");p.add_argument("--figures",default="docs/papers/paper0_5/figures/rnn_comparison");main(p.parse_args())
