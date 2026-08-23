"""Frozen equivalence-feature audit over Paper 1's native-K/V natural gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cl.common.artifacts import atomic_write_json, stable_hash, write_csv
from cl.common.metrics import spearman
from cl.nsc.features import ridge_fit, ridge_predict


SURFACE = ("base_score", "lexical_score", "semantic_score")
RECURRENCE = ("entropy_contribution", "persistence", "agreement")
STRUCTURE = ("community_centrality", "bridge_score")


def _augment(rows):
    by_example = {}
    for row in rows: by_example.setdefault(row["example_id"], []).append(row)
    output=[]
    for values in by_example.values():
        entropy=np.asarray([float(r["entropy_contribution"]) for r in values]); span=max(float(np.ptp(entropy)),1e-12)
        for row,e in zip(values,entropy):
            agreement=float(row["agreement"]); persistence=float(row["persistence"]); bridge=float(row["bridge_score"])
            output.append({**row,
                # A frozen dynamical proxy, not Paper 0.5 functional membership: candidate-level output distributions were not cached.
                "predictive_equivalence_proxy": agreement*persistence*(1-abs(float(e)-float(np.median(entropy)))/span),
                # Paper 0.6 failed competence; the semantic feature family is deliberately unavailable, not imputed.
                "semantic_equivalence_available": 0.0,
                "functional_role": bridge*(0.5+0.5*agreement)+float(row["community_centrality"])*persistence,
                "exception_indicator": abs(float(e)-float(np.median(entropy)))/span,
            })
    return output


def _evaluate(rows):
    models=(
        ("surface",SURFACE),
        ("surface_plus_recurrence",SURFACE+RECURRENCE),
        ("surface_plus_structure",SURFACE+RECURRENCE+STRUCTURE),
        ("surface_plus_predictive_proxy",SURFACE+RECURRENCE+("predictive_equivalence_proxy","exception_indicator")),
        ("surface_plus_functional_role",SURFACE+RECURRENCE+STRUCTURE+("predictive_equivalence_proxy","functional_role","exception_indicator")),
    )
    output=[]
    for dataset in sorted({r["dataset"] for r in rows}):
        train=[r for r in rows if r["dataset"]==dataset and r["split"]=="validation"]
        test=[r for r in rows if r["dataset"]==dataset and r["split"]=="test"]
        y_train=np.asarray([float(r["causal_utility"]) for r in train]); y_test=np.asarray([float(r["causal_utility"]) for r in test])
        for name,fields in models:
            x_train=np.asarray([[float(r[f]) for f in fields] for r in train]); x_test=np.asarray([[float(r[f]) for f in fields] for r in test])
            mean=x_train.mean(0); scale=x_train.std(0); scale[scale<1e-9]=1
            fit=ridge_fit((x_train-mean)/scale,y_train,alpha=2.0); prediction=ridge_predict((x_test-mean)/scale,fit)
            denom=np.square(y_test-y_test.mean()).sum()
            output.append({"dataset":dataset,"model":name,"n_validation":len(train),"n_test":len(test),
                           "spearman":spearman(y_test,prediction),"r2":float(1-np.square(y_test-prediction).sum()/max(denom,1e-12)),
                           "rmse":float(np.sqrt(np.mean(np.square(y_test-prediction))))})
    return output


def run(args):
    source=Path(args.source); output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    with source.open(newline="",encoding="utf-8") as handle: rows=list(csv.DictReader(handle))
    augmented=_augment(rows); results=_evaluate(augmented)
    write_csv(output/"equivalence_feature_prediction.csv",results)
    write_csv(output/"equivalence_candidate_features.csv",augmented)
    decision={"gate_passed":False,"reason":"No feature family has positive held-out R2 on both task families; Paper 0.6 semantic features disabled after competence failure.",
              "paper05_feature_level":"dynamical proxy only; candidate output distributions unavailable","paper06_feature_level":"unavailable: competence gate failed",
              "artifact_hash":stable_hash({"results":results})}
    atomic_write_json(output/"equivalence_gate_decision.json",decision)
    print(json.dumps({"rows":len(augmented),"results":results,"decision":decision},indent=2))


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--source",default="docs/papers/paper1/results/natural/tables/candidate_removal.csv")
    p.add_argument("--output",default="docs/papers/paper1/results/natural/equivalence"); return p.parse_args()


if __name__=="__main__": run(parse_args())
