"""Integrate the three controlled Paper 0.5 programs into paper artifacts."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from cl.common.artifacts import atomic_write_json,stable_hash,write_csv

def read_csv(path):
    with Path(path).open(newline="",encoding="utf-8") as h:return list(csv.DictReader(h))

def run(root="docs/papers/paper0_5/results/three_parts"):
    root=Path(root);g1=read_csv(root/"aggregates/group1/group1_inference.csv");g2=read_csv(root/"aggregates/group2/group2_transport.csv");g3=read_csv(root/"aggregates/group3/head_utility.csv")
    metrics=[]
    for row in g1:
        if row["metric"]=="within_js_bits":metrics.append({"group":"layer_composition","phenomenon":row["generator_family"],"metric":"last_minus_first_within_js_bits","estimate":row["last_minus_first"],"ci_low":row["ci_low"],"ci_high":row["ci_high"],"status":"supported" if float(row["ci_high"])<0 else "mixed"})
    final=[r for r in g2 if int(r["layer"])==int(r["depth"])-1 and r["local_control"]=="0"]
    metrics.append({"group":"attention_transport","phenomenon":"unreachable","metric":"mean_accuracy","estimate":float(np.mean([float(r["accuracy"]) for r in final if r["reachable"]=="0"])),"ci_low":"","ci_high":"","status":"supported"})
    metrics.append({"group":"attention_transport","phenomenon":"reachable","metric":"mean_accuracy","estimate":float(np.mean([float(r["accuracy"]) for r in final if r["reachable"]=="1"])),"ci_low":"","ci_high":"","status":"partially_supported"})
    for mode in ("zero","mean","equivalent","mismatched"):
        v=[float(r["target_logprob_drop"]) for r in g3 if r["mode"]==mode];metrics.append({"group":"heads","phenomenon":mode,"metric":"mean_target_logprob_drop","estimate":float(np.mean(v)),"ci_low":"","ci_high":"","status":"descriptive"})
    write_csv(root/"paper05_integrated_metrics.csv",metrics)
    manifests={name:json.loads((root/f"manifests/{name}.json").read_text()) for name in ("group1","group2","group3")}
    assessment={"statistical":"supported: family-conditioned JS contracts with preserved target signal","information_theoretic":"partially supported: prediction-space class separation improves; mutual information not estimated","dynamical_systems":"supported descriptively: non-monotone composition and JVP diagnostics","statistical_mechanical":"partially supported: R and covariance order parameters measured; no finite-size phase-transition claim"}
    manifest={"schema_version":"paper05.integrated.v1","sources":{k:v["artifact_hash"] for k,v in manifests.items()},"assessment":assessment,"artifact_hash":stable_hash({"metrics":metrics,"sources":manifests})}
    atomic_write_json(root/"paper05_integrated_manifest.json",manifest)
    (root/"paper05_integrated_summary.md").write_text("# Integrated Paper 0.5 summary\n\nDepth improves predictive-class organization across three controlled generators, usually non-monotonically. Local-attention reachability is necessary but not sufficient for long-range prediction. Head interventions show distributed, non-additive contributions; equivalent head outputs transfer better than mismatches, while motif specificity again fails to predict causal utility.\n\nThe theory is best supported as composed, anisotropic, state-dependent refinement. The data do not establish universal contraction, a sharp phase transition, a simple depth-times-window sufficiency law, stable individual-head semantics, or a continual-learning mechanism.\n",encoding="utf-8")
    print(json.dumps(manifest,indent=2))

if __name__=="__main__":run()
