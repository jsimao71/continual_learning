"""Strict complete-only publication analysis for Paper 1 natural replication v2."""
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from cl.common.artifacts import atomic_write_json,stable_hash,write_csv
from cl.experiments.paper1_natural_replication import _paired_identity_rows,_read_csv,validate_seed_output

def _jsonl(path):
 with path.open(encoding="utf-8") as h:return [json.loads(line) for line in h if line.strip()]

def validate_complete(root,config):
 root=Path(root);preregistered=root/"preregistered_config.json"
 if preregistered.exists() and json.loads(preregistered.read_text())!=config:raise RuntimeError("preregistered protocol hash mismatch")
 audits=[];selector=[];prediction=[];identities=defaultdict(lambda:defaultdict(set))
 for seed in config["sampling_seeds"]:
  directory=root/"seeds"/f"seed-{seed}"
  try:audit=validate_seed_output(directory,config,seed)
  except ValueError as error:raise RuntimeError(f"seed {seed} incomplete: {error}") from error
  marker=directory/"replication_complete.json"
  if not marker.exists() or json.loads(marker.read_text())!=audit:raise RuntimeError(f"seed {seed} completion marker mismatch")
  candidates=_jsonl(directory/"raw"/"candidate_features.jsonl");traces=_jsonl(directory/"raw"/"selection_traces.jsonl")
  expected_candidates=2*2*config["per_split"]*config["candidates"]
  expected_selector=2*config["per_split"]*len(config["selectors"])*len(config["budget_chunks"])
  if len(candidates)!=expected_candidates or len(traces)!=expected_selector:raise RuntimeError(f"seed {seed} raw sample counts differ")
  for dataset in ("hotpotqa","qasper"):
   for split in ("validation","test"):
    ids={r["identity_id"] for r in candidates if r["dataset"]==dataset and r["split"]==split}
    if len(ids)!=config["per_split"]:raise RuntimeError(f"seed {seed} {dataset}/{split} identity count differs")
    counts={identity:sum(r["identity_id"]==identity and r["dataset"]==dataset and r["split"]==split for r in candidates) for identity in ids}
    if set(counts.values())!={config["candidates"]}:raise RuntimeError(f"seed {seed} {dataset}/{split} candidate allocation differs")
   validation={r["identity_id"] for r in candidates if r["dataset"]==dataset and r["split"]=="validation"};test={r["identity_id"] for r in candidates if r["dataset"]==dataset and r["split"]=="test"}
   if validation&test:raise RuntimeError(f"seed {seed} {dataset} identity leakage")
  values=_read_csv(directory/"tables"/"selector_by_example.csv")
  expected_grid={(dataset,identity,condition,budget) for dataset in ("hotpotqa","qasper")
                 for identity in {r["identity_id"] for r in values if r["dataset"]==dataset}
                 for condition in config["selectors"] for budget in config["budget_tokens"]}
  observed_grid={(r["dataset"],r["identity_id"],r["condition"],int(r["budget_tokens"])) for r in values}
  if observed_grid!=expected_grid or len(observed_grid)!=len(values):raise RuntimeError(f"seed {seed} selector grid differs")
  for row in values:row["sampling_seed"]=seed;selector.append(row);identities[row["dataset"]][seed].add(row["identity_id"])
  values=_read_csv(directory/"tables"/"causal_utility_prediction.csv")
  for row in values:row["sampling_seed"]=seed;prediction.append(row)
  audits.append(audit)
 return audits,selector,prediction,identities

def causal_summary(rows,config,samples=None):
 samples=samples or config["bootstrap_samples"];rng=np.random.default_rng(config["sampling_seeds"][0]+17);out=[]
 for dataset in sorted({r["dataset"] for r in rows}):
  lookup={(int(r["sampling_seed"]),r["model"]):float(r["r2"]) for r in rows if r["dataset"]==dataset}
  deltas=np.asarray([lookup[(s,"surface_plus_structure")]-lookup[(s,"surface_controls")] for s in config["sampling_seeds"]])
  boot=np.asarray([rng.choice(deltas,len(deltas)).mean() for _ in range(samples)])
  out.append({"dataset":dataset,"seeds":len(deltas),"mean_incremental_r2":float(deltas.mean()),"ci_low":float(np.quantile(boot,.025)),"ci_high":float(np.quantile(boot,.975)),"min_incremental_r2":float(deltas.min()),"max_incremental_r2":float(deltas.max()),"positive_seeds":int((deltas>0).sum())})
 return out

def overlap_rows(identities,config):
 out=[]
 for dataset,by_seed in sorted(identities.items()):
  for i,left in enumerate(config["sampling_seeds"]):
   for right in config["sampling_seeds"][i+1:]:out.append({"dataset":dataset,"seed_left":left,"seed_right":right,"left_identities":len(by_seed[left]),"right_identities":len(by_seed[right]),"overlap_identities":len(by_seed[left]&by_seed[right]),"union_identities":len(by_seed[left]|by_seed[right])})
  out.append({"dataset":dataset,"seed_left":"all","seed_right":"all","left_identities":sum(map(len,by_seed.values())),"right_identities":"","overlap_identities":"","union_identities":len(set().union(*by_seed.values()))})
 return out

def publication(out,selector,paired,causal):
 tables=out/"tables";figures=out/"figures";tables.mkdir(parents=True,exist_ok=True);figures.mkdir(parents=True,exist_ok=True)
 grouped=defaultdict(list)
 for r in selector:grouped[(r["dataset"],r["condition"],int(r["budget_tokens"]))].append(float(r["answer_logprob"]))
 quality=[{"dataset":d,"condition":c,"budget_tokens":b,"n_evaluations":len(v),"mean_answer_logprob":float(np.mean(v))} for (d,c,b),v in sorted(grouped.items())];write_csv(tables/"quality_frontier.csv",quality)
 write_csv(tables/"paired_unique_identity.csv",paired);write_csv(tables/"causal_incremental_r2.csv",causal)
 tex=["\\begin{table}[t]\\centering\\small","\\begin{tabular}{llrr}\\toprule","Dataset & Tokens & Bridge--base $\\Delta$ & 95\\% CI \\\\ ","\\midrule"]
 for r in paired:tex.append(f"{r['dataset'].upper()} & {r['budget_tokens']} & {r['mean_answer_logprob_delta']:+.3f} & [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] \\\\")
 tex += ["\\bottomrule\\end{tabular}","\\caption{Three-seed replication pooled by unique held-out identity. Repeated identities across sampling seeds are averaged before the paired bootstrap.}","\\label{tab:paper1-replication}","\\end{table}"];(tables/"natural_replication_results.tex").write_text("\n".join(tex)+"\n")
 decision_front={d:any(r["dataset"]==d and float(r["ci_low"])>0 for r in paired) for d in ("hotpotqa","qasper")};decision_causal={r["dataset"]:bool(r["mean_incremental_r2"]>0 and r["positive_seeds"]>=2) for r in causal};gate=int(all(decision_front.values()) or all(decision_causal.values()))
 macros=[f"\\newcommand{{\\PaperOneReplicationGate}}{{{gate}}}",f"\\newcommand{{\\PaperOneReplicationUniqueHotpot}}{{{next(r['n_unique_test_identities'] for r in paired if r['dataset']=='hotpotqa')}}}",f"\\newcommand{{\\PaperOneReplicationUniqueQasper}}{{{next(r['n_unique_test_identities'] for r in paired if r['dataset']=='qasper')}}}"];(tables/"natural_replication_macros.tex").write_text("\n".join(macros)+"\n")
 fig,axes=plt.subplots(1,2,figsize=(8.8,3.6))
 for ax,dataset in zip(axes,("hotpotqa","qasper")):
  values=[r for r in paired if r["dataset"]==dataset];x=[r["budget_tokens"] for r in values];y=np.asarray([r["mean_answer_logprob_delta"] for r in values]);lo=y-np.asarray([r["ci_low"] for r in values]);hi=np.asarray([r["ci_high"] for r in values])-y;ax.errorbar(x,y,yerr=[lo,hi],marker="o",capsize=4);ax.axhline(0,color="black",lw=.8);ax.set_title(dataset.upper());ax.set_xlabel("materialized tokens");ax.grid(alpha=.25)
 axes[0].set_ylabel("bridge minus base answer log probability");fig.tight_layout();fig.savefig(figures/"replication_paired_frontier.png",dpi=180);plt.close(fig)
 return {"frontier_by_dataset":decision_front,"causal_prediction_by_dataset":decision_causal,"persistent_learning_gate":gate}

def analyze(root,out,config):
 audits,selector,prediction,identities=validate_complete(root,config);paired=_paired_identity_rows(selector,config["bootstrap_samples"],config["sampling_seeds"][0]);causal=causal_summary(prediction,config);out=Path(out);(out/"tables").mkdir(parents=True,exist_ok=True);overlap=overlap_rows(identities,config);write_csv(out/"tables"/"identity_overlap_audit.csv",overlap);decision=publication(out,selector,paired,causal);atomic_write_json(out/"gate_decision.json",decision)
 manifest={"schema_version":"paper1.natural_replication.analysis.v1","complete_only":True,"config_hash":stable_hash(config),"completed_seeds":len(audits),"seed_audits":audits,"pooled_selector_rows":len(selector),"identity_overlap_hash":stable_hash(overlap),"paired_hash":stable_hash(paired),"causal_hash":stable_hash(causal),"gate_decision":decision};atomic_write_json(out/"analysis_manifest.json",manifest);return manifest

def main(args=None):
 p=argparse.ArgumentParser();p.add_argument("--config",default="configs/paper1/natural_replication_v2.json");p.add_argument("--root",default="docs/papers/paper1/results/natural_replication_v2");p.add_argument("--output",default="docs/papers/paper1/results/natural_replication_v2_analysis");ns=p.parse_args(args);print(json.dumps(analyze(ns.root,ns.output,json.loads(Path(ns.config).read_text())),indent=2))
if __name__=="__main__":main()
