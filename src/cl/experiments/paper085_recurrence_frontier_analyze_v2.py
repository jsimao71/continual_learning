"""Aggregate and validate the reduced Paper 0.85 recurrence-frontier pass."""
from __future__ import annotations
import argparse,csv,json,random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from cl.common.artifacts import atomic_write_json,write_csv
from cl.experiments.paper085_recurrence_frontier_v2 import online_training_pool
from cl.semantic.recurrence_chains import recurrence_pair_split

CONDITIONS=("baseline","partial_starts","high_diversity","partial_diversity")
METRICS=("final_correct","trajectory_exact","transition_accuracy","termination_correct","premature_termination","delayed_termination")

def read(path):
    with path.open(newline="",encoding="utf-8") as handle:return list(csv.DictReader(handle))

def replay_dataset_audit(condition,seed,cfg,train_pairs):
    """Reconstruct sampling-only audit state without loading a model or accelerator."""
    rng=random.Random(seed+701);batch=cfg["batch_size"];seen_chains=set();seen_transitions=set();diversity_seen=set()
    for _ in range(cfg["updates"]):
        pool=online_training_pool(train_pairs,condition,cfg,batch,rng,diversity_seen)
        per=batch//len(pool);selected=[]
        for depth in sorted(pool):selected.extend(rng.choice(pool[depth]) for _ in range(per))
        rng.shuffle(selected)  # Keep the RNG stream identical to batch_from_pool.
        seen_chains.update(row.latent_chain for row in selected)
        seen_transitions.update(edge for row in selected for edge in zip(row.chain,row.chain[1:]))
    partial=condition in {"partial_starts","partial_diversity"};high=condition in {"high_diversity","partial_diversity"}
    return {"condition":condition,"seed":seed,"partial_starts":partial,"high_diversity":high,
        "sampling_policy":"online_nonrepeating_depth_gt1" if high else "fresh_online_stage1_matched",
        "audit_provenance":"deterministic_sampling_replay","updates":cfg["updates"],"examples_seen":cfg["updates"]*batch,
        "processed_token_budget":cfg["updates"]*batch*(cfg["training_sequence_length"]-1),
        "supervised_target_tokens_last_invocation":cfg["updates"]*288,
        "observed_distinct_latent_chains":len(seen_chains),"observed_unique_transitions":len(seen_transitions),
        "token_budget_definition":"examples times fixed padded causal length; identical across conditions"}

def main(args):
    out=Path(args.output);cfg=json.loads(Path(args.config).read_text());rows=read(out/"condition_seed_results.csv")
    expected=len(CONDITIONS)*len(cfg["model_seeds"])*len(cfg["test_depths"])
    keys={(r["condition"],int(r["seed"]),int(r["depth"])) for r in rows}
    if len(rows)!=expected or len(keys)!=expected:raise RuntimeError(f"incomplete/duplicate cells: {len(rows)}, {len(keys)}, expected {expected}")
    aggregate=[]
    for condition in CONDITIONS:
      for depth in cfg["test_depths"]:
        selected=[r for r in rows if r["condition"]==condition and int(r["depth"])==depth]
        aggregate.append({"condition":condition,"depth":depth,"seeds":len(selected),
            **{f"mean_{metric}":float(np.mean([float(r[metric]) for r in selected])) for metric in METRICS},
            **{f"min_{metric}":float(np.min([float(r[metric]) for r in selected])) for metric in METRICS}})
    write_csv(out/"condition_depth_summary.csv",aggregate)
    write_csv(out/"partial_start_results.csv",[r for r in aggregate if r["condition"] in {"baseline","partial_starts"}])
    write_csv(out/"diversity_results.csv",[r for r in aggregate if r["condition"] in {"baseline","high_diversity"}])
    frontier=[]
    for condition in CONDITIONS:
        seed_frontiers=[]
        for seed in cfg["model_seeds"]:
            seed_rows=[r for r in rows if r["condition"]==condition and int(r["seed"])==seed]
            seed_frontiers.append(int(float(seed_rows[0]["k_ar_seed"])))
        frontier.append({"condition":condition,"seed_frontiers":";".join(map(str,seed_frontiers)),
            "stable_k_ar":min(seed_frontiers),"p_acquire_k4":sum(value>=4 for value in seed_frontiers)/len(seed_frontiers),
            "delta_frontier_vs_baseline":min(seed_frontiers)-3})
    write_csv(out/"frontier_condition_summary.csv",frontier)
    colors={"baseline":"#555555","partial_starts":"#0072B2","high_diversity":"#D55E00","partial_diversity":"#009E73"}
    fig,ax=plt.subplots(figsize=(6.3,3.7))
    for condition in CONDITIONS:
        selected=[r for r in aggregate if r["condition"]==condition]
        ax.plot([r["depth"] for r in selected],[r["mean_final_correct"] for r in selected],marker="o",label=condition.replace("_"," "),color=colors[condition])
    ax.axvline(3.5,color="black",ls="--",lw=.8);ax.set(xlabel="test proof depth K",ylabel="three-seed mean final-position accuracy",ylim=(-.03,1.03));ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(out/"frontier_by_condition.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.4),sharey=True)
    for ax,condition,title in zip(axes,("partial_starts","high_diversity"),("partial starts","high diversity")):
        for name in ("baseline",condition):
            selected=[r for r in aggregate if r["condition"]==name]
            ax.plot([r["depth"] for r in selected],[r["mean_transition_accuracy"] for r in selected],marker="o",label=name.replace("_"," "),color=colors[name])
        ax.set_title(title);ax.set_xlabel("test proof depth K");ax.grid(alpha=.25);ax.legend(fontsize=8)
    axes[0].set_ylabel("conditional transition accuracy");fig.tight_layout();fig.savefig(out/"intervention_transition_accuracy.png",dpi=180);plt.close(fig)
    audit=read(out/"dataset_audit.csv")
    # Replace the older baseline replay, which omitted the batch shuffle and thus
    # did not follow the training RNG stream exactly. Rows emitted by training are
    # retained; absent rows are reconstructed from sampling alone below.
    audit=[r for r in audit if r.get("sampling_policy")!="fresh_online_stage1_matched_replayed_audit"]
    for row in audit:
        if not row.get("audit_provenance"):row["audit_provenance"]="training_run"
    present={(r["condition"],int(r["seed"])) for r in audit}
    train_pairs=recurrence_pair_split(cfg["symbol_count"],cfg["pair_split_seed"],cfg["test_pair_fraction"])[1]
    for condition in CONDITIONS:
      for seed in cfg["model_seeds"]:
        if (condition,seed) not in present:audit.append(replay_dataset_audit(condition,seed,cfg,train_pairs))
    audit.sort(key=lambda r:(CONDITIONS.index(r["condition"]),int(r["seed"])))
    write_csv(out/"dataset_audit.csv",audit)
    if len({(r["condition"],int(r["seed"])) for r in audit})!=12:raise RuntimeError("dataset audit is not 12-cell complete")
    budgets={int(r["processed_token_budget"]) for r in audit};examples={int(r["examples_seen"]) for r in audit}
    if budgets!={4512000} or examples!={96000}:raise RuntimeError("training budget mismatch")
    atomic_write_json(out/"analysis_manifest.json",{"schema_version":"paper085.recurrence_frontier_v2.analysis.v1",
        "official_cells":expected,"unique_cells":len(keys),"conditions":list(CONDITIONS),"three_seed_complete":True,
        "stable_frontiers":{r["condition"]:r["stable_k_ar"] for r in frontier},"all_p_acquire_k4_zero":all(r["p_acquire_k4"]==0 for r in frontier),
        "exact_processed_token_budget":budgets.pop(),"examples_per_run":examples.pop(),"o0_followup_gate_open":False,
        "excluded_from_aggregation":["smoke/","invalid_low_diversity_baseline/","recurrence_stage1_v1/ query-target-leakage pilot"],
        "invalid_design_pilots_excluded":["invalid_low_diversity_baseline","invalid_partial_only","superseded_padding127"]})

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/paper085/recurrence_frontier_v2.json")
    p.add_argument("--output",default="docs/papers/paper0_85/results/recurrence_frontier_v2");main(p.parse_args())
