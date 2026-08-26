"""Consolidate Paper 0.8 D0--D4 and replication evidence."""
from __future__ import annotations
import csv,glob
from pathlib import Path
import numpy as np
from cl.common.artifacts import atomic_write_json,write_csv

ROOT=Path("docs/papers/paper0_8/results")
def read(p):
    with open(p,newline="") as h:return list(csv.DictReader(h))
def main():
    coarse=read(ROOT/"phase/icl_phase_grid.csv");rep=[]
    for path in glob.glob(str(ROOT/"d4_replication/*/phase/icl_phase_grid.csv")):
        arch=Path(path).parents[1].name
        for row in read(path):rep.append({"architecture":arch,**row})
    write_csv(ROOT/"phase/icl_d4_replication.csv",rep)
    aggregate=[]
    for arch in sorted({r["architecture"] for r in rep}):
        q=[r for r in rep if r["architecture"]==arch]
        aggregate.append({"architecture":arch,"seed_count":len(q),"passing_seeds":sum(int(r["competent"]) for r in q),"mean_correct_accuracy":np.mean([float(r["correct_accuracy"]) for r in q]),"mean_context_free_rank":np.mean([float(r["context_free_mean_rank"]) for r in q]),"maximum_control_accuracy":max(float(r["max_matched_control_accuracy"]) for r in q),"central_status":"pass" if all(int(r["competent"]) for r in q) else "seed_unstable"})
    write_csv(ROOT/"phase/icl_d4_replication_aggregate.csv",aggregate)
    stages=[]
    for stage in ("D0","D1","D2","D3","D4"):
        q=[r for r in coarse if r["stage"]==stage and r["layers"]=="2" and r["width"]=="32" and r["heads"]=="2"]
        stages.append({"stage":stage,"seed_count":len(q),"passing_seeds":sum(int(r["competent"]) for r in q),"mean_correct_accuracy":np.mean([float(r["correct_accuracy"]) for r in q]),"maximum_control_accuracy":max(float(r["max_matched_control_accuracy"]) for r in q),"status":"stable" if q and all(int(r["competent"]) for r in q) else "not_seed_stable"})
    write_csv(ROOT/"phase/icl_dataset_ladder_summary.csv",stages)
    summary="# Paper 0.8 D4 phase summary\n\n"
    summary+="The 54-cell coarse sweep contains 16 individually competent cells, but the reference L2/W32/H2 cell passes only seed 11. D0 fails; D1--D3 can attain high correct-context accuracy but are not selectively competent across seeds. D4 likewise reaches 100% correct-context accuracy at the reference architecture for all seeds, yet seeds either begin with the answer already favored or permit negative controls to solve the task.\n\n"
    summary+="Five Pareto-small seed-11 positives were replicated. No architecture passes all three seeds. L2/W16/H1 and L6/W8/H1 pass two of three; the other three pass one of three. Therefore the present D4 result is a seed-sensitive acquisition frontier, not a stable central ICL mechanism. Full D4 tracing remains competence-blocked; the copy calibration may validate instrumentation independently.\n"
    (ROOT/"summaries").mkdir(exist_ok=True);(ROOT/"summaries/d4_phase_summary.md").write_text(summary)
    atomic_write_json(ROOT/"phase/d4_analysis_manifest.json",{"coarse_cells":len(coarse),"coarse_competent_cells":sum(int(r["competent"]) for r in coarse),"replicated_architectures":len(aggregate),"three_seed_stable_architectures":sum(r["central_status"]=="pass" for r in aggregate),"d4_tracing_status":"blocked_pending_stable_competence"})
if __name__=="__main__":main()
