"""Family-level inference and crossed-factor robustness summaries for Paper 0.5."""

from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path

import numpy as np

from cl.common.artifacts import write_csv
from cl.common.metrics import bootstrap_ci


def read(path):
    with Path(path).open(newline="",encoding="utf-8") as handle: return list(csv.DictReader(handle))


def controlled_contrasts(rows):
    selected=[r for r in rows if r["training_stage"]=="trained" and r["location"]=="post_block"]
    units=defaultdict(dict)
    for r in selected:
        key=(r["model_setting"],r["seed"],r["relation_id"],r["control_type"],r["target_control"],r["prefix_evidence"],r["noise_level"])
        units[key][int(r["layer"])]=float(r["mean_js_to_pattern_centroid"])
    groups=defaultdict(list)
    for key,depth in units.items():
        if len(depth)>1: groups[key[3:]].append(depth[max(depth)]-depth[min(depth)])
    output=[]
    for key,values in sorted(groups.items()):
        estimate,low,high=bootstrap_ci(values,samples=2000,seed=53)
        output.append({"control_type":key[0],"target_control":key[1],"prefix_evidence":key[2],"noise_level":key[3],
                       "contrast":"last_minus_first_js_dispersion","n_relation_run_units":len(values),"estimate":estimate,"ci_low":low,"ci_high":high})
    return output


def pretrained_factorial(rows):
    """Descriptive fixed-factor regression; clustered bootstrap is the primary inference."""
    model_levels=sorted({(r["model_id"],r["revision"]) for r in rows})
    x=[]; targets={"mean_js_dispersion":[],"mean_target_probability":[],"snr_js_ratio":[]}
    for r in rows:
        syntax=float(r["syntax"]=="mapping"); semantic=float(r["semantic"]=="color"); model=model_levels.index((r["model_id"],r["revision"]))
        x.append([1,syntax,semantic,syntax*semantic,float(r["evidence"]),float(r["noise"]),float(r["layer"]),float(model)])
        for name in targets: targets[name].append(float(r[name]))
    design=np.asarray(x); names=("intercept","mapping_syntax","color_semantics","syntax_x_semantics","evidence","noise","depth","model_checkpoint_index")
    output=[]
    for outcome,y in targets.items():
        coefficients=np.linalg.lstsq(design,np.asarray(y),rcond=None)[0]; prediction=design@coefficients; total=np.square(np.asarray(y)-np.mean(y)).sum(); r2=1-np.square(np.asarray(y)-prediction).sum()/max(total,1e-12)
        output.extend({"outcome":outcome,"term":name,"coefficient":float(value),"descriptive_r2":float(r2),"n_cells":len(y)} for name,value in zip(names,coefficients))
    return output


def causal_family_contrasts(rows):
    unit=defaultdict(list)
    for r in rows: unit[(r["model_id"],r["revision"],r["family"],r["mode"])].append(float(r["target_logprob_change"]))
    output=[]
    comparisons=("replace_syntax_mismatch","replace_semantic_mismatch","replace_nonequivalent","project_remove")
    for model,revision in sorted({(r["model_id"],r["revision"]) for r in rows}):
        for comparison in comparisons:
            deltas=[]
            for family in sorted({r["family"] for r in rows if r["model_id"]==model and r["revision"]==revision}):
                equivalent=np.mean(unit[(model,revision,family,"replace_equivalent")]); other=np.mean(unit[(model,revision,family,comparison)])
                deltas.append(equivalent-other)
            estimate,low,high=bootstrap_ci(deltas,samples=2000,seed=59)
            output.append({"model_id":model,"revision":revision,"contrast":f"equivalent_minus_{comparison}","n_families":len(deltas),"estimate":estimate,"ci_low":low,"ci_high":high})
    return output


def run(root="docs/papers/paper0_5/results"):
    root=Path(root); tables=root/"tables"; pretrained=root/"pretrained"/"tables"
    write_csv(tables/"variance_family_bootstrap.csv",controlled_contrasts(read(tables/"variance_by_depth_prefix_noise.csv")))
    variance=read(pretrained/"pretrained_variance.csv"); causal=read(pretrained/"pretrained_causal_mediation.csv")
    write_csv(pretrained/"factorial_regression.csv",pretrained_factorial(variance))
    write_csv(pretrained/"causal_family_contrasts.csv",causal_family_contrasts(causal))


if __name__=="__main__": run()
