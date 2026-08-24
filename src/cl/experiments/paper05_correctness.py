"""Correctness-aware order parameters for Paper 0.5 Group 1."""
from __future__ import annotations
import argparse,json,random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from cl.analysis.equivalence import jensen_shannon_bits
from cl.common.artifacts import RunMetadata,atomic_write_json,stable_hash,write_csv,write_jsonl
from cl.common.metrics import bootstrap_ci
from cl.experiments.paper05_layer_composition import generate_records,train_model

EPS=1e-8

def boundaries(trace):
    yield 0,"pre_sa",trace.layers[0].pre_sa
    depth=1
    for layer,values in enumerate(trace.layers):
        yield depth,f"layer{layer}_post_sa",values.post_sa;depth+=1
        yield depth,f"layer{layer}_post_block",values.post_block;depth+=1

@torch.no_grad()
def collect(model,records,seed,device):
    rows=[]
    for record in records:
        inputs=torch.tensor([record.tokens],device=device);_,trace=model(inputs,capture=True);assert trace is not None
        for depth,boundary,state in boundaries(trace):
            final=state[:,-1];logits=model.diagnostic_logits(final)[0];p=torch.softmax(logits,-1);target=record.target_token
            masked=logits.clone();masked[target]=float("-inf");competitor=int(masked.argmax());top=int(logits.argmax())
            rank=int((logits>logits[target]).sum())+1
            rows.append({**asdict(record),"model_seed":seed,"depth_index":depth,"boundary":boundary,"target_token":target,
                "target_logit":float(logits[target]),"target_probability":float(p[target]),"target_logprob":float(torch.log(p[target].clamp_min(1e-30))),
                "target_rank":rank,"strongest_competitor_token":competitor,"strongest_competitor_logit":float(logits[competitor]),
                "target_margin":float(logits[target]-logits[competitor]),"top1_token":top,"top1_probability":float(p[top]),
                "prediction_correct":int(top==target),"output_entropy":float(-(p*torch.log2(p.clamp_min(1e-12))).sum()),
                "full_output_distribution_hash":stable_hash(p.cpu().numpy().astype(np.float16).tobytes().hex()),
                "probabilities":p.cpu().numpy().tolist(),"residual":final[0].cpu().numpy().tolist()})
    return rows

def decisions(rows):
    groups=defaultdict(list)
    keyfields=("model_seed","generator_family","predictive_family_id","surface_identity_id","nuisance_count")
    for row in rows:groups[tuple(row[k] for k in keyfields)].append(row)
    output=[]
    for key,values in sorted(groups.items()):
        values=sorted(values,key=lambda r:r["depth_index"]);correct=[bool(r["prediction_correct"]) for r in values]
        first=next((i for i,x in enumerate(correct) if x),None);stable=next((i for i in range(len(correct)) if all(correct[i:])),None)
        reversals=sum(a!=b for a,b in zip(correct,correct[1:]));output.append({**dict(zip(keyfields,key)),
            "first_top1_layer":"" if first is None else first,"stable_top1_layer":"" if stable is None else stable,
            "settling_delay":"" if first is None or stable is None else stable-first,"top1_reversal_count":reversals,
            "final_correct":int(correct[-1])})
    return output

def aggregate(rows,decision_rows):
    fields=("model_seed","generator_family","predictive_family_id","nuisance_count","depth_index","boundary");groups=defaultdict(list)
    decision_map={(r["model_seed"],r["predictive_family_id"],r["surface_identity_id"],r["nuisance_count"]):r for r in decision_rows}
    for row in rows:groups[tuple(row[k] for k in fields)].append(row)
    output=[]
    for key,values in sorted(groups.items()):
        margin=np.asarray([r["target_margin"] for r in values]);prob=np.asarray([r["target_probability"] for r in values]);p=np.asarray([r["probabilities"] for r in values]);centroid=p.mean(0)
        residual=np.asarray([r["residual"] for r in values]);cov=np.cov(residual,rowvar=False);mad=np.median(np.abs(margin-np.median(margin)))
        stable=[decision_map[(r["model_seed"],r["predictive_family_id"],r["surface_identity_id"],r["nuisance_count"])] for r in values]
        depth=key[4]
        output.append({**dict(zip(fields,key)),"n_realizations":len(values),"mean_target_probability":float(prob.mean()),"var_target_probability":float(prob.var(ddof=1)),
            "mean_target_logprob":float(np.mean([r["target_logprob"] for r in values])),"mean_target_margin":float(margin.mean()),"var_target_margin":float(margin.var(ddof=1)),
            "mean_target_rank":float(np.mean([r["target_rank"] for r in values])),"fraction_top1":float(np.mean([r["prediction_correct"] for r in values])),
            "fraction_stably_top1":float(np.mean([d["stable_top1_layer"]!="" and int(d["stable_top1_layer"])<=depth for d in stable])),
            "mean_output_entropy":float(np.mean([r["output_entropy"] for r in values])),"within_family_JS":float(np.mean([jensen_shannon_bits(x,centroid) for x in p])),
            "correct_target_signal":float(margin.mean()),"fluctuation_variance":float(margin.var(ddof=1)),"margin_snr":float(margin.mean()/np.sqrt(margin.var(ddof=1)+EPS)),
            "robust_margin_snr":float(np.median(margin)/(1.4826*mad+EPS)),"representation_covariance_trace":float(np.trace(cov)),"distribution_centroid":centroid.tolist(),"residual_centroid":residual.mean(0).tolist()})
    matched=defaultdict(list)
    for r in output:matched[(r["model_seed"],r["nuisance_count"],r["depth_index"])].append(r)
    for values in matched.values():
        q=float(np.mean([r["representation_covariance_trace"] for r in values]));pairs=[]
        for i,a in enumerate(values):
            for b in values[i+1:]:pairs.append(float(np.square(np.asarray(a["residual_centroid"])-b["residual_centroid"]).sum()))
        mclass=float(np.mean(pairs));
        for r in values:
            other=[jensen_shannon_bits(r["distribution_centroid"],x["distribution_centroid"]) for x in values if x["predictive_family_id"]!=r["predictive_family_id"]]
            r["between_family_JS"]=float(np.mean(other));r["R"]=r["between_family_JS"]/max(r["within_family_JS"],EPS);r["q_representation"]=q;r["class_centroid_separation"]=mclass;r["chi_representation"]=q/max(mclass,EPS)
    for r in output:r.pop("distribution_centroid");r.pop("residual_centroid")
    return output

def layer_updates(rows):
    example=defaultdict(list)
    for r in rows:example[(r["model_seed"],r["predictive_family_id"],r["surface_identity_id"],r["nuisance_count"])].append(r)
    updates=[]
    for key,values in example.items():
        values=sorted(values,key=lambda r:r["depth_index"])
        for a,b in zip(values,values[1:]):updates.append({"model_seed":key[0],"predictive_family_id":key[1],"surface_identity_id":key[2],"nuisance_count":key[3],"from_depth":a["depth_index"],"to_depth":b["depth_index"],"margin_update":b["target_margin"]-a["target_margin"]})
    groups=defaultdict(list)
    for r in updates:groups[(r["model_seed"],r["predictive_family_id"],r["nuisance_count"],r["from_depth"],r["to_depth"])].append(r["margin_update"])
    summary=[]
    for key,v in sorted(groups.items()):
        x=np.asarray(v);summary.append({"model_seed":key[0],"predictive_family_id":key[1],"nuisance_count":key[2],"from_depth":key[3],"to_depth":key[4],
            "mean_margin_update":float(x.mean()),"var_margin_update":float(x.var(ddof=1)),"margin_update_snr":float(x.mean()/(x.std(ddof=1)+EPS)),"fraction_positive_margin_update":float(np.mean(x>0))})
    covariance=[]
    grouped=defaultdict(list)
    for r in updates:grouped[(r["model_seed"],r["predictive_family_id"],r["nuisance_count"],r["surface_identity_id"])].append(r)
    cells=defaultdict(list)
    for key,v in grouped.items():cells[key[:3]].append([x["margin_update"] for x in sorted(v,key=lambda z:z["from_depth"])])
    for key,matrix in cells.items():
        matrix=np.asarray(matrix);cov=np.cov(matrix,rowvar=False);means=matrix.mean(0);cumulative=float(means.sum());cumvar=float(np.ones(len(means))@cov@np.ones(len(means)))
        for i in range(cov.shape[0]):
            for j in range(cov.shape[1]):covariance.append({"model_seed":key[0],"predictive_family_id":key[1],"nuisance_count":key[2],"layer_i":i,"layer_j":j,"update_covariance":float(cov[i,j]),"cumulative_signal":cumulative,"cumulative_fluctuation_variance":cumvar})
    return updates,summary,covariance

def fit_form(name,x,y):
    if name=="linear":X=np.c_[np.ones(len(x)),x];params=np.linalg.lstsq(X,y,rcond=None)[0];pred=X@params
    elif name=="logarithmic":X=np.c_[np.ones(len(x)),np.log1p(x)];params=np.linalg.lstsq(X,y,rcond=None)[0];pred=X@params
    elif name=="power_law":
        shift=max(0,-y.min()+EPS);X=np.c_[np.ones(len(x)),np.log1p(x)];p=np.linalg.lstsq(X,np.log(y+shift),rcond=None)[0];params=np.r_[np.exp(p[0]),p[1],shift];pred=params[0]*(x+1)**params[1]-shift
    else:
        best=None
        for tau in np.geomspace(.1,100,80):
            basis=1-np.exp(-x/tau) if name=="exponential_saturation" else x/(x+tau)
            X=np.c_[np.ones(len(x)),basis];p=np.linalg.lstsq(X,y,rcond=None)[0];candidate=(np.square(y-X@p).sum(),np.r_[p,tau],X@p)
            if best is None or candidate[0]<best[0]:best=candidate
        _,params,pred=best
    return np.asarray(params),np.asarray(pred)

def predict_form(name,params,x):
    if name=="linear":return params[0]+params[1]*x
    if name=="logarithmic":return params[0]+params[1]*np.log1p(x)
    if name=="power_law":return params[0]*(x+1)**params[1]-params[2]
    basis=1-np.exp(-x/params[2]) if name=="exponential_saturation" else x/(x+params[2])
    return params[0]+params[1]*basis

def scaling_fits(order,decision):
    datasets=[]
    formed=[r for r in order if r["mean_target_margin"]>0]
    datasets.append(("margin_snr_vs_depth",np.asarray([r["depth_index"] for r in formed],float),np.asarray([r["margin_snr"] for r in formed],float),formed))
    valid=[r for r in decision if r["stable_top1_layer"]!=""]
    datasets.append(("stable_depth_vs_nuisance",np.asarray([r["nuisance_count"] for r in valid],float),np.asarray([r["stable_top1_layer"] for r in valid],float),valid))
    output=[]
    for outcome,x,y,source in datasets:
        for name in ("linear","logarithmic","power_law","exponential_saturation","rational_saturation"):
            params,pred=fit_form(name,x,y);rss=float(np.square(y-pred).sum());k=len(params);r2=1-rss/max(float(np.square(y-y.mean()).sum()),EPS);bic=len(y)*np.log(max(rss/len(y),EPS))+k*np.log(len(y))
            errors=[]
            for seed in sorted({int(r["model_seed"]) for r in source}):
                train=np.asarray([int(r["model_seed"])!=seed for r in source]);p,_=fit_form(name,x[train],y[train]);combined=predict_form(name,p,x[~train])
                errors.extend(np.abs(y[~train]-combined).tolist())
            rng=np.random.default_rng(97);boot=[]
            for _ in range(300):
                idx=rng.integers(0,len(y),len(y));boot.append(fit_form(name,x[idx],y[idx])[0])
            width=max(map(len,boot));ci=[]
            for j in range(width):
                vals=[p[j] for p in boot if len(p)>j];ci.append([float(np.quantile(vals,.025)),float(np.quantile(vals,.975))])
            output.append({"outcome":outcome,"form":name,"fit_parameters":json.dumps(params.tolist()),"parameter_CI":json.dumps(ci),"R2":r2,"BIC":bic,"heldout_MAE":float(np.mean(errors)),"fit_range":f"{x.min()}..{x.max()}","n_independent_cells":len(y)})
    return output

def plots(output,order,decision,updates,covariance,scaling):
    figdir=output/"figures";figdir.mkdir(parents=True,exist_ok=True)
    def depthplot(metric,name,ylabel):
        fig,ax=plt.subplots(figsize=(7.2,4.4));
        for noise in sorted({r["nuisance_count"] for r in order}):
            depths=sorted({r["depth_index"] for r in order});ax.plot(depths,[np.median([r[metric] for r in order if r["nuisance_count"]==noise and r["depth_index"]==d]) for d in depths],marker="o",label=f"nuisance {noise}")
        ax.set_xlabel("sublayer depth");ax.set_ylabel(ylabel);ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(figdir/name,dpi=180);plt.close(fig)
    for metric,name,label in (("mean_target_probability","g1_target_probability_vs_depth.png","target probability"),("mean_target_rank","g1_target_rank_vs_depth.png","target rank"),("mean_target_margin","g1_target_margin_vs_depth.png","target margin"),("fraction_top1","g1_fraction_top1_vs_depth.png","fraction top-1"),("margin_snr","g1_margin_snr_vs_depth.png","margin SNR"),("mean_output_entropy","g1_entropy_vs_margin_snr.png","entropy")) :depthplot(metric,name,label)
    fig,ax=plt.subplots(figsize=(6,5));v=[r for r in decision if r["first_top1_layer"]!="" and r["stable_top1_layer"]!=""];ax.scatter([r["first_top1_layer"] for r in v],[r["stable_top1_layer"] for r in v],alpha=.3);ax.set_xlabel("first top-1 depth");ax.set_ylabel("stable top-1 depth");fig.tight_layout();fig.savefig(figdir/"g1_first_vs_stable_top1_layer.png",dpi=180);plt.close(fig)
    for name,xkey,ykey in (("g1_signal_vs_fluctuation_trajectory.png","mean_target_probability","var_target_probability"),("g1_signal_dispersion_phase_plane.png","correct_target_signal","fluctuation_variance")):
        fig,ax=plt.subplots(figsize=(6,5));ax.scatter([r[xkey] for r in order],[r[ykey] for r in order],c=[r["depth_index"] for r in order],alpha=.5);ax.set_xlabel(xkey);ax.set_ylabel(ykey);fig.colorbar(ax.collections[0],ax=ax,label="depth");fig.tight_layout();fig.savefig(figdir/name,dpi=180);plt.close(fig)
    depthplot("fraction_stably_top1","g1_decision_depth_vs_nuisance.png","fraction stably top-1")
    fig,ax=plt.subplots(figsize=(7,4.4));depths=sorted({r["from_depth"] for r in updates});ax.plot(depths,[np.median([r["margin_update_snr"] for r in updates if r["from_depth"]==d]) for d in depths],marker="o");ax.set_xlabel("from depth");ax.set_ylabel("margin-update SNR");fig.tight_layout();fig.savefig(figdir/"g1_margin_update_snr_vs_depth.png",dpi=180);plt.close(fig)
    matrix=np.zeros((8,8));counts=np.zeros((8,8))
    for r in covariance:matrix[r["layer_i"],r["layer_j"]]+=r["update_covariance"];counts[r["layer_i"],r["layer_j"]]+=1
    fig,ax=plt.subplots(figsize=(6,5));im=ax.imshow(matrix/np.maximum(counts,1));fig.colorbar(im,ax=ax,label="update covariance");fig.tight_layout();fig.savefig(figdir/"g1_layer_update_covariance.png",dpi=180);plt.close(fig)
    for filename,outcome in (("g1_decision_depth_scaling_fits.png","stable_depth_vs_nuisance"),("g1_snr_scaling_fits.png","margin_snr_vs_depth")):
        fig,ax=plt.subplots(figsize=(7,4.4));v=[r for r in scaling if r["outcome"]==outcome];ax.bar([r["form"] for r in v],[r["BIC"] for r in v]);ax.tick_params(axis="x",rotation=25);ax.set_ylabel("BIC (lower better)");fig.tight_layout();fig.savefig(figdir/filename,dpi=180);plt.close(fig)

def run(args):
    repo=Path(args.repo).resolve();output=Path(args.output);config=json.loads(Path(args.config).read_text());device=torch.device(args.device)
    records=sum((generate_records(config,"test",n,config["probe_realizations"]) for n in config["nuisance_counts"]),[]);rows=[];metadata=[]
    for seed in config["model_seeds"]:
        model,_=train_model(config,seed,device);rows.extend(collect(model,records,seed,device));metadata.append(RunMetadata.capture(repo=repo,run_id=f"paper05-correctness-{seed}",config=config,model_id=f"tiny-controlled-seed{seed}",dataset_id="paper05-three-family-v1",seed=seed,device=str(device),dtype="float32",data_hash=stable_hash([asdict(r) for r in records])).as_dict())
    decision=decisions(rows);order=aggregate(rows,decision);raw_updates,updates,covariance=layer_updates(rows);scaling=scaling_fits(order,decision)
    raw=output/"raw/group1_correctness";agg=output/"aggregates/group1";raw.mkdir(parents=True,exist_ok=True);agg.mkdir(parents=True,exist_ok=True)
    compact=[{k:v for k,v in r.items() if k not in {"probabilities","residual"}} for r in rows];write_jsonl(raw/"correctness_observations.jsonl",compact);write_jsonl(raw/"metadata.jsonl",metadata);write_jsonl(raw/"margin_updates.jsonl",raw_updates)
    write_csv(agg/"group1_correctness_order_parameters.csv",order);write_csv(agg/"group1_decision_depth.csv",decision);write_csv(agg/"group1_layer_signal_noise.csv",updates);write_csv(agg/"group1_layer_update_covariance.csv",covariance);write_csv(agg/"group1_scaling_law_fits.csv",scaling)
    write_csv(agg/"group1_entropy_correctness_joint.csv",[{k:r[k] for k in ("model_seed","generator_family","predictive_family_id","nuisance_count","depth_index","mean_output_entropy","mean_target_probability","mean_target_margin","margin_snr","within_family_JS","fraction_top1")} for r in order]);plots(output,order,decision,updates,covariance,scaling)
    manifest={"schema_version":"paper05.correctness.v1","rows":len(rows),"decision_rows":len(decision),"order_parameter_rows":len(order),"update_rows":len(raw_updates),"covariance_rows":len(covariance),"scaling_fits":len(scaling),"sentinel":"empty string for never-top1","artifact_hash":stable_hash({"order":order,"decision":decision,"updates":updates,"covariance":covariance,"scaling":scaling})};atomic_write_json(output/"manifests/group1_correctness.json",manifest);print(json.dumps(manifest,indent=2))

def parse_args():
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/paper05/group1.json");p.add_argument("--repo",default=".");p.add_argument("--output",default="docs/papers/paper0_5/results/three_parts");p.add_argument("--device",default="cpu");return p.parse_args()
if __name__=="__main__":run(parse_args())
