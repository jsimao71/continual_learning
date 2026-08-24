"""Train controlled models and measure Dataset V2 decision trajectories."""
from __future__ import annotations
import argparse,hashlib,json,math,random
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from cl.common.artifacts import atomic_write_json,stable_hash,write_csv
from cl.common.model_adapter import TinyTransformerLM
from cl.experiments.paper05_dataset_v2 import generate

def device(config=None):
    if config and config.get("device"):return torch.device(config["device"])
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def file_hash(path):
    digest=hashlib.sha256()
    with open(path,"rb") as handle:
        for chunk in iter(lambda:handle.read(1<<20),b""):digest.update(chunk)
    return digest.hexdigest()

def batch_rows(rows, size, rng):
    chosen=[rows[rng.randrange(len(rows))] for _ in range(size)]
    return torch.tensor([r["tokens"] for r in chosen],dtype=torch.long),torch.tensor([r["target_token"] for r in chosen],dtype=torch.long)

def train_model(config,seed,output,steps=None):
    torch.manual_seed(seed);random.seed(seed);dev=device(config)
    model=TinyTransformerLM(64,config["sequence_length"],**config["model"]).to(dev)
    rows=generate(config,"train");rng=random.Random(seed);optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"])
    losses=[]
    for step in range(steps or config["train_steps"]):
        x,y=batch_rows(rows,config["batch_size"],rng);x=x.to(dev);y=y.to(dev);model.train();optimizer.zero_grad(set_to_none=True)
        logits,_=model(x);loss=F.cross_entropy(logits[:,-1],y);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();losses.append(float(loss.detach().cpu()))
    path=output/f"seed_{seed}.pt";path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":model.to("cpu").state_dict(),"seed":seed,"config":config},path)
    return model,path,losses

def boundaries(model,x):
    logits,trace=model(x,capture=True);states=[trace.layers[0].pre_sa]
    names=["embedding"]
    for index,layer in enumerate(trace.layers):states.extend((layer.post_sa,layer.post_block));names.extend((f"layer_{index}_sa",f"layer_{index}_ff"))
    return names,states,logits

def metric_rows(model,rows,seed,batch_size=128):
    dev=next(model.parameters()).device;out=[];model.eval()
    for start in range(0,len(rows),batch_size):
        batch=rows[start:start+batch_size];x=torch.tensor([r["tokens"] for r in batch],device=dev)
        with torch.no_grad(): names,states,_=boundaries(model,x)
        for boundary_index,(name,state) in enumerate(zip(names,states)):
            with torch.no_grad():
                logits=model.diagnostic_logits(state[:,-1]);prob=logits.softmax(-1);entropy=-(prob*prob.clamp_min(1e-12).log()).sum(-1)
                target=torch.tensor([r["target_token"] for r in batch],device=dev);tp=prob.gather(1,target[:,None]).squeeze(1)
                masked=logits.clone();masked.scatter_(1,target[:,None],float("-inf"));competitor=masked.max(-1).values;margin=logits.gather(1,target[:,None]).squeeze(1)-competitor
                rank=1+(logits>logits.gather(1,target[:,None])).sum(-1);pred=logits.argmax(-1)
                prob_cpu=prob.cpu().numpy();tp_cpu=tp.cpu().numpy();margin_cpu=margin.cpu().numpy();rank_cpu=rank.cpu().numpy();pred_cpu=pred.cpu().numpy();target_cpu=target.cpu().numpy();entropy_cpu=entropy.cpu().numpy();logits_cpu=logits.cpu().numpy()
                for i,r in enumerate(batch):
                    short=r.get("short_target_token","");sp=sm=""
                    if short!="":
                        sp=float(prob_cpu[i,int(short)]);short_logits=logits_cpu[i].copy();short_value=short_logits[int(short)];short_logits[int(short)]=-np.inf;sm=float(short_value-short_logits.max())
                    out.append({"model_seed":seed,"generator_family":r["generator_family"],"predictive_family_id":r["predictive_family_id"],"surface_identity_id":r["surface_identity_id"],
                        "pattern_length":r["pattern_length"],"dependency_span":r["dependency_span"],"nuisance_type":r["nuisance_type"],"nuisance_count":r["nuisance_count"],"nuisance_difficulty":r["nuisance_difficulty"],
                        "competing_pattern_count":r["competing_pattern_count"],"composition_depth":r["composition_depth"],"boundary":name,"boundary_index":boundary_index,
                        "target_probability":float(tp_cpu[i]),"target_rank":int(rank_cpu[i]),"target_margin":float(margin_cpu[i]),"top1_correct":int(pred_cpu[i]==target_cpu[i]),"output_entropy":float(entropy_cpu[i]),
                        "short_target_probability":sp,"long_target_probability":float(tp_cpu[i]),"short_target_margin":sm,"long_target_margin":float(margin_cpu[i])})
    return out

def aggregate(rows):
    keys=("model_seed","generator_family","nuisance_type","nuisance_count","boundary") ;groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    output=[]
    for key,values in groups.items():
        margins=np.array([r["target_margin"] for r in values]);probs=np.array([r["target_probability"] for r in values])
        output.append({**dict(zip(keys,key)),"n":len(values),"mean_target_probability":float(probs.mean()),"mean_target_rank":float(np.mean([r["target_rank"] for r in values])),
            "mean_target_margin":float(margins.mean()),"margin_std":float(margins.std()),"margin_SNR":float(margins.mean()/(margins.std()+1e-9)),
            "fraction_top1":float(np.mean([r["top1_correct"] for r in values])),"mean_output_entropy":float(np.mean([r["output_entropy"] for r in values]))})
    return output

def decisions(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r["model_seed"],r["predictive_family_id"],r["surface_identity_id"])].append(r)
    out=[]
    for (seed,family,identity),values in groups.items():
        values=sorted(values,key=lambda r:r["boundary_index"]);correct=[r["top1_correct"] for r in values]
        first=next((i for i,x in enumerate(correct) if x),"");stable=next((i for i in range(len(correct)) if all(correct[i:])),"")
        crossings=sum(a!=b for a,b in zip(correct,correct[1:]));override="";stable_override=""
        if values[0]["short_target_probability"]!="":
            wins=[r["long_target_probability"]>r["short_target_probability"] for r in values]
            override=next((i for i,x in enumerate(wins) if x),"");stable_override=next((i for i in range(len(wins)) if all(wins[i:])),"")
        out.append({"model_seed":seed,"predictive_family_id":family,"surface_identity_id":identity,"generator_family":values[0]["generator_family"],
            "nuisance_type":values[0]["nuisance_type"],"nuisance_count":values[0]["nuisance_count"],"pattern_length":values[0]["pattern_length"],"dependency_span":values[0]["dependency_span"],
            "first_top1_layer":first,"stable_top1_layer":stable,"settling_delay":stable-first if isinstance(first,int) and isinstance(stable,int) else "","top1_reversal_count":crossings,
            "override_crossing_layer":override,"stable_override_layer":stable_override})
    return out

def final_accuracy(rows):
    final=max(r["boundary_index"] for r in rows);groups=defaultdict(list)
    for r in rows:
        if r["boundary_index"]==final:groups[(r["model_seed"],r["generator_family"],r["nuisance_type"],r["nuisance_count"])].append(r["top1_correct"])
    return [{"model_seed":k[0],"generator_family":k[1],"nuisance_type":k[2],"nuisance_count":k[3],"n":len(v),"accuracy":sum(v)/len(v)} for k,v in groups.items()]

def geometry(model,rows,seed,batch_size=256):
    """Sufficient statistics for representation covariance and predictive JS."""
    dev=next(model.parameters()).device;stats={};model.eval()
    for start in range(0,len(rows),batch_size):
        batch=rows[start:start+batch_size];x=torch.tensor([r["tokens"] for r in batch],device=dev)
        with torch.no_grad():names,states,_=boundaries(model,x)
        for name,state in zip(names,states):
            with torch.no_grad():p=model.diagnostic_logits(state[:,-1]).softmax(-1).cpu().numpy();h=state[:,-1].cpu().numpy()
            for i,r in enumerate(batch):
                key=(r["generator_family"],r["nuisance_type"],r["nuisance_count"],name);family=r["predictive_family_id"]
                if key not in stats:stats[key]={"n":0,"sum_h":np.zeros(h.shape[1]),"sum_hh":np.zeros((h.shape[1],h.shape[1])),"sum_p":np.zeros(p.shape[1]),"sum_entropy":0.,"families":{}}
                s=stats[key];s["n"]+=1;s["sum_h"]+=h[i];s["sum_hh"]+=np.outer(h[i],h[i]);s["sum_p"]+=p[i];s["sum_entropy"]+=float(-(p[i]*np.log2(np.maximum(p[i],1e-12))).sum())
                f=s["families"].setdefault(family,[0,np.zeros(h.shape[1]),np.zeros(p.shape[1])]);f[0]+=1;f[1]+=h[i];f[2]+=p[i]
    out=[]
    for key,s in stats.items():
        mean_h=s["sum_h"]/s["n"];cov=s["sum_hh"]/s["n"]-np.outer(mean_h,mean_h);eig=np.maximum(np.linalg.eigvalsh(cov),0);trace=float(eig.sum());effective=float(trace**2/(np.square(eig).sum()+1e-12))
        mean_p=s["sum_p"]/s["n"];entropy=lambda q:float(-(q*np.log2(np.maximum(q,1e-12))).sum());within=entropy(mean_p)-s["sum_entropy"]/s["n"]
        family_means=[(v[0],v[1]/v[0],v[2]/v[0]) for v in s["families"].values()];between=entropy(mean_p)-sum(n*entropy(mp) for n,_,mp in family_means)/s["n"]
        between_h=sum(n*float(np.square(mh-mean_h).sum()) for n,mh,_ in family_means)/s["n"];within_h=max(trace-between_h,0.)
        out.append({"model_seed":seed,"generator_family":key[0],"nuisance_type":key[1],"nuisance_count":key[2],"boundary":key[3],"n":s["n"],"within_family_JS_bits":within,"between_family_JS_bits":between,"R_between_within":between_h/(within_h+1e-9),"covariance_trace":trace,"effective_rank":effective})
    return out

def load_model(config,path):
    payload=torch.load(path,map_location="cpu",weights_only=False);model=TinyTransformerLM(64,config["sequence_length"],**config["model"]);model.load_state_dict(payload["state_dict"]);return model.to(device(config))

def main(args):
    config=json.loads(Path(args.config).read_text());root=Path(args.output);check=root/"checkpoints";all_rows=[];training=[]
    test=generate(config,"test")
    if args.evaluate_only:
        geo=[]
        for seed in config["model_seeds"]:geo.extend(geometry(load_model(config,check/f"seed_{seed}.pt"),test,seed))
        write_csv(root/"aggregates/dataset_v2_geometry.csv",geo);print(json.dumps({"geometry_rows":len(geo),"artifact_hash":stable_hash(geo)},indent=2));return
    for seed in config["model_seeds"]:
        model,path,losses=train_model(config,seed,check,args.steps);model=model.to(device(config))
        rows=metric_rows(model,test,seed);all_rows.extend(rows);training.append({"model_seed":seed,"steps":args.steps or config["train_steps"],"initial_loss":losses[0],"final_loss":losses[-1],"checkpoint":str(path),"checkpoint_hash":file_hash(path)})
    agg=aggregate(all_rows);decision=decisions(all_rows);accuracy=final_accuracy(all_rows)
    (root/"aggregates").mkdir(parents=True,exist_ok=True);write_csv(root/"aggregates/dataset_v2_layer_metrics.csv",agg);write_csv(root/"aggregates/dataset_v2_decision_depth.csv",decision);write_csv(root/"aggregates/dataset_v2_accuracy.csv",accuracy)
    manifest={"schema_version":"paper05.dataset_v2.training.v1","config":config,"training":training,"observation_rows":len(all_rows),"aggregate_rows":len(agg),"decision_rows":len(decision),"artifact_hash":stable_hash({"aggregate":agg,"decision":decision,"accuracy":accuracy})};atomic_write_json(root/"training_manifest.json",manifest);print(json.dumps(manifest,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/paper05/dataset_v2.json");p.add_argument("--output",default="docs/papers/paper0_5/results/dataset_v2_training");p.add_argument("--steps",type=int);p.add_argument("--evaluate-only",action="store_true");main(p.parse_args())
