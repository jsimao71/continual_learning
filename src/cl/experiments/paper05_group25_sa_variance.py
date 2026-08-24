"""Experiment 2.5: SA window, predictive span, and nuisance distance."""
from __future__ import annotations
import argparse,hashlib,json,random
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from cl.analysis.equivalence import jensen_shannon_bits
from cl.common.artifacts import atomic_write_json,stable_hash,write_csv
from cl.common.model_adapter import TinyTransformerLM

A=tuple(range(4,8));B=tuple(range(8,12));Y=tuple(range(20,24));NEUTRAL=50;QUERY=52

def regime(span,distance,window,depth):
    if window is None:return "full_attention"
    signal=depth*window>=span;nuisance=depth*window>=distance
    if not signal and nuisance:return "nuisance_reachable_before_full_signal"
    if not signal:return "signal_unreachable"
    if signal and not nuisance:return "signal_only_reachable"
    return "signal_and_nuisance_reachable"

def row_tokens(span,distance,kind,a,b,realization,length,seed,mask="none"):
    rng=random.Random(seed+span*100003+distance*1009+a*101+b*17+realization);t=length-1;tokens=[NEUTRAL]*length;signal=(t-span,t-span+1);tokens[signal[0]]=A[a];tokens[signal[1]]=B[b];nuisance=(t-distance,t-distance+1)
    if set(signal)&set(nuisance):return None
    if kind=="N1":values=(rng.choice(A+B),rng.choice(A+B))
    elif kind=="N2":values=(A[rng.randrange(4)],NEUTRAL)
    elif kind=="N3":
        ca,cb=rng.randrange(4),rng.randrange(4)
        while (ca+cb)%4==(a+b)%4:ca,cb=rng.randrange(4),rng.randrange(4)
        values=(A[ca],B[cb])
    else:values=(NEUTRAL,NEUTRAL)
    tokens[nuisance[0]],tokens[nuisance[1]]=values;tokens[t]=QUERY
    if mask=="nuisance":tokens[nuisance[0]]=tokens[nuisance[1]]=NEUTRAL
    elif mask=="signal":tokens[signal[0]]=tokens[signal[1]]=NEUTRAL
    elif mask=="random":
        candidates=[p for p in range(t) if p not in signal+nuisance];chosen=rng.sample(candidates,2);tokens[chosen[0]]=tokens[chosen[1]]=NEUTRAL
    return {"tokens":tokens,"target":Y[(a+b)%4],"family":f"{a}:{b}","signal_positions":signal,"nuisance_positions":nuisance}

def examples(config,span,split="test",mask="none"):
    rows=[];offset=0 if split=="train" else 10_000_000
    for distance in config["nuisance_distances"]:
        for kind in config["nuisance_types"]:
            for a in range(4):
                for b in range(4):
                    for realization in range(config["nuisance_realizations"]):
                        row=row_tokens(span,distance,kind,a,b,realization,config["sequence_length"],config["dataset_seed"]+offset,mask)
                        if row:rows.append({**row,"predictive_span":span,"nuisance_distance":distance,"nuisance_type":kind,"realization":realization,"split":split,"mask":mask})
    return rows

def sample_batch(rows,size,rng):
    batch=[rows[rng.randrange(len(rows))] for _ in range(size)];return torch.tensor([r["tokens"] for r in batch]),torch.tensor([r["target"] for r in batch])

def train(config,span,depth,seed,window=None):
    torch.manual_seed(seed);model=TinyTransformerLM(config["vocab_size"],config["sequence_length"],config["width"],depth,config["heads"],attention_window=window);optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"]);rows=examples(config,span,"train");rng=random.Random(seed+span*19+depth)
    losses=[]
    for _ in range(config["train_steps"]):
        x,y=sample_batch(rows,config["batch_size"],rng);optimizer.zero_grad(set_to_none=True);logits,_=model(x);loss=F.cross_entropy(logits[:,-1],y);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append(float(loss.detach()))
    return model.eval(),losses

def custom_trace(model,x,window,blocked=()):
    length=x.shape[1];positions=torch.arange(length);state=model.token_embedding(x)+model.position_embedding(positions)[None];mask=model.causal_mask(length,x.device)
    if window!=model.attention_window:
        query=torch.arange(length)[:,None];key=torch.arange(length)[None,:];allowed=key<=query
        if window is not None:allowed&=key>=query-window
        mask=torch.where(allowed,torch.tensor(0.),torch.tensor(float("-inf")))
    for position in blocked:mask[position+1:,position]=float("-inf")
    layers=[]
    for block in model.blocks:
        pre=state;norm=block.norm_sa(pre);state,trace=block(pre,causal_mask=mask,capture=True,intervention=None);layers.append((pre,norm,trace.post_sa,trace.post_block,trace.delta_sa,trace.delta_ff,trace.attention,trace.head_outputs))
    return layers

def state_metrics(model,state,target):
    logits=model.diagnostic_logits(state[:,-1]);p=logits.softmax(-1);chosen=logits.gather(1,target[:,None]).squeeze(1);masked=logits.clone();masked.scatter_(1,target[:,None],float("-inf"));competitor=masked.argmax(-1);margin=chosen-masked.max(-1).values;rank=1+(logits>chosen[:,None]).sum(-1);entropy=-(p*p.clamp_min(1e-12).log()).sum(-1)
    return logits,p,margin,rank,competitor,entropy

def evaluate(model,config,span,depth,seed,trained_window=None):
    rows=examples(config,span);output=[];attention=[];batch_size=128
    windows=config["windows"] if trained_window is None else [trained_window]
    for window in windows:
        for start in range(0,len(rows),batch_size):
            batch=rows[start:start+batch_size];x=torch.tensor([r["tokens"] for r in batch]);target=torch.tensor([r["target"] for r in batch]);layers=custom_trace(model,x,window)
            boundaries=[("initial",layers[0][0])]
            for layer,(pre,norm,post_sa,post_ff,*_) in enumerate(layers):boundaries.extend(((f"l{layer}_preSA",pre),(f"l{layer}_postNorm",norm),(f"l{layer}_postSA",post_sa),(f"l{layer}_postFF",post_ff)))
            with torch.no_grad():
                for boundary_index,(boundary,state) in enumerate(boundaries):
                    logits,p,margin,rank,competitor,entropy=state_metrics(model,state,target)
                    for i,r in enumerate(batch):output.append({"model_seed":seed,"trained_window":trained_window if trained_window is not None else "full","predictive_span":span,"nuisance_distance":r["nuisance_distance"],"window":window if window is not None else "full","depth":depth,"regime":regime(span,r["nuisance_distance"],window,depth),"nuisance_type":r["nuisance_type"],"family":r["family"],"realization":r["realization"],"boundary":boundary,"boundary_index":boundary_index,"layer":-1 if boundary=="initial" else int(boundary[1]),"target_logit":float(logits[i,r["target"]]),"target_probability":float(p[i,r["target"]]),"target_rank":int(rank[i]),"target_margin":float(margin[i]),"strongest_competitor":int(competitor[i]),"top1_correct":int(competitor[i]!=r["target"] and logits[i].argmax()==r["target"]),"output_entropy":float(entropy[i])})
                for layer,values in enumerate(layers):
                    weights=values[6][:,:,-1];heads=values[7][:,:,-1]
                    for i,r in enumerate(batch):attention.append({"model_seed":seed,"trained_window":trained_window if trained_window is not None else "full","predictive_span":span,"nuisance_distance":r["nuisance_distance"],"window":window if window is not None else "full","depth":depth,"nuisance_type":r["nuisance_type"],"family":r["family"],"realization":r["realization"],"layer":layer,"sa_signal_attention_mass":float(weights[i,:,list(r["signal_positions"])].sum(-1).mean()),"sa_nuisance_attention_mass":float(weights[i,:,list(r["nuisance_positions"])].sum(-1).mean()),"sa_update_norm":float(values[4][i,-1].norm()),"ff_update_norm":float(values[5][i,-1].norm()),"per_head_output_norm_mean":float(heads[i].norm(dim=-1).mean())})
    return output,attention

def aggregate(raw):
    keys=("model_seed","trained_window","predictive_span","nuisance_distance","window","depth","regime","nuisance_type","boundary","boundary_index","layer");groups=defaultdict(list);out=[]
    for r in raw:groups[tuple(r[k] for k in keys)].append(r)
    for key,values in groups.items():
        margins=np.array([r["target_margin"] for r in values]);probs=np.array([r["target_probability"] for r in values]);families=sorted({r["family"] for r in values});centroids={f:np.mean([r["target_probability"] for r in values if r["family"]==f]) for f in families};within=float(np.mean([(r["target_probability"]-centroids[r["family"]])**2 for r in values]));between=float(np.var(list(centroids.values())))
        out.append({**dict(zip(keys,key)),"n":len(values),"mean_target_probability":float(probs.mean()),"mean_target_margin":float(margins.mean()),"margin_variance":float(margins.var()),"margin_SNR":float(margins.mean()/(margins.std()+1e-9)),"coefficient_of_variation":float(margins.std()/(abs(margins.mean())+1e-9)),"accuracy":float(np.mean([r["top1_correct"] for r in values])),"mean_output_entropy":float(np.mean([r["output_entropy"] for r in values])),"within_family_probability_variance":within,"between_family_probability_variance":between,"R":between/(within+1e-12),"residual_covariance_trace":"computed in selected trace only"})
    return out

def sublayer(agg):
    lookup={(r["model_seed"],r["trained_window"],r["predictive_span"],r["nuisance_distance"],r["window"],r["depth"],r["nuisance_type"],r["boundary"]):r for r in agg};rows=[]
    bases=sorted({k[:-1] for k in lookup},key=str)
    for base in bases:
        for layer in range(int(base[5])):
            pre=lookup.get((*base,f"l{layer}_preSA"));norm=lookup.get((*base,f"l{layer}_postNorm"));sa=lookup.get((*base,f"l{layer}_postSA"));ff=lookup.get((*base,f"l{layer}_postFF"))
            if not all((pre,norm,sa,ff)):continue
            rows.append({**{k:pre[k] for k in ("model_seed","trained_window","predictive_span","nuisance_distance","window","depth","regime","nuisance_type")},"layer":layer,"pre_sa_variance":pre["margin_variance"],"post_norm_variance":norm["margin_variance"],"post_sa_variance":sa["margin_variance"],"post_ff_variance":ff["margin_variance"],"delta_variance_norm":norm["margin_variance"]-pre["margin_variance"],"delta_variance_sa":sa["margin_variance"]-pre["margin_variance"],"delta_variance_ff":ff["margin_variance"]-sa["margin_variance"],"pre_sa_signal":pre["mean_target_margin"],"post_sa_signal":sa["mean_target_margin"],"post_ff_signal":ff["mean_target_margin"]})
    return rows

def decision_rows(raw):
    groups=defaultdict(list);out=[]
    for r in raw:groups[(r["model_seed"],r["trained_window"],r["predictive_span"],r["nuisance_distance"],r["window"],r["depth"],r["nuisance_type"],r["family"],r["realization"])].append(r)
    for key,v in groups.items():
        v=sorted(v,key=lambda r:r["boundary_index"]);correct=[r["top1_correct"] for r in v];stable=next((i for i in range(len(v)) if all(correct[i:])),"");switch=sum(a["strongest_competitor"]!=b["strongest_competitor"] for a,b in zip(v,v[1:]));reversal=sum(a!=b for a,b in zip(correct,correct[1:]));out.append({**dict(zip(("model_seed","trained_window","predictive_span","nuisance_distance","window","depth","nuisance_type","family","realization"),key)),"stable_decision_depth":stable,"competitor_switch_count":switch,"top1_reversal_count":reversal})
    return out

def aggregate_attention(rows):
    keys=("model_seed","trained_window","predictive_span","nuisance_distance","window","depth","nuisance_type","layer");groups=defaultdict(list);out=[]
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    for key,v in groups.items():out.append({**dict(zip(keys,key)),"n":len(v),**{field:float(np.mean([r[field] for r in v])) for field in ("sa_signal_attention_mass","sa_nuisance_attention_mass","sa_update_norm","ff_update_norm","per_head_output_norm_mean")}})
    return out

def interventions(model,config,span,depth,seed):
    out=[];base=examples(config,span)
    for mask in ("none","nuisance","signal","random"):
        for distance in sorted({r["nuisance_distance"] for r in base}):
            rows=[r for r in base if r["nuisance_distance"]==distance];x=torch.tensor([r["tokens"] for r in rows]);y=torch.tensor([r["target"] for r in rows]);signal=rows[0]["signal_positions"];nuisance=rows[0]["nuisance_positions"]
            if mask=="nuisance":blocked=nuisance
            elif mask=="signal":blocked=signal
            elif mask=="random":blocked=tuple(p for p in range(config["sequence_length"]-2,-1,-1) if p not in signal+nuisance)[:2]
            else:blocked=()
            layers=custom_trace(model,x,None,blocked);_,p,m,_,_,_=state_metrics(model,layers[-1][3],y)
            for i,r in enumerate(rows):out.append({"model_seed":seed,"predictive_span":span,"depth":depth,"mask":mask,"nuisance_distance":r["nuisance_distance"],"nuisance_type":r["nuisance_type"],"family":r["family"],"realization":r["realization"],"target_margin":float(m[i]),"target_probability":float(p[i,r["target"]]),"top1_correct":int(p[i].argmax()==r["target"])})
    return out

def file_hash(path):
    h=hashlib.sha256();h.update(Path(path).read_bytes());return h.hexdigest()

def main(args):
    config=json.loads(Path(args.config).read_text());root=Path(args.output);check=root/"checkpoints";check.mkdir(parents=True,exist_ok=True);surfaces=[];sublayers=[];attention=[];decisions=[];training=[];window_interventions=[];raw_count=0
    if args.interventions_only:
        for seed in config["model_seeds"]:
            for span in config["predictive_spans"]:
                depth=max(config["depths"]);model=TinyTransformerLM(config["vocab_size"],config["sequence_length"],config["width"],depth,config["heads"]);model.load_state_dict(torch.load(check/f"full_s{span}_l{depth}_seed{seed}.pt",map_location="cpu",weights_only=True));window_interventions+=interventions(model.eval(),config,span,depth,seed)
        write_csv(root/"group25_window_interventions.csv",window_interventions);print(json.dumps({"intervention_rows":len(window_interventions),"mode":"attention_key_blocking","artifact_hash":stable_hash(window_interventions)},indent=2));return
    for seed in config["model_seeds"]:
        for span in config["predictive_spans"]:
            for depth in config["depths"]:
                path=check/f"full_s{span}_l{depth}_seed{seed}.pt"
                if path.exists():
                    model=TinyTransformerLM(config["vocab_size"],config["sequence_length"],config["width"],depth,config["heads"]);model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True));model.eval();loss=["checkpoint_reused","checkpoint_reused"]
                else:model,loss=train(config,span,depth,seed);torch.save(model.state_dict(),path)
                training.append({"seed":seed,"span":span,"depth":depth,"trained_window":"full","initial_loss":loss[0],"final_loss":loss[-1],"checkpoint_hash":file_hash(path)});r,a=evaluate(model,config,span,depth,seed);raw_count+=len(r);cell_agg=aggregate(r);surfaces+=cell_agg;sublayers+=sublayer(cell_agg);attention+=aggregate_attention(a);decisions+=decision_rows(r)
                if depth==max(config["depths"]):window_interventions+=interventions(model,config,span,depth,seed)
        for cell in config["local_training_cells"]:
            model,loss=train(config,cell["predictive_span"],cell["depth"],seed,cell["window"]);r,a=evaluate(model,config,cell["predictive_span"],cell["depth"],seed,cell["window"]);raw_count+=len(r);cell_agg=aggregate(r);surfaces+=cell_agg;sublayers+=sublayer(cell_agg);attention+=aggregate_attention(a);decisions+=decision_rows(r);training.append({"seed":seed,**cell,"trained_window":cell["window"],"initial_loss":loss[0],"final_loss":loss[-1]})
    root.mkdir(parents=True,exist_ok=True);write_csv(root/"group25_surface_metrics.csv",surfaces);write_csv(root/"group25_sublayer_variance.csv",sublayers);write_csv(root/"group25_sa_ff_variance_attribution.csv",sublayers);write_csv(root/"group25_attention_mass.csv",attention);write_csv(root/"group25_competitor_dynamics.csv",decisions);write_csv(root/"group25_window_interventions.csv",window_interventions)
    manifest={"schema_version":"paper05.group25.v1","config":config,"training":training,"raw_rows":raw_count,"surface_rows":len(surfaces),"sublayer_rows":len(sublayers),"attention_rows":len(attention),"decision_rows":len(decisions),"artifact_hash":stable_hash({"surface":surfaces,"sublayer":sublayers,"decision":decisions})};atomic_write_json(root/"group25_manifest.json",manifest);print(json.dumps(manifest,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/paper05/group25_sa_window_variance.json");p.add_argument("--output",default="docs/papers/paper0_5/results/group25");p.add_argument("--interventions-only",action="store_true");main(p.parse_args())
