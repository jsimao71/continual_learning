"""Competence-gated nested pattern length-versus-depth experiment."""
from __future__ import annotations
import argparse,itertools,json,math,random
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from cl.common.artifacts import atomic_write_json,stable_hash,write_csv
from cl.common.model_adapter import Intervention,TinyTransformerLM

ROLES=tuple(tuple(range(4+4*i,8+4*i)) for i in range(8));SUPPORT=tuple(range(40,44));LENGTH=tuple(range(60,65));MARKER={"redundant":70,"supportive":71,"necessary":72};TARGET=tuple(range(80,84));DESIGNATE=88;QUERY=89;NEUTRAL=90
def rule(values):return sum(values)%4
def mi(x,y):
    n=len(x);joint=Counter(zip(x,y));cx=Counter(x);cy=Counter(y);return sum((c/n)*math.log2((c/n)/((cx[a]/n)*(cy[b]/n))) for (a,b),c in joint.items())

def validation(config):
    rows=[];fail=[]
    for regime,n in itertools.product(config["regimes"],config["pattern_lengths"]):
        if regime=="necessary":
            values=list(itertools.product(range(4),repeat=n));targets=[rule(v) for v in values];single=max(mi([v[i] for v in values],targets) for i in range(n));proper=max(mi([tuple(v[i] for i in subset) for v in values],targets) for size in range(1,n) for subset in itertools.combinations(range(n),size)) if n<=4 else 0.;full=mi(values,targets);pstar=n
        else:
            values=list(itertools.product(range(4),repeat=2));targets=[rule(v) for v in values];single=max(mi([v[i] for v in values],targets) for i in range(2));proper=single;full=mi(values,targets);pstar=1 if regime=="supportive" and n>2 else 2
        passed=single<1e-9 and proper<1e-9 and abs(full-2)<1e-9;rows.append({"regime":regime,"pattern_length":n,"max_singleton_MI_bits":single,"max_proper_subset_MI_bits":proper,"full_pattern_MI_bits":full,"predictive_order_pstar":pstar,"indispensable_tokens":n if regime=="necessary" else 2,"passed":passed});
        if not passed:fail.append((regime,n))
    result={"schema_version":"paper05.nested_length.validation.v1","passed":not fail,"failures":fail,"note":"For n=6,8 necessary rules, proper-subset independence follows analytically from uniform modular secret sharing; exhaustive singleton and full-pattern checks are retained."};return result,rows

def make_example(config,regime,n,nuisance,span_mode,position_mode,index,split="test"):
    rng=random.Random(config["dataset_seed"]+(0 if split=="train" else 10_000_000)+index*1543+n*101+nuisance*17+MARKER[regime]);values=[rng.randrange(4) for _ in range(n)];target=rule(values if regime=="necessary" else values[:2]);pattern=[ROLES[i][v] for i,v in enumerate(values)]
    if regime=="supportive" and n>2:pattern=[*pattern[:2],*[SUPPORT[target]]*(n-2)]
    if span_mode=="sparse":
        sparse=[]
        for token in pattern:sparse.extend((token,NEUTRAL))
        pattern=sparse[:-1]
    noise=[rng.choice(sum((list(role) for role in ROLES),[])) for _ in range(nuisance)];body=[MARKER[regime],LENGTH[config["pattern_lengths"].index(n)],DESIGNATE,*pattern];space=config["sequence_length"]-len(noise)-len(body)-1;before=space if position_mode=="aligned" else rng.randrange(space+1);tokens=[NEUTRAL]*before+noise+body+[NEUTRAL]*(space-before)+[QUERY]
    return {"tokens":tokens,"target":TARGET[target],"regime":regime,"pattern_length":n,"predictive_order":n if regime=="necessary" else (1 if regime=="supportive" and n>2 else 2),"indispensable_tokens":n if regime=="necessary" else 2,"dependency_span":len(pattern),"nuisance_count":nuisance,"span_mode":span_mode,"position_mode":position_mode,"family_id":f"{regime}:{tuple(values[:2])}:{index%8}","split":split}

def evaluation(config):
    return [make_example(config,r,n,k,s,p,i) for r,n,k,s,p in itertools.product(config["regimes"],config["pattern_lengths"],config["nuisance_counts"],config["span_modes"],config["position_modes"]) for i in range(config["evaluation_examples_per_cell"])]

def training_batch(config,rng,step,total):
    rows=[];lengths=[n for n in config["pattern_lengths"] if n<=4 or step>total*.30]
    for _ in range(config["batch_size"]):
        regime=rng.choices(config["regimes"],weights=(1,1,3))[0];rows.append(make_example(config,regime,rng.choice(lengths),rng.choice(config["nuisance_counts"]),rng.choice(config["span_modes"]),rng.choice(config["position_modes"]),rng.randrange(1_000_000),"train"))
    return torch.tensor([r["tokens"] for r in rows]),torch.tensor([r["target"] for r in rows])

def train(config,depth,seed,steps,path):
    torch.manual_seed(seed);model=TinyTransformerLM(config["vocab_size"],config["sequence_length"],config["width"],depth,config["heads"]);opt=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"]);rng=random.Random(seed+depth);loss=[]
    for step in range(steps):
        x,y=training_batch(config,rng,step,steps);opt.zero_grad(set_to_none=True);z,_=model(x);value=F.cross_entropy(z[:,-1],y);value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();loss.append(float(value.detach()))
    path.parent.mkdir(parents=True,exist_ok=True);torch.save(model.state_dict(),path);return model.eval(),loss

def trace_metrics(model,rows,seed,architecture_depth,batch_size=128):
    output=[]
    for start in range(0,len(rows),batch_size):
        batch=rows[start:start+batch_size];x=torch.tensor([r["tokens"] for r in batch]);target=torch.tensor([r["target"] for r in batch]);logits,trace=model(x,capture=True);states=[trace.layers[0].pre_sa];names=["initial"]
        for layer in trace.layers:states.extend((layer.post_sa,layer.post_block));names.extend(("postSA","postFF"))
        with torch.no_grad():
            for boundary,(name,state) in enumerate(zip(names,states)):
                z=model.diagnostic_logits(state[:,-1]);p=z.softmax(-1);chosen=z.gather(1,target[:,None]).squeeze(1);masked=z.clone();masked.scatter_(1,target[:,None],float("-inf"));margin=chosen-masked.max(-1).values;rank=1+(z>chosen[:,None]).sum(-1);competitor=masked.argmax(-1);entropy=-(p*p.clamp_min(1e-12).log()).sum(-1)
                for i,r in enumerate(batch):output.append({**{k:r[k] for k in ("regime","pattern_length","predictive_order","indispensable_tokens","dependency_span","nuisance_count","span_mode","position_mode","family_id")},"model_seed":seed,"architecture_depth":architecture_depth,"boundary_index":boundary,"boundary":name,"layer":-1 if boundary==0 else (boundary-1)//2,"target_logit":float(chosen[i]),"target_probability":float(p[i,r["target"]]),"target_rank":int(rank[i]),"target_margin":float(margin[i]),"top1_correct":int(z[i].argmax()==target[i]),"strongest_competitor":int(competitor[i]),"entropy":float(entropy[i])})
    return output

def decisions(raw):
    groups=defaultdict(list);out=[]
    for r in raw:groups[(r["model_seed"],r["architecture_depth"],r["regime"],r["pattern_length"],r["nuisance_count"],r["span_mode"],r["position_mode"],r["family_id"])].append(r)
    for key,v in groups.items():
        v=sorted(v,key=lambda r:r["boundary_index"]);correct=[r["top1_correct"] for r in v];first=next((i for i,x in enumerate(correct) if x),"");stable=next((i for i in range(len(v)) if all(correct[i:])),"");out.append({**dict(zip(("model_seed","architecture_depth","regime","pattern_length","nuisance_count","span_mode","position_mode","family_id"),key)),"predictive_order":v[0]["predictive_order"],"dependency_span":v[0]["dependency_span"],"first_top1_layer":first,"stable_top1_layer":stable,"settling_delay":stable-first if isinstance(first,int) and isinstance(stable,int) else "","top1_reversal_count":sum(a!=b for a,b in zip(correct,correct[1:]))})
    return out

def accuracy(raw):
    final=max(r["boundary_index"] for r in raw);groups=defaultdict(list);out=[]
    for r in raw:
        if r["boundary_index"]==final:groups[(r["model_seed"],r["architecture_depth"],r["regime"],r["pattern_length"],r["nuisance_count"],r["span_mode"],r["position_mode"])].append(r)
    for key,v in groups.items():out.append({**dict(zip(("model_seed","architecture_depth","regime","pattern_length","nuisance_count","span_mode","position_mode"),key)),"n":len(v),"accuracy":float(np.mean([r["top1_correct"] for r in v])),"mean_final_margin":float(np.mean([r["target_margin"] for r in v]))})
    return out

def reuse_and_heads(model,config,seed,depth):
    rows=[];heads=[]
    base=[make_example(config,r,2,0,"contiguous","aligned",i) for r in ("redundant","supportive") for i in range(32)]
    for n in config["pattern_lengths"][1:]:
        long=[make_example(config,r,n,0,"contiguous","aligned",i) for r in ("redundant","supportive") for i in range(32)];sx=torch.tensor([r["tokens"] for r in base]);lx=torch.tensor([r["tokens"] for r in long]);_,st=model(sx,capture=True);_,lt=model(lx,capture=True)
        for layer in range(depth):
            sb=st.layers[layer].post_block-st.layers[layer].pre_sa;lb=lt.layers[layer].post_block-lt.layers[layer].pre_sa
            for i,(short,longrow) in enumerate(zip(base,long)):rows.append({"model_seed":seed,"architecture_depth":depth,"regime":longrow["regime"],"short_length":2,"long_length":n,"layer":layer,"block_update_cosine":float(F.cosine_similarity(sb[i,-1][None],lb[i,-1][None])),"sa_update_cosine":float(F.cosine_similarity(st.layers[layer].delta_sa[i,-1][None],lt.layers[layer].delta_sa[i,-1][None])),"ff_update_cosine":float(F.cosine_similarity(st.layers[layer].delta_ff[i,-1][None],lt.layers[layer].delta_ff[i,-1][None]))})
        for layer in range(depth):
            us=[];ul=[]
            ts=torch.tensor([r["target"] for r in base]);tl=torch.tensor([r["target"] for r in long]);bs=model.target_logprob(sx,ts);bl=model.target_logprob(lx,tl)
            for head in range(config["heads"]):
                iv=Intervention(layer,"sa","head_zero",head=head);us.append((bs-model.target_logprob(sx,ts,iv)).numpy());ul.append((bl-model.target_logprob(lx,tl,iv)).numpy())
            us=np.stack(us,1);ul=np.stack(ul,1)
            for i,longrow in enumerate(long):
                a=set(np.argsort(-us[i])[:2]);b=set(np.argsort(-ul[i])[:2]);heads.append({"model_seed":seed,"architecture_depth":depth,"regime":longrow["regime"],"short_length":2,"long_length":n,"layer":layer,"top2_overlap":len(a&b)/2,"new_head_count":len(b-a),"last_newly_recruited_layer":layer if b-a else ""})
    return rows,heads

def main(args):
    config=json.loads(Path(args.config).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=True);valid,info=validation(config);atomic_write_json(out/"nested_length_dataset_validation.json",valid);write_csv(out/"nested_length_mi_validation.csv",info)
    if not valid["passed"]:raise RuntimeError(valid)
    depth=config["smoke_depth"];seed=config["smoke_seed"];model,loss=train(config,depth,seed,config["smoke_train_steps"],out/f"checkpoints/smoke_l{depth}_seed{seed}.pt");raw=trace_metrics(model,evaluation(config),seed,depth);acc=accuracy(raw)
    gate={f"{r}:n{n}":float(np.mean([x["accuracy"] for x in acc if x["regime"]==r and x["pattern_length"]==n])) for r in config["regimes"] for n in config["pattern_lengths"]}
    passed=all(v>=config["minimum_competence"] for v in gate.values());decision=decisions(raw);reuse,heads=reuse_and_heads(model,config,seed,depth);write_csv(out/"nested_length_metrics.csv",raw);write_csv(out/"nested_length_decision_depth.csv",decision);write_csv(out/"nested_length_accuracy_by_model_depth.csv",acc);write_csv(out/"nested_length_update_reuse.csv",reuse);write_csv(out/"nested_length_head_recruitment.csv",heads);write_csv(out/"nested_length_predictive_order.csv",info)
    status={"schema_version":"paper05.nested_length.smoke.v2","gate_unit":"regime_by_pattern_length","gate_passed":passed,"competence":gate,"minimum_threshold":config["minimum_competence"],"preferred_threshold":config["preferred_competence"],"architecture_sweep_status":"eligible" if passed else "blocked: at least one regime-by-length smoke cell failed competence; deeper models are not evidence unless they recover competence","smoke_initial_loss":loss[0],"smoke_final_loss":loss[-1],"artifact_hash":stable_hash({"accuracy":acc,"decisions":decision,"reuse":reuse,"heads":heads})};atomic_write_json(out/"nested_length_smoke_decision.json",status);print(json.dumps(status,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/paper05/nested_length_depth.json");p.add_argument("--output",default="docs/papers/paper0_5/results/nested_length_depth");main(p.parse_args())
