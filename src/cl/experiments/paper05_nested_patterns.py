"""Nested predictive-pattern representation bridge for Papers 0.5/0.6."""
from __future__ import annotations
import argparse,hashlib,itertools,json,math,random
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from cl.analysis.equivalence import jensen_shannon_bits
from cl.common.artifacts import atomic_write_json,stable_hash,write_csv
from cl.common.model_adapter import Intervention,TinyTransformerLM

ROLES=(tuple(range(4,8)),tuple(range(8,12)),tuple(range(12,16)),tuple(range(16,20)));Y=tuple(range(24,28));X=tuple(range(28,32));SUPPORT=tuple(range(32,36));MARKERS={name:40+i for i,name in enumerate(("irrelevant_extension","supportive_extension","refining_extension","override_extension","multi_level_hierarchy"))};SHORT=45;LONG=46;DESIGNATE=47;QUERY=48;NEUTRAL=49

def mi(x,y):
    n=len(x);joint=Counter(zip(x,y));cx=Counter(x);cy=Counter(y);return sum((c/n)*math.log2((c/n)/((cx[a]/n)*(cy[b]/n))) for (a,b),c in joint.items())

def validate(config):
    rows=[];fail=[]
    for n in config["base_lengths"]:
        tuples=list(itertools.product(range(4),repeat=n));targets=[sum(v)%4 for v in tuples];single=max(mi([v[p] for v in tuples],targets) for p in range(n));proper=max(mi([tuple(v[p] for p in subset) for v in tuples],targets) for size in range(1,n) for subset in itertools.combinations(range(n),size));full=mi(tuples,targets);rows.append({"base_length":n,"n":len(tuples),"max_singleton_MI_bits":single,"max_proper_subset_MI_bits":proper,"full_pattern_MI_bits":full,"target_entropy_bits":2.0});
        if single>1e-9 or proper>1e-9 or abs(full-2)>1e-9:fail.append(n)
    return {"schema_version":"paper05.nested.validation.v1","passed":not fail,"failures":fail,"checks":rows,"supportive_extension_exception":"extension token intentionally has I(X;Y)>0 while the core remains sufficient","single_token_hierarchy_exception":"C->Y0 is an explicitly controlled short-rule condition"},rows

def tokens_for(indices):return [ROLES[i][value] for i,value in enumerate(indices)]
def encode(relation,level,content,position,length,seed):
    body=[MARKERS[relation],level,DESIGNATE,*content];space=length-len(body)-1;rng=random.Random(seed);before=space if position=="aligned" else rng.randrange(space+1);return [NEUTRAL]*before+body+[NEUTRAL]*(space-before)+[QUERY]

def alternative(indices,same_target):
    values=list(indices);values[0]=(values[0]+1)%4
    if same_target:values[1]=(values[1]-1)%4
    return tuple(values)

def make_pair(config,relation,n,extension,position,index,split="test"):
    rng=random.Random(config["dataset_seed"]+(0 if split=="train" else 10_000_000)+index*7919+n*101+extension*17+MARKERS[relation]);indices=tuple(rng.randrange(4) for _ in range(n));short_y=sum(indices)%4;extra=[rng.randrange(4) for _ in range(extension)]
    if relation=="irrelevant_extension":long_y=short_y;extra_tokens=[X[v] for v in extra]
    elif relation=="supportive_extension":long_y=short_y;extra_tokens=[SUPPORT[short_y]]*extension
    elif relation in {"refining_extension","multi_level_hierarchy"}:long_y=(short_y+extra[0])%4;extra_tokens=[X[v] for v in extra]
    else:
        extra[0]=1+rng.randrange(3);long_y=(short_y+extra[0])%4;extra_tokens=[X[v] for v in extra]
    core=tokens_for(indices);prefix=index%2==0;long_content=extra_tokens+core if prefix else core+extra_tokens;same=alternative(indices,True);noneq=alternative(indices,False)
    seed=config["dataset_seed"]+index;common={"relation_type":relation,"base_length":n,"extension_length":extension,"position_mode":position,"family_id":f"{relation}:n{n}:k{extension}:f{index}","short_target":Y[short_y],"long_target":Y[long_y],"target_preserved":int(short_y==long_y),"split":split}
    return {**common,"short_tokens":encode(relation,SHORT,core,position,config["sequence_length"],seed),"long_tokens":encode(relation,LONG,long_content,position,config["sequence_length"],seed+1),"same_target_tokens":encode(relation,SHORT,tokens_for(same),position,config["sequence_length"],seed+2),"nonequivalent_tokens":encode(relation,SHORT,tokens_for(noneq),position,config["sequence_length"],seed+3),"same_target":Y[short_y],"nonequivalent_target":Y[sum(noneq)%4],"core_indices":list(indices),"extension_indices":extra}

def evaluation_pairs(config):
    rows=[]
    for relation,n,k,position in itertools.product(config["relation_types"],config["base_lengths"],config["extension_lengths"],config["positions"]):
        for index in range(config["evaluation_families_per_cell"]):rows.append(make_pair(config,relation,n,k,position,index))
    return rows

def training_batch(config,rng):
    pairs=[]
    for _ in range(config["batch_size"]//4):
        relation=rng.choice(config["relation_types"]);n=rng.choice(config["base_lengths"]);k=rng.choice(config["extension_lengths"]);position=rng.choice(config["positions"]);pairs.append(make_pair(config,relation,n,k,position,rng.randrange(1_000_000),"train"))
    x=[];y=[]
    for p in pairs:
        for field,target in (("short_tokens","short_target"),("long_tokens","long_target"),("same_target_tokens","same_target"),("nonequivalent_tokens","nonequivalent_target")):x.append(p[field]);y.append(p[target])
    return torch.tensor(x),torch.tensor(y)

def file_hash(path):
    h=hashlib.sha256();h.update(Path(path).read_bytes());return h.hexdigest()

def train(config,seed,path):
    torch.manual_seed(seed);model=TinyTransformerLM(config["vocab_size"],config["sequence_length"],config["width"],config["layers"],config["heads"]);opt=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"]);rng=random.Random(seed);losses=[]
    for _ in range(config["train_steps"]):
        x,y=training_batch(config,rng);opt.zero_grad(set_to_none=True);logits,_=model(x);loss=F.cross_entropy(logits[:,-1],y);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()))
    path.parent.mkdir(parents=True,exist_ok=True);torch.save(model.state_dict(),path);return model.eval(),losses

def capture(model,ids):
    with torch.no_grad():logits,trace=model(ids,capture=True)
    states=[trace.layers[0].pre_sa]+[layer.post_block for layer in trace.layers];sa=[layer.delta_sa for layer in trace.layers];ff=[layer.delta_ff for layer in trace.layers];heads=[layer.head_outputs for layer in trace.layers]
    return logits,states,sa,ff,heads

def cosine(a,b):return F.cosine_similarity(a.flatten(1),b.flatten(1)).cpu().numpy()
def l2(a,b):return torch.linalg.vector_norm((a-b).flatten(1),dim=1).cpu().numpy()

def metrics_for_seed(model,pairs,seed,trained=True,batch_size=128):
    pair_rows=[];update_rows=[];head_rows=[];cache={}
    for start in range(0,len(pairs),batch_size):
        batch=pairs[start:start+batch_size];forms={name:torch.tensor([p[name+"_tokens"] for p in batch]) for name in ("short","long","same_target","nonequivalent")};capt={name:capture(model,x) for name,x in forms.items()}
        centered={}
        for layer in range(len(capt["short"][1])):
            for _,group in itertools.groupby(range(len(batch)),key=lambda i:(batch[i]["relation_type"],batch[i]["base_length"],batch[i]["extension_length"],batch[i]["position_mode"])):
                ix=list(group);s=capt["short"][1][layer][ix,-1];l=capt["long"][1][layer][ix,-1];sc=s-s.mean(0);lc=l-l.mean(0);cka=float(torch.square(sc.T@lc).sum()/(torch.square(sc.T@sc).sum().sqrt()*torch.square(lc.T@lc).sum().sqrt()).clamp_min(1e-12))
                for local,j in enumerate(ix):centered[(j,layer)]=(float(F.cosine_similarity(sc[local][None],lc[local][None])),cka)
        for i,p in enumerate(batch):
            for layer in range(len(capt["short"][1])):
                short=capt["short"][1][layer][i,-1];long=capt["long"][1][layer][i,-1];same=capt["same_target"][1][layer][i,-1];noneq=capt["nonequivalent"][1][layer][i,-1];alpha=float((long@short)/(short@short).clamp_min(1e-12));context=long-alpha*short
                logits=model.diagnostic_logits(long);target=p["long_target"];masked=logits.clone();masked[target]=float("-inf");margin=float(logits[target]-masked.max());pair_rows.append({**{k:p[k] for k in ("relation_type","base_length","extension_length","position_mode","family_id","target_preserved")},"model_seed":seed,"trained":int(trained),"layer":layer,"raw_residual_cosine":float(F.cosine_similarity(short[None],long[None])),"centered_residual_cosine":centered[(i,layer)][0],"linear_CKA":centered[(i,layer)][1],"same_target_unrelated_cosine":float(F.cosine_similarity(same[None],long[None])),"nonequivalent_cosine":float(F.cosine_similarity(noneq[None],long[None])),"L2_distance":float(torch.linalg.vector_norm(short-long)),"alpha_depth":alpha,"context_component_norm":float(context.norm()),"context_component_target_alignment":float((context@model.lm_head.weight[target])/(context.norm()*model.lm_head.weight[target].norm()).clamp_min(1e-12)),"long_target_margin":margin,"short_target_margin_on_long":float(logits[p["short_target"]]-torch.cat((logits[:p["short_target"]],logits[p["short_target"]+1:])).max()),"top1_correct":int(logits.argmax()==target)})
            for layer in range(len(capt["short"][2])):
                block_s=capt["short"][1][layer+1][i,-1]-capt["short"][1][layer][i,-1];block_l=capt["long"][1][layer+1][i,-1]-capt["long"][1][layer][i,-1];control=capt["nonequivalent"][1][layer+1][i,-1]-capt["nonequivalent"][1][layer][i,-1];nested=float(F.cosine_similarity(block_s[None],block_l[None]));base=float(F.cosine_similarity(control[None],block_l[None]));sd=model.lm_head.weight[p["short_target"]];ld=model.lm_head.weight[p["long_target"]];update_rows.append({**{k:p[k] for k in ("relation_type","base_length","extension_length","position_mode","family_id","target_preserved")},"model_seed":seed,"trained":int(trained),"layer":layer,"block_update_cosine":nested,"sa_update_cosine":float(F.cosine_similarity(capt["short"][2][layer][i,-1][None],capt["long"][2][layer][i,-1][None])),"ff_update_cosine":float(F.cosine_similarity(capt["short"][3][layer][i,-1][None],capt["long"][3][layer][i,-1][None])),"nonequivalent_update_cosine":base,"normalized_retention":(nested-base)/(1-base+1e-9),"short_target_alignment":float((block_l@sd)/(block_l.norm()*sd.norm()).clamp_min(1e-12)),"long_target_alignment":float((block_l@ld)/(block_l.norm()*ld.norm()).clamp_min(1e-12))})
                for head in range(model.blocks[layer].attention.num_heads):
                    hs=capt["short"][4][layer][i,head,-1];hl=capt["long"][4][layer][i,head,-1];head_rows.append({"model_seed":seed,"relation_type":p["relation_type"],"family_id":p["family_id"],"base_length":p["base_length"],"extension_length":p["extension_length"],"position_mode":p["position_mode"],"layer":layer,"head":head,"head_output_cosine":float(F.cosine_similarity(hs[None],hl[None])),"head_output_norm_ratio":float(hl.norm()/hs.norm().clamp_min(1e-12)),"head_target_alignment":float((hl@model.lm_head.weight[p["long_target"]])/(hl.norm()*model.lm_head.weight[p["long_target"]].norm()).clamp_min(1e-12))})
    return pair_rows,update_rows,head_rows

def replacement(model,pairs,seed,batch_size=128):
    rows=[]
    for start in range(0,len(pairs),batch_size):
        batch=pairs[start:start+batch_size];long=torch.tensor([p["long_tokens"] for p in batch]);donors={name:torch.tensor([p[name+"_tokens"] for p in batch]) for name in ("short","same_target","nonequivalent")};base_logits,_=model(long)
        for donor_name,ids in donors.items():
            _,trace=model(ids,capture=True)
            for layer in range(len(model.blocks)):
                for component,update in (("sa",trace.layers[layer].delta_sa),("ff",trace.layers[layer].delta_ff)):
                    changed,_=model(long,intervention=Intervention(layer,component,"replace",update));
                    for i,p in enumerate(batch):
                        target=p["long_target"];base=base_logits[i,-1];new=changed[i,-1];bp=base.log_softmax(-1);np_=new.log_softmax(-1);bm=base[target]-torch.cat((base[:target],base[target+1:])).max();nm=new[target]-torch.cat((new[:target],new[target+1:])).max();rows.append({"model_seed":seed,"relation_type":p["relation_type"],"family_id":p["family_id"],"base_length":p["base_length"],"extension_length":p["extension_length"],"position_mode":p["position_mode"],"target_preserved":p["target_preserved"],"layer":layer,"component":component,"donor_type":donor_name,"target_logprob_delta":float(np_[target]-bp[target]),"target_margin_delta":float(nm-bm),"JS_to_intact":jensen_shannon_bits(base.softmax(-1).detach().numpy(),new.softmax(-1).detach().numpy()),"final_top1_change":int(base.argmax()!=new.argmax())})
    return rows

def head_utilities(model,pairs,seed,batch_size=128):
    rows=[];overlap=[]
    for start in range(0,len(pairs),batch_size):
        batch=pairs[start:start+batch_size];short=torch.tensor([p["short_tokens"] for p in batch]);long=torch.tensor([p["long_tokens"] for p in batch]);targets_s=torch.tensor([p["short_target"] for p in batch]);targets_l=torch.tensor([p["long_target"] for p in batch]);base_s=model.target_logprob(short,targets_s);base_l=model.target_logprob(long,targets_l);utility_s=np.zeros((len(batch),len(model.blocks),model.blocks[0].attention.num_heads));utility_l=np.zeros_like(utility_s)
        for layer in range(len(model.blocks)):
            for head in range(model.blocks[0].attention.num_heads):
                intervention=Intervention(layer,"sa","head_zero",head=head);utility_s[:,layer,head]=(base_s-model.target_logprob(short,targets_s,intervention)).numpy();utility_l[:,layer,head]=(base_l-model.target_logprob(long,targets_l,intervention)).numpy()
        for i,p in enumerate(batch):
            for layer in range(len(model.blocks)):
                a=utility_s[i,layer];b=utility_l[i,layer];rho=None if np.ptp(a)==0 or np.ptp(b)==0 else float(spearmanr(a,b).statistic);ka=set(np.argsort(-a)[:2]);kb=set(np.argsort(-b)[:2]);overlap.append({"model_seed":seed,"relation_type":p["relation_type"],"family_id":p["family_id"],"base_length":p["base_length"],"extension_length":p["extension_length"],"position_mode":p["position_mode"],"layer":layer,"spearman_utility_correlation":rho,"top2_head_overlap":len(ka&kb)/2,"jaccard_overlap":len(ka&kb)/len(ka|kb)})
                for head in range(a.size):rows.append({"model_seed":seed,"relation_type":p["relation_type"],"family_id":p["family_id"],"layer":layer,"head":head,"short_causal_utility":a[head],"long_causal_utility":b[head]})
    return rows,overlap

def divergence(pair_rows,updates):
    out=[];by=defaultdict(list)
    for r in pair_rows:by[(r["model_seed"],r["family_id"])].append(r)
    for key,v in by.items():
        v=sorted(v,key=lambda r:r["layer"]);base=np.array([r["raw_residual_cosine"]-r["nonequivalent_cosine"] for r in v]);threshold=base[0]-.15;div=next((r["layer"] for r,d in zip(v,base) if d<threshold),"");cross=next((r["layer"] for r in v if r["long_target_margin"]>r["short_target_margin_on_long"]),"");out.append({"model_seed":key[0],"family_id":key[1],"relation_type":v[0]["relation_type"],"base_length":v[0]["base_length"],"extension_length":v[0]["extension_length"],"position_mode":v[0]["position_mode"],"representational_divergence_layer":div,"behavioral_target_crossing_layer":cross,"divergence_minus_crossing":div-cross if isinstance(div,int) and isinstance(cross,int) else ""})
    return out

def hierarchy(pair_rows,updates,overlap):
    rows=[]
    for source,metric,kind in ((pair_rows,"raw_residual_cosine","residual"),(updates,"block_update_cosine","update"),(overlap,"spearman_utility_correlation","head_utility")):
        selected=[r for r in source if r["relation_type"]=="multi_level_hierarchy"]
        groups=defaultdict(list)
        for r in selected:
            if r[metric] is not None:groups[(r["model_seed"],r["layer"],r["base_length"],r["extension_length"])].append(float(r[metric]))
        for key,v in groups.items():rows.append({"model_seed":key[0],"layer":key[1],"short_level":key[2],"extension_length":key[3],"metric_type":kind,"similarity":float(np.mean(v)),"n":len(v)})
    return rows

def main(args):
    config=json.loads(Path(args.config).read_text());out=Path(args.output);checks,info=validate(config);out.mkdir(parents=True,exist_ok=True);atomic_write_json(out/"nested_validation.json",checks);write_csv(out/"nested_information_checks.csv",info)
    if not checks["passed"]:raise RuntimeError(checks)
    pairs=evaluation_pairs(config);pair_rows=[];updates=[];replacements=[];heads=[];utilities=[];overlaps=[];training=[];check=out/"checkpoints"
    for seed in config["model_seeds"]:
        path=check/f"seed_{seed}.pt";model,loss=train(config,seed,path);training.append({"seed":seed,"initial_loss":loss[0],"final_loss":loss[-1],"checkpoint_hash":file_hash(path)});p,u,h=metrics_for_seed(model,pairs,seed);pair_rows+=p;updates+=u;heads+=h;replacements+=replacement(model,pairs,seed);util,ov=head_utilities(model,pairs,seed);utilities+=util;overlaps+=ov
        random_model=TinyTransformerLM(config["vocab_size"],config["sequence_length"],config["width"],config["layers"],config["heads"]);rp,ru,_=metrics_for_seed(random_model,pairs,seed,False);pair_rows+=rp;updates+=ru
    div=divergence([r for r in pair_rows if r["trained"]],updates);hier=hierarchy(pair_rows,updates,overlaps);write_csv(out/"nested_pair_metrics.csv",pair_rows);write_csv(out/"nested_update_similarity.csv",updates);write_csv(out/"nested_replacement_results.csv",replacements);write_csv(out/"nested_divergence_layers.csv",div);write_csv(out/"nested_head_similarity.csv",heads);write_csv(out/"nested_head_causal_utilities.csv",utilities);write_csv(out/"nested_head_causal_overlap.csv",overlaps);write_csv(out/"nested_hierarchy_metrics.csv",hier)
    manifest={"schema_version":"paper05.nested.v1","config":config,"pairs":len(pairs),"training":training,"pair_metric_rows":len(pair_rows),"update_rows":len(updates),"replacement_rows":len(replacements),"head_rows":len(heads),"overlap_rows":len(overlaps),"artifact_hash":stable_hash({"pairs":pair_rows,"updates":updates,"replacement":replacements,"overlap":overlaps})};atomic_write_json(out/"nested_manifest.json",manifest);print(json.dumps(manifest,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/paper05/nested_patterns.json");p.add_argument("--output",default="docs/papers/paper0_5/results/nested_patterns");main(p.parse_args())
