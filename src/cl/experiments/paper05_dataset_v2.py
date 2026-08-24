"""No-shortcut controlled Dataset V2 for Paper 0.5."""
from __future__ import annotations
import argparse,json,math,random
from collections import Counter,defaultdict
from pathlib import Path
from cl.common.artifacts import atomic_write_json,stable_hash,write_csv

A=tuple(range(4,8));B=tuple(range(8,12));C=tuple(range(12,16));Y=tuple(range(20,24));FUNCTORS=(40,41)
NEUTRAL=50;DESIGNATE=51;QUERY=52;SHARED_POOL=A+B+C

def mutual_information(x,y):
    n=len(x);joint=Counter(zip(x,y));cx=Counter(x);cy=Counter(y);value=0.
    for (a,b),count in joint.items():
        p=count/n;value+=p*math.log2(p/((cx[a]/n)*(cy[b]/n)))
    return value

def entropy(values):
    count=Counter(values);n=len(values);return -sum((v/n)*math.log2(v/n) for v in count.values())

def rule(generator,indices,functor=0):
    if generator in {"balanced_pair","sparse_pair"}:return (indices[0]+indices[1])%4
    if generator=="balanced_ngram3":return sum(indices)%4
    if generator=="functor":return (indices[0]+indices[1])%4 if functor==0 else indices[0]^indices[1]
    if generator in {"nested_short","nested_override"}:return sum(indices)%4
    raise ValueError(generator)

def base_patterns(generator,K=4):
    rows=[]
    if generator in {"balanced_pair","sparse_pair"}:
        for i in range(K):
            for j in range(K):rows.append(((A[i],B[j]),(i,j),0))
    elif generator=="balanced_ngram3":
        for i in range(K):
            for j in range(K):
                for k in range(K):rows.append(((A[i],B[j],C[k]),(i,j,k),0))
    elif generator=="functor":
        for f in range(2):
            for i in range(K):
                for j in range(K):rows.append(((FUNCTORS[f],A[i],B[j]),(i,j),f))
    elif generator=="nested_override":
        for i in range(K):
            for j in range(K):
                for k in range(K):rows.append(((A[i],B[j],C[k]),(i,j,k),0))
    elif generator=="nested_short":
        for j in range(K):
            for k in range(K):rows.append(((B[j],C[k]),(j,k),0))
    return rows

def nuisance_tokens(level,count,rng,target_pair=(0,0)):
    if level=="N0" or count==0:return ()
    if level=="N2":return tuple(rng.choice(SHARED_POOL) for _ in range(count))
    if level=="N3":
        fragments=[]
        for _ in range(count):fragments.append(rng.choice(A+B))
        return tuple(fragments)
    if level=="N4":
        values=[]
        for _ in range(count):values.extend((rng.choice(A),rng.choice(B)))
        return tuple(values)
    raise ValueError(level)

def position_context(pattern,nuisance,mode,length,rng,generator):
    if generator=="sparse_pair":pattern=(pattern[0],NEUTRAL,NEUTRAL,pattern[1])
    designated=(DESIGNATE,*pattern)
    available=length-len(designated)-len(nuisance)-1
    if available<0:raise ValueError("sequence too short")
    before=available if mode=="aligned" else rng.randrange(available+1)
    after=available-before
    return (NEUTRAL,)*before+nuisance+designated+(NEUTRAL,)*after+(QUERY,)

def generate(config,split="train"):
    rows=[];length=config["sequence_length"]
    generators=list(config["generator_families"])+(["nested_short"] if "nested_override" in config["generator_families"] else [])
    for generator in generators:
        for pattern,indices,functor in base_patterns(generator,config["K"]):
            target_index=rule(generator,indices,functor);target=Y[target_index]
            for nuisance_level in config["nuisance_levels"]:
                counts=[0] if nuisance_level=="N0" else [x for x in config["nuisance_counts"] if x>0]
                for count in counts:
                    for mode in config["position_modes"]:
                        for identity in range(config["identity_realizations"]):
                            seed=config["generator_seed"]+(0 if split=="train" else 10_000_000)+stable_int(generator,nuisance_level,mode,pattern,count,identity)
                            rng=random.Random(seed);nuisance=nuisance_tokens(nuisance_level,count,rng,indices[:2]);tokens=position_context(pattern,nuisance,mode,length,rng,generator)
                            rows.append({"generator_family":generator,"predictive_family_id":f"{generator}:{functor}:{'-'.join(map(str,indices))}",
                                "surface_identity_id":f"{split}:{generator}:{nuisance_level}:{mode}:{indices}:{identity}","target_token":target,
                                "pattern_tokens":list(pattern),"pattern_length":len(pattern),"dependency_span":len(pattern) if generator!="sparse_pair" else 4,
                                "predictive_arity":len(indices)+(1 if generator=="functor" else 0),"nuisance_tokens":list(nuisance),"nuisance_count":count,
                                "nuisance_type":nuisance_level,"nuisance_difficulty":int(nuisance_level[1]),"competing_pattern_count":count if nuisance_level=="N4" else 0,
                                "answer_changing_context":False,"continuation_entropy":0.0,"single_token_target_MI":None,"subset_target_MI":None,"full_pattern_target_MI":None,
                                "position_mode":mode,"train_frequency":config["identity_realizations"],"split":split,"generator_seed":seed,
                                "rule_signature":signature(generator,functor),"rule_inputs":list(indices),"rule_output":target,"tokens":list(tokens),
                                "composition_depth":1,"intermediate_latent_class":"",
                                "short_target_token":Y[(indices[-2]+indices[-1])%4] if generator=="nested_override" else ""})
    # Matched answer-changing controls: same B,C suffix, added A changes the valid target.
    for j in range(config["K"]):
        for k in range(config["K"]):
            short=(B[j],C[k]);short_y=Y[(j+k)%4]
            for i in range(1,config["K"]):
                long=(A[i],*short);long_y=Y[(i+j+k)%4]
                rows.append({"generator_family":"nested_override_control","predictive_family_id":f"override:{i}:{j}:{k}","surface_identity_id":f"{split}:override:{i}:{j}:{k}",
                    "target_token":long_y,"pattern_tokens":list(long),"pattern_length":3,"dependency_span":3,"predictive_arity":3,"nuisance_tokens":[],"nuisance_count":0,
                    "nuisance_type":"N6","nuisance_difficulty":6,"competing_pattern_count":1,"answer_changing_context":True,"continuation_entropy":0.0,
                    "single_token_target_MI":None,"subset_target_MI":None,"full_pattern_target_MI":None,"position_mode":"aligned","train_frequency":1,"split":split,"generator_seed":config["generator_seed"],
                    "rule_signature":"nested_mod4_override","rule_inputs":[i,j,k],"rule_output":long_y,"tokens":list(position_context(long,(),"aligned",length,random.Random(0),"nested_override")),
                    "composition_depth":1,"intermediate_latent_class":short_y,"short_target_token":short_y})
    annotate_information(rows)
    return rows

def stable_int(*parts):return int(stable_hash(parts)[:12],16)%1_000_000
def signature(generator,functor):
    return {"balanced_pair":"latin_square_mod4","balanced_ngram3":"modular_sum3","sparse_pair":"sparse_latin_square_mod4","functor":("functor_add_mod4" if functor==0 else "functor_xor2"),"nested_short":"nested_short_mod4","nested_override":"nested_mod4"}[generator]

def annotate_information(rows):
    base=[r for r in rows if not r["answer_changing_context"] and r["nuisance_type"]=="N0" and r["position_mode"]=="aligned"]
    metrics={}
    for generator in {r["generator_family"] for r in base}:
        values=[r for r in base if r["generator_family"]==generator];targets=[r["target_token"] for r in values];width=max(len(r["pattern_tokens"]) for r in values)
        single=max(mutual_information([r["pattern_tokens"][i] for r in values],targets) for i in range(width))
        metrics[generator]=(single,mutual_information([tuple(r["pattern_tokens"][:-1]) for r in values],targets),mutual_information([tuple(r["pattern_tokens"]) for r in values],targets))
    for row in rows:
        source="nested_override" if row["generator_family"]=="nested_override_control" else row["generator_family"]
        row["single_token_target_MI"],row["subset_target_MI"],row["full_pattern_target_MI"]=metrics[source]

def information_checks(rows):
    output=[]
    base=[r for r in rows if not r["answer_changing_context"] and r["nuisance_type"]=="N0" and r["position_mode"]=="aligned"]
    for generator in sorted({r["generator_family"] for r in base}):
        values=[r for r in base if r["generator_family"]==generator];targets=[r["target_token"] for r in values];maxlen=max(len(r["pattern_tokens"]) for r in values)
        single=[]
        for position in range(maxlen):single.append(mutual_information([r["pattern_tokens"][position] for r in values],targets))
        subset=[tuple(r["pattern_tokens"][:-1]) for r in values];full=[tuple(r["pattern_tokens"]) for r in values]
        output.append({"generator_family":generator,"n":len(values),"target_entropy_bits":entropy(targets),"max_singleton_target_MI_bits":max(single),
            "subset_target_MI_bits":mutual_information(subset,targets),"full_pattern_target_MI_bits":mutual_information(full,targets),"single_token_values":json.dumps(single)})
    # Conditional nuisance check within each exact predictive family.
    nuisance=[]
    for family in sorted({r["predictive_family_id"] for r in rows if r["nuisance_type"] in {"N2","N3","N4"}}):
        values=[r for r in rows if r["predictive_family_id"]==family and r["nuisance_type"] in {"N2","N3","N4"}]
        nuisance.append(mutual_information([tuple(r["nuisance_tokens"]) for r in values],[r["target_token"] for r in values]))
    return output,max(nuisance,default=0.)

def validate(config,train,test):
    checks,max_cond=information_checks(train);threshold=config["singleton_mi_threshold_bits"];fail=[]
    for row in checks:
        if row["max_singleton_target_MI_bits"]>threshold:fail.append(f"singleton shortcut {row['generator_family']}")
        if row["full_pattern_target_MI_bits"]<1.9:fail.append(f"full pattern weak {row['generator_family']}")
    if max_cond>threshold:fail.append("conditional nuisance MI")
    filler_rows=[r for r in train if r["nuisance_tokens"]]
    filler_mi=mutual_information([r["nuisance_tokens"][0] for r in filler_rows],[r["target_token"] for r in filler_rows])
    if filler_mi>threshold:fail.append("filler target MI")
    if {r["surface_identity_id"] for r in train}&{r["surface_identity_id"] for r in test}:fail.append("split overlap")
    controls=[r for r in train if r["answer_changing_context"]]
    if not controls or any(r["target_token"]==r["short_target_token"] for r in controls):fail.append("answer-changing control")
    targets=Counter(r["target_token"] for r in train if not r["answer_changing_context"])
    if max(targets.values())!=min(targets.values()):fail.append("target imbalance")
    # Position mode is crossed with every underlying predictive row, so it cannot leak target.
    position_mi=mutual_information([r["position_mode"] for r in train if not r["answer_changing_context"]],[r["target_token"] for r in train if not r["answer_changing_context"]])
    competing_mi=mutual_information([r["competing_pattern_count"] for r in train if not r["answer_changing_context"]],[r["target_token"] for r in train if not r["answer_changing_context"]])
    validation={"schema_version":"paper05.dataset_v2.validation.v1","passed":not fail,"failures":fail,"train_rows":len(train),"test_rows":len(test),
        "singleton_threshold_bits":threshold,"max_conditional_nuisance_target_MI_bits":max_cond,"position_mode_target_MI_bits":position_mi,
        "filler_target_MI_bits":filler_mi,"competing_count_target_MI_bits":competing_mi,"train_test_identity_overlap":0,"answer_changing_controls":len(controls),
        "target_counts":dict(targets),"artifact_hash":stable_hash({"train":train,"test":test,"checks":checks})}
    return validation,checks

def examples_markdown(rows):
    lines=["# Dataset V2 examples","","Tokens are integer symbols; `51` designates the predictive pattern and `52` is the query marker.",""]
    for generator in sorted({r["generator_family"] for r in rows}):
        r=next(x for x in rows if x["generator_family"]==generator);lines.extend([f"## {generator}","",f"- rule: `{r['rule_signature']}`",f"- inputs: `{r['rule_inputs']}` -> target `{r['target_token']}`",f"- nuisance: `{r['nuisance_type']}` / `{r['nuisance_tokens']}`",f"- tokens: `{r['tokens']}`",""])
    return "\n".join(lines)

def run(args):
    config=json.loads(Path(args.config).read_text());train=generate(config,"train");test=generate(config,"test");validation,checks=validate(config,train,test);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    atomic_write_json(out/"dataset_v2_validation.json",validation);write_csv(out/"dataset_v2_information_checks.csv",checks);(out/"dataset_v2_examples.md").write_text(examples_markdown(train),encoding="utf-8")
    if not validation["passed"]:raise RuntimeError(json.dumps(validation,indent=2))
    print(json.dumps(validation,indent=2))

def parse_args():
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/paper05/dataset_v2.json");p.add_argument("--output",default="docs/papers/paper0_5/results/dataset_v2");return p.parse_args()
if __name__=="__main__":run(parse_args())
