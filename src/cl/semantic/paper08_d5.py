"""Leakage-audited identifiable episodes for learned Paper 0.8 D5 tests."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,math,random
from .paper08_taxonomy import FunctionHypothesis,answer_distribution,entropy,posterior,hypotheses

FAMILIES=("successor","add_n","multiply_a","affine","square")
PARAMETRIC={"add_n","multiply_a","affine"}

@dataclass(frozen=True)
class D5Episode:
    episode_id:str;split:str;family:str;modulus:int;parameters:tuple[int,...]
    demonstrations:tuple[tuple[int,int],...];query:int;target:int
    consistent_hypotheses:int;answer_entropy_bits:float;minimum_examples_needed:int

def parameter_partition(modulus:int,family:str,seed:int,test_fraction:float=.25):
    pool=list(hypotheses(family,modulus))
    if family not in PARAMETRIC:return tuple(pool),()
    # A stable hash avoids Python hash randomization and guarantees nonempty splits.
    pool.sort(key=lambda h:hashlib.sha256(f"{seed}:{modulus}:{family}:{h.parameters}".encode()).hexdigest())
    n=max(1,round(len(pool)*test_fraction));return tuple(pool[n:]),tuple(pool[:n])

def declared_prior(modulus:int):
    return tuple(h for family in FAMILIES for h in hypotheses(family,modulus))

def _id(split,family,k,params,seed):
    return hashlib.sha256(f"{split}:{family}:{k}:{params}:{seed}".encode()).hexdigest()[:16]

def identifiable_episode(h:FunctionHypothesis,split:str,seed:int,prior=None)->D5Episode:
    prior=tuple(prior or declared_prior(h.modulus));rng=random.Random(seed);query=rng.randrange(h.modulus)
    inputs=[x for x in range(h.modulus) if x!=query];rng.shuffle(inputs);demos=[]
    for x in inputs:
        demos.append((x,h(x)));post=posterior(prior,demos);dist=answer_distribution(post,query)
        if entropy(dist)==0:break
    else:raise ValueError("declared prior does not identify query answer")
    target=h(query);uncertainty=entropy(dist)
    if uncertainty!=0 or any(x==query for x,_ in demos):raise ValueError("invalid D5 episode")
    return D5Episode(_id(split,h.family,h.modulus,h.parameters,seed),split,h.family,h.modulus,h.parameters,
        tuple(demos),query,target,len(post),uncertainty,len(demos))

def episode_signature(row):return (row.modulus,row.demonstrations,row.query,row.target)

def generate_split(moduli,split,episodes_per_hypothesis,seed,test_fraction=.25,excluded_signatures=None):
    rows=[];split_offset={"train":0,"seen_parameter":10_000_000,"unseen_parameter":20_000_000,"unseen_family":30_000_000}[split]
    used=set(excluded_signatures or ())
    for k in moduli:
      prior=declared_prior(k)
      for family in FAMILIES:
        train,test=parameter_partition(k,family,seed,test_fraction)
        if split=="seen_parameter":members=train
        elif split=="unseen_parameter":members=test
        elif split=="unseen_family":members=hypotheses("square",k) if family=="square" else ()
        elif split=="train":members=() if family=="square" else train
        else:raise ValueError(split)
        # Square is held out as a family, not also counted as seen parameters.
        if family=="square" and split in {"train","seen_parameter","unseen_parameter"}:members=()
        for j,h in enumerate(members):
          for repeat in range(episodes_per_hypothesis):
            for attempt in range(10000):
                row=identifiable_episode(h,split,seed+split_offset+100003*k+1009*j+repeat+104729*attempt,prior)
                signature=episode_signature(row)
                if signature not in used:used.add(signature);rows.append(row);break
            else:raise ValueError(f"cannot construct disjoint {split} episodes for {family} mod {k}")
    if not rows:raise ValueError(f"empty split {split}")
    ids=[r.episode_id for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError("episode-id collision")
    return rows

def split_audit(train,seen,unseen,unseen_family):
    groups={"train":train,"seen_parameter":seen,"unseen_parameter":unseen,"unseen_family":unseen_family}
    ids={name:{r.episode_id for r in rows} for name,rows in groups.items()}
    if any(ids[a]&ids[b] for i,a in enumerate(ids) for b in list(ids)[i+1:]):raise ValueError("episode leakage")
    signatures={name:{episode_signature(r) for r in rows} for name,rows in groups.items()}
    if any(signatures[a]&signatures[b] for i,a in enumerate(signatures) for b in list(signatures)[i+1:]):raise ValueError("serialized episode leakage")
    train_params={(r.modulus,r.family,r.parameters) for r in train}
    unseen_params={(r.modulus,r.family,r.parameters) for r in unseen}
    if train_params&unseen_params:raise ValueError("held-out parameter leakage")
    if any(r.family!="square" for r in unseen_family) or any(r.family=="square" for r in train):raise ValueError("family leakage")
    all_rows=sum(groups.values(),[])
    return {"train_episodes":len(train),"seen_parameter_episodes":len(seen),"unseen_parameter_episodes":len(unseen),
        "unseen_family_episodes":len(unseen_family),"episode_id_overlap":0,"serialized_episode_overlap":0,"heldout_parameter_overlap":0,
        "square_training_episodes":0,"all_answer_entropy_zero":all(r.answer_entropy_bits==0 for r in all_rows),
        "bayes_oracle_ceiling":1.0,"max_identifying_examples":max(r.minimum_examples_needed for r in all_rows),
        "target_query_pair_explicit_in_demonstrations":False}
