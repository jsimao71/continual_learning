"""Leakage-audited episodic generators for Paper 0.8."""
from __future__ import annotations
from dataclasses import dataclass
import math,random
from collections import Counter

PAD,SEP,QUERY=0,1,2

@dataclass(frozen=True)
class ICLExample:
    tokens: tuple[int,...]
    target: int
    query: int
    family: int
    stage: str
    mapping: tuple[int,...]
    condition: str="correct"

def _symbols(family:int,chain_length:int,offset:int)->tuple[tuple[int,...],tuple[int,...]]:
    # Shared roles make the held-out object tokens trained symbols; ``family``
    # denotes an episode/relation family, not a new output vocabulary.
    return tuple(range(offset,offset+chain_length)),tuple(range(4,4+chain_length))

def episodic_examples(stage:str,families:int,episodes:int,chain_length:int,seed:int,
                      heldout_families:tuple[int,...]=(),offset:int=16,
                      exclude_critical:bool=False,critical_only:bool=False,
                      family_offset:int=0)->list[ICLExample]:
    """Generate stages D0--D4; D4 varies the value permutation every episode."""
    if stage not in {"D0","D1","D2","D3","D4"}:raise ValueError(stage)
    rng=random.Random(seed);rows=[]
    for family in range(family_offset,family_offset+families):
        keys,values=_symbols(family,chain_length,offset)
        for episode in range(episodes):
            mapping=list(values)
            if stage in {"D0","D4"}:rng.shuffle(mapping)
            query_index=(episode+family)%chain_length
            critical=(query_index==chain_length-1 and mapping[query_index]==values[-1])
            if critical_only:
                query_index=chain_length-1;mapping=list(values)
            elif exclude_critical and critical:
                mapping[-1],mapping[0]=mapping[0],mapping[-1]
            if stage in {"D0","D1","D2"}:
                # No paired demonstrations: only independent local chains/noise.
                context=list(keys[:-1] if stage!="D0" else rng.sample(list(keys),chain_length-1))
                context += [SEP]+list(values[:-1] if stage!="D0" else rng.sample(list(values),chain_length-1))
            else:
                context=[]
                for i in range(chain_length):
                    if i!=query_index:context += [keys[i],mapping[i],SEP]
            tokens=tuple(context+[QUERY,keys[query_index]])
            rows.append(ICLExample(tokens,mapping[query_index],keys[query_index],family,stage,tuple(mapping)))
    return rows

def controlled(row:ICLExample,condition:str,seed:int)->ICLExample:
    rng=random.Random(seed);tokens=list(row.tokens);q=tokens.index(QUERY)
    if condition=="none":tokens=tokens[q:]
    elif condition in {"shuffled","reversed"}:
        prefix=tokens[:q]
        pairs=[prefix[i:i+3] for i in range(0,len(prefix),3)] if row.stage in {"D3","D4"} else []
        if condition=="shuffled":rng.shuffle(pairs)
        else:pairs.reverse()
        if condition=="shuffled" and len(pairs)>1:
            vals=[p[1] for p in pairs];vals=vals[1:]+vals[:1]
            for p,v in zip(pairs,vals):p[1]=v
        tokens=[x for p in pairs for x in p]+tokens[q:] if pairs else list(reversed(prefix))+tokens[q:]
    elif condition in {"irrelevant","wrong_chain"}:
        delta=1000 if condition=="irrelevant" else 2*len(row.mapping)
        tokens=[x+delta if x not in {PAD,SEP,QUERY} else x for x in tokens[:q]]+tokens[q:]
    elif condition!="correct":raise ValueError(condition)
    return ICLExample(tuple(tokens),row.target,row.query,row.family,row.stage,row.mapping,condition)

def validation(train:list[ICLExample],test:list[ICLExample])->dict:
    train_families={r.family for r in train};test_families={r.family for r in test}
    direct=Counter((r.query,r.target) for r in train)
    critical=sum(direct[(r.query,r.target)] for r in test)
    targets=Counter(r.target for r in test);n=sum(targets.values())
    entropy=-sum((c/n)*math.log2(c/n) for c in targets.values()) if n else 0
    return {"train_examples":len(train),"test_examples":len(test),"train_test_family_overlap":len(train_families&test_families),"critical_direct_mapping_count":critical,"test_target_entropy_bits":entropy,"test_target_count":len(targets),"target_balance_applicable":len(targets)>1,"passed":not(train_families&test_families) and critical==0}
