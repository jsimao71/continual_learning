"""Pair-disjoint shared-vocabulary proof chains for Paper 0.85 recurrence."""
from __future__ import annotations
from dataclasses import dataclass
import random
from cl.semantic.r0_implication import split_implication_pairs,symbol_ids

PAD,BOS,FACT,RULE,ARROW,QUERY,END,STATE,DERIVE,ANSWER=range(10)
SYMBOL_OFFSET=10

@dataclass(frozen=True)
class ChainTokens:
    chain:tuple[int,...]
    prompt:tuple[int,...]
    pair_split:str
    depth:int

    def target(self,condition):
        if condition=="O0":return (self.chain[-1],)
        if condition=="O1":return (*self.chain[1:],END)
        if condition in {"O2","O3"}:
            out=[STATE,self.chain[0]]
            for node in self.chain[1:]:out.extend((DERIVE,node))
            out.extend((ANSWER,self.chain[-1],END));return tuple(out)
        raise ValueError(condition)

def recurrence_pair_split(symbol_count,seed,test_fraction=.2):
    # r0 symbols are shifted to this serialization's lexical offset.
    raw=symbol_ids(0,symbol_count);train,test=split_implication_pairs(raw,seed,test_fraction)
    shift=SYMBOL_OFFSET-min(raw)
    convert=lambda pairs:tuple((a+shift,b+shift) for a,b in pairs)
    return tuple(x+shift for x in raw),convert(train),convert(test)

def generate_chains(pairs,depth,count,seed,pair_split):
    rng=random.Random(seed);outgoing={}
    for a,b in pairs:outgoing.setdefault(a,[]).append(b)
    viable=[a for a in outgoing]
    rows=[];attempts=0
    while len(rows)<count and attempts<count*100:
        attempts+=1;chain=[rng.choice(viable)]
        for _ in range(depth):
            candidates=[b for b in outgoing.get(chain[-1],()) if b not in chain]
            if not candidates:break
            chain.append(rng.choice(candidates))
        if len(chain)!=depth+1:continue
        prompt=[BOS,FACT,chain[0]]
        for a,b in zip(chain,chain[1:]):prompt.extend((RULE,a,ARROW,b))
        # QUERY is an instruction token, not the answer. The conclusion remains
        # present only as a rule consequent and must be selected/composed.
        prompt.append(QUERY);rows.append(ChainTokens(tuple(chain),tuple(prompt),pair_split,depth))
    if len(rows)<count:raise RuntimeError("could not generate enough split-pure chains")
    return rows

def validate_chain_splits(train_pairs,test_pairs,train_rows,test_rows):
    train=set(train_pairs);test=set(test_pairs)
    train_observed={(a,b) for r in train_rows for a,b in zip(r.chain,r.chain[1:])}
    test_observed={(a,b) for r in test_rows for a,b in zip(r.chain,r.chain[1:])}
    report={"pair_partition_disjoint":train.isdisjoint(test),"observed_pair_leakage":len(train_observed&test_observed),
        "train_chains_within_train_pairs":train_observed<=train,"test_chains_within_heldout_pairs":test_observed<=test,
        "query_field_contains_no_target":all(r.prompt[-1]==QUERY for r in (*train_rows,*test_rows))}
    report["valid"]=report["pair_partition_disjoint"] and report["observed_pair_leakage"]==0 and report["train_chains_within_train_pairs"] and report["test_chains_within_heldout_pairs"] and report["query_field_contains_no_target"]
    return report
