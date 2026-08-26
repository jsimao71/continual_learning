"""Leakage-controlled one-step implication instrument for Paper 0.85."""
from __future__ import annotations
from dataclasses import dataclass
import random

SPECIAL={"PAD":0,"BOS":1,"FACT":2,"RULE":3,"ARROW":4,"QUERY":5,"NO_ENTAIL":6}
SYMBOL_OFFSET=7

@dataclass(frozen=True)
class R0Example:
    tokens: tuple[int,...]
    target: int
    condition: str
    lhs: int
    rhs: int
    fact: int

def symbol_ids(start:int,count:int)->tuple[int,...]: return tuple(range(SYMBOL_OFFSET+start,SYMBOL_OFFSET+start+count))

def make_example(lhs:int,rhs:int,fact:int,condition:str)->R0Example:
    return R0Example((SPECIAL["BOS"],SPECIAL["FACT"],fact,SPECIAL["RULE"],lhs,SPECIAL["ARROW"],rhs,SPECIAL["QUERY"]),
        rhs if fact==lhs else SPECIAL["NO_ENTAIL"],condition,lhs,rhs,fact)

def generate_r0(symbols:tuple[int,...],examples:int,seed:int,conditions=("entailed","fact_mismatch","rule_lhs_mismatch"))->list[R0Example]:
    if len(symbols)<4: raise ValueError("at least four symbols required")
    rng=random.Random(seed); rows=[]
    for i in range(examples):
        condition=conditions[i%len(conditions)]; lhs,rhs,other,alternate=rng.sample(symbols,4)
        if condition=="entailed": row=make_example(lhs,rhs,lhs,condition)
        elif condition=="fact_mismatch": row=make_example(lhs,rhs,other,condition)
        elif condition=="rule_lhs_mismatch": row=make_example(other,rhs,lhs,condition)
        elif condition=="consequent_swap": row=make_example(lhs,alternate,lhs,condition)
        else: raise ValueError(condition)
        rows.append(row)
    return rows

def validate_r0(train_symbols:tuple[int,...],test_symbols:tuple[int,...],rows:list[R0Example])->dict:
    disjoint=set(train_symbols).isdisjoint(test_symbols)
    exact=all(row.target==(row.rhs if row.fact==row.lhs else SPECIAL["NO_ENTAIL"]) for row in rows)
    balanced=max((sum(r.condition==c for r in rows) for c in set(r.condition for r in rows)),default=0)-min(
        (sum(r.condition==c for r in rows) for c in set(r.condition for r in rows)),default=0)<=1
    return {"train_test_symbol_identities_disjoint":disjoint,"exact_targets":exact,"conditions_balanced":balanced,
        "valid":disjoint and exact and balanced}

def split_implication_pairs(symbols:tuple[int,...],seed:int,test_fraction:float=.2):
    """Partition ordered non-identity pairs while retaining shared lexical support."""
    pairs=[(a,b) for a in symbols for b in symbols if a!=b];rng=random.Random(seed);rng.shuffle(pairs)
    cut=max(1,int(len(pairs)*test_fraction));test=tuple(pairs[:cut]);train=tuple(pairs[cut:])
    return train,test

def coverage_examples(train_pairs:tuple[tuple[int,int],...],symbols:tuple[int,...])->list[R0Example]:
    """Return positives covering every symbol as antecedent and consequent."""
    rows=[]
    for symbol in symbols:
        lhs_pair=next(pair for pair in train_pairs if pair[0]==symbol)
        rhs_pair=next(pair for pair in train_pairs if pair[1]==symbol)
        rows.extend((make_example(*lhs_pair,lhs_pair[0],"entailed"),make_example(*rhs_pair,rhs_pair[0],"entailed")))
    return rows

def generate_r0_pairs(pairs:tuple[tuple[int,int],...],symbols:tuple[int,...],examples:int,seed:int,
                      conditions=("entailed","fact_mismatch","rule_lhs_mismatch","consequent_swap")):
    rng=random.Random(seed);rows=[]
    by_rhs={rhs:[pair for pair in pairs if pair[1]==rhs] for rhs in symbols}
    for i in range(examples):
        condition=conditions[i%len(conditions)];lhs,rhs=rng.choice(pairs)
        if condition in {"entailed","consequent_swap"}: row=make_example(lhs,rhs,lhs,condition)
        elif condition=="fact_mismatch":
            fact=rng.choice(tuple(s for s in symbols if s!=lhs));row=make_example(lhs,rhs,fact,condition)
        elif condition=="rule_lhs_mismatch":
            alternatives=[pair for pair in by_rhs[rhs] if pair[0]!=lhs]
            rule_lhs,_=rng.choice(alternatives);row=make_example(rule_lhs,rhs,lhs,condition)
        else: raise ValueError(condition)
        rows.append(row)
    return rows

def validate_pair_split(symbols,train_pairs,test_pairs,training_rows,test_rows):
    train_set=set(train_pairs);test_set=set(test_pairs);observed_train={(r.lhs,r.rhs) for r in training_rows};observed_test={(r.lhs,r.rhs) for r in test_rows}
    input_coverage={r.fact for r in training_rows}|{r.lhs for r in training_rows}
    positive_target_coverage={r.target for r in training_rows if r.target!=SPECIAL["NO_ENTAIL"]}
    report={"pair_partition_disjoint":train_set.isdisjoint(test_set),"observed_pair_leakage":len(observed_train&observed_test),
        "all_symbols_input_covered":set(symbols)<=input_coverage,"all_symbols_positive_target_covered":set(symbols)<=positive_target_coverage,
        "test_pairs_within_holdout":observed_test<=test_set,"train_pairs_within_train":observed_train<=train_set,
        "exact_test_targets":all(r.target==(r.rhs if r.fact==r.lhs else SPECIAL["NO_ENTAIL"]) for r in test_rows)}
    report["valid"]=all(v for k,v in report.items() if k!="observed_pair_leakage") and report["observed_pair_leakage"]==0
    return report
