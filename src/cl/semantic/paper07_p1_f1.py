"""Leakage-audited P1 modus-ponens and F1 variable-binding gates."""
from __future__ import annotations

from dataclasses import dataclass
import random

from cl.semantic.paper07_gates import PAD, FACT, RULE, ARROW, QUERY, UNIFY, WITH, PRED, SYMBOL, YES, NO

VAR, BIND, CONST, DISTRACTOR = 11, 12, 13, 14
SYMBOL_COUNT = 12


@dataclass(frozen=True)
class StepGateExample:
    stage: str
    split: str
    namespace: str
    example_id: str
    tokens: tuple[int, ...]
    target: int
    label: str
    raw_symbols: tuple[str, ...]
    template: str


def _pad(tokens: list[int], length: int) -> tuple[int, ...]:
    if len(tokens) > length:
        raise ValueError("P1/F1 sequence exceeds configured length")
    return tuple([PAD] * (length - len(tokens)) + tokens)


def p1_examples(split: str, count: int, seed: int, length: int = 20) -> list[StepGateExample]:
    if split not in {"train", "validation", "test"} or count % 6:
        raise ValueError("P1 requires a valid split and count divisible by six")
    rng=random.Random(seed);namespace={"train":"tr","validation":"va","test":"te"}[split];rows=[]
    templates=("fact_match", "rule_lhs_match", "consequent_match")
    for index in range(count):
        positive=index%2==0;template=templates[(index//2)%len(templates)]
        a,b,c,d=rng.sample(range(SYMBOL,SYMBOL+SYMBOL_COUNT),4)
        fact,lhs,consequent,query=a,a,b,b
        if not positive:
            if template=="fact_match": fact=c
            elif template=="rule_lhs_match": lhs=c
            else: query=c
        tokens=[FACT,fact,RULE,lhs,ARROW,consequent,QUERY,query]
        raw=tuple(f"{namespace}_{seed}_{index}_{role}" for role in ("a","b","c","d"))
        rows.append(StepGateExample("P1",split,namespace,f"P1:{split}:{index}",_pad(tokens,length),
                                    YES if positive else NO,"entailed" if positive else "not_entailed",raw,template))
    rng.shuffle(rows);return rows


def f1_examples(split: str, count: int, seed: int, length: int = 20) -> list[StepGateExample]:
    if split not in {"train", "validation", "test"} or count % 48:
        raise ValueError("F1 requires a valid split and count divisible by 48")
    rng=random.Random(seed);namespace={"train":"tr","validation":"va","test":"te"}[split];rows=[]
    templates=("canonical", "reversed", "constant_distractor", "predicate_distractor")
    for index in range(count):
        target=SYMBOL+(index%SYMBOL_COUNT);available=[v for v in range(SYMBOL,SYMBOL+SYMBOL_COUNT) if v!=target]
        predicate,other_predicate,distractor=rng.sample(available,3);template=templates[(index//SYMBOL_COUNT)%4]
        variable=available[(index+3)%len(available)]
        left=[PRED,predicate,VAR,variable];right=[PRED,predicate,CONST,target]
        if template=="reversed": left,right=right,left
        tokens=[UNIFY,*left,WITH,*right]
        if template=="constant_distractor": tokens += [DISTRACTOR,CONST,distractor]
        elif template=="predicate_distractor": tokens += [DISTRACTOR,PRED,other_predicate,CONST,distractor]
        tokens += [QUERY,BIND,variable]
        raw=tuple(f"{namespace}_{seed}_{index}_{role}" for role in ("predicate","other_predicate","constant","variable"))
        rows.append(StepGateExample("F1",split,namespace,f"F1:{split}:{index}",_pad(tokens,length),
                                    target,"binding",raw,template))
    rng.shuffle(rows);return rows


def step_gate_examples(stage: str, split: str, count: int, seed: int, length: int = 20) -> list[StepGateExample]:
    if stage=="P1": return p1_examples(split,count,seed,length)
    if stage=="F1": return f1_examples(split,count,seed,length)
    raise ValueError(stage)


def validate_step_gates(rows: list[StepGateExample]) -> dict:
    expected={"P1":{"fact_match","rule_lhs_match","consequent_match"},
              "F1":{"canonical","reversed","constant_distractor","predicate_distractor"}}
    namespaces={split:{r.namespace for r in rows if r.split==split} for split in ("train","validation","test")}
    template_counts={};checks={}
    for stage in ("P1","F1"):
      for split in ("train","validation","test"):
        selected=[r for r in rows if r.stage==stage and r.split==split]
        template_counts[f"{stage}:{split}"]={t:sum(r.template==t for r in selected) for t in sorted(expected[stage])}
        checks[f"{stage}:{split}:templates"]={r.template for r in selected}==expected[stage]
        if stage=="P1":
            checks[f"{stage}:{split}:balanced"]=(sum(r.target==YES for r in selected)==sum(r.target==NO for r in selected))
            for template in expected[stage]:
                subset=[r for r in selected if r.template==template]
                checks[f"{stage}:{split}:{template}:balanced"]=(sum(r.target==YES for r in subset)==sum(r.target==NO for r in subset))
        else:
            target_counts={target:sum(r.target==target for r in selected) for target in range(SYMBOL,SYMBOL+SYMBOL_COUNT)}
            checks[f"{stage}:{split}:target_balanced"]=len(set(target_counts.values()))==1
            checks[f"{stage}:{split}:target_in_prompt"]=all(r.target in r.tokens for r in selected)
    checks["split_namespaces_disjoint"]=all(namespaces[a].isdisjoint(namespaces[b]) for a,b in
        (("train","validation"),("train","test"),("validation","test")))
    checks["unique_ids"]=len({r.example_id for r in rows})==len(rows)
    return {"valid":all(checks.values()),"rows":len(rows),"checks":checks,"template_counts":template_counts,
            "P1_constant_baseline_accuracy":.5,"F1_symbol_baseline_accuracy":1/SYMBOL_COUNT}
