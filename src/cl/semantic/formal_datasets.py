"""Validated symbolic dataset ladder for Paper 0.7.

This module owns serialization and split construction; exact reasoning remains in
``formal_logic``.  Records deliberately expose complexity axes for stratified
evaluation rather than relying on token length as a proxy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Any, Iterable

from cl.semantic.formal_logic import (Fun, PropositionProblem, Rule, Term, Var,
    chain_problem, serialize_term, term_depth, unify, validate_proposition,
    validate_unifier)


@dataclass(frozen=True)
class FormalRecord:
    record_id: str
    family: str
    stage: str
    split: str
    input_text: str
    target_text: str
    label: str
    proof_depth: int
    term_depth: int
    arity: int
    variable_count: int
    variable_reuse: int
    branching: int
    distractors: int
    symbol_namespace: str
    ground_truth: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _prop_text(problem: PropositionProblem) -> str:
    facts = " ; ".join(f"FACT {x}" for x in problem.facts)
    rules = " ; ".join(
        f"RULE {' & '.join(rule.premises)} -> {rule.conclusion}" for rule in problem.rules
    )
    return f"{facts} ; {rules} ; QUERY {problem.query}"


def _prop_record(stage: str, split: str, index: int, problem: PropositionProblem,
                 *, branching: int = 1, namespace: str) -> FormalRecord:
    if not validate_proposition(problem):
        raise ValueError(f"invalid proposition ground truth: {stage}/{split}/{index}")
    depth = problem.minimal_proof_depth or 0
    return FormalRecord(f"{stage}:{split}:{index}", "proposition", stage, split,
        _prop_text(problem), "ENTAILED" if problem.entailed else "NOT_ENTAILED",
        str(problem.entailed).lower(), depth, 0, max((len(r.premises) for r in problem.rules), default=0),
        0, 0, branching, sum(r.premises[0].startswith("d") for r in problem.rules if r.premises),
        namespace, {"entailed": problem.entailed, "minimal_proof_depth": problem.minimal_proof_depth})


def proposition_ladder(split: str, count: int, seed: int,
                       chain_lengths: Iterable[int]) -> list[FormalRecord]:
    """Generate balanced P0--P6 examples with split-specific symbol identities."""
    rng = random.Random(seed); rows: list[FormalRecord] = []
    ns = {"train": "tr", "validation": "va", "test": "te"}[split]
    lengths = tuple(chain_lengths)
    for i in range(count):
        stage = f"P{i % 7}"; occurrence=i//7; base = i * 100; prefix = f"{ns}{base}_"
        a, b, c = (prefix+x for x in "abc")
        negative = (occurrence // len(lengths)) % 2 == 1
        if stage == "P0":
            kind=i%3
            if kind==0: premise,query=f"not_not_{a}",a
            elif kind==1: premise,query=f"and_{a}_{b}",a
            else: premise,query=a,f"or_{a}_{b}"
            # The explicit rules make the normalization convention auditable.
            problem=PropositionProblem((premise,),(Rule((premise,),query),),query,True,1,split)
        elif stage == "P1": problem=PropositionProblem((a,),(Rule((a,),b),),b,True,1,split)
        elif stage == "P2": problem=PropositionProblem((f"not_{b}",),(Rule((f"not_{b}",),f"not_{a}"),),f"not_{a}",True,1,split)
        elif stage == "P3": problem=PropositionProblem((a,),(Rule((a,),b),Rule((b,),c)),c,True,2,split)
        elif stage == "P4":
            length=lengths[occurrence%len(lengths)]
            problem=chain_problem(length,offset=10_000*(seed+1)+base,distractors=0,
                                  negative=negative,split=split)
        elif stage == "P5":
            depth=lengths[occurrence%len(lengths)]; atoms=[f"{prefix}p{j}" for j in range(depth+1)]
            rules=[Rule((atoms[j],),atoms[j+1]) for j in range(depth)]
            branches=1+(i%3)
            for j in range(branches): rules.append(Rule((atoms[0],),f"{prefix}branch{j}"))
            for j in range(i%4): rules.append(Rule((f"d{prefix}{j}",),f"e{prefix}{j}"))
            query=f"{prefix}absent" if negative else atoms[-1]
            problem=PropositionProblem((atoms[0],),tuple(rules),query,not negative,None if negative else depth,split)
        else:
            rules=(Rule((a,b),c),); facts=(a,) if negative else (a,b)
            problem=PropositionProblem(facts,rules,c,not negative,None if negative else 1,split)
        rows.append(_prop_record(stage,split,i,problem,branching=(1+i%3 if stage=="P5" else 1),namespace=ns))
    rng.shuffle(rows)
    return rows


def _nested(name: str, inner: Term, depth: int) -> Term:
    for level in range(depth): inner=Fun(f"{name}{level}",(inner,))
    return inner


def _subst_text(result) -> str:
    if not result.success: return f"FAIL:{result.failure_reason}"
    return ",".join(f"?{name}={serialize_term(term)}" for name,term in result.substitution) or "IDENTITY"


def unification_ladder(split: str, count: int, seed: int,
                       functor_depths: Iterable[int]) -> list[FormalRecord]:
    """Generate F0--F9 exact-unification instances with held-out namespaces."""
    rng=random.Random(seed); rows=[]; ns={"train":"tr","validation":"va","test":"te"}[split]
    depths=tuple(functor_depths)
    for i in range(count):
        stage=f"F{i%10}"; occurrence=i//10; x=Var(f"{ns}x{i}"); y=Var(f"{ns}y{i}")
        a=Fun(f"{ns}a{i}"); b=Fun(f"{ns}b{i}"); p=f"{ns}P{i}"
        negative=(occurrence//len(depths))%2==1; depth=depths[occurrence%len(depths)] if stage in {"F4","F8","F9"} else 0
        if stage=="F0": left=right=Fun(p,(a,))
        elif stage=="F1": left,right=Fun(p,(x,)),Fun(p,(a,))
        elif stage=="F2": left,right=Fun(p,(Fun("f",(x,)),)),Fun(p,(Fun("f",(a,)),))
        elif stage=="F3":
            arity=2+i%2; left=Fun(p,tuple([x]+[y]*(arity-1))); right=Fun(p,tuple([a]+[b]*(arity-1)))
        elif stage=="F4": left,right=Fun(p,(_nested("f",x,depth),)),Fun(p,(_nested("f",a,depth),))
        elif stage=="F5": left=Fun(p,(Fun("f",(x,x)),)); right=Fun(p,(Fun("f",(a,b if negative else a)),))
        elif stage=="F6": left=Fun("PAIR",(Fun(p,(x,)),Fun(f"{ns}Q{i}",(x,)))); right=Fun("PAIR",(Fun(p,(a,)),Fun(f"{ns}Q{i}",(a,))))
        elif stage=="F7": left,right=Fun("IMPLIES",(Fun(p,(x,)),Fun(f"{ns}Q{i}",(x,)))),Fun("IMPLIES",(Fun(p,(a,)),Fun(f"{ns}Q{i}",(a,))))
        elif stage=="F8": left,right=Fun("CHAIN",(_nested("f",x,depth),)),Fun("CHAIN",(_nested("f",a,depth),))
        else: left,right=Fun(p,(Fun("f",(x,)),)),Fun(p,(Fun("f",(a if not negative else Fun("g",(a,)),)),))
        result=unify(left,right)
        if not validate_unifier(left,right,result): raise ValueError(f"invalid unifier: {stage}/{split}/{i}")
        max_depth=max(term_depth(left),term_depth(right)); arity=max((len(left.args),len(right.args)))
        text=f"UNIFY {serialize_term(left)} WITH {serialize_term(right)}"
        rows.append(FormalRecord(f"{stage}:{split}:{i}","unification",stage,split,text,
            _subst_text(result),"success" if result.success else result.failure_reason or "failure",
            0,max_depth,arity,2 if stage in {"F3"} else 1,1 if stage in {"F5","F6"} else 0,
            0,0,ns,{"success":result.success,"substitution":_subst_text(result),"failure_reason":result.failure_reason}))
    rng.shuffle(rows); return rows


def validate_dataset(rows: Iterable[FormalRecord]) -> dict[str, Any]:
    materialized=list(rows); ids=[r.record_id for r in materialized]
    namespaces={split:{r.symbol_namespace for r in materialized if r.split==split} for split in ("train","validation","test")}
    disjoint=all(namespaces[a].isdisjoint(namespaces[b]) for a,b in (("train","validation"),("train","test"),("validation","test")))
    return {"rows":len(materialized),"unique_ids":len(ids)==len(set(ids)),"split_namespaces_disjoint":disjoint,
        "stages":sorted({r.stage for r in materialized}),"valid":bool(materialized) and len(ids)==len(set(ids)) and disjoint}
