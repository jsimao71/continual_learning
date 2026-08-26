"""Exact propositional and first-order ground truth for Paper 0.7."""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import TypeAlias

@dataclass(frozen=True)
class Rule:
    premises: tuple[str,...]
    conclusion: str

@dataclass(frozen=True)
class PropositionProblem:
    facts: tuple[str,...]
    rules: tuple[Rule,...]
    query: str
    entailed: bool
    minimal_proof_depth: int|None
    split: str

def proposition_closure(facts:tuple[str,...],rules:tuple[Rule,...])->dict[str,int]:
    depth={f:0 for f in facts};changed=True
    while changed:
        changed=False
        for rule in rules:
            if all(p in depth for p in rule.premises):
                candidate=1+max((depth[p] for p in rule.premises),default=-1)
                if rule.conclusion not in depth or candidate<depth[rule.conclusion]:depth[rule.conclusion]=candidate;changed=True
    return depth

def validate_proposition(row:PropositionProblem)->bool:
    closure=proposition_closure(row.facts,row.rules);actual=row.query in closure
    return actual==row.entailed and row.minimal_proof_depth==(closure.get(row.query) if actual else None)

def chain_problem(length:int,offset:int=0,distractors:int=0,negative:bool=False,split:str="train")->PropositionProblem:
    if length<1:raise ValueError("length must be positive")
    atoms=[f"p{offset+i}" for i in range(length+1)];rules=[Rule((atoms[i],),atoms[i+1]) for i in range(length)]
    for i in range(distractors):rules.append(Rule((f"d{offset+i}",),f"e{offset+i}"))
    query=f"z{offset}" if negative else atoms[-1]
    return PropositionProblem((atoms[0],),tuple(rules),query,not negative,None if negative else length,split)

@dataclass(frozen=True)
class Var: name:str
@dataclass(frozen=True)
class Fun:
    name:str
    args:tuple["Term",...]=()
Term:TypeAlias=Var|Fun

@dataclass(frozen=True)
class UnificationResult:
    success:bool
    substitution:tuple[tuple[str,Term],...]
    failure_reason:str|None

def _walk(term:Term,subst:dict[str,Term])->Term:
    while isinstance(term,Var) and term.name in subst:term=subst[term.name]
    return term
def _occurs(name:str,term:Term,subst:dict[str,Term])->bool:
    term=_walk(term,subst)
    return isinstance(term,Var) and term.name==name or isinstance(term,Fun) and any(_occurs(name,a,subst) for a in term.args)
def _apply(term:Term,subst:dict[str,Term])->Term:
    term=_walk(term,subst)
    return Fun(term.name,tuple(_apply(a,subst) for a in term.args)) if isinstance(term,Fun) else term

def unify(left:Term,right:Term)->UnificationResult:
    pending=deque([(left,right)]);subst:dict[str,Term]={}
    while pending:
        a,b=map(lambda x:_walk(x,subst),pending.popleft())
        if a==b:continue
        if isinstance(a,Var):
            if _occurs(a.name,b,subst):return UnificationResult(False,(),"occurs_check")
            subst[a.name]=b;continue
        if isinstance(b,Var):pending.appendleft((b,a));continue
        if a.name!=b.name:return UnificationResult(False,(),"symbol_mismatch")
        if len(a.args)!=len(b.args):return UnificationResult(False,(),"arity_mismatch")
        pending.extend(zip(a.args,b.args))
    canonical=tuple((name,_apply(subst[name],subst)) for name in sorted(subst))
    return UnificationResult(True,canonical,None)

def term_depth(term:Term)->int:
    return 0 if isinstance(term,Var) or not term.args else 1+max(term_depth(a) for a in term.args)

def serialize_term(term:Term)->str:
    if isinstance(term,Var):return f"?{term.name}"
    return term.name if not term.args else f"{term.name}({','.join(map(serialize_term,term.args))})"

def validate_unifier(left:Term,right:Term,result:UnificationResult)->bool:
    exact=unify(left,right)
    return result==exact and (not result.success or _apply(left,dict(result.substitution))==_apply(right,dict(result.substitution)))
