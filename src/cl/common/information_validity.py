"""Discrete information-validity checks for learned-machine task generators."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import asdict, dataclass
from math import log2
from typing import Hashable, Iterable

Observation = tuple[Hashable, Hashable, Hashable, Hashable]

@dataclass(frozen=True)
class InformationValidity:
    h_y_given_x: float
    h_y_given_xd: float
    conditional_mutual_information: float
    declared_prior: str
    bayes_ceiling: float
    consistent_hypotheses: int | None
    minimum_context_identifiability: int | None
    observations: int

    def as_dict(self): return asdict(self)

def _conditional_entropy(rows: list[Observation], condition: tuple[int,...]) -> float:
    groups: dict[tuple[Hashable,...],dict[Hashable,int]]=defaultdict(lambda:defaultdict(int))
    for row in rows: groups[tuple(row[i] for i in condition)][row[2]] += 1
    total=len(rows); result=0.0
    for counts in groups.values():
        n=sum(counts.values()); entropy=-sum((c/n)*log2(c/n) for c in counts.values())
        result += (n/total)*entropy
    return result

def _bayes_ceiling(rows: list[Observation]) -> float:
    groups: dict[tuple[Hashable,...],dict[Hashable,int]]=defaultdict(lambda:defaultdict(int))
    for x,d,y,r in rows: groups[(x,d,r)][y] += 1
    return sum(max(counts.values()) for counts in groups.values())/len(rows)

def information_validity(rows: Iterable[Observation], declared_prior: str,
                         consistent_hypotheses: int|None=None,
                         minimum_context_identifiability: int|None=None) -> InformationValidity:
    """Compute empirical discrete quantities for rows ``(X,D,Y,R)``."""
    data=list(rows)
    if not data: raise ValueError("at least one observation is required")
    h_x=_conditional_entropy(data,(0,3)); h_xd=_conditional_entropy(data,(0,1,3))
    return InformationValidity(h_x,h_xd,max(0.0,h_x-h_xd),declared_prior,
        _bayes_ceiling(data),consistent_hypotheses,minimum_context_identifiability,len(data))

def independent_trajectory_success(local_error: float, steps: int) -> float:
    if not 0<=local_error<=1 or steps<0: raise ValueError("invalid error or step count")
    return (1-local_error)**steps

def acquisition_mixture(p_acquire: float, success_if_acquired: float,
                        success_if_not: float) -> float:
    for value in (p_acquire,success_if_acquired,success_if_not):
        if not 0<=value<=1: raise ValueError("probabilities must lie in [0,1]")
    return p_acquire*success_if_acquired+(1-p_acquire)*success_if_not
