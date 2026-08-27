"""Discrete information-validity checks for learned-machine task generators."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import product
from math import comb, log, log2
from typing import Hashable, Iterable, Mapping, Sequence

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

def _probability_vector(values: Sequence[float], name: str) -> tuple[float, ...]:
    result=tuple(float(value) for value in values)
    if not result or any(value < 0 or value > 1 for value in result):
        raise ValueError(f"{name} must be a nonempty probability vector")
    if abs(sum(result)-1.0)>1e-12:
        raise ValueError(f"{name} must sum to one")
    return result

def sampling_accuracy(true_distribution: Sequence[float],
                      predictive_distribution: Sequence[float]) -> float:
    """Expected match accuracy for conditionally independent target/prediction draws."""
    truth=_probability_vector(true_distribution,"true_distribution")
    prediction=_probability_vector(predictive_distribution,"predictive_distribution")
    if len(truth)!=len(prediction): raise ValueError("distribution sizes must match")
    return sum(p*q for p,q in zip(truth,prediction))

def bayes_accuracy(true_distribution: Sequence[float]) -> float:
    """Accuracy ceiling for the optimal deterministic class decision."""
    return max(_probability_vector(true_distribution,"true_distribution"))

def deterministic_accuracy(true_distribution: Sequence[float], decision: int) -> float:
    """Expected accuracy of one declared deterministic class decision."""
    truth=_probability_vector(true_distribution,"true_distribution")
    if decision<0 or decision>=len(truth): raise ValueError("decision is outside the outcome set")
    return truth[decision]

def cross_entropy(true_distribution: Sequence[float],
                  predictive_distribution: Sequence[float]) -> float:
    """Population natural-log loss for a known finite generator."""
    truth=_probability_vector(true_distribution,"true_distribution")
    prediction=_probability_vector(predictive_distribution,"predictive_distribution")
    if len(truth)!=len(prediction): raise ValueError("distribution sizes must match")
    if any(p>0 and q==0 for p,q in zip(truth,prediction)): return float("inf")
    return -sum(p*log(q) for p,q in zip(truth,prediction) if p)

def iid_majority_accuracy(per_run_accuracy: float, runs: int) -> float:
    """Binary odd-run majority accuracy under an explicit iid assumption."""
    if not 0<=per_run_accuracy<=1 or runs<1 or runs%2==0:
        raise ValueError("accuracy must be in [0,1] and runs must be positive odd")
    threshold=(runs+1)//2
    return sum(comb(runs,j)*per_run_accuracy**j*(1-per_run_accuracy)**(runs-j)
               for j in range(threshold,runs+1))

def enumerated_iid_majority_accuracy(per_run_accuracy: float, runs: int) -> float:
    """Exact small-domain enumeration used to validate the closed form."""
    if not 0<=per_run_accuracy<=1 or runs<1 or runs%2==0:
        raise ValueError("accuracy must be in [0,1] and runs must be positive odd")
    threshold=(runs+1)//2
    return sum(per_run_accuracy**sum(bits)*(1-per_run_accuracy)**(runs-sum(bits))
               for bits in product((0,1),repeat=runs) if sum(bits)>=threshold)

def iid_candidate_coverage(per_run_accuracy: float, runs: int) -> float:
    """Oracle coverage: probability at least one iid candidate is correct."""
    if not 0<=per_run_accuracy<=1 or runs<1: raise ValueError("invalid accuracy or run count")
    return 1-(1-per_run_accuracy)**runs

def aggregate_pattern_metrics(pattern_probabilities: Mapping[tuple[bool,...],float]) -> tuple[float,float]:
    """Return (majority accuracy, oracle coverage) for a declared joint law."""
    if not pattern_probabilities: raise ValueError("at least one pattern is required")
    lengths={len(pattern) for pattern in pattern_probabilities}
    if len(lengths)!=1 or next(iter(lengths))<1 or next(iter(lengths))%2==0:
        raise ValueError("patterns must have one common positive odd length")
    probabilities=_probability_vector(tuple(pattern_probabilities.values()),"pattern probabilities")
    runs=next(iter(lengths)); threshold=(runs+1)//2
    majority=sum(prob for (pattern,_),prob in zip(pattern_probabilities.items(),probabilities)
                 if sum(pattern)>=threshold)
    coverage=sum(prob for (pattern,_),prob in zip(pattern_probabilities.items(),probabilities)
                 if any(pattern))
    return majority,coverage
