"""Exact symbolic chains and autoregressive traces for Paper 0.85."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import random
from typing import Iterable

@dataclass(frozen=True)
class Implication:
    lhs: str
    rhs: str

@dataclass(frozen=True)
class ProofExample:
    example_id: str
    split: str
    stage: str
    chain: tuple[str,...]
    rules: tuple[Implication,...]
    query: str
    proof_depth: int
    distractors: int
    branching: int
    shuffled: bool
    namespace: str

    def prompt(self) -> str:
        rules=" ; ".join(f"{r.lhs} -> {r.rhs}" for r in self.rules)
        return f"FACT {self.chain[0]} ; RULES {rules} ; QUERY {self.query}"

    def target(self, condition: str) -> str:
        if condition=="O0": return self.query
        if condition=="O1": return " ".join((*self.chain[1:],"END"))
        if condition in {"O2","O3"}:
            states=" ".join(f"DERIVE {node}" for node in self.chain[1:])
            return f"STATE {self.chain[0]} {states} ANSWER {self.query} END"
        raise ValueError(f"unknown output condition {condition}")

    def as_dict(self) -> dict:
        row=asdict(self); row["rules"]=[asdict(r) for r in self.rules]
        row["prompt"]=self.prompt(); row["targets"]={o:self.target(o) for o in ("O0","O1","O2","O3")}
        row["prompt_tokens"]=len(self.prompt().split())
        row["generated_tokens"]={o:len(self.target(o).split()) for o in ("O0","O1","O2","O3")}
        return row

def generate_chain(*, split: str, depth: int, seed: int, index: int,
                   distractors: int=0, branching: int=1, shuffled: bool=False,
                   stage: str="R2") -> ProofExample:
    if depth<1 or branching<1 or distractors<0: raise ValueError("invalid complexity")
    ns={"train":"tr","validation":"va","test":"te"}[split]
    prefix=f"{ns}_{seed}_{index}_"; chain=tuple(f"{prefix}p{i}" for i in range(depth+1))
    rules=[Implication(chain[i],chain[i+1]) for i in range(depth)]
    # Competing branches are dead ends, preserving a unique target proof.
    for i in range(depth):
        for branch in range(1,branching): rules.append(Implication(chain[i],f"{prefix}b{i}_{branch}"))
    for j in range(distractors): rules.append(Implication(f"{prefix}d{j}",f"{prefix}e{j}"))
    rng=random.Random(seed*1_000_003+index*97+depth*13+distractors*7+branching)
    if shuffled: rng.shuffle(rules)
    return ProofExample(f"{stage}:{split}:{seed}:{index}",split,stage,chain,tuple(rules),chain[-1],depth,
                        distractors,branching,shuffled,ns)

def validate_example(example: ProofExample) -> dict:
    outgoing={}
    for rule in example.rules: outgoing.setdefault(rule.lhs,[]).append(rule.rhs)
    valid_path=all(example.chain[i+1] in outgoing.get(example.chain[i],[]) for i in range(example.proof_depth))
    namespace_ok=all(node.startswith(example.namespace+"_") for node in example.chain)
    unique_target_path=sum(1 for rule in example.rules if rule.rhs==example.query)==1
    return {"valid":valid_path and namespace_ok and unique_target_path,
        "valid_path":valid_path,"namespace_ok":namespace_ok,"unique_target_path":unique_target_path}

def evaluate_trace(example: ProofExample, output: str, condition: str) -> dict:
    expected=example.target(condition).split(); actual=output.split()
    exact=actual==expected; end_positions=[i for i,t in enumerate(actual) if t=="END"]
    premature=bool(end_positions and end_positions[0]<len(expected)-1)
    delayed=bool(end_positions and end_positions[0]>len(expected)-1) or (not end_positions)
    if condition=="O0": transition_correct=int(bool(actual) and actual[0]==example.query); transitions=1
    else:
        expected_states=list(example.chain[1:]); observed=[t for t in actual if t in set(example.chain[1:])]
        transition_correct=sum(a==b for a,b in zip(observed,expected_states)); transitions=len(expected_states)
    return {"exact":exact,"final_correct":example.query in actual,"transition_correct":transition_correct,
        "transitions":transitions,"termination_correct":len(end_positions)==1 and end_positions[0]==len(actual)-1,
        "premature_termination":premature,"delayed_or_missing_termination":delayed}

def generate_split(split: str, seed: int, examples_per_cell: int,
                   depths: Iterable[int], distractors: Iterable[int]) -> list[ProofExample]:
    rows=[]; index=0
    for depth in depths:
        for nuisance in distractors:
            for shuffled in (False,True):
                for branching in (1,2):
                    for _ in range(examples_per_cell):
                        stage="R0" if depth==1 and nuisance==0 and not shuffled and branching==1 else (
                            "R1" if depth==2 and nuisance==0 and not shuffled and branching==1 else
                            "R5" if branching>1 else "R4" if nuisance else "R3" if shuffled else "R2")
                        rows.append(generate_chain(split=split,depth=depth,seed=seed,index=index,
                            distractors=nuisance,branching=branching,shuffled=shuffled,stage=stage)); index+=1
    return rows
