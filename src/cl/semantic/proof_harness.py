"""Minimal typed proof tools for matched Paper 0.85/0.9 comparisons."""
from __future__ import annotations
from dataclasses import dataclass
from cl.semantic.autoregressive_proofs import Implication, ProofExample

@dataclass(frozen=True)
class ProofState:
    current: str
    target: str
    calls: int=0
    terminated: bool=False

@dataclass(frozen=True)
class ToolResult:
    valid: bool
    value: str|tuple[Implication,...]|None
    failure: str|None

def apply_rule(current: str, rule: Implication) -> ToolResult:
    """T1 executes one selected rule but performs no rule search."""
    if rule.lhs!=current: return ToolResult(False,None,"wrong_tool_arguments")
    return ToolResult(True,rule.rhs,None)

def find_applicable(current: str, rules: tuple[Implication,...]) -> ToolResult:
    """T2 supplies search candidates but does not choose among branches."""
    candidates=tuple(rule for rule in rules if rule.lhs==current)
    return ToolResult(bool(candidates),candidates,None if candidates else "no_applicable_rule")

def next_proof_state(example: ProofExample, current: str) -> ToolResult:
    """T3 upper control: return the next state on the certified proof."""
    try: index=example.chain.index(current)
    except ValueError: return ToolResult(False,None,"state_not_on_proof")
    if index==len(example.chain)-1: return ToolResult(False,None,"already_at_target")
    return ToolResult(True,example.chain[index+1],None)

def oracle_harness(example: ProofExample, strength: str, max_steps: int|None=None) -> tuple[dict,list[dict]]:
    """Validate harness semantics with an explicitly oracle selecting controller."""
    limit=max_steps if max_steps is not None else example.proof_depth+1
    state=ProofState(example.chain[0],example.query); transitions=[]; failure=None
    for step in range(limit):
        if state.current==state.target:
            state=ProofState(state.current,state.target,state.calls,True); break
        expected=example.chain[state.calls+1] if state.calls+1<len(example.chain) else None
        if strength=="T1":
            selected=next(rule for rule in example.rules if rule.lhs==state.current and rule.rhs==expected)
            result=apply_rule(state.current,selected); candidates=1
        elif strength=="T2":
            lookup=find_applicable(state.current,example.rules); candidates=len(lookup.value or ())
            selected=next((rule for rule in (lookup.value or ()) if rule.rhs==expected),None)
            result=apply_rule(state.current,selected) if selected else ToolResult(False,None,"branching_search_failure")
        elif strength=="T3": result=next_proof_state(example,state.current); candidates=1
        else: raise ValueError(f"unknown tool strength {strength}")
        transitions.append({"step":step,"current":state.current,"expected":expected,"result":result.value,
            "valid":result.valid,"failure":result.failure,"candidate_count":candidates})
        if not result.valid: failure=result.failure; break
        state=ProofState(result.value,state.target,state.calls+1)
    if not state.terminated and state.current==state.target: state=ProofState(state.current,state.target,state.calls,True)
    if not state.terminated and failure is None: failure="nontermination"
    summary={"correct":state.terminated and state.current==example.query,"termination_correct":state.terminated,
        "tool_calls":state.calls,"model_forwards":state.calls+1,"generated_action_tokens":4*state.calls+2,
        "failure":failure,"oracle_controller":True,"tool_execution_is_model_reasoning":False}
    return summary,transitions

def example_from_dict(row:dict) -> ProofExample:
    return ProofExample(row["example_id"],row["split"],row["stage"],tuple(row["chain"]),
        tuple(Implication(**r) for r in row["rules"]),row["query"],row["proof_depth"],row["distractors"],
        row["branching"],row["shuffled"],row["namespace"])
