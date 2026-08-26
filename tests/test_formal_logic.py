from cl.semantic.formal_logic import *

def test_chain_minimal_depth_and_negative():
    assert validate_proposition(chain_problem(8,distractors=12))
    assert validate_proposition(chain_problem(3,negative=True))
    row=chain_problem(3);assert proposition_closure(row.facts,row.rules)[row.query]==3

def test_multi_premise_depth():
    rules=(Rule(("a",),"b"),Rule(("a","b"),"c"))
    row=PropositionProblem(("a",),rules,"c",True,2,"test")
    assert validate_proposition(row)

def test_binding_and_nested_functors():
    x=Var("x");left=Fun("P",(Fun("f",(x,)),));right=Fun("P",(Fun("f",(Fun("a"),)),))
    result=unify(left,right)
    assert result.success and result.substitution==(("x",Fun("a")),)
    assert validate_unifier(left,right,result) and term_depth(left)==2

def test_repeated_variable_and_occurs_check():
    x=Var("x");a,b=Fun("a"),Fun("b")
    assert unify(Fun("f",(x,x)),Fun("f",(a,b))).failure_reason=="symbol_mismatch"
    assert unify(x,Fun("f",(x,))).failure_reason=="occurs_check"

def test_arity_and_symbol_failures_are_distinct():
    assert unify(Fun("f",(Fun("a"),)),Fun("f",())).failure_reason=="arity_mismatch"
    assert unify(Fun("f"),Fun("g")).failure_reason=="symbol_mismatch"
