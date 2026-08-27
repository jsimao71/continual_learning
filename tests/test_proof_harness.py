from cl.semantic.autoregressive_proofs import Implication, generate_chain
from cl.semantic.proof_harness import apply_rule, find_applicable, next_proof_state, oracle_harness
from cl.experiments.paper09_paper085_comparison import matched_recurrence_examples

def test_t1_applies_only_selected_applicable_rule():
    rule=Implication("a","b")
    assert apply_rule("a",rule).value=="b"
    assert apply_rule("x",rule).failure=="wrong_tool_arguments"

def test_t2_search_is_distinct_from_selection():
    rules=(Implication("a","b"),Implication("a","c"),Implication("x","y"))
    result=find_applicable("a",rules)
    assert result.valid and len(result.value)==2

def test_t3_is_explicit_oracle_upper_control():
    row=generate_chain(split="test",depth=3,seed=37,index=0)
    assert next_proof_state(row,row.chain[1]).value==row.chain[2]

def test_oracle_controller_validates_all_strengths_and_cost_scales():
    row=generate_chain(split="test",depth=8,seed=37,index=1,distractors=4,branching=2,shuffled=True)
    for strength in ("T1","T2","T3"):
        summary,trace=oracle_harness(row,strength)
        assert summary["correct"] and summary["tool_calls"]==8 and len(trace)==8
        assert summary["model_forwards"]==9 and summary["oracle_controller"]

def test_matched_comparison_uses_corrected_stage1_depths():
    cfg={"symbol_count":16,"pair_split_seed":8502,"test_pair_fraction":.2,"test_depths":[1,4],"eval_per_depth":3}
    rows=matched_recurrence_examples(cfg)
    assert len(rows)==6 and {r.proof_depth for r in rows}=={1,4}
