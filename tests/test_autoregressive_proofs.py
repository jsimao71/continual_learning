from cl.semantic.autoregressive_proofs import evaluate_trace, generate_chain, generate_split, validate_example
from cl.experiments.paper085_phase import run

def test_exact_chain_and_all_output_conditions():
    row=generate_chain(split="test",depth=8,seed=37,index=0,distractors=4,branching=2,shuffled=True)
    assert validate_example(row)["valid"]
    for output in ("O0","O1","O2","O3"):
        score=evaluate_trace(row,row.target(output),output)
        assert score["exact"] and score["final_correct"]
        if output!="O0": assert score["termination_correct"] and score["transition_correct"]==8

def test_wrong_trace_separates_transition_and_termination():
    row=generate_chain(split="test",depth=3,seed=37,index=1)
    score=evaluate_trace(row,f"{row.chain[1]} wrong END","O1")
    assert not score["exact"] and score["termination_correct"] and score["transition_correct"]==1

def test_splits_are_symbol_disjoint_and_deterministic():
    train=generate_split("train",11,1,[1,2],[0]); test=generate_split("test",37,1,[1,2],[0])
    assert train==generate_split("train",11,1,[1,2],[0])
    train_symbols={x for r in train for x in r.chain}; test_symbols={x for r in test for x in r.chain}
    assert train_symbols.isdisjoint(test_symbols)

def test_phase_keeps_training_depth_bounded():
    config={"smoke_depths":[1,2,4],"test_depths":[1,2,4],"train_depths":[1,2],
        "smoke_distractors":[0],"distractors":[0],"smoke_examples_per_cell":1,"examples_per_cell":1,
        "generator_seeds":{"train":11,"validation":23,"test":37},"output_conditions":["O0"],
        "smoke_grid":{"layers":[1],"widths":[16],"heads":[1],"budgets":[1]},
        "architecture_grid":{"layers":[1],"widths":[16],"heads":[1],"budgets":[1]},"model_seeds":[11]}
    rows,_,_,_=run(config,True)
    assert max(r.proof_depth for r in rows if r.split=="train")==2
    assert max(r.proof_depth for r in rows if r.split=="test")==4
