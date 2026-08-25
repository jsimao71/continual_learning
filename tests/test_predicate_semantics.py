import json
from pathlib import Path
import torch
from cl.common.model_adapter import TinyTransformerLM
from cl.semantic.predicates import evaluation_matrix,predicate_example,validate

CONFIG=json.loads(Path("configs/paper06/predicate_v4.json").read_text())

def test_predicate_generator_validation_and_replay():
    result=validate(CONFIG);assert result["passed"] and result["train_test_query_identity_overlap"]==0 and result["unseen_test_target_labels"]==0 and result["max_sequence_length"]<=CONFIG["max_length"]
    kwargs=dict(split="test",predicate="ancestor_k",total_depth=12,required_path=6,branching=4,distractors=16,template=3,position_mode="randomized",index=7,tree_seed=101,model_seed=11)
    assert predicate_example(CONFIG,**kwargs)==predicate_example(CONFIG,**kwargs)

def test_predicate_targets_and_hard_negative_balance():
    rows=evaluation_matrix(CONFIG,11,2);assert {r.predicate for r in rows}==set(CONFIG["predicates"])
    for row in rows:
        if row.predicate=="isAncestor":assert row.target in (26,27)
        else:assert row.target==row.node_path[row.required_path] and row.target in row.tokens
    binary=[r.target for r in rows if r.predicate=="isAncestor"];assert binary.count(26)==binary.count(27)

def test_sinusoidal_position_model_handles_unseen_lengths():
    model=TinyTransformerLM(384,128,32,2,4,position_encoding="sinusoidal")
    short,_=model(torch.zeros(2,16,dtype=torch.long));deep,_=model(torch.zeros(2,112,dtype=torch.long))
    assert short.shape==(2,16,384) and deep.shape==(2,112,384)
