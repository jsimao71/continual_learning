from cl.experiments.paper05_layer_composition import classify, generate_records, reachable
from cl.experiments.paper05_correctness import decisions


CONFIG={"seed":1,"generators":["contiguous_ngram","skip_gram","binary_functor"],
        "relations_per_generator":3,"train_examples_per_relation":10}


def test_generator_replay_and_metadata_contract():
    left=generate_records(CONFIG,"test",4,3); right=generate_records(CONFIG,"test",4,3)
    assert left == right and len(left) == 27
    assert all(row.nuisance_count == len(row.nuisance_tokens) == 4 for row in left)
    assert all(not row.answer_changing_context for row in left)
    assert len({row.predictive_family_id for row in left}) == 9


def test_trajectory_classifier_preserves_nonmonotonicity():
    assert classify(.4,[.5,.2],.1) == "expansion_then_contraction"
    assert classify(.4,[.3,.2],.1) == "monotone_contraction"
    assert classify(.1,[.2,.3],.2) == "no_contraction"


def test_graph_reachability_matches_window_times_depth_bound():
    assert not reachable(span=8, window=2, depth=3)
    assert reachable(span=8, window=2, depth=4)
    assert reachable(span=32, window=None, depth=1)


def test_first_and_stable_top1_are_distinct_and_missing_is_explicit():
    base={"model_seed":1,"generator_family":"g","predictive_family_id":"g:r","surface_identity_id":"x","nuisance_count":4}
    rows=[{**base,"depth_index":i,"prediction_correct":value} for i,value in enumerate((0,1,0,1,1))]
    result=decisions(rows)[0]
    assert result["first_top1_layer"]==1 and result["stable_top1_layer"]==3
    assert result["settling_delay"]==2 and result["top1_reversal_count"]==3
    never=decisions([{**base,"depth_index":i,"prediction_correct":0} for i in range(3)])[0]
    assert never["first_top1_layer"]==never["stable_top1_layer"]==""
