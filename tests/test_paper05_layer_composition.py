from cl.experiments.paper05_layer_composition import classify, generate_records


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
