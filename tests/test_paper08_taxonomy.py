import math

from cl.semantic.paper08_taxonomy import (
    FunctionHypothesis,
    answer_distribution,
    dataset_taxonomy,
    entropy,
    hypotheses,
    make_identifiable_episode,
    posterior,
)


def test_d4_is_constraint_completion_not_rule_induction():
    d4 = next(row for row in dataset_taxonomy(3) if row["dataset"] == "D4")
    assert d4["taxonomy"] == "T3"
    assert d4["mapping_space_size"] == 6
    assert d4["demonstrations_per_episode"] == 2
    assert d4["h_y_given_x_bits"] == math.log2(3)
    assert d4["h_y_given_x_context_bits"] == 0
    assert d4["admissible_as_rule_induction"] is False


def test_affine_posterior_can_retain_parameters_but_identify_answer():
    prior = hypotheses("affine", 5)
    post = posterior(prior, ((0, 1),))
    assert len(post) == 5
    assert entropy(answer_distribution(post, 0)) == 0


def test_nonidentifiable_query_has_positive_entropy():
    prior = hypotheses("add_n", 5)
    assert entropy(answer_distribution(prior, 2)) == math.log2(5)


def test_d5_generator_enforces_zero_answer_entropy():
    for family, parameters in (("successor", ()), ("add_n", (3,)), ("multiply_a", (2,)), ("affine", (2, 3)), ("square", ())):
        row = make_identifiable_episode(family, 7, parameters, seed=9)
        assert row.answer_entropy_bits == 0
        assert row.target == FunctionHypothesis(family, parameters, 7)(row.query)
        assert row.minimum_examples_needed <= len(row.demonstrations)
