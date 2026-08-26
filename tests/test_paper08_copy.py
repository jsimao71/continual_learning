from cl.semantic.paper08_copy import controlled_copy, generate_copy_examples, validate_copy


def test_copy_generator_balances_literal_mapping():
    rows = generate_copy_examples("C2", 4, 24, pairs_per_prompt=3, seed=7)
    audit = validate_copy(rows)
    assert audit["passed"] == 1
    assert audit["max_p_y_given_x"] == 1 / 3
    assert {len(row.tokens) for row in rows} == {11}


def test_controls_preserve_length_and_break_association():
    row = generate_copy_examples("C2", 4, 24, pairs_per_prompt=3, seed=9)[0]
    shuffled = controlled_copy(row, "shuffled_pairings", 2)
    assert len(shuffled.tokens) == len(row.tokens)
    assert shuffled.tokens[-1] == row.query_key
    assert next(v for k, v in shuffled.pairs if k == row.query_key) != row.target
    assert controlled_copy(row, "none").tokens == (2, row.query_key)


def test_v2_is_explicitly_degenerate():
    audit = validate_copy(generate_copy_examples("C0", 2, 8, seed=1))
    assert audit["v2_degenerate"] == 1
    assert audit["passed"] == 0
