from cl.semantic.r0_implication import (SPECIAL,coverage_examples,generate_r0,generate_r0_pairs,
    split_implication_pairs,symbol_ids,validate_pair_split,validate_r0)

def test_r0_exact_targets_and_controls():
    symbols=symbol_ids(0,8);rows=generate_r0(symbols,40,11)
    assert all(r.target==(r.rhs if r.fact==r.lhs else SPECIAL["NO_ENTAIL"]) for r in rows)
    assert {r.condition for r in rows}=={"entailed","fact_mismatch","rule_lhs_mismatch"}

def test_train_test_identities_are_disjoint():
    train=symbol_ids(0,8);test=symbol_ids(8,4);rows=generate_r0(test,16,37,conditions=("entailed","consequent_swap"))
    report=validate_r0(train,test,rows);assert report["valid"] and report["train_test_symbol_identities_disjoint"]

def test_v2_shared_vocabulary_has_pair_disjointness_and_full_coverage():
    symbols=symbol_ids(0,16);train_pairs,test_pairs=split_implication_pairs(symbols,8502,.2)
    train=coverage_examples(train_pairs,symbols)+generate_r0_pairs(train_pairs,symbols,64,11)
    test=generate_r0_pairs(test_pairs,symbols,64,37)
    report=validate_pair_split(symbols,train_pairs,test_pairs,train,test)
    assert report["valid"] and report["observed_pair_leakage"]==0
    assert report["all_symbols_input_covered"] and report["all_symbols_positive_target_covered"]

def test_v2_controls_remain_balanced_and_exact():
    symbols=symbol_ids(0,16);_,test_pairs=split_implication_pairs(symbols,8502,.2)
    rows=generate_r0_pairs(test_pairs,symbols,40,37)
    assert {r.condition for r in rows}=={"entailed","fact_mismatch","rule_lhs_mismatch","consequent_swap"}
    assert all(r.target==(r.rhs if r.fact==r.lhs else SPECIAL["NO_ENTAIL"]) for r in rows)
