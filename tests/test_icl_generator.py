from cl.semantic.icl import QUERY,controlled,episodic_examples,validation

def test_d4_is_episode_specific_and_leakage_free():
    train=episodic_examples("D4",8,12,3,11,exclude_critical=True)
    test=episodic_examples("D4",1,6,3,23,critical_only=True)
    report=validation(train,test)
    assert report["passed"] and report["critical_direct_mapping_count"]==0
    assert len({r.mapping for r in train if r.family==0})>1

def test_controls_preserve_query_and_target():
    row=episodic_examples("D4",1,1,3,11)[0]
    for name in ("none","shuffled","reversed","irrelevant","wrong_chain"):
        changed=controlled(row,name,13)
        assert changed.tokens[-2]==QUERY and changed.query==row.query and changed.target==row.target
    shuffled=controlled(row,"shuffled",13)
    assert row.target in shuffled.tokens and shuffled.tokens!=row.tokens

def test_dataset_ladder_adds_paired_context_only_at_d3():
    d2=episodic_examples("D2",1,1,3,1)[0]
    d3=episodic_examples("D3",1,1,3,1)[0]
    assert d2.tokens.count(QUERY)==d3.tokens.count(QUERY)==1
    assert len(d3.tokens)>len(d2.tokens)
