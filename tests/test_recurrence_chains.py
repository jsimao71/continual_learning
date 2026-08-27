from cl.semantic.recurrence_chains import generate_chains,recurrence_pair_split,validate_chain_splits

def test_pair_disjoint_chains_and_targets():
    symbols,train,test=recurrence_pair_split(32,8502,.2)
    tr=generate_chains(train,3,20,11,"train");te=generate_chains(test,8,20,37,"test")
    report=validate_chain_splits(train,test,tr,te);assert report["valid"] and report["observed_pair_leakage"]==0
    for row in te:
        assert row.target("O0")==row.chain[-1:]
        assert row.target("O1")[-1] < min(symbols)
        assert row.prompt[-1] < min(symbols) and row.prompt[-1] != row.chain[-1]

def test_all_output_serializations_are_deterministic():
    _,train,_=recurrence_pair_split(16,8502,.2);row=generate_chains(train,2,1,11,"train")[0]
    assert len(row.target("O1"))==3 and len(row.target("O2"))==9 and row.target("O2")==row.target("O3")
