from cl.semantic.formal_datasets import proposition_ladder, unification_ladder, validate_dataset

def test_all_stages_and_split_namespaces_are_valid():
    rows=[]
    for split,seed in (("train",11),("validation",23),("test",37)):
        rows += proposition_ladder(split,70,seed,[1,2,4,8])
        rows += unification_ladder(split,100,seed,[1,2,4,8])
    report=validate_dataset(rows)
    assert report["valid"]
    assert set(report["stages"])=={*(f"P{i}" for i in range(7)),*(f"F{i}" for i in range(10))}

def test_generation_is_deterministic_and_balances_failures():
    a=unification_ladder("test",100,37,[1,2,4])
    b=unification_ladder("test",100,37,[1,2,4])
    assert a==b
    f5=[r for r in a if r.stage=="F5"]
    assert {r.label for r in f5}=={"success","symbol_mismatch"}

def test_complexity_axes_are_explicit_not_length_proxies():
    rows=proposition_ladder("train",280,11,[1,2,4,8])
    p4=[r for r in rows if r.stage=="P4" and r.label=="true"]
    assert {r.proof_depth for r in p4} >= {1,2,4,8}
    f4=[r for r in unification_ladder("train",100,11,[1,2,4,8]) if r.stage=="F4"]
    assert len({r.term_depth for r in f4}) >= 3
