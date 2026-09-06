import json,random
from cl.semantic.paper08_d5 import episode_signature,generate_split,split_audit
from cl.experiments.paper08_d5_acquisition import serialize

CFG=json.load(open("configs/paper08/d5_acquisition_v1.json"))

def data():
    out={};used=set()
    for s in ("train","seen_parameter","unseen_parameter","unseen_family"):
        out[s]=generate_split(CFG["moduli"],s,1,CFG["dataset_seed"],CFG["parameter_test_fraction"],used)
        used.update(episode_signature(r) for r in out[s])
    return out

def test_exact_posterior_gate_and_splits_have_no_parameter_or_family_leakage():
    p=data();audit=split_audit(p["train"],p["seen_parameter"],p["unseen_parameter"],p["unseen_family"])
    assert audit["all_answer_entropy_zero"] and audit["episode_id_overlap"]==0
    assert audit["serialized_episode_overlap"]==0 and audit["heldout_parameter_overlap"]==0 and audit["square_training_episodes"]==0
    assert {r.family for r in p["unseen_family"]}=={"square"}

def test_query_pair_is_absent_and_serialization_has_fixed_length():
    p=data()
    for rows in p.values():
      for row in rows:
        assert row.query not in {x for x,_ in row.demonstrations}
        assert (row.query,row.target) not in row.demonstrations
        for condition in ("correct","none","shuffled","wrong_rule"):
            assert len(serialize(row,CFG["moduli"],CFG["max_demonstrations"],condition))==CFG["max_length"]

def test_corruption_controls_change_demonstration_content_not_query():
    row=data()["seen_parameter"][0]
    correct=serialize(row,CFG["moduli"],CFG["max_demonstrations"],"correct")
    for condition in ("shuffled","wrong_rule","none"):
        changed=serialize(row,CFG["moduli"],CFG["max_demonstrations"],condition)
        assert changed[-2:]==correct[-2:] and changed!=correct
