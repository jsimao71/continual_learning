import json
from pathlib import Path

from cl.semantic.v2 import LABEL_MODE_TOKEN,rule_evaluation,rule_validation,s1_evaluation,s1_example,s1_validation

CONFIG=json.loads(Path("configs/paper06/semantic_v2.json").read_text())


def test_s1_validation_and_matrix():
    result=s1_validation(CONFIG);assert result["passed"] and result["train_test_identity_overlap"]==0
    rows=s1_evaluation(CONFIG,2);expected=2*2*2*2*3*4*2*2
    assert len(rows)==expected and all(len(r.tokens)==CONFIG["sequence_length"] for r in rows)


def test_s1_replay_and_heldout_identity():
    a=s1_example(CONFIG,4,3,16,"arbitrary","ancestor",2,"randomized",17,"test")
    b=s1_example(CONFIG,4,3,16,"arbitrary","ancestor",2,"randomized",17,"test")
    train=s1_example(CONFIG,4,3,16,"arbitrary","ancestor",2,"randomized",17,"train")
    assert a==b and a.entity_id!=train.entity_id and a.target in a.tokens
    assert a.tokens[-2:] == (5,a.tokens[-1])


def test_s1_all_deep_taxonomy_labels_are_in_vocab_and_facts_valid():
    for mode in CONFIG["label_modes"]:
        for template in CONFIG["templates"]:
            row=s1_example(CONFIG,4,3,16,mode,"root",template,"randomized",63,"test")
            assert max(row.tokens)<CONFIG["vocab_size"] and row.target<CONFIG["vocab_size"]
            compact=[token for token in row.tokens if token]
            assert compact[-2:]==[6,compact[-1]]
            assert compact.count(2)==3


def test_rule_generators_hold_out_identities_and_combinations():
    for stage in ("s2","s3"):
        validation=rule_validation(CONFIG,stage);assert validation["passed"]
        rows=rule_evaluation(CONFIG,stage,2);assert rows and all(len(r.tokens)==CONFIG["sequence_length"] for r in rows)
        assert all(r.tokens[-1] in range(100,180) for r in rows)
    s2=rule_evaluation(CONFIG,"s2",1)
    assert all(LABEL_MODE_TOKEN[r.label_mode] in r.tokens for r in s2)
