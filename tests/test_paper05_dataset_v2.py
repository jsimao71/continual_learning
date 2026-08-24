import json
from pathlib import Path
from cl.experiments.paper05_dataset_v2 import generate,validate

CONFIG=json.loads(Path("configs/paper05/dataset_v2.json").read_text())

def test_dataset_v2_no_shortcut_validation_passes():
    train=generate(CONFIG,"train");test=generate(CONFIG,"test");result,checks=validate(CONFIG,train,test)
    assert result["passed"],result["failures"]
    assert all(float(row["max_singleton_target_MI_bits"])<=CONFIG["singleton_mi_threshold_bits"] for row in checks)
    assert all(float(row["full_pattern_target_MI_bits"])>=1.9 for row in checks)

def test_dataset_v2_contract_and_split_are_complete():
    train=generate(CONFIG,"train");test=generate(CONFIG,"test")
    required={"generator_family","predictive_family_id","surface_identity_id","target_token","pattern_tokens","pattern_length","dependency_span","predictive_arity","nuisance_tokens","nuisance_count","nuisance_type","nuisance_difficulty","competing_pattern_count","answer_changing_context","continuation_entropy","single_token_target_MI","subset_target_MI","full_pattern_target_MI","position_mode","train_frequency","split","generator_seed","rule_signature","rule_inputs","rule_output"}
    assert required<=train[0].keys()
    assert all(row["single_token_target_MI"] is not None and row["full_pattern_target_MI"]>=1.9 for row in train)
    assert not ({r["surface_identity_id"] for r in train}&{r["surface_identity_id"] for r in test})
    assert {r["nuisance_type"] for r in train}>={"N0","N2","N3","N4","N6"}
