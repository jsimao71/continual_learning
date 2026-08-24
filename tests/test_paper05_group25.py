import json
from pathlib import Path
from cl.experiments.paper05_group25_sa_variance import examples,regime

CONFIG=json.loads(Path("configs/paper05/group25_sa_window_variance.json").read_text())

def test_group25_balanced_rule_has_no_single_token_shortcut():
    rows=examples(CONFIG,8)
    base=[r for r in rows if r["nuisance_distance"]==16 and r["nuisance_type"]=="N1" and r["realization"]==0]
    for position in (27,28):
        mapping={}
        for row in base:mapping.setdefault(row["tokens"][position],set()).add(row["target"])
        assert all(len(targets)==4 for targets in mapping.values())
    assert len({(row["tokens"][27],row["tokens"][28]):row["target"] for row in base})==16

def test_group25_regime_tags_reachability_cases():
    assert regime(8,16,4,2)=="signal_only_reachable"
    assert regime(16,4,2,2)=="nuisance_reachable_before_full_signal"
    assert regime(8,4,4,2)=="signal_and_nuisance_reachable"
    assert regime(8,16,None,2)=="full_attention"
