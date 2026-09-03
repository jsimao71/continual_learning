import json
from pathlib import Path
import random
from cl.semantic.predicate_stress import PAIRWISE,axis,stress_design,stress_example,training_batch,validate_stress
from cl.experiments.paper06_stress_analyze import contiguous_frontier

CONFIG=json.loads(Path("configs/paper06/stress_v6.json").read_text())

def test_stress_generator_replay_topology_and_pairwise_truth():
    kw=dict(split="test",predicate="sameAncestorAtLevel_k",total_depth=8,required_path=3,branching=4,distractors=16,template=3,position_mode="randomized",index=2,tree_seed=101,model_seed=11)
    row=stress_example(CONFIG,**kw);assert row==stress_example(CONFIG,**kw)
    assert row.target==28 and row.paths[0][3:]==row.paths[1][3:]
    neg=stress_example(CONFIG,**{**kw,"index":3});assert neg.target==27 and neg.paths[0][3]!=neg.paths[1][3] and neg.paths[0][4:]==neg.paths[1][4:]
    root_neg=stress_example(CONFIG,**{**kw,"total_depth":3,"required_path":3,"index":3})
    assert root_neg.target==27 and root_neg.paths[0][3]!=root_neg.paths[1][3]
    assert row.tokens.count(7)>=8*4*2

def test_stress_design_is_factorized_and_balanced():
    rows=stress_design(CONFIG,11,2);assert {r.predicate for r in rows}==set(CONFIG["predicates"])
    assert all(r.branching==2 and r.distractors==4 for r in rows if axis(r)=="depth_path")
    for p in (*PAIRWISE,"isAncestor"):
        targets=[r.target for r in rows if r.predicate==p];assert targets.count(27)==targets.count(28)
    assert validate_stress(CONFIG)["passed"]

def test_stress_design_cycles_presentations_and_training_is_balanced():
    rows=stress_design(CONFIG,11,8)
    assert {r.template_id for r in rows}==set(CONFIG["templates"])
    assert {r.position_mode for r in rows}==set(CONFIG["position_modes"])
    batch=training_batch(CONFIG,random.Random(5),11,["isAncestor"],8)
    assert len(batch)==8 and {r.predicate for r in batch}=={"isAncestor"}
    assert [r.positive for r in batch].count(0)==[r.positive for r in batch].count(1)==4

def test_frontier_requires_a_contiguous_competent_prefix():
    assert contiguous_frontier([(1,.91),(2,.83),(4,.79),(8,.95)],.8)==2
    assert contiguous_frontier([(1,.79),(2,.99)],.8) is None
