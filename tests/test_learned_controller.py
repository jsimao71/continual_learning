import random,torch
from cl.experiments.paper09_learned_controller_v1 import context,training_batch
from cl.experiments.paper09_learned_controller_analyze import aggregate, corrected_row
from cl.semantic.recurrence_chains import ANSWER,generate_chains,recurrence_pair_split

def test_m4_training_targets_next_edge_or_answer():
    _,train,_=recurrence_pair_split(16,8502,.2);cfg={"train_depths":[1,2,3]}
    x,y,lengths=training_batch("M4",train,cfg,random.Random(11),12,torch.device("cpu"))
    assert x.shape[0]==12 and y.shape==(12,) and all(int(v)==ANSWER or int(v)>=10 for v in y)
    assert lengths.min()>0

def test_context_appends_typed_results():
    _,train,_=recurrence_pair_split(16,8502,.2);row=generate_chains(train,3,1,11,"train")[0]
    base=context(row,[]);extended=context(row,list(row.chain[1:3]))
    assert len(extended)==len(base)+4 and extended[-1]==row.chain[2]


def test_m3_legacy_metric_is_relabelled_as_one_call_coverage():
    row={"machine":"M3","seed":"11","depth":"3","example_id":"0","final_correct":"1","invalid_call":"0",
         "nontermination":"0","tool_calls":"1","model_forwards":"2","rule_selection_accuracy":"1",
         "transition_accuracy":"0.3333333333333333","termination_correct":"1"}
    fixed=corrected_row(row)
    assert fixed["selected_edge_valid"] == 1
    assert fixed["one_call_edge_coverage"] == 1/3
    assert fixed["per_transition_accuracy"] == ""


def test_frontier_requires_contiguous_all_seed_machine_specific_gate():
    rows=[]
    for machine in ("M3","M4"):
      for seed in (11,23,37):
       for depth in (1,2,3,4):
        passing=depth<=3
        row={"machine":machine,"seed":seed,"depth":depth,"example_id":0,"final_correct":int(passing),
             "invalid_call":0,"nontermination":0,"tool_calls":1,"model_forwards":2}
        if machine=="M3": row.update(selected_edge_valid=int(passing),one_call_edge_coverage=int(passing)/depth,
                                      post_tool_answer_correct=int(passing),per_transition_accuracy="",exact_trajectory_correct="",
                                      stop_emitted="",termination_correct="")
        else: row.update(selected_edge_valid="",one_call_edge_coverage="",post_tool_answer_correct="",
                         per_transition_accuracy=float(passing),exact_trajectory_correct=int(passing),
                         stop_emitted=1,termination_correct=int(passing))
        rows.append(row)
    _,_,frontiers=aggregate(rows)
    assert {r["machine"]:r["contiguous_frontier"] for r in frontiers} == {"M3":3,"M4":3}
