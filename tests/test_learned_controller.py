import random,torch
from cl.experiments.paper09_learned_controller_v1 import context,evaluate_m4,train,training_batch
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


def test_cpu_resume_exactly_replays_uninterrupted_training(tmp_path):
    _,train_pairs,test_pairs=recurrence_pair_split(16,8502,.2)
    cfg={"train_depths":[1,2,3],"max_length":48,"model":{"width":16,"layers":1,"heads":1,"mlp_ratio":2},
         "learning_rate":.001,"log_every":1,"checkpoint_every":2}
    device=torch.device("cpu")
    full,full_losses=train("M4",11,cfg,device,tmp_path/"full.pt",train_pairs,4,6)
    train("M4",11,cfg,device,tmp_path/"resume.pt",train_pairs,2,6)
    resumed,resumed_losses=train("M4",11,cfg,device,tmp_path/"resume.pt",train_pairs,4,6)
    assert full_losses == resumed_losses
    assert all(torch.equal(full.state_dict()[key],resumed.state_dict()[key]) for key in full.state_dict())
    rows=generate_chains(test_pairs,3,8,9903,"test")
    assert evaluate_m4(full,rows,device,2) == evaluate_m4(resumed,rows,device,2)
    payload=torch.load(tmp_path/"resume.pt",map_location="cpu",weights_only=False)
    assert payload["master_stream_cursor"] == 24
    assert {"python_rng","data_rng","numpy_rng","torch_rng","config_sha256","dataset_sha256"} <= payload.keys()
