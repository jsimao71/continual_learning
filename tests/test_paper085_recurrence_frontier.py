from cl.experiments.paper085_oracle_prefix import oracle_prompt,score_oracle_output
from cl.experiments.paper085_recurrence_frontier_v2 import batch_from_pool,build_training_pool
from cl.semantic.recurrence_chains import END,generate_chains,recurrence_pair_split
import json,random,torch


def test_oracle_prefix_is_proper_and_target_free():
    _,_,test=recurrence_pair_split(32,8502,.2);row=generate_chains(test,4,1,11,"test")[0]
    for prefix in range(4):
        prompt=oracle_prompt(row,prefix)
        assert row.chain[-1] not in prompt[len(row.prompt):]
        assert prompt[-prefix:]==row.chain[1:1+prefix] if prefix else prompt==row.prompt


def test_oracle_scoring_separates_transition_final_and_stop():
    _,_,test=recurrence_pair_split(32,8502,.2);row=generate_chains(test,4,1,11,"test")[0]
    correct=[*row.chain[3:],END]
    assert score_oracle_output(row,correct,2)=={"next_state_correct":1,"final_correct":1,
        "termination_correct":1,"remaining_trajectory_exact":1,"generated_tokens":3}
    wrong=[row.chain[3],END]
    score=score_oracle_output(row,wrong,2)
    assert score["next_state_correct"]==1 and score["final_correct"]==0 and score["termination_correct"]==0


def test_partial_starts_change_absolute_start_without_target_leakage():
    cfg=json.loads(open("configs/paper085/recurrence_frontier_v2.json").read())
    cfg["low_diversity_latent_chains"]=6;cfg["high_diversity_latent_chains"]=8
    _,train,_=recurrence_pair_split(32,8502,.2)
    pool,audit=build_training_pool(train,"partial_starts",cfg,11)
    assert audit["partial_starts"] and max(audit["start_indices"])>0
    for rows in pool.values():
        for row in rows:
            assert row.chain[-1] not in row.prompt[len(row.prompt):]
            assert row.residual_depth<=3 and row.start_index==row.latent_depth-row.residual_depth


def test_fixed_padding_makes_processed_budget_condition_invariant():
    cfg=json.loads(open("configs/paper085/recurrence_frontier_v2.json").read())
    cfg["low_diversity_latent_chains"]=6;cfg["high_diversity_latent_chains"]=8
    _,train,_=recurrence_pair_split(32,8502,.2)
    shapes=[]
    for condition in cfg["conditions"]:
        pool,_=build_training_pool(train,condition,cfg,11)
        x,_,_,_=batch_from_pool(pool,6,random.Random(1),cfg["max_length"],torch.device("cpu"));shapes.append(tuple(x.shape))
    assert shapes==[(6,cfg["max_length"]-1)]*4
