"""Oracle-prefix diagnostic for the leakage-free Paper 0.85 Stage-1 O1 models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cl.common.artifacts import atomic_write_json, write_csv
from cl.experiments.paper05_predictive_order_phase import resolve_device
from cl.experiments.paper085_recurrence_stage1 import model
from cl.semantic.recurrence_chains import END, SYMBOL_OFFSET, generate_chains, recurrence_pair_split


def oracle_prompt(row, forced_prefix_length: int) -> tuple[int, ...]:
    """Append only proper intermediate states; never append the held-out target."""
    if forced_prefix_length < 0 or forced_prefix_length >= row.depth:
        raise ValueError("forced prefix must contain zero to depth-1 intermediate states")
    prefix=row.chain[1:1+forced_prefix_length]
    if row.chain[-1] in prefix:
        raise AssertionError("oracle prefix leaked the final target")
    return (*row.prompt,*prefix)


def score_oracle_output(row, output: list[int], forced_prefix_length: int) -> dict[str,float|int]:
    remaining=list(row.chain[1+forced_prefix_length:])
    expected=remaining+[END]
    first=remaining[0]
    end_positions=[i for i,token in enumerate(output) if token==END]
    return {
        "next_state_correct":int(bool(output) and output[0]==first),
        "final_correct":int(len(output)>=len(remaining) and output[len(remaining)-1]==row.chain[-1]),
        "termination_correct":int(end_positions==[len(remaining)]),
        "remaining_trajectory_exact":int(output==expected),
        "generated_tokens":len(output),
    }


@torch.no_grad()
def generate_after_prefix(model_,row,prefix_length,cfg,device):
    seq=torch.tensor([oracle_prompt(row,prefix_length)],device=device)
    output=[];max_new=row.depth-prefix_length+cfg["generation_slack"]
    for _ in range(max_new):
        logits,_=model_(seq);token=int(logits[0,-1].argmax())
        output.append(token);seq=torch.cat((seq,torch.tensor([[token]],device=device)),dim=1)
        if token==END:break
    return output


def main(args):
    cfg=json.loads(Path(args.config).read_text());device=resolve_device(args.device)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    _,_,test_pairs=recurrence_pair_split(cfg["symbol_count"],cfg["pair_split_seed"],cfg["test_pair_fraction"])
    raw=[]
    for seed in cfg["model_seeds"]:
        checkpoint=Path(args.checkpoints)/f"O1_L{cfg['model']['layers']}_W{cfg['model']['width']}_H{cfg['model']['heads']}_seed{seed}.pt"
        payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
        net=model(cfg,device);net.load_state_dict(payload["model"]);net.eval()
        rows=generate_chains(test_pairs,args.depth,args.examples,seed*1000+args.depth,"test")
        for prefix in range(args.depth):
            for example_id,row in enumerate(rows):
                prompt=oracle_prompt(row,prefix)
                output=generate_after_prefix(net,row,prefix,cfg,device)
                raw.append({"seed":seed,"depth":args.depth,"forced_prefix_length":prefix,
                    "example_id":example_id,"prompt_contains_answer_field":0,
                    "forced_prefix_contains_final_target":int(row.chain[-1] in prompt[len(row.prompt):]),
                    **score_oracle_output(row,output,prefix)})
    write_csv(out/"oracle_prefix_raw.csv",raw)
    summary=[]
    for seed in cfg["model_seeds"]:
        for prefix in range(args.depth):
            rows=[row for row in raw if row["seed"]==seed and row["forced_prefix_length"]==prefix]
            summary.append({"seed":seed,"depth":args.depth,"forced_prefix_length":prefix,"examples":len(rows),
                **{key:float(np.mean([row[key] for row in rows])) for key in
                   ("next_state_correct","final_correct","termination_correct","remaining_trajectory_exact")}})
    write_csv(out/"oracle_prefix_rescue.csv",summary)
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,3,figsize=(9,3),sharey=True)
    for seed in cfg["model_seeds"]:
        rows=[row for row in summary if row["seed"]==seed]
        x=[row["forced_prefix_length"] for row in rows]
        for axis,key,title in zip(axes,("next_state_correct","final_correct","termination_correct"),
                                  ("next state","final answer","termination")):
            axis.plot(x,[row[key] for row in rows],marker="o",label=f"seed {seed}");axis.set_title(title)
            axis.set_xlabel("forced prefix length");axis.set_ylim(-.03,1.03);axis.grid(alpha=.25)
    axes[0].set_ylabel("accuracy");axes[-1].legend(fontsize=7);figure.tight_layout()
    figure.savefig(out/"oracle_prefix_rescue.png",dpi=180);plt.close(figure)
    atomic_write_json(out/"oracle_prefix_manifest.json",{
        "schema_version":"paper085.oracle_prefix.v1","source_stage":"recurrence_stage1_v2","condition":"O1",
        "checkpoint_steps":sorted({int(torch.load(Path(args.checkpoints)/f"O1_L{cfg['model']['layers']}_W{cfg['model']['width']}_H{cfg['model']['heads']}_seed{s}.pt",map_location="cpu",weights_only=False)["step"]) for s in cfg["model_seeds"]}),
        "depth":args.depth,"prefix_lengths":list(range(args.depth)),"examples_per_seed":args.examples,
        "target_leakage_audit":{"answer_field_in_prompt":False,"final_target_in_forced_prefix":False,
            "proper_prefix_only":True,"heldout_pair_split":True},"rows":len(raw),"device":str(device)})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/paper085/recurrence_stage1_v2.json")
    parser.add_argument("--checkpoints",default="docs/papers/paper0_85/results/recurrence_stage1_v2/checkpoints")
    parser.add_argument("--output",default="docs/papers/paper0_85/results/recurrence_frontier_v2/oracle_prefix")
    parser.add_argument("--device",default="cpu");parser.add_argument("--depth",type=int,default=4)
    parser.add_argument("--examples",type=int,default=64);main(parser.parse_args())
