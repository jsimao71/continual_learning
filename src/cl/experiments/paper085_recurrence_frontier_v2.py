"""Reduced recurrence-oriented data intervention for Paper 0.85."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cl.common.artifacts import atomic_write_json,write_csv
from cl.experiments.paper05_predictive_order_phase import resolve_device
from cl.experiments.paper085_recurrence_stage1 import model
from cl.semantic.recurrence_chains import BOS,END,FACT,PAD,QUERY,RULE,ARROW,SYMBOL_OFFSET,generate_chains,recurrence_pair_split


@dataclass(frozen=True)
class FrontierExample:
    chain:tuple[int,...]
    prompt:tuple[int,...]
    residual_depth:int
    latent_depth:int
    start_index:int
    latent_chain:tuple[int,...]

    @property
    def target(self):return (*self.chain[1:],END)


def _latent_to_examples(row,partial: bool,train_depths: tuple[int,...]) -> list[FrontierExample]:
    depths=train_depths if partial else (row.depth,)
    examples=[]
    for residual in depths:
        if residual>row.depth:continue
        start=row.depth-residual
        chain=row.chain[start:]
        prompt=[BOS,FACT,chain[0]]
        for a,b in zip(row.chain,row.chain[1:]):prompt.extend((RULE,a,ARROW,b))
        prompt.append(QUERY)
        examples.append(FrontierExample(chain,tuple(prompt),residual,row.depth,start,row.chain))
    return examples


def build_training_pool(pairs,condition,cfg,seed):
    partial=condition in {"partial_starts","partial_diversity"}
    high=condition in {"high_diversity","partial_diversity"}
    count=cfg["high_diversity_latent_chains"] if high else cfg["low_diversity_latent_chains"]
    rng=random.Random(seed+sum(map(ord,condition)));examples=[];latent=[]
    while len(latent)<count:
        if partial:depth=rng.choice(cfg["partial_latent_depths"])
        else:depth=rng.choice(cfg["train_depths"])
        row=generate_chains(pairs,depth,1,rng.randrange(2**31),"train")[0]
        if row.chain in {item.chain for item in latent}:continue
        latent.append(row);examples.extend(_latent_to_examples(row,partial,tuple(cfg["train_depths"])))
    by_depth={depth:[row for row in examples if row.residual_depth==depth] for depth in cfg["train_depths"]}
    if any(not rows for rows in by_depth.values()):raise RuntimeError("training pool lacks a requested residual depth")
    return by_depth,{"condition":condition,"partial_starts":partial,"high_diversity":high,
        "latent_chains":len({row.latent_chain for row in examples}),"suffix_examples":len(examples),
        "unique_transitions":len({edge for row in examples for edge in zip(row.chain,row.chain[1:])}),
        "start_indices":sorted({row.start_index for row in examples}),
        "residual_depth_counts":{str(depth):len(rows) for depth,rows in by_depth.items()}}


def batch_from_pool(pool,batch_size,rng,max_length,device):
    per=batch_size//len(pool);rows=[]
    for depth in sorted(pool):rows.extend(rng.choice(pool[depth]) for _ in range(per))
    while len(rows)<batch_size:rows.append(rng.choice(pool[rng.choice(sorted(pool))]))
    sequences=[(*row.prompt,*row.target) for row in rows];inputs=[];targets=[];masks=[]
    for row,seq in zip(rows,sequences):
        if len(seq)>max_length:raise ValueError("sequence exceeds max_length")
        inp=list(seq[:-1]);tar=list(seq[1:]);mask=[False]*(len(row.prompt)-1)+[True]*len(row.target)
        pad=max_length-1-len(inp);inputs.append(inp+[PAD]*pad);targets.append(tar+[PAD]*pad);masks.append(mask+[False]*pad)
    return (torch.tensor(inputs,device=device),torch.tensor(targets,device=device),
        torch.tensor(masks,device=device),rows)


def _fingerprint(cfg,condition,seed,smoke):
    payload=json.dumps({"cfg":cfg,"condition":condition,"seed":seed,"smoke":smoke},sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def train(condition,seed,cfg,device,checkpoint,smoke):
    pool,audit=build_training_pool(recurrence_pair_split(cfg["symbol_count"],cfg["pair_split_seed"],cfg["test_pair_fraction"])[1],condition,cfg,seed)
    updates=cfg["smoke_updates"] if smoke else cfg["updates"];batch=cfg["smoke_batch_size"] if smoke else cfg["batch_size"]
    fingerprint=_fingerprint(cfg,condition,seed,smoke);torch.manual_seed(seed);rng=random.Random(seed+701)
    net=model(cfg,device);optimizer=torch.optim.AdamW(net.parameters(),lr=cfg["learning_rate"]);start=0;losses=[]
    if checkpoint.exists():
        payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
        if payload["fingerprint"]!=fingerprint:raise RuntimeError("checkpoint/config fingerprint mismatch")
        net.load_state_dict(payload["model"]);optimizer.load_state_dict(payload["optimizer"]);start=payload["step"];losses=payload["losses"]
        for state in optimizer.state.values():
            for key,value in state.items():
                if torch.is_tensor(value):state[key]=value.to(device)
    net.train();supervised=0
    for step in range(start,updates):
        x,y,mask,_=batch_from_pool(pool,batch,rng,cfg["max_length"],device);supervised+=int(mask.sum())
        optimizer.zero_grad(set_to_none=True);logits,_=net(x);loss=torch.nn.functional.cross_entropy(logits[mask],y[mask])
        loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);optimizer.step()
        if step==0 or (step+1)%cfg["log_every"]==0 or step+1==updates:losses.append({"step":step+1,"loss":float(loss.detach())})
        if (step+1)%cfg["checkpoint_every"]==0 or step+1==updates:
            checkpoint.parent.mkdir(parents=True,exist_ok=True);torch.save({"model":net.state_dict(),"optimizer":optimizer.state_dict(),
                "step":step+1,"losses":losses,"fingerprint":fingerprint,"condition":condition,"seed":seed},checkpoint)
    audit.update({"updates":updates,"examples_seen":updates*batch,
        "processed_token_budget":updates*batch*(cfg["max_length"]-1),"supervised_target_tokens_last_invocation":supervised,
        "token_budget_definition":"examples times fixed padded causal length; identical across conditions"})
    return net.eval(),losses,audit


@torch.no_grad()
def evaluate(net,row,cfg,device):
    seq=torch.tensor([row.prompt],device=device);output=[]
    for _ in range(row.depth+cfg["generation_slack"]):
        logits,_=net(seq);token=int(logits[0,-1].argmax());output.append(token)
        seq=torch.cat((seq,torch.tensor([[token]],device=device)),1)
        if token==END:break
    truth=list(row.chain[1:]);correct=[int(i<len(output) and output[i]==token) for i,token in enumerate(truth)]
    return {"final_correct":int(len(output)>=len(truth) and output[len(truth)-1]==row.chain[-1]),"trajectory_exact":int(output==truth+[END]),
        "transition_accuracy":float(np.mean(correct)),"termination_correct":int(len(output)==len(truth)+1 and output[-1]==END),
        "premature_termination":int(END in output[:len(truth)]),"delayed_termination":int(not (len(output)==len(truth)+1 and output[-1]==END) and END not in output[:len(truth)]),
        "generated_tokens":len(output)},correct


def _read(path):
    if not path.exists():return []
    with path.open(newline="",encoding="utf-8") as handle:return list(csv.DictReader(handle))


def main(args):
    cfg=json.loads(Path(args.config).read_text());device=resolve_device(args.device or cfg["device"]);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    _,train_pairs,test_pairs=recurrence_pair_split(cfg["symbol_count"],cfg["pair_split_seed"],cfg["test_pair_fraction"])
    smoke=args.smoke;depths=cfg["smoke_test_depths"] if smoke else cfg["test_depths"];eval_per=cfg["smoke_eval_per_depth"] if smoke else cfg["eval_per_depth"]
    conditions=cfg["conditions"];seeds=cfg["model_seeds"][:1] if smoke else cfg["model_seeds"]
    cells=_read(out/"condition_seed_results.csv") if args.resume else [];raw=_read(out/"frontier_raw.csv") if args.resume else []
    losses=_read(out/"training_loss.csv") if args.resume else [];audits=[];done={(r["condition"],int(r["seed"])) for r in cells}
    for condition in conditions:
      for seed in seeds:
        if (condition,seed) in done:continue
        checkpoint=out/"checkpoints"/f"{condition}_O1_L2_W64_H4_seed{seed}.pt"
        net,cell_losses,audit=train(condition,seed,cfg,device,checkpoint,smoke);audits.append({"seed":seed,**audit});cell_raw=[]
        for depth in depths:
            rows=generate_chains(test_pairs,depth,eval_per,seed*1000+depth,"test")
            for example_id,row in enumerate(rows):
                scores,steps=evaluate(net,row,cfg,device);cell_raw.append({"condition":condition,"seed":seed,"depth":depth,"example_id":example_id,**scores})
                for step,value in enumerate(steps,1):raw.append({"record_type":"transition","condition":condition,"seed":seed,"depth":depth,"example_id":example_id,"step":step,"correct":value})
        raw.extend({"record_type":"example","condition":condition,"seed":seed,**row} for row in cell_raw)
        by_depth=[]
        for depth in depths:
            rows=[row for row in cell_raw if row["depth"]==depth]
            by_depth.append({"condition":condition,"seed":seed,"depth":depth,**{key:float(np.mean([row[key] for row in rows])) for key in
                ("final_correct","trajectory_exact","transition_accuracy","termination_correct","premature_termination","delayed_termination")}})
        passed=[row["depth"] for row in by_depth if row["final_correct"]>=cfg["competence_accuracy"] and row["termination_correct"]>=cfg["termination_accuracy"] and row["trajectory_exact"]>=cfg["trajectory_accuracy"]]
        frontier=0
        for depth in sorted(depths):
            if depth in passed and (frontier==0 or depth==frontier+1):frontier=depth
            else:break
        cells.extend({**row,"k_ar_seed":frontier} for row in by_depth);losses.extend({"condition":condition,"seed":seed,**row} for row in cell_losses)
        write_csv(out/"condition_seed_results.csv",cells);write_csv(out/"frontier_raw.csv",raw);write_csv(out/"training_loss.csv",losses)
        print(f"{condition} seed={seed} K_AR={frontier}",flush=True)
    write_csv(out/"dataset_audit.csv",audits)
    # Required schema views; these remain explicitly O1-only until a positive triggers O0.
    write_csv(out/"conditional_transition_accuracy.csv",[row for row in raw if row["record_type"]=="transition"])
    write_csv(out/"termination_decomposition.csv",cells)
    frontiers=[]
    for condition in conditions:
        condition_rows=[row for row in cells if row["condition"]==condition]
        for seed in seeds:
            rows=[row for row in condition_rows if int(row["seed"])==seed]
            if rows:frontiers.append({"condition":condition,"seed":seed,"k_ar":int(float(rows[0]["k_ar_seed"])),"o1_only":1})
    write_csv(out/"frontier_summary.csv",frontiers)
    atomic_write_json(out/"run_manifest.json",{"schema_version":cfg["schema_version"],"device":str(device),"smoke":smoke,
        "conditions":conditions,"seeds":seeds,"train_depths":cfg["train_depths"],"test_depths":depths,
        "output_regime":"O1_free_trace","exact_processed_token_matching":True,"checkpoint_resume":True,
        "gates":{"final":cfg["competence_accuracy"],"termination":cfg["termination_accuracy"],"trajectory":cfg["trajectory_accuracy"]},
        "completed_condition_seeds":len(frontiers),"planned_condition_seeds":len(conditions)*len(seeds),
        "o0_gate":"run only if O1 reaches unseen K>=4"})


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--config",default="configs/paper085/recurrence_frontier_v2.json")
    parser.add_argument("--output",default="docs/papers/paper0_85/results/recurrence_frontier_v2/reduced_first_pass")
    parser.add_argument("--device");parser.add_argument("--smoke",action="store_true");parser.add_argument("--resume",action="store_true");main(parser.parse_args())
