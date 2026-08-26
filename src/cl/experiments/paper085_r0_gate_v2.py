"""Shared-vocabulary, pair-disjoint Paper 0.85 R0 competence gate."""
from __future__ import annotations
import argparse,csv,json,random
from pathlib import Path
import numpy as np
import torch
from cl.common.artifacts import atomic_write_json,write_csv
from cl.common.model_adapter import TinyTransformerLM
from cl.experiments.paper05_predictive_order_phase import resolve_device
from cl.semantic.r0_implication import (SPECIAL,SYMBOL_OFFSET,coverage_examples,generate_r0_pairs,
    split_implication_pairs,symbol_ids,validate_pair_split)

CONDITIONS=("entailed","fact_mismatch","rule_lhs_mismatch","consequent_swap")
def batch_tensor(rows,device): return (torch.tensor([r.tokens for r in rows],device=device),torch.tensor([r.target for r in rows],device=device))

def train(seed,cfg,device,checkpoint,updates,batch_size,train_pairs,symbols,coverage):
    torch.manual_seed(seed);rng=random.Random(seed+701);model=TinyTransformerLM(SYMBOL_OFFSET+len(symbols),8,
        cfg["model"]["width"],cfg["model"]["layers"],cfg["model"]["heads"],cfg["model"]["mlp_ratio"]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"]);start=0;losses=[]
    if checkpoint.exists():
        payload=torch.load(checkpoint,map_location="cpu",weights_only=False);model.load_state_dict(payload["model"]);optimizer.load_state_dict(payload["optimizer"])
        start=int(payload["step"]);losses=list(payload.get("losses",[]))
        for state in optimizer.state.values():
            for key,value in state.items():
                if torch.is_tensor(value):state[key]=value.to(device)
    model.train()
    for step in range(start,updates):
        rows=generate_r0_pairs(train_pairs,symbols,batch_size, rng.randrange(2**31),CONDITIONS)
        # Deterministically cycle positive coverage so every lexical row receives output supervision.
        rows[0]=coverage[(2*step)%len(coverage)];rows[1]=coverage[(2*step+1)%len(coverage)]
        tokens,target=batch_tensor(rows,device);optimizer.zero_grad(set_to_none=True);logits,_=model(tokens)
        loss=torch.nn.functional.cross_entropy(logits[:,-1],target);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        if step==0 or (step+1)%cfg["log_every"]==0 or step+1==updates:losses.append({"step":step+1,"loss":float(loss.detach())})
        if (step+1)%cfg["checkpoint_every"]==0 or step+1==updates:
            checkpoint.parent.mkdir(parents=True,exist_ok=True);torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                "step":step+1,"losses":losses,"seed":seed},checkpoint)
    return model.eval(),losses

@torch.no_grad()
def evaluate(model,rows,device):
    tokens,targets=batch_tensor(rows,device);logits,_=model(tokens);pred=logits[:,-1].argmax(-1);raw=[]
    for i,row in enumerate(rows):
        values=logits[i,-1];competitor=torch.cat((values[:row.target],values[row.target+1:])).max()
        raw.append({"example_id":i,"condition":row.condition,"target":row.target,"prediction":int(pred[i]),
            "correct":int(pred[i]==targets[i]),"rank":int((values>values[row.target]).sum())+1,"margin":float(values[row.target]-competitor)})
    return raw

def aggregate(raw,cfg):
    by={c:[r for r in raw if r["condition"]==c] for c in CONDITIONS};acc=lambda c:float(np.mean([r["correct"] for r in by[c]]))
    positive=acc("entailed");swap=acc("consequent_swap");fact=acc("fact_mismatch");lhs=acc("rule_lhs_mismatch")
    return {"heldout_entailed_accuracy":positive,"heldout_consequent_swap_accuracy":swap,
        "heldout_fact_mismatch_accuracy":fact,"heldout_rule_lhs_mismatch_accuracy":lhs,
        "heldout_negative_control_accuracy":min(fact,lhs),"heldout_accuracy":float(np.mean([r["correct"] for r in raw])),
        "competent":int(min(positive,swap)>=cfg["competence_accuracy"] and min(fact,lhs)>=cfg["control_accuracy"])}

def read_csv(path):
    if not path.exists():return []
    with path.open(newline="",encoding="utf-8") as h:return list(csv.DictReader(h))

def main(args):
    cfg=json.loads(Path(args.config).read_text());device=resolve_device(args.device or cfg["device"]);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    symbols=symbol_ids(0,cfg["symbol_count"]);train_pairs,test_pairs=split_implication_pairs(symbols,cfg["pair_split_seed"],cfg["test_pair_fraction"])
    coverage=coverage_examples(train_pairs,symbols);eval_n=cfg["smoke_eval_examples"] if args.smoke else cfg["eval_examples"]
    audit_train=coverage+generate_r0_pairs(train_pairs,symbols,max(128,eval_n),cfg["pair_split_seed"]+1,CONDITIONS)
    audit_test=generate_r0_pairs(test_pairs,symbols,eval_n,cfg["pair_split_seed"]+2,CONDITIONS);audit=validate_pair_split(symbols,train_pairs,test_pairs,audit_train,audit_test)
    if not audit["valid"]:raise RuntimeError(audit)
    updates=cfg["smoke_updates"] if args.smoke else cfg["updates"];batch=cfg["smoke_batch_size"] if args.smoke else cfg["batch_size"]
    seeds=cfg["model_seeds"][:1] if args.smoke else cfg["model_seeds"];cells=read_csv(out/"local_step_competence.csv") if args.resume else []
    done={int(r["seed"]) for r in cells};raw_all=[];loss_all=[]
    for index,seed in enumerate(seeds):
        if seed in done:print(f"skip completed seed {seed}",flush=True);continue
        checkpoint=out/"checkpoints"/f"r0v2_L{cfg['model']['layers']}_W{cfg['model']['width']}_H{cfg['model']['heads']}_seed{seed}.pt"
        model,losses=train(seed,cfg,device,checkpoint,updates,batch,train_pairs,symbols,coverage)
        rows=generate_r0_pairs(test_pairs,symbols,eval_n,seed+100003,CONDITIONS);raw=evaluate(model,rows,device);summary=aggregate(raw,cfg)
        cells.append({"seed":seed,**cfg["model"],"updates":updates,"eval_examples":eval_n,**summary,"checkpoint":str(checkpoint)})
        raw_all += [{"seed":seed,**r} for r in raw];loss_all += [{"seed":seed,**r} for r in losses]
        write_csv(out/"local_step_competence.csv",cells);write_csv(out/"local_step_raw_latest.csv",raw_all);write_csv(out/"training_loss_latest.csv",loss_all)
        print(f"[{index+1}/{len(seeds)}] seed={seed} entailed={summary['heldout_entailed_accuracy']:.3f} swap={summary['heldout_consequent_swap_accuracy']:.3f} controls={summary['heldout_negative_control_accuracy']:.3f} competent={summary['competent']}",flush=True)
    atomic_write_json(out/"r0_manifest.json",{"schema_version":cfg["schema_version"],"device":str(device),"smoke":args.smoke,"updates":updates,
        "seeds_completed":sorted(int(r["seed"]) for r in cells),"required_seeds":cfg["model_seeds"],"dataset_audit":audit,
        "train_pairs":len(train_pairs),"heldout_pairs":len(test_pairs),"shared_symbol_vocabulary":True,"v1_interpretation":"lexical_transfer_stress",
        "three_seed_gate_passed":len(cells)>=3 and all(int(r["competent"]) for r in cells),"checkpoint_resume":True})

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/paper085/r0_gate_v2.json");p.add_argument("--output",default="docs/papers/paper0_85/results/r0_gate_v2")
    p.add_argument("--device");p.add_argument("--smoke",action="store_true");p.add_argument("--resume",action="store_true");main(p.parse_args())
