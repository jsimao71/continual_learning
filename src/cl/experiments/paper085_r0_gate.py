"""Resumable three-seed learned local-implication gate for Paper 0.85."""
from __future__ import annotations
import argparse,csv,json,random
from pathlib import Path
import numpy as np
import torch
from cl.common.artifacts import atomic_write_json,write_csv
from cl.common.model_adapter import TinyTransformerLM
from cl.experiments.paper05_predictive_order_phase import resolve_device
from cl.semantic.r0_implication import SPECIAL,SYMBOL_OFFSET,generate_r0,symbol_ids,validate_r0

CONDITIONS=("entailed","fact_mismatch","rule_lhs_mismatch","consequent_swap")

def settings(cfg):
    return [{"seed":seed,**cfg["model"]} for seed in cfg["model_seeds"]]

def batch_tensor(rows,device):
    return torch.tensor([r.tokens for r in rows],dtype=torch.long,device=device),torch.tensor([r.target for r in rows],dtype=torch.long,device=device)

def train(setting,cfg,device,checkpoint,updates,batch_size):
    torch.manual_seed(setting["seed"]);rng=random.Random(setting["seed"]+701)
    vocab=SYMBOL_OFFSET+cfg["train_symbol_count"]+cfg["test_symbol_count"]
    model=TinyTransformerLM(vocab,8,setting["width"],setting["layers"],setting["heads"],setting["mlp_ratio"]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"])
    start=0;losses=[]
    if checkpoint.exists():
        payload=torch.load(checkpoint,map_location="cpu",weights_only=False);model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"]);start=int(payload["step"]);losses=list(payload.get("losses",[]))
        for state in optimizer.state.values():
            for key,value in state.items():
                if torch.is_tensor(value): state[key]=value.to(device)
    train_symbols=symbol_ids(0,cfg["train_symbol_count"])
    model.train()
    for step in range(start,updates):
        rows=generate_r0(train_symbols,batch_size,rng.randrange(2**31),conditions=("entailed","fact_mismatch","rule_lhs_mismatch","consequent_swap"))
        tokens,target=batch_tensor(rows,device);optimizer.zero_grad(set_to_none=True);logits,_=model(tokens)
        loss=torch.nn.functional.cross_entropy(logits[:,-1],target);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        if step==0 or (step+1)%cfg["log_every"]==0 or step+1==updates: losses.append({"step":step+1,"loss":float(loss.detach())})
        if (step+1)%cfg["checkpoint_every"]==0 or step+1==updates:
            checkpoint.parent.mkdir(parents=True,exist_ok=True)
            torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"step":step+1,"losses":losses,"setting":setting},checkpoint)
    return model.eval(),losses

@torch.no_grad()
def evaluate(model,cfg,seed,device,examples):
    test_symbols=symbol_ids(cfg["train_symbol_count"],cfg["test_symbol_count"])
    rows=generate_r0(test_symbols,examples,seed+100_003,conditions=CONDITIONS);tokens,targets=batch_tensor(rows,device)
    logits,_=model(tokens);pred=logits[:,-1].argmax(-1);raw=[]
    for i,row in enumerate(rows):
        values=logits[i,-1];competitor=torch.cat((values[:row.target],values[row.target+1:])).max()
        raw.append({"example_id":i,"condition":row.condition,"target":row.target,"prediction":int(pred[i]),
            "correct":int(pred[i]==targets[i]),"rank":int((values>values[row.target]).sum())+1,
            "margin":float(values[row.target]-competitor)})
    return rows,raw

def aggregate(raw,cfg):
    by={c:[r for r in raw if r["condition"]==c] for c in CONDITIONS}
    acc=lambda c:float(np.mean([r["correct"] for r in by[c]]))
    positive=acc("entailed");swap=acc("consequent_swap");negative=min(acc("fact_mismatch"),acc("rule_lhs_mismatch"))
    return {"heldout_entailed_accuracy":positive,"heldout_consequent_swap_accuracy":swap,
        "heldout_negative_control_accuracy":negative,"heldout_accuracy":float(np.mean([r["correct"] for r in raw])),
        "competent":int(positive>=cfg["competence_accuracy"] and swap>=cfg["competence_accuracy"] and negative>=cfg["control_accuracy"])}

def existing(path):
    if not path.exists(): return []
    with path.open(newline="",encoding="utf-8") as handle:return list(csv.DictReader(handle))

def main(args):
    cfg=json.loads(Path(args.config).read_text());device=resolve_device(args.device or cfg["device"]);output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    run_settings=settings(cfg);updates=cfg["smoke_updates"] if args.smoke else cfg["updates"];batch=cfg["smoke_batch_size"] if args.smoke else cfg["batch_size"]
    eval_examples=cfg["smoke_eval_examples"] if args.smoke else cfg["eval_examples"]
    if args.smoke: run_settings=run_settings[:1]
    cells=existing(output/"local_step_competence.csv") if args.resume else [];done={int(r["seed"]) for r in cells};raw_all=[];loss_all=[]
    train_symbols=symbol_ids(0,cfg["train_symbol_count"]);test_symbols=symbol_ids(cfg["train_symbol_count"],cfg["test_symbol_count"])
    audit_rows=generate_r0(test_symbols,eval_examples,37+100_003,conditions=CONDITIONS);audit=validate_r0(train_symbols,test_symbols,audit_rows)
    if not audit["valid"]:raise RuntimeError(audit)
    for index,setting in enumerate(run_settings):
        if setting["seed"] in done: print(f"skip completed seed {setting['seed']}",flush=True);continue
        ckpt=output/"checkpoints"/f"r0_L{setting['layers']}_W{setting['width']}_H{setting['heads']}_seed{setting['seed']}.pt"
        model,losses=train(setting,cfg,device,ckpt,updates,batch);test,raw=evaluate(model,cfg,setting["seed"],device,eval_examples);summary=aggregate(raw,cfg)
        cells.append({**setting,"updates":updates,"eval_examples":eval_examples,**summary,"checkpoint":str(ckpt)})
        raw_all += [{"seed":setting["seed"],**r} for r in raw];loss_all += [{"seed":setting["seed"],**r} for r in losses]
        write_csv(output/"local_step_competence.csv",cells);write_csv(output/"local_step_raw_latest.csv",raw_all);write_csv(output/"training_loss_latest.csv",loss_all)
        print(f"[{index+1}/{len(run_settings)}] seed={setting['seed']} entailed={summary['heldout_entailed_accuracy']:.3f} controls={summary['heldout_negative_control_accuracy']:.3f} competent={summary['competent']}",flush=True)
    atomic_write_json(output/"r0_manifest.json",{"schema_version":cfg["schema_version"],"device":str(device),"smoke":args.smoke,
        "updates":updates,"seeds_completed":sorted(int(r["seed"]) for r in cells),"required_seeds":cfg["model_seeds"],
        "dataset_audit":audit,"three_seed_gate_passed":len(cells)>=3 and all(int(r["competent"]) for r in cells),
        "heldout_symbol_identities":True,"checkpoint_resume":True})

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/paper085/r0_gate_v1.json")
    p.add_argument("--output",default="docs/papers/paper0_85/results/r0_gate_v1");p.add_argument("--device");p.add_argument("--smoke",action="store_true");p.add_argument("--resume",action="store_true");main(p.parse_args())
