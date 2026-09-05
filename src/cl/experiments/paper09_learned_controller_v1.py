"""Learned M3/M4 controllers over corrected Paper 0.85 chains and T1."""
from __future__ import annotations
import argparse,csv,hashlib,json,random
from pathlib import Path
import numpy as np
import torch
from cl.common.artifacts import atomic_write_json,write_csv
from cl.common.model_adapter import TinyTransformerLM
from cl.experiments.paper05_predictive_order_phase import resolve_device
from cl.semantic.recurrence_chains import ANSWER,PAD,STATE,SYMBOL_OFFSET,generate_chains,recurrence_pair_split,validate_chain_splits

RESULT=SYMBOL_OFFSET+64
INVALID=RESULT+1
VOCAB_SIZE=INVALID+1

def context(row,history):
    tokens=[*row.prompt,STATE,row.chain[0]]
    for value in history:tokens.extend((RESULT,value))
    return tokens

def make_model(cfg,device):return TinyTransformerLM(VOCAB_SIZE,cfg["max_length"],cfg["model"]["width"],cfg["model"]["layers"],cfg["model"]["heads"],cfg["model"]["mlp_ratio"]).to(device)

def stable_sha256(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),default=list).encode()).hexdigest()

def dataset_sha256(pairs):
    return stable_sha256([list(pair) for pair in pairs])

def capture_replay_state(rng,step,batch_size,device):
    state={"python_rng":random.getstate(),"data_rng":rng.getstate(),"numpy_rng":np.random.get_state(),
           "torch_rng":torch.get_rng_state(),"master_stream_cursor":step*batch_size}
    if device.type=="mps":
        state["mps_rng_if_available"]=torch.mps.get_rng_state()
    return state

def restore_replay_state(payload,rng,device):
    random.setstate(payload["python_rng"]);rng.setstate(payload["data_rng"]);np.random.set_state(payload["numpy_rng"])
    torch.set_rng_state(payload["torch_rng"])
    if device.type=="mps" and "mps_rng_if_available" in payload:torch.mps.set_rng_state(payload["mps_rng_if_available"])

def training_batch(machine,pairs,cfg,rng,batch_size,device):
    samples=[]
    for i in range(batch_size):
        depth=cfg["train_depths"][i%len(cfg["train_depths"])];row=generate_chains(pairs,depth,1,rng.randrange(2**31),"train")[0]
        if machine=="M4":
            step=rng.randrange(depth+1);history=list(row.chain[1:step+1]);target=row.chain[step+1] if step<depth else ANSWER
        else:
            stage=rng.randrange(2);history=[] if stage==0 else [row.chain[1]];target=row.chain[1] if stage==0 else row.chain[-1]
        samples.append((context(row,history),target))
    length=max(len(x) for x,_ in samples)
    x=torch.tensor([tokens+[PAD]*(length-len(tokens)) for tokens,_ in samples],device=device)
    y=torch.tensor([target for _,target in samples],device=device);lengths=torch.tensor([len(tokens) for tokens,_ in samples],device=device)
    return x,y,lengths

def train(machine,seed,cfg,device,checkpoint,pairs,updates,batch_size):
    torch.manual_seed(seed);rng=random.Random(seed+(300 if machine=="M3" else 400));model=make_model(cfg,device);opt=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"]);start=0;losses=[]
    config_hash=stable_sha256(cfg);data_hash=dataset_sha256(pairs)
    if checkpoint.exists():
        p=torch.load(checkpoint,map_location="cpu",weights_only=False);model.load_state_dict(p["model"]);opt.load_state_dict(p["optimizer"]);start=p["step"];losses=p.get("losses",[])
        if p.get("config_sha256")!=config_hash or p.get("dataset_sha256")!=data_hash:raise RuntimeError("checkpoint config/dataset hash mismatch")
        if p.get("master_stream_cursor")!=start*batch_size:raise RuntimeError("checkpoint data cursor mismatch")
        for state in opt.state.values():
            for key,value in state.items():
                if torch.is_tensor(value):state[key]=value.to(device)
        restore_replay_state(p,rng,device)
    model.train()
    for step in range(start,updates):
        x,y,lengths=training_batch(machine,pairs,cfg,rng,batch_size,device);opt.zero_grad(set_to_none=True);logits,_=model(x)
        selected=logits[torch.arange(len(x),device=device),lengths-1];loss=torch.nn.functional.cross_entropy(selected,y)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step==0 or (step+1)%cfg["log_every"]==0 or step+1==updates:losses.append({"step":step+1,"loss":float(loss.detach())})
        if (step+1)%cfg["checkpoint_every"]==0 or step+1==updates:
            checkpoint.parent.mkdir(parents=True,exist_ok=True);torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),"step":step+1,
                "losses":losses,"machine":machine,"seed":seed,"config_sha256":config_hash,"dataset_sha256":data_hash,
                **capture_replay_state(rng,step+1,batch_size,device)},checkpoint)
    return model.eval(),losses

@torch.no_grad()
def predict(model,token_lists,device):
    length=max(map(len,token_lists));x=torch.tensor([tokens+[PAD]*(length-len(tokens)) for tokens in token_lists],device=device)
    lengths=torch.tensor([len(tokens) for tokens in token_lists],device=device);logits,_=model(x)
    return logits[torch.arange(len(x),device=device),lengths-1].argmax(-1).tolist()

def evaluate_m3(model,rows,device):
    selected=predict(model,[context(r,[]) for r in rows],device);result=[]
    valid=[]
    for row,value in zip(rows,selected):valid.append(value==row.chain[1])
    # T1 rejects an invalid selected edge and returns a typed failure token.
    final=predict(model,[context(r,[value if ok else INVALID]) for r,value,ok in zip(rows,selected,valid)],device)
    for row,choice,ok,answer in zip(rows,selected,valid,final):
        final_correct=int(answer==row.chain[-1])
        result.append({"final_correct":final_correct,"selected_edge_valid":int(ok),
            "one_call_edge_coverage":int(ok)/row.depth,"post_tool_answer_correct":final_correct,
            "tool_calls":1,"model_forwards":2,"invalid_call":int(not ok),"nontermination":0})
    return result

@torch.no_grad()
def evaluate_m4(model,rows,device,max_extra):
    histories=[[] for _ in rows];active=[True]*len(rows);valid_steps=[0]*len(rows);invalid=[0]*len(rows);answered=[False]*len(rows);calls=[0]*len(rows);forwards=[0]*len(rows)
    for _ in range(max(r.depth for r in rows)+max_extra):
        indices=[i for i,a in enumerate(active) if a]
        if not indices:break
        outputs=predict(model,[context(rows[i],histories[i]) for i in indices],device)
        for i,value in zip(indices,outputs):
            forwards[i]+=1
            current=rows[i].chain[len(histories[i])]
            if value==ANSWER:
                answered[i]=current==rows[i].chain[-1];active[i]=False;continue
            calls[i]+=1;expected_index=len(histories[i])+1
            if expected_index>=len(rows[i].chain) or value!=rows[i].chain[expected_index]:invalid[i]=1;active[i]=False;continue
            valid_steps[i]+=1;histories[i].append(value)
    result=[]
    for i,row in enumerate(rows):
        correct=int(answered[i]);stop_emitted=int(not active[i] and not invalid[i])
        result.append({"final_correct":correct,"per_transition_accuracy":valid_steps[i]/row.depth,
            "exact_trajectory_correct":correct,"termination_correct":correct,"stop_emitted":stop_emitted,"tool_calls":calls[i],
            "model_forwards":forwards[i],"invalid_call":invalid[i],"nontermination":int(active[i])})
    return result

def read(path):
    if not path.exists():return []
    with path.open(newline="",encoding="utf-8") as h:return list(csv.DictReader(h))

def main(args):
    cfg=json.loads(Path(args.config).read_text());r0=json.loads(Path(cfg["r0_gate_manifest"]).read_text())
    if not r0["three_seed_gate_passed"]:raise RuntimeError("R0 gate closed")
    _,train_pairs,test_pairs=recurrence_pair_split(cfg["symbol_count"],cfg["pair_split_seed"],cfg["test_pair_fraction"])
    depths=cfg["smoke_test_depths"] if args.smoke else cfg["test_depths"];eval_per=cfg["smoke_eval_per_depth"] if args.smoke else cfg["eval_per_depth"]
    audit=validate_chain_splits(train_pairs,test_pairs,sum((generate_chains(train_pairs,d,4,100+d,"train") for d in cfg["train_depths"]),[]),
        sum((generate_chains(test_pairs,d,4,200+d,"test") for d in depths),[]))
    if not audit["valid"]:raise RuntimeError(audit)
    device=resolve_device(args.device or cfg["device"]);out=Path(args.output);out.mkdir(parents=True,exist_ok=True);updates=cfg["smoke_updates"] if args.smoke else cfg["updates"];batch=cfg["smoke_batch_size"] if args.smoke else cfg["batch_size"]
    machines=cfg["machines"][:1] if args.smoke else cfg["machines"];seeds=cfg["model_seeds"][:1] if args.smoke else cfg["model_seeds"]
    cells=read(out/"learned_controller_cells.csv") if args.resume else [];done={(r["machine"],int(r["seed"])) for r in cells};raw_all=read(out/"learned_controller_raw.csv") if args.resume else [];loss_all=read(out/"learned_controller_loss.csv") if args.resume else []
    total=len(machines)*len(seeds);index=0
    for machine in machines:
      for seed in seeds:
        index+=1
        if (machine,seed) in done:print(f"skip completed {machine} seed {seed}",flush=True);continue
        ckpt=out/"checkpoints"/f"{machine}_L2_W64_H4_seed{seed}.pt";model,losses=train(machine,seed,cfg,device,ckpt,train_pairs,updates,batch);cell=[]
        for depth in depths:
            rows=generate_chains(test_pairs,depth,eval_per,seed*1000+depth,"test");metrics=evaluate_m3(model,rows,device) if machine=="M3" else evaluate_m4(model,rows,device,cfg["max_extra_steps"])
            cell += [{"depth":depth,"example_id":i,**m} for i,m in enumerate(metrics)]
        mean=lambda k:float(np.mean([r[k] for r in cell]));summary={"machine":machine,"tool_strength":"T1","seed":seed,"updates":updates,
            "final_accuracy":mean("final_correct"),"invalid_call_rate":mean("invalid_call")}
        if machine=="M3":
            summary.update({"selected_edge_validity":mean("selected_edge_valid"),"one_call_edge_coverage":mean("one_call_edge_coverage"),
                "post_tool_answer_accuracy":mean("post_tool_answer_correct"),"competent":int(mean("final_correct")>=.95 and mean("selected_edge_valid")>=.95)})
        else:
            summary.update({"per_transition_accuracy":mean("per_transition_accuracy"),"exact_trajectory_accuracy":mean("exact_trajectory_correct"),
                "termination_accuracy":mean("termination_correct"),"competent":int(mean("final_correct")>=.95 and mean("per_transition_accuracy")>=.95 and mean("termination_correct")>=.95)})
        cells.append(summary);raw_all += [{"machine":machine,"seed":seed,**r} for r in cell];loss_all += [{"machine":machine,"seed":seed,**r} for r in losses]
        write_csv(out/"learned_controller_cells.csv",cells);write_csv(out/"learned_controller_raw.csv",raw_all);write_csv(out/"learned_controller_loss.csv",loss_all)
        print(f"[{index}/{total}] {machine} seed={seed} final={summary['final_accuracy']:.3f} invalid={summary['invalid_call_rate']:.3f}",flush=True)
    atomic_write_json(out/"learned_controller_manifest.json",{"schema_version":cfg["schema_version"],"device":str(device),"smoke":args.smoke,"dataset_audit":audit,
        "tool_strength":"T1_local_apply_no_search","learned_model_results":True,"train_depths":cfg["train_depths"],"test_depths":depths,
        "cells_completed":len(cells),"required_cells":6,"all_three_seed_conditions_complete":len(cells)>=6,"T2_status":"delegation_control_pending","T3_status":"oracle_control_only","checkpoint_resume":True})

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/paper09/learned_controller_v1.json");p.add_argument("--output",default="docs/papers/paper0_9/results/learned_controller_v1")
    p.add_argument("--device");p.add_argument("--smoke",action="store_true");p.add_argument("--resume",action="store_true");main(p.parse_args())
