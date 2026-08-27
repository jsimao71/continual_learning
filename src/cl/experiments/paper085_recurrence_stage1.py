"""Gated O0--O3 recurrence comparison at the passing R0-v2 architecture."""
from __future__ import annotations
import argparse,csv,json,random
from pathlib import Path
import numpy as np
import torch
from cl.common.artifacts import atomic_write_json,write_csv
from cl.common.model_adapter import TinyTransformerLM
from cl.experiments.paper05_predictive_order_phase import resolve_device
from cl.semantic.recurrence_chains import (ANSWER,DERIVE,END,PAD,STATE,SYMBOL_OFFSET,generate_chains,
    recurrence_pair_split,validate_chain_splits)

CONDITIONS=("O0","O1","O2","O3")
def model(cfg,device):return TinyTransformerLM(SYMBOL_OFFSET+cfg["symbol_count"],cfg["max_length"],cfg["model"]["width"],cfg["model"]["layers"],cfg["model"]["heads"],cfg["model"]["mlp_ratio"]).to(device)

def training_batch(rows,condition,device):
    sequences=[(*r.prompt,*r.target(condition)) for r in rows];length=max(map(len,sequences));inputs=[];targets=[];masks=[]
    for row,seq in zip(rows,sequences):
        inp=list(seq[:-1]);tar=list(seq[1:]);mask=[False]*(len(row.prompt)-1)+[True]*len(row.target(condition))
        if condition=="O2":
            # Teach serialization control and final answer, but not intermediate proposition states.
            start=len(row.prompt)-1
            for j,token in enumerate(row.target(condition)):
                mask[start+j]=token in {STATE,DERIVE,ANSWER,END} or (j>0 and row.target(condition)[j-1]==ANSWER)
        pad=length-1-len(inp);inputs.append(inp+[PAD]*pad);targets.append(tar+[PAD]*pad);masks.append(mask+[False]*pad)
    return torch.tensor(inputs,device=device),torch.tensor(targets,device=device),torch.tensor(masks,device=device)

def train_cell(condition,seed,cfg,device,checkpoint,train_pairs,updates,batch_size):
    condition_seed={"O0":0,"O1":101,"O2":202,"O3":303}[condition]
    torch.manual_seed(seed);rng=random.Random(seed+condition_seed);m=model(cfg,device);opt=torch.optim.AdamW(m.parameters(),lr=cfg["learning_rate"]);start=0;losses=[]
    if checkpoint.exists():
        p=torch.load(checkpoint,map_location="cpu",weights_only=False);m.load_state_dict(p["model"]);opt.load_state_dict(p["optimizer"]);start=p["step"];losses=p.get("losses",[])
        for state in opt.state.values():
            for key,value in state.items():
                if torch.is_tensor(value):state[key]=value.to(device)
    m.train()
    for step in range(start,updates):
        depths=cfg["train_depths"];per=max(1,batch_size//len(depths));rows=[]
        for depth in depths:rows+=generate_chains(train_pairs,depth,per,rng.randrange(2**31),"train")
        rows=rows[:batch_size];x,y,mask=training_batch(rows,condition,device);opt.zero_grad(set_to_none=True);logits,_=m(x)
        loss=torch.nn.functional.cross_entropy(logits[mask],y[mask]);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
        if step==0 or (step+1)%cfg["log_every"]==0 or step+1==updates:losses.append({"step":step+1,"loss":float(loss.detach())})
        if (step+1)%cfg["checkpoint_every"]==0 or step+1==updates:
            checkpoint.parent.mkdir(parents=True,exist_ok=True);torch.save({"model":m.state_dict(),"optimizer":opt.state_dict(),"step":step+1,"losses":losses,"condition":condition,"seed":seed},checkpoint)
    return m.eval(),losses

@torch.no_grad()
def generate(model_,rows,condition,cfg,device):
    prompts=[list(r.prompt) for r in rows];seq=torch.tensor(prompts,device=device);generated=[[] for _ in rows]
    max_new=1 if condition=="O0" else max(len(r.target(condition)) for r in rows)+cfg["generation_slack"]
    active=torch.ones(len(rows),dtype=torch.bool,device=device)
    for _ in range(max_new):
        logits,_=model_(seq);nxt=logits[:,-1].argmax(-1);seq=torch.cat((seq,nxt[:,None]),1)
        for i,value in enumerate(nxt.tolist()):
            if active[i]:generated[i].append(value)
        if condition!="O0":active &= nxt.ne(END)
        if not active.any():break
    return generated

def score(row,output,condition):
    expected=list(row.target(condition));exact=output==expected
    if condition=="O0":return {"final_correct":int(bool(output) and output[0]==row.chain[-1]),"trajectory_exact":int(exact),"transition_accuracy":int(exact),"termination_correct":1,"generated_steps":1}
    end=[i for i,x in enumerate(output) if x==END];termination=len(end)==1 and end[0]==len(output)-1
    if condition=="O1":observed=[x for x in output if x>=SYMBOL_OFFSET]
    else:observed=[output[i+1] for i,x in enumerate(output[:-1]) if x==DERIVE]
    truth=list(row.chain[1:]);correct=sum(a==b for a,b in zip(observed,truth));final=row.chain[-1] in output
    return {"final_correct":int(final),"trajectory_exact":int(exact),"transition_accuracy":correct/len(truth),"termination_correct":int(termination),"generated_steps":len(output)}

def read(path):
    if not path.exists():return []
    with path.open(newline="",encoding="utf-8") as h:return list(csv.DictReader(h))

def main(args):
    cfg=json.loads(Path(args.config).read_text());r0=json.loads(Path(cfg["r0_gate_manifest"]).read_text())
    if not r0["three_seed_gate_passed"]:raise RuntimeError("R0-v2 prerequisite is closed")
    device=resolve_device(args.device or cfg["device"]);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    symbols,train_pairs,test_pairs=recurrence_pair_split(cfg["symbol_count"],cfg["pair_split_seed"],cfg["test_pair_fraction"])
    smoke=args.smoke;depths=cfg["smoke_test_depths"] if smoke else cfg["test_depths"];eval_per=cfg["smoke_eval_per_depth"] if smoke else cfg["eval_per_depth"]
    audit_train=sum((generate_chains(train_pairs,d,8,100+d,"train") for d in cfg["train_depths"]),[])
    audit_test=sum((generate_chains(test_pairs,d,8,200+d,"test") for d in depths),[]);audit=validate_chain_splits(train_pairs,test_pairs,audit_train,audit_test)
    if not audit["valid"]:raise RuntimeError(audit)
    updates=cfg["smoke_updates"] if smoke else cfg["updates"];batch=cfg["smoke_batch_size"] if smoke else cfg["batch_size"]
    conditions=CONDITIONS[:1] if smoke else CONDITIONS;seeds=cfg["model_seeds"][:1] if smoke else cfg["model_seeds"]
    cells=read(out/"stage1_cells.csv") if args.resume else [];done={(r["condition"],int(r["seed"])) for r in cells}
    raw_all=read(out/"stage1_raw_latest.csv") if args.resume else [];loss_all=read(out/"stage1_loss_latest.csv") if args.resume else []
    total=len(conditions)*len(seeds);index=0
    for condition in conditions:
      for seed in seeds:
        index+=1
        if (condition,seed) in done:print(f"skip completed {condition} seed {seed}",flush=True);continue
        ckpt=out/"checkpoints"/f"{condition}_L{cfg['model']['layers']}_W{cfg['model']['width']}_H{cfg['model']['heads']}_seed{seed}.pt"
        m,losses=train_cell(condition,seed,cfg,device,ckpt,train_pairs,updates,batch);cell_raw=[]
        for depth in depths:
            rows=generate_chains(test_pairs,depth,eval_per,seed*1000+depth,"test");outputs=generate(m,rows,condition,cfg,device)
            cell_raw += [{"depth":depth,"example_id":i,**score(row,output,condition)} for i,(row,output) in enumerate(zip(rows,outputs))]
        agg=lambda key:float(np.mean([r[key] for r in cell_raw]));summary={"condition":condition,"seed":seed,**cfg["model"],"updates":updates,
            "final_accuracy":agg("final_correct"),"trajectory_accuracy":agg("trajectory_exact"),"transition_accuracy":agg("transition_accuracy"),
            "termination_accuracy":agg("termination_correct"),"competent":int(agg("final_correct")>=cfg["competence_accuracy"] and agg("termination_correct")>=cfg["termination_accuracy"])}
        cells.append(summary);raw_all += [{"condition":condition,"seed":seed,**r} for r in cell_raw];loss_all += [{"condition":condition,"seed":seed,**r} for r in losses]
        write_csv(out/"stage1_cells.csv",cells);write_csv(out/"stage1_raw_latest.csv",raw_all);write_csv(out/"stage1_loss_latest.csv",loss_all)
        print(f"[{index}/{total}] {condition} seed={seed} final={summary['final_accuracy']:.3f} trajectory={summary['trajectory_accuracy']:.3f} transition={summary['transition_accuracy']:.3f} termination={summary['termination_accuracy']:.3f}",flush=True)
    atomic_write_json(out/"stage1_manifest.json",{"schema_version":cfg["schema_version"],"r0_gate_passed":True,"device":str(device),"smoke":smoke,
        "train_depths":cfg["train_depths"],"test_depths":depths,"dataset_audit":audit,"cells_completed":len(cells),"required_cells":12,
        "all_conditions_three_seed_complete":len(cells)>=12,"checkpoint_resume":True,"supervision":{"O0":"final_only","O1":"free_trace",
        "O2":"structured_grammar_and_final_without_intermediate_state_labels","O3":"fully_teacher_forced_structured_trace"}})

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/paper085/recurrence_stage1_v1.json");p.add_argument("--output",default="docs/papers/paper0_85/results/recurrence_stage1_v1")
    p.add_argument("--device");p.add_argument("--smoke",action="store_true");p.add_argument("--resume",action="store_true");main(p.parse_args())
