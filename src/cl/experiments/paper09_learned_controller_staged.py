"""Shared deterministic selection and training protocol for Paper 0.9 Stages D/E."""
from __future__ import annotations
import csv, hashlib, json, random
from pathlib import Path
import numpy as np
import torch
from cl.common.artifacts import atomic_write_json,write_csv
from cl.experiments.paper05_predictive_order_phase import resolve_device
from cl.experiments.paper09_learned_controller_stage_a import read_csv
from cl.experiments.paper09_learned_controller_stage_b import summarize
from cl.experiments.paper09_learned_controller_v1 import evaluate_m3,evaluate_m4
from cl.experiments.paper09_learned_controller_v1 import context, dataset_sha256, make_model, stable_sha256
from cl.semantic.recurrence_chains import ANSWER, PAD, generate_chains
from cl.semantic.recurrence_chains import recurrence_pair_split

def rows(path):
    with Path(path).open(newline="",encoding="utf-8") as h:return list(csv.DictReader(h))

def select_architecture(plan, cells_path, stage_b, stage_c):
    sources=[]; observed=[]
    for stage,path in (("B",Path(stage_b)),("C",Path(stage_c))):
        manifest_path=path/f"stage_{stage.lower()}_manifest.json"; frontier_path=path/f"stage_{stage.lower()}_frontiers.csv"
        if not manifest_path.exists() or not frontier_path.exists():raise RuntimeError(f"Stage {stage} selection prerequisite absent")
        manifest=json.loads(manifest_path.read_text())
        if manifest.get("completed") is not True:raise RuntimeError(f"Stage {stage} selection prerequisite partial")
        selection_hash=""
        if stage=="C":
            selection_path=path/"stage_c_selection_manifest.json"
            if not selection_path.exists():raise RuntimeError("Stage C selection manifest absent")
            selection_payload=json.loads(selection_path.read_text());selection_hash=stable_sha256(selection_payload)
            if manifest.get("selection_manifest_hash")!=selection_hash:raise RuntimeError("Stage C selection manifest hash mismatch")
        values=rows(frontier_path)
        expected={(a,m) for a in plan["stages"][stage]["architectures"] for m in plan["common"]["machines"]}
        seen={(r["architecture"],r["machine"]) for r in values if int(r.get("gate_complete",0))==1 and r.get("contiguous_frontier","")!=""}
        if seen!=expected:raise RuntimeError(f"Stage {stage} frontiers incomplete")
        observed.extend(values);sources.append({"stage":stage,"manifest_hash":stable_sha256(manifest),"frontier_hash":stable_sha256(values),"selection_hash":selection_hash})
    specs={r["condition"]:r for r in rows(cells_path) if r["stage"] in ("B","C")}
    scores={}
    for architecture in specs:
        values=[int(r["contiguous_frontier"]) for r in observed if r["architecture"]==architecture]
        if len(values)==2:scores[architecture]={"min_frontier":min(values),"sum_frontier":sum(values),"parameters":int(specs[architecture]["parameters"])}
    winner=max(sorted(scores),key=lambda a:(scores[a]["min_frontier"],scores[a]["sum_frontier"],-scores[a]["parameters"]))
    return specs[winner],{"rule":"maximize minimum M3/M4 frontier, then sum, then minimize parameters, then lexical name",
                          "selected_architecture":winner,"scores":scores,"sources":sources}

def _sample(machine,pairs,depths,seed,index):
    digest=hashlib.sha256(f"paper09-v3|{machine}|{seed}|{index}".encode()).digest();rng=random.Random(int.from_bytes(digest[:8],"big"))
    depth=depths[index%len(depths)];row=generate_chains(pairs,depth,1,rng.randrange(2**31),"train")[0]
    if machine=="M4":
        step=rng.randrange(depth+1);history=list(row.chain[1:step+1]);target=row.chain[step+1] if step<depth else ANSWER
    else:
        stage=rng.randrange(2);history=[] if stage==0 else [row.chain[1]];target=row.chain[1] if stage==0 else row.chain[-1]
    return context(row,history),target

def protocol_index(protocol,absolute):
    if protocol["kind"]=="diversity":return absolute%protocol["pool_size"],[1,2,3]
    shared=protocol["shared_prefix_draws"]
    if absolute<shared:return absolute,[1,2,3]
    return absolute,[1,2,3,4] if protocol["train_kmax"]==4 else [1,2,3]

def train_protocol(machine,seed,cfg,device,checkpoint,pairs,updates,batch_size,protocol):
    torch.manual_seed(seed);model=make_model(cfg,device);opt=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"]);start=0;losses=[]
    protocol_hash=stable_sha256(protocol);config_hash=stable_sha256(cfg);data_hash=dataset_sha256(pairs)
    if checkpoint.exists():
        p=torch.load(checkpoint,map_location="cpu",weights_only=False)
        if (p.get("protocol_sha256"),p.get("config_sha256"),p.get("dataset_sha256"))!=(protocol_hash,config_hash,data_hash):raise RuntimeError("checkpoint protocol/config/dataset hash mismatch")
        model.load_state_dict(p["model"]);opt.load_state_dict(p["optimizer"]);start=p["step"];losses=p["losses"]
        if p.get("master_stream_cursor")!=start*batch_size:raise RuntimeError("checkpoint data cursor mismatch")
        random.setstate(p["python_rng"]);np.random.set_state(p["numpy_rng"]);torch.set_rng_state(p["torch_rng"])
        for state in opt.state.values():
            for key,value in state.items():
                if torch.is_tensor(value):state[key]=value.to(device)
    model.train()
    for step in range(start,updates):
        samples=[]
        for absolute in range(step*batch_size,(step+1)*batch_size):
            index,depths=protocol_index(protocol,absolute);samples.append(_sample(machine,pairs,depths,seed,index))
        length=max(len(x) for x,_ in samples);x=torch.tensor([v+[PAD]*(length-len(v)) for v,_ in samples],device=device);y=torch.tensor([v for _,v in samples],device=device)
        lengths=torch.tensor([len(v) for v,_ in samples],device=device);opt.zero_grad(set_to_none=True);logits,_=model(x)
        loss=torch.nn.functional.cross_entropy(logits[torch.arange(len(x),device=device),lengths-1],y);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step==0 or (step+1)%cfg["log_every"]==0 or step+1==updates:losses.append({"step":step+1,"loss":float(loss.detach())})
        if (step+1)%cfg["checkpoint_every"]==0 or step+1==updates:
            checkpoint.parent.mkdir(parents=True,exist_ok=True);torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),"step":step+1,"losses":losses,
                "master_stream_cursor":(step+1)*batch_size,"protocol_sha256":protocol_hash,"config_sha256":config_hash,"dataset_sha256":data_hash,
                "python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state()},checkpoint)
    return model.eval(),losses

def run_cells(stage,plan,specifications,selection,output,device_name,resume,smoke,protocol_for):
    common,dataset=plan["common"],plan["dataset"];output=Path(output);output.mkdir(parents=True,exist_ok=True);device=resolve_device(device_name)
    _,train_pairs,test_pairs=recurrence_pair_split(dataset["symbol_count"],dataset["pair_split_seed"],dataset["test_pair_fraction"])
    machines=common["machines"][:1] if smoke else common["machines"];seeds=common["model_seeds"][:1] if smoke else common["model_seeds"]
    specifications=specifications[:1] if smoke else specifications;depths=dataset["test_depths"][:2] if smoke else dataset["test_depths"]
    eval_per=4 if smoke else dataset["eval_per_depth_per_seed"];updates=2 if smoke else plan["stages"][stage]["updates"];batch=4 if smoke else common["batch_size"]
    prefix=f"stage_{stage.lower()}";raw=read_csv(output/f"{prefix}_raw.csv") if resume else [];losses=read_csv(output/f"{prefix}_loss.csv") if resume else [];required_rows=len(depths)*eval_per
    counts={}
    for spec in specifications:
      condition=spec["condition"];model_cfg={"layers":int(spec["layers"]),"width":int(spec["width"]),"heads":int(spec["heads"]),"mlp_ratio":common["mlp_ratio"]}
      cfg={"train_depths":dataset["train_depth_default"],"max_length":common["max_length"],"model":model_cfg,"learning_rate":common["learning_rate"],"log_every":50,"checkpoint_every":common["checkpoint_every"]}
      for machine in machines:
       for seed in seeds:
        key=(condition,machine,seed);count=sum(1 for r in raw if (r["architecture"],r["machine"],int(r["seed"]))==key);counts[key]=count
        if count==required_rows:print(f"skip complete {condition} {machine} seed={seed}",flush=True);continue
        raw=[r for r in raw if (r["architecture"],r["machine"],int(r["seed"]))!=key]
        protocol=protocol_for(spec,machine,seed,updates*batch,smoke);checkpoint=output/"checkpoints"/f"{condition}_{machine}_seed{seed}.pt"
        model,train_losses=train_protocol(machine,seed,cfg,device,checkpoint,train_pairs,updates,batch,protocol);cell=[]
        for depth in depths:
            examples=generate_chains(test_pairs,depth,eval_per,seed*1000+depth,"test");metrics=evaluate_m3(model,examples,device) if machine=="M3" else evaluate_m4(model,examples,device,common["max_extra_steps"])
            cell.extend({"architecture":condition,"machine":machine,"seed":seed,"depth":depth,"example_id":i,**row} for i,row in enumerate(metrics))
        raw.extend(cell);losses=[r for r in losses if not (r["architecture"]==condition and r["machine"]==machine and int(r["seed"])==seed)]
        losses.extend({"architecture":condition,"machine":machine,"seed":seed,**r} for r in train_losses);write_csv(output/f"{prefix}_raw.csv",raw);write_csv(output/f"{prefix}_loss.csv",losses)
        by_depth,frontiers=summarize(raw,plan["gates"]["competence_threshold"]);write_csv(output/f"{prefix}_by_seed_depth.csv",by_depth);write_csv(output/f"{prefix}_frontiers.csv",frontiers);print(f"complete {condition} {machine} seed={seed}",flush=True)
    by_depth,frontiers=summarize(raw,plan["gates"]["competence_threshold"]);write_csv(output/f"{prefix}_by_seed_depth.csv",by_depth);write_csv(output/f"{prefix}_frontiers.csv",frontiers)
    required=len(specifications)*len(machines)*len(seeds);complete=sum(sum(1 for r in raw if (r["architecture"],r["machine"],int(r["seed"]))==(s["condition"],m,z))==required_rows for s in specifications for m in machines for z in seeds)
    manifest={"schema_version":f"paper09.learned_controller.stage_{stage.lower()}.v1","device":str(device),"smoke":smoke,"completed":complete==required,"complete_cells":complete,"required_cells":required,
      "exact_resume":True,"selection_manifest_hash":stable_sha256(selection),"config_sha256":stable_sha256(plan),"cell_spec_sha256":stable_sha256(specifications),"dataset_sha256":dataset_sha256(train_pairs),
      "updates":updates,"matched_draws":updates*batch,"conditions":[s["condition"] for s in specifications],"machines":machines,"seeds":seeds,"raw_rows":len(raw),"frontiers":frontiers}
    atomic_write_json(output/f"{prefix}_manifest.json",manifest);return manifest
