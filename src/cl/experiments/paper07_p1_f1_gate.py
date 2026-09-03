"""Resumable narrow three-seed P1/F1 acquisition gate with template strata."""
from __future__ import annotations

import argparse,csv,json,random,time
from pathlib import Path
import torch
import torch.nn.functional as F

from cl.common.artifacts import atomic_write_json,stable_hash,write_csv
from cl.common.model_adapter import TinyTransformerLM
from cl.experiments.paper05_predictive_order_phase import resolve_device
from cl.semantic.paper07_gates import SYMBOL,YES,NO
from cl.semantic.paper07_p1_f1 import SYMBOL_COUNT,StepGateExample,step_gate_examples,validate_step_gates


def build(config:dict,smoke:bool=False)->list[StepGateExample]:
    rows=[]
    for stage in ("P1","F1"):
      for split,seed in config["split_seeds"].items():
        count=config["smoke_examples"] if smoke else config["examples"][split]
        rows.extend(step_gate_examples(stage,split,count,seed,config["sequence_length"]))
    return rows


def _save(path:Path,payload:dict):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(".tmp");torch.save(payload,temporary);temporary.replace(path)


def _read(path:Path)->list[dict]:
    with path.open(newline="",encoding="utf-8") as handle:return list(csv.DictReader(handle))


def train(config:dict,stage:str,seed:int,rows:list[StepGateExample],device:torch.device,output:Path,smoke:bool,resume:bool):
    arch=config["architecture"];steps=config["smoke_steps"] if smoke else config["steps_by_stage"][stage]
    batch_size=config["smoke_batch_size"] if smoke else config["batch_size"]
    torch.manual_seed(seed+(10000 if stage=="F1" else 0));model=TinyTransformerLM(config["vocab_size"],config["sequence_length"],
        arch["width"],arch["layers"],arch["heads"],arch["mlp_ratio"]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"]);rng=random.Random(seed+(21000 if stage=="F1" else 11000))
    losses=[];first=0;state_path=output/"training_state.pt"
    if resume and state_path.exists():
        state=torch.load(state_path,map_location=device,weights_only=False);model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"])
        rng.setstate(state["rng"]);losses=state["losses"];first=state["step"]
    train_rows=[r for r in rows if r.stage==stage and r.split=="train"];started=time.perf_counter()
    for step in range(first,steps):
        batch=[train_rows[rng.randrange(len(train_rows))] for _ in range(batch_size)]
        x=torch.tensor([r.tokens for r in batch],device=device);y=torch.tensor([r.target for r in batch],device=device)
        optimizer.zero_grad(set_to_none=True);logits,_=model(x);loss=F.cross_entropy(logits[:,-1],y);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        if step==first or (step+1)%50==0 or step+1==steps:losses.append({"step":step+1,"loss":float(loss.detach().cpu())})
        if (step+1)%config["checkpoint_every"]==0 or step+1==steps:
            _save(state_path,{"model":model.state_dict(),"optimizer":optimizer.state_dict(),"rng":rng.getstate(),"losses":losses,"step":step+1})
    torch.save(model.to("cpu").state_dict(),output/"checkpoint.pt")
    return model.eval(),losses,time.perf_counter()-started


@torch.no_grad()
def evaluate(model,rows,stage,seed,device):
    model=model.to(device);output=[]
    for split in ("validation","test"):
      selected=[r for r in rows if r.stage==stage and r.split==split]
      for start in range(0,len(selected),128):
        batch=selected[start:start+128];x=torch.tensor([r.tokens for r in batch],device=device);logits,_=model(x);logits=logits[:,-1]
        unrestricted=logits.argmax(-1).tolist()
        if stage=="P1": restricted=logits[:,[YES,NO]];predictions=[(YES,NO)[i] for i in restricted.argmax(-1).tolist()]
        else: predictions=(logits[:,SYMBOL:SYMBOL+SYMBOL_COUNT].argmax(-1)+SYMBOL).tolist()
        for index,(row,prediction,unrestricted_prediction) in enumerate(zip(batch,predictions,unrestricted)):
            candidates=[NO] if stage=="P1" and row.target==YES else [YES] if stage=="P1" else [v for v in range(SYMBOL,SYMBOL+SYMBOL_COUNT) if v!=row.target]
            competitor=max(float(logits[index,v]) for v in candidates)
            output.append({"stage":stage,"model_seed":seed,"split":split,"example_id":row.example_id,"template":row.template,
                "label":row.label,"target":row.target,"prediction":prediction,"correct":int(prediction==row.target),
                "unrestricted_prediction":unrestricted_prediction,
                "unrestricted_correct":int(unrestricted_prediction==row.target),
                "target_margin":float(logits[index,row.target])-competitor})
    return output


def aggregate(raw:list[dict],threshold:float):
    rows=[]
    keys=sorted({(r["stage"],int(r["model_seed"]),r["split"],r["template"]) for r in raw})
    for stage,seed,split,template in keys:
        selected=[r for r in raw if (r["stage"],int(r["model_seed"]),r["split"],r["template"])==(stage,seed,split,template)]
        accuracy=sum(int(r["correct"]) for r in selected)/len(selected)
        rows.append({"stage":stage,"model_seed":seed,"split":split,"template":template,"examples":len(selected),
                     "accuracy":accuracy,"mean_target_margin":sum(float(r["target_margin"]) for r in selected)/len(selected),
                     "competent":int(accuracy>=threshold)})
    return rows


def main(args):
    config=json.loads(Path(args.config).read_text());rows=build(config,args.smoke);audit=validate_step_gates(rows)
    if not audit["valid"]:raise RuntimeError(audit)
    output=Path(args.output);(output/"models").mkdir(parents=True,exist_ok=True);atomic_write_json(output/"gate_validation.json",audit)
    device=resolve_device(args.device);specs=[(stage,seed) for stage in ("P1","F1") for seed in config["model_seeds"]]
    if args.only_stage:specs=[s for s in specs if s[0]==args.only_stage]
    if args.max_models:specs=specs[:args.max_models]
    for index,(stage,seed) in enumerate(specs,1):
        model_dir=output/"models"/f"{stage}_seed{seed}";model_dir.mkdir(parents=True,exist_ok=True)
        if args.resume and (model_dir/"complete.json").exists():print(f"[{index}/{len(specs)}] skip {stage} seed {seed}",flush=True);continue
        print(f"[{index}/{len(specs)}] train {stage} seed={seed} on {device}",flush=True)
        model,losses,seconds=train(config,stage,seed,rows,device,model_dir,args.smoke,args.resume);raw=evaluate(model,rows,stage,seed,device)
        write_csv(model_dir/"raw.csv",raw);write_csv(model_dir/"loss.csv",losses)
        atomic_write_json(model_dir/"complete.json",{"stage":stage,"model_seed":seed,"rows":len(raw),"artifact_hash":stable_hash(raw),"seconds":seconds})
    raw=[];losses=[];runtimes=[]
    for complete in sorted((output/"models").glob("*/complete.json")):
        meta=json.loads(complete.read_text());raw.extend(_read(complete.parent/"raw.csv"));losses.extend({"stage":meta["stage"],"model_seed":meta["model_seed"],**r} for r in _read(complete.parent/"loss.csv"))
        runtimes.append({"stage":meta["stage"],"model_seed":meta["model_seed"],"seconds":meta["seconds"]})
    write_csv(output/"gate_raw.csv",raw);write_csv(output/"gate_training_loss.csv",losses);write_csv(output/"gate_runtime.csv",runtimes)
    by_template=aggregate(raw,config["competence_threshold"]);write_csv(output/"gate_accuracy_by_template.csv",by_template)
    overall=[]
    for stage,seed,split in sorted({(r["stage"],int(r["model_seed"]),r["split"]) for r in raw}):
        selected=[r for r in raw if (r["stage"],int(r["model_seed"]),r["split"])==(stage,seed,split)];accuracy=sum(int(r["correct"]) for r in selected)/len(selected)
        templates=[r for r in by_template if (r["stage"],int(r["model_seed"]),r["split"])==(stage,seed,split)]
        overall.append({"stage":stage,"model_seed":seed,"split":split,"examples":len(selected),"accuracy":accuracy,
            "worst_template_accuracy":min(float(r["accuracy"]) for r in templates),
            "competent":int(accuracy>=config["competence_threshold"] and all(int(r["competent"]) for r in templates))})
    write_csv(output/"gate_accuracy.csv",overall);tests=[r for r in overall if r["split"]=="test"]
    dataset_payload=[{"stage":r.stage,"split":r.split,"example_id":r.example_id,"tokens":r.tokens,
                      "target":r.target,"template":r.template} for r in rows]
    atomic_write_json(output/"gate_manifest.json",{"schema_version":config["schema_version"],"device":str(device),"smoke":args.smoke,
        "config_hash":stable_hash(config),"dataset_hash":stable_hash(dataset_payload),
        "planned_models":len(specs),"completed_models":len(list((output/"models").glob("*/complete.json"))),"dataset_audit":audit,
        "template_stratified":True,
        "decoding":{"P1":"argmax over the declared {YES,NO} output grammar",
                    "F1":"argmax over the declared 12-symbol binding-output grammar",
                    "unrestricted_argmax_also_recorded":True},
        "three_seed_gate":{stage:int(sum(r["stage"]==stage for r in tests)==3 and all(int(r["competent"]) for r in tests if r["stage"]==stage)) for stage in ("P1","F1")}})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/paper07/p1_f1_gate_v1.json")
    p.add_argument("--output",default="docs/papers/paper0_7/results/p1_f1_gate_v1");p.add_argument("--device",default="auto")
    p.add_argument("--smoke",action="store_true");p.add_argument("--resume",action="store_true");p.add_argument("--max-models",type=int)
    p.add_argument("--only-stage",choices=("P1","F1"));main(p.parse_args())
