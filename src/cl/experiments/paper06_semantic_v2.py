"""Run competence-gated Paper 0.6 S1 taxonomy experiments."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cl.common.artifacts import atomic_write_json,stable_hash,write_csv
from cl.common.model_adapter import TinyTransformerLM
from cl.experiments.paper05_predictive_order_phase import resolve_device
from cl.semantic.v2 import SemanticExample,s1_evaluation,s1_training_batch,s1_validation


def parameter_count(model):return sum(p.numel() for p in model.parameters())


def train_s1(config:dict,seed:int,device:torch.device):
    torch.manual_seed(seed);rng=random.Random(seed+701);model=TinyTransformerLM(config["vocab_size"],config["sequence_length"],config["width"],config["depth"],config["heads"]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"]);loss=[]
    for step in range(config["s1_steps"]):
        rows=s1_training_batch(config,rng);x=torch.tensor([r.tokens for r in rows],device=device);y=torch.tensor([r.target for r in rows],device=device)
        optimizer.zero_grad(set_to_none=True);z,_=model(x);value=F.cross_entropy(z[:,-1],y);value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        if step in (0,config["s1_steps"]-1) or (step+1)%100==0:loss.append({"model_seed":seed,"step":step+1,"loss":float(value.detach().cpu())})
    return model.eval(),loss


@torch.no_grad()
def evaluate_s1(model,rows:list[SemanticExample],seed:int,batch_size:int=128):
    device=next(model.parameters()).device;raw=[];layer=[]
    for start in range(0,len(rows),batch_size):
        batch=rows[start:start+batch_size];x=torch.tensor([r.tokens for r in batch],device=device);y=torch.tensor([r.target for r in batch],device=device);z,_=model(x)
        final=z[:,-1];p=final.softmax(-1);chosen=final.gather(1,y[:,None]).squeeze(1);other=final.clone();other.scatter_(1,y[:,None],float("-inf"));margin=chosen-other.max(1).values
        for i,r in enumerate(batch):raw.append({"model_seed":seed,**{k:getattr(r,k) for k in ("example_id","ontology_family","label_mode","query_level","template_id","position_mode","entity_id","category_id","parent_id","root_id")},"top1_correct":int(final[i].argmax()==y[i]),"target_probability":float(p[i,y[i]]),"target_rank":int(1+(final[i]>chosen[i]).sum()),"target_margin":float(margin[i]),"output_entropy":float(-(p[i]*p[i].clamp_min(1e-12).log()).sum())})
    # Trace a balanced canonical subset after competence is known.
    selected=[rows[i] for i in np.linspace(0,len(rows)-1,min(768,len(rows)),dtype=int)]
    for start in range(0,len(selected),batch_size):
        batch=selected[start:start+batch_size];x=torch.tensor([r.tokens for r in batch],device=device);y=torch.tensor([r.target for r in batch],device=device);_,trace=model(x,capture=True);states=[trace.layers[0].pre_sa]
        for block in trace.layers:states.extend((block.post_sa,block.post_block))
        for boundary,state in enumerate(states):
            z=model.diagnostic_logits(state[:,-1]);p=z.softmax(-1);chosen=z.gather(1,y[:,None]).squeeze(1);other=z.clone();other.scatter_(1,y[:,None],float("-inf"));margin=chosen-other.max(1).values
            for i,r in enumerate(batch):layer.append({"model_seed":seed,"example_id":r.example_id,"ontology_family":r.ontology_family,"label_mode":r.label_mode,"query_level":r.query_level,"template_id":r.template_id,"position_mode":r.position_mode,"entity_id":r.entity_id,"category_id":r.category_id,"parent_id":r.parent_id,"root_id":r.root_id,"boundary":boundary,"target_probability":float(p[i,y[i]]),"target_rank":int(1+(z[i]>chosen[i]).sum()),"target_margin":float(margin[i]),"top1_correct":int(z[i].argmax()==y[i]),"output_entropy":float(-(p[i]*p[i].clamp_min(1e-12).log()).sum()),"residual":":".join(f"{v:.6g}" for v in state[i,-1].cpu().tolist())})
    return raw,layer


def main(args)->None:
    config=json.loads(Path(args.config).read_text());out=Path(args.output)/"s1_taxonomy";out.mkdir(parents=True,exist_ok=True);validation=s1_validation(config);atomic_write_json(out/"s1_generator_validation.json",validation)
    if not validation["passed"]:raise RuntimeError(validation)
    if args.smoke:config={**config,"s1_steps":4,"batch_size":8,"model_seeds":[11]}
    device=resolve_device(args.device or config["device"]);rows=s1_evaluation(config,args.eval_examples);raw=[];layers=[];losses=[];models=[]
    for i,seed in enumerate(config["model_seeds"],1):
        print(f"[{i}/{len(config['model_seeds'])}] S1 seed={seed}",flush=True)
        if args.evaluate_only:
            model=TinyTransformerLM(config["vocab_size"],config["sequence_length"],config["width"],config["depth"],config["heads"]);model.load_state_dict(torch.load(out/f"s1_seed{seed}.pt",map_location="cpu",weights_only=True));model=model.to(device).eval();loss=[]
        else:model,loss=train_s1(config,seed,device)
        r,l=evaluate_s1(model,rows,seed);raw.extend(r);layers.extend(l);losses.extend(loss);models.append({"model_seed":seed,"parameter_count":parameter_count(model),"depth":config["depth"],"width":config["width"],"steps":config["s1_steps"]})
        if not args.evaluate_only:torch.save(model.to("cpu").state_dict(),out/f"s1_seed{seed}.pt")
        write_csv(out/"s1_competence_raw.csv",raw);write_csv(out/"s1_layer_raw.csv",layers)
    if losses:write_csv(out/"s1_training_loss.csv",losses)
    write_csv(out/"s1_models.csv",models)
    cells=[]
    groups=defaultdict(list)
    for r in raw:groups[(r["model_seed"],r["ontology_family"],r["label_mode"],r["query_level"],r["position_mode"])].append(r)
    for key,values in groups.items():cells.append({**dict(zip(("model_seed","ontology_family","label_mode","query_level","position_mode"),key)),"accuracy":float(np.mean([v["top1_correct"] for v in values])),"mean_margin":float(np.mean([v["target_margin"] for v in values])),"n":len(values)})
    write_csv(out/"s1_competence.csv",cells);minimum=min(r["accuracy"] for r in cells);mean=float(np.mean([r["accuracy"] for r in cells]));passed=minimum>=config["competence_threshold"]
    decision={"schema_version":"paper06.s1.gate.v1","gate_passed":passed,"mean_accuracy":mean,"minimum_cell_accuracy":minimum,"threshold":config["competence_threshold"],"mechanism_status":"eligible" if passed else "blocked","config":config};decision["artifact_hash"]=stable_hash({"cells":cells,"layers":layers});atomic_write_json(out/"s1_gate.json",decision);print(json.dumps(decision,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/paper06/semantic_v2.json");p.add_argument("--output",default="docs/papers/paper0_6/results/v2");p.add_argument("--device");p.add_argument("--smoke",action="store_true");p.add_argument("--evaluate-only",action="store_true");p.add_argument("--eval-examples",type=int);main(p.parse_args())
