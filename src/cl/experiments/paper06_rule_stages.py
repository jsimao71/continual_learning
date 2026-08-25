"""Train and evaluate Paper 0.6 S2 relational and S3 compositional generators."""
from __future__ import annotations
import argparse,json,random,time
from collections import defaultdict
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from cl.common.artifacts import atomic_write_json,stable_hash,write_csv
from cl.common.model_adapter import TinyTransformerLM
from cl.experiments.paper05_predictive_order_phase import resolve_device
from cl.semantic.v2 import rule_evaluation,rule_training_batch,rule_validation

def train(config,stage,seed,device):
    torch.manual_seed(seed);rng=random.Random(seed+900);model=TinyTransformerLM(config["vocab_size"],config["sequence_length"],config["width"],config["depth"],config["heads"]).to(device);opt=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"]);loss=[];steps=config[f"{stage}_steps"]
    started=time.time()
    for step in range(steps):
        rows=rule_training_batch(config,stage,rng);x=torch.tensor([r.tokens for r in rows],device=device);y=torch.tensor([r.target for r in rows],device=device);opt.zero_grad(set_to_none=True);z,_=model(x);value=F.cross_entropy(z[:,-1],y);value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step==0 or (step+1)%100==0 or step+1==steps:
            loss.append({"model_seed":seed,"step":step+1,"loss":float(value.detach().cpu()),"elapsed_seconds":time.time()-started});print(f"{stage} seed={seed} step={step+1}/{steps} loss={loss[-1]['loss']:.4f}",flush=True)
    return model.eval(),loss

@torch.no_grad()
def evaluate(model,rows,stage,seed,batch_size=128):
    device=next(model.parameters()).device;raw=[];layer=[]
    for start in range(0,len(rows),batch_size):
        batch=rows[start:start+batch_size];x=torch.tensor([r.tokens for r in batch],device=device);y=torch.tensor([r.target for r in batch],device=device);z,_=model(x);p=z[:,-1].softmax(-1);chosen=z[:,-1].gather(1,y[:,None]).squeeze(1);other=z[:,-1].clone();other.scatter_(1,y[:,None],float("-inf"));margin=chosen-other.max(1).values
        for i,r in enumerate(batch):raw.append({"stage":stage,"model_seed":seed,"example_id":r.example_id,"label_mode":r.label_mode,"template_id":r.template_id,"position_mode":r.position_mode,"entity_id":r.entity_id,"class_id":r.class_id,"predictive_order":r.predictive_order,"feature_bits":"".join(map(str,r.feature_bits)),"top1_correct":int(z[i,-1].argmax()==y[i]),"target_probability":float(p[i,y[i]]),"target_margin":float(margin[i]),"output_entropy":float(-(p[i]*p[i].clamp_min(1e-12).log()).sum())})
    selected=rows[::max(1,len(rows)//512)][:512]
    for start in range(0,len(selected),batch_size):
        batch=selected[start:start+batch_size];x=torch.tensor([r.tokens for r in batch],device=device);y=torch.tensor([r.target for r in batch],device=device);_,trace=model(x,capture=True);states=[trace.layers[0].pre_sa]
        for block in trace.layers:states.extend((block.post_sa,block.post_block))
        for boundary,state in enumerate(states):
            z=model.diagnostic_logits(state[:,-1]);chosen=z.gather(1,y[:,None]).squeeze(1);other=z.clone();other.scatter_(1,y[:,None],float("-inf"));margin=chosen-other.max(1).values
            for i,r in enumerate(batch):layer.append({"stage":stage,"model_seed":seed,"example_id":r.example_id,"class_id":r.class_id,"predictive_order":r.predictive_order,"template_id":r.template_id,"position_mode":r.position_mode,"boundary":boundary,"target_margin":float(margin[i]),"top1_correct":int(z[i].argmax()==y[i]),"residual":":".join(f"{v:.6g}" for v in state[i,-1].cpu().tolist())})
    return raw,layer

def main(args):
    config=json.loads(Path(args.config).read_text());stage=args.stage;out=Path(args.output)/({"s2":"s2_relational","s3":"s3_compositional"}[stage]);out.mkdir(parents=True,exist_ok=True);validation=rule_validation(config,stage);atomic_write_json(out/f"{stage}_generator_validation.json",validation)
    if not validation["passed"]:raise RuntimeError(validation)
    if args.steps is not None:config={**config,f"{stage}_steps":args.steps}
    if args.smoke:config={**config,f"{stage}_steps":4,"batch_size":8,"model_seeds":[11]}
    rows=rule_evaluation(config,stage,args.eval_examples);device=resolve_device(args.device or config["device"]);raw=[];layers=[];losses=[]
    for seed in config["model_seeds"]:
        model,loss=train(config,stage,seed,device);r,l=evaluate(model,rows,stage,seed);raw+=r;layers+=l;losses+=loss;torch.save(model.to("cpu").state_dict(),out/f"{stage}_seed{seed}.pt");write_csv(out/f"{stage}_competence_raw.csv",raw);write_csv(out/f"{stage}_layer_raw.csv",layers)
    write_csv(out/f"{stage}_training_loss.csv",losses);groups=defaultdict(list)
    for row in raw:groups[(row["model_seed"],row["label_mode"],row["predictive_order"],row["position_mode"])].append(row)
    cells=[{"model_seed":k[0],"label_mode":k[1],"predictive_order":k[2],"position_mode":k[3],"accuracy":float(np.mean([r["top1_correct"] for r in v])),"mean_margin":float(np.mean([r["target_margin"] for r in v])),"n":len(v)} for k,v in groups.items()];write_csv(out/f"{stage}_competence.csv",cells);minimum=min(r["accuracy"] for r in cells);decision={"schema_version":f"paper06.{stage}.gate.v1","gate_passed":minimum>=config["competence_threshold"],"mean_accuracy":float(np.mean([r["accuracy"] for r in cells])),"minimum_cell_accuracy":minimum,"threshold":config["competence_threshold"],"artifact_hash":stable_hash({"cells":cells,"layers":layers})};atomic_write_json(out/f"{stage}_gate.json",decision);print(json.dumps(decision,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=("s2","s3"),required=True);p.add_argument("--config",default="configs/paper06/semantic_v2.json");p.add_argument("--output",default="docs/papers/paper0_6/results/v2");p.add_argument("--device");p.add_argument("--steps",type=int);p.add_argument("--smoke",action="store_true");p.add_argument("--eval-examples",type=int);main(p.parse_args())
