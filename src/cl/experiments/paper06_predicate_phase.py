"""Run the Paper 0.6 v4 shallow-train/deep-test predicate phase."""
from __future__ import annotations
import argparse,csv,json,random,time
from collections import defaultdict
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from cl.common.artifacts import atomic_write_json,stable_hash,write_csv
from cl.common.model_adapter import TinyTransformerLM
from cl.experiments.paper05_predictive_order_phase import resolve_device
from cl.semantic.predicates import PredicateExample,evaluation_matrix,training_batch,validate

def make_model(config,depth,device):return TinyTransformerLM(config["vocab_size"],config["max_length"],config["width"],depth,config["heads"],position_encoding=config["position_encoding"]).to(device)
def params(model):return sum(p.numel() for p in model.parameters())
def token_tensor(rows,device):
    length=max(len(r.tokens) for r in rows)
    return torch.tensor([[0]*(length-len(r.tokens))+list(r.tokens) for r in rows],device=device)

def train(config,depth,seed,steps,device):
    torch.manual_seed(seed+depth*1009);rng=random.Random(seed+depth*701);model=make_model(config,depth,device);opt=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"]);loss=[];started=time.time()
    for step in range(steps):
        rows=training_batch(config,rng,seed);x=token_tensor(rows,device);y=torch.tensor([r.target for r in rows],device=device);opt.zero_grad(set_to_none=True);z,_=model(x);value=F.cross_entropy(z[:,-1],y);value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step==0 or (step+1)%100==0 or step+1==steps:
            loss.append({"model_depth":depth,"model_seed":seed,"step":step+1,"loss":float(value.detach().cpu()),"elapsed_seconds":time.time()-started});print(f"predicate L={depth} seed={seed} step={step+1}/{steps} loss={loss[-1]['loss']:.4f}",flush=True)
    return model.eval(),loss

def axis(row:PredicateExample)->str:
    marker=int(row.example_id.rsplit(":i",1)[1]);return "distractors" if marker>=20_000 else "branching" if marker>=10_000 else "depth_path"

@torch.no_grad()
def evaluate(model,rows,depth,seed,batch_size=128):
    device=next(model.parameters()).device;raw=[];by_length=defaultdict(list)
    for row in rows:by_length[len(row.tokens)].append(row)
    for length,bucket in sorted(by_length.items()):
        for start in range(0,len(bucket),batch_size):
            batch=bucket[start:start+batch_size];x=torch.tensor([r.tokens for r in batch],device=device);y=torch.tensor([r.target for r in batch],device=device);z,_=model(x);final=z[:,-1];p=final.softmax(-1);chosen=final.gather(1,y[:,None]).squeeze(1);other=final.clone();other.scatter_(1,y[:,None],float("-inf"));margin=chosen-other.max(1).values
            for i,r in enumerate(batch):raw.append({"model_depth":depth,"model_seed":seed,"tree_seed":r.tree_seed,"example_id":r.example_id,"evaluation_axis":axis(r),"predicate":r.predicate,"total_depth":r.total_depth,"required_path":r.required_path,"branching":r.branching,"distractors":r.distractors,"template_id":r.template_id,"position_mode":r.position_mode,"label_mode":r.label_mode,"sequence_length":length,"top1_correct":int(final[i].argmax()==y[i]),"target_probability":float(p[i,y[i]]),"target_rank":int(1+(final[i]>chosen[i]).sum()),"target_margin":float(margin[i]),"output_entropy":float(-(p[i]*p[i].clamp_min(1e-12).log()).sum())})
    return raw

def aggregate(raw):
    groups=defaultdict(list)
    keys=("model_depth","model_seed","evaluation_axis","predicate","total_depth","required_path","branching","distractors")
    for row in raw:groups[tuple(row[k] for k in keys)].append(row)
    return [{**dict(zip(keys,key)),"accuracy":float(np.mean([int(r["top1_correct"]) for r in values])),"mean_margin":float(np.mean([float(r["target_margin"]) for r in values])),"mean_target_probability":float(np.mean([float(r["target_probability"]) for r in values])),"n":len(values)} for key,values in groups.items()]

def existing(path):
    if not path.exists():return []
    with path.open(newline="") as handle:return list(csv.DictReader(handle))

def main(args):
    config=json.loads(Path(args.config).read_text());out=Path(args.output)/"s1_predicates";out.mkdir(parents=True,exist_ok=True);validation=validate(config);atomic_write_json(out/"s1_predicate_generator_validation.json",validation)
    if not validation["passed"]:raise RuntimeError(validation)
    depths=[int(x) for x in args.depths.split(",")] if args.depths else config["model_depths"];seeds=[int(x) for x in args.seeds.split(",")] if args.seeds else config["model_seeds"]
    steps=args.steps or config["training_steps"]
    if args.smoke:depths=[2];seeds=[11];steps=4;config={**config,"batch_size":8}
    device=resolve_device(args.device or config["device"]);raw_path=out/"s1_predicate_competence_raw.csv";loss_path=out/"s1_predicate_training_loss.csv";raw=[] if args.fresh or args.smoke else existing(raw_path);losses=[] if args.fresh or args.smoke else existing(loss_path);done={(int(r["model_depth"]),int(r["model_seed"])) for r in raw}
    for depth in depths:
        for seed in seeds:
            if (depth,seed) in done and not args.force:print(f"skip completed L={depth} seed={seed}",flush=True);continue
            checkpoint=out/f"predicate_L{depth}_seed{seed}.pt"
            if args.evaluate_only:
                model=make_model(config,depth,device);model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True));model=model.eval();new_loss=[]
            else:model,new_loss=train(config,depth,seed,steps,device);torch.save(model.to("cpu").state_dict(),checkpoint);model=model.to(device)
            rows=evaluation_matrix(config,seed,args.eval_examples);new=evaluate(model,rows,depth,seed);raw=[r for r in raw if (int(r["model_depth"]),int(r["model_seed"]))!=(depth,seed)]+new;losses=[r for r in losses if (int(r["model_depth"]),int(r["model_seed"]))!=(depth,seed)]+new_loss;write_csv(raw_path,raw);write_csv(loss_path,losses);write_csv(out/"s1_predicate_competence.csv",aggregate(raw));print(f"saved L={depth} seed={seed}: {len(new)} held-out rows",flush=True)
    cells=aggregate(raw);depth_cells=[r for r in cells if r["evaluation_axis"]=="depth_path"];by_pred=defaultdict(list)
    for row in depth_cells:by_pred[row["predicate"]].append(row)
    score={predicate:{"mean_accuracy":float(np.mean([r["accuracy"] for r in values])),"minimum_cell_accuracy":float(min(r["accuracy"] for r in values)),"training_depth_accuracy":float(np.mean([r["accuracy"] for r in values if int(r["total_depth"])<=4 and int(r["required_path"])<=3])),"deep_accuracy":float(np.mean([r["accuracy"] for r in values if int(r["total_depth"])>=8 or int(r["required_path"])>=4]))} for predicate,values in by_pred.items()}
    decision={"schema_version":"paper06.predicate_v4.phase.v1","config":config,"trained_cells":[{"model_depth":d,"model_seed":s} for d,s in sorted({(int(r["model_depth"]),int(r["model_seed"])) for r in raw})],"predicate_score":score,"mechanism_eligibility":{p:v["deep_accuracy"]>=config["competence_threshold"] for p,v in score.items()},"artifact_hash":stable_hash(cells)};atomic_write_json(out/"s1_predicate_phase.json",decision);print(json.dumps(decision,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/paper06/predicate_v4.json");p.add_argument("--output",default="docs/papers/paper0_6/results/v4");p.add_argument("--device");p.add_argument("--depths");p.add_argument("--seeds");p.add_argument("--steps",type=int);p.add_argument("--eval-examples",type=int);p.add_argument("--smoke",action="store_true");p.add_argument("--fresh",action="store_true");p.add_argument("--force",action="store_true");p.add_argument("--evaluate-only",action="store_true");main(p.parse_args())
