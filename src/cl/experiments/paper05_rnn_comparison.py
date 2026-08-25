"""Matched recurrent/Transformer sensitivity comparison on controlled axes."""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from cl.common.artifacts import atomic_write_json, stable_hash, write_csv
from cl.common.model_adapter import TinyTransformerLM
from cl.experiments.paper05_predictive_order_phase import TARGETS, VALUE_ROLES, resolve_device, target_rule

AXIS_MARKER={"length":100,"span":101,"predictive_order":102,"nuisance":103}
QUERY,NEUTRAL=89,90


def make_example(config: dict, axis: str, value: int, index: int, split: str) -> dict:
    seed=config["dataset_seed"]+(0 if split=="train" else 30_000_000)+index*1613+value*37+AXIS_MARKER[axis]
    rng=random.Random(seed)
    order=value if axis=="predictive_order" else 2
    raw_length=value if axis=="length" else order
    span=value if axis=="span" else order-1
    nuisance=value if axis=="nuisance" else 0
    values=[rng.randrange(4) for _ in range(order)];target=target_rule(values)
    if axis=="span":positions=[0,span]
    else:positions=list(range(order))
    pattern=[NEUTRAL]*(positions[-1]+1)
    for role,(position,item) in enumerate(zip(positions,values)):pattern[position]=VALUE_ROLES[role][item]
    pattern.extend([NEUTRAL]*max(0,raw_length-len(pattern)))
    noise=[rng.choice(sum((list(role) for role in VALUE_ROLES),[])) for _ in range(nuisance)]
    body=[AXIS_MARKER[axis],48+order,*pattern];padding=config["sequence_length"]-len(noise)-len(body)-1
    if padding<0:raise ValueError((axis,value,len(body),nuisance))
    tokens=[NEUTRAL]*padding+noise+body+[QUERY]
    return {"tokens":tokens,"target":TARGETS[target],"axis":axis,"axis_value":value,"predictive_order":order,
            "raw_length":raw_length,"dependency_span":span,"nuisance_count":nuisance,"family_id":f"{axis}:{value}:{tuple(values)}:{index%16}"}


def axis_values(config: dict) -> dict[str,list[int]]:
    return {"length":config["lengths"],"span":config["spans"],"predictive_order":config["predictive_orders"],"nuisance":config["nuisance_counts"]}


def evaluation(config: dict, examples: int | None = None) -> list[dict]:
    count=examples or config["evaluation_examples_per_cell"]
    return [make_example(config,axis,value,i,"test") for axis,values in axis_values(config).items() for value in values for i in range(count)]


def training_batch(config: dict,rng:random.Random,device:torch.device):
    axes=axis_values(config);rows=[]
    for _ in range(config["batch_size"]):
        axis=rng.choice(list(axes));value=rng.choice(axes[axis]);rows.append(make_example(config,axis,value,rng.randrange(10_000_000),"train"))
    return torch.tensor([r["tokens"] for r in rows],device=device),torch.tensor([r["target"] for r in rows],device=device)


class RecurrentLM(nn.Module):
    def __init__(self,vocab_size:int,hidden_size:int,kind:str):
        super().__init__();self.vocab_size=vocab_size;self.embedding=nn.Embedding(vocab_size,hidden_size)
        cls={"rnn":nn.RNN,"gru":nn.GRU,"lstm":nn.LSTM}[kind];self.recurrent=cls(hidden_size,hidden_size,batch_first=True)
        self.norm=nn.LayerNorm(hidden_size);self.head=nn.Linear(hidden_size,vocab_size,bias=False);self.head.weight=self.embedding.weight

    def forward(self,input_ids:torch.Tensor):
        hidden,_=self.recurrent(self.embedding(input_ids));return self.head(self.norm(hidden)),hidden


def build_model(config:dict,kind:str,device:torch.device):
    if kind=="transformer":return TinyTransformerLM(config["vocab_size"],config["sequence_length"],config["transformer_width"],config["transformer_depth"],config["transformer_heads"]).to(device)
    return RecurrentLM(config["vocab_size"],config["hidden_size"],kind).to(device)


def parameter_count(model:nn.Module)->int:return sum(p.numel() for p in model.parameters())


def train(config:dict,kind:str,device:torch.device,steps:int):
    seed=config["model_seed"]+sum(map(ord,kind));torch.manual_seed(seed);rng=random.Random(seed);model=build_model(config,kind,device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"]);loss=[]
    for step in range(steps):
        x,y=training_batch(config,rng,device);optimizer.zero_grad(set_to_none=True)
        output=model(x);logits=output[0];value=F.cross_entropy(logits[:,-1],y);value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        if step in (0,steps-1) or (step+1)%100==0:loss.append({"model_type":kind,"step":step+1,"loss":float(value.detach().cpu())})
    return model.eval(),loss


@torch.no_grad()
def evaluate(model:nn.Module,kind:str,rows:list[dict],batch_size:int=128):
    device=next(model.parameters()).device;results=[];internal=[]
    for start in range(0,len(rows),batch_size):
        batch=rows[start:start+batch_size];x=torch.tensor([r["tokens"] for r in batch],device=device);y=torch.tensor([r["target"] for r in batch],device=device)
        if kind=="transformer":
            logits,trace=model(x,capture=True);states=[trace.layers[0].pre_sa]+[layer.post_block for layer in trace.layers]
            trajectory=[model.diagnostic_logits(state[:,-1]) for state in states];hidden=None
        else:
            logits,hidden=model(x);sample_times=sorted(set([0,1,3,7,15,31,63,x.shape[1]-1]));trajectory=[logits[:,t] for t in sample_times]
        chosen=logits[:,-1].gather(1,y[:,None]).squeeze(1);other=logits[:,-1].clone();other.scatter_(1,y[:,None],float("-inf"));margin=chosen-other.max(1).values
        for i,r in enumerate(batch):
            results.append({"model_type":kind,**{k:r[k] for k in ("axis","axis_value","predictive_order","raw_length","dependency_span","nuisance_count","family_id")},
                            "top1_correct":int(logits[i,-1].argmax()==y[i]),"target_margin":float(margin[i])})
            if i<16:
                for j,z in enumerate(trajectory):
                    c=z[i,y[i]];o=z[i].clone();o[y[i]]=float("-inf")
                    internal.append({"model_type":kind,"axis":r["axis"],"axis_value":r["axis_value"],"family_id":r["family_id"],
                                     "computation_index":j if kind=="transformer" else sample_times[j],"target_margin":float(c-o.max()),
                                     "top1_correct":int(z[i].argmax()==y[i]),"hidden_norm":float(hidden[i,sample_times[j]].norm()) if hidden is not None else ""})
    return results,internal


def main(args)->None:
    config=json.loads(Path(args.config).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if args.smoke:config={**config,"training_steps":4,"batch_size":8}
    if args.steps:config={**config,"training_steps":args.steps}
    device=resolve_device(args.device or config["device"]);rows=evaluation(config,args.eval_examples)
    results=[];internal=[];parameters=[];losses=[]
    kinds=config["model_types"][:args.max_models] if args.max_models else config["model_types"]
    for index,kind in enumerate(kinds,1):
        print(f"[{index}/{len(kinds)}] {kind}",flush=True);model,loss=train(config,kind,device,config["training_steps"]);result,trace=evaluate(model,kind,rows)
        count=parameter_count(model);parameters.append({"model_type":kind,"parameter_count":count,"training_steps":config["training_steps"],"training_tokens":config["training_steps"]*config["batch_size"]*config["sequence_length"]})
        results.extend(result);internal.extend(trace);losses.extend(loss);torch.save(model.to("cpu").state_dict(),out/f"{kind}_checkpoint.pt")
        write_csv(out/"rnn_transformer_results.csv",results);write_csv(out/"rnn_transformer_internal.csv",internal)
    write_csv(out/"rnn_transformer_parameter_match.csv",parameters);write_csv(out/"rnn_transformer_training_loss.csv",losses)
    manifest={"schema_version":"paper05.rnn_comparison.run.v1","device":str(device),"config":config,"models":kinds,"evaluation_examples":len(rows),
              "artifact_hash":stable_hash({"parameters":parameters,"results":results})};atomic_write_json(out/"rnn_transformer_manifest.json",manifest);print(json.dumps(manifest,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/paper05/rnn_comparison.json");p.add_argument("--output",default="docs/papers/paper0_5/results/rnn_comparison");p.add_argument("--device");p.add_argument("--smoke",action="store_true");p.add_argument("--steps",type=int);p.add_argument("--eval-examples",type=int);p.add_argument("--max-models",type=int);main(p.parse_args())
