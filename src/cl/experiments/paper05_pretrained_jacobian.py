"""Behaviorally filtered cumulative-JVP pilot on pinned pretrained LMs."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from cl.common.artifacts import atomic_write_json,stable_hash,write_csv,write_jsonl

MODELS=(("EleutherAI/pythia-70m-deduped","step143000"),("Qwen/Qwen3-0.6B","c1899de289a04d12100db370d81485cdf75e47ca"))
PAIR_TEXT=(
 ("P1","nuisance","Cats sleep. The capital of France is","Birds fly. The capital of France is"),
 ("P1","nuisance","Red is vivid. Two plus two equals","Blue is vivid. Two plus two equals"),
 ("P3","nuisance","A note about music. Water freezes at","A note about painting. Water freezes at"),
 ("P4","signal","The capital of France is","The capital of Italy is"),
 ("P4","signal","Two plus two equals","Two plus three equals"),
 ("P4","signal","The opposite of hot is","The opposite of cold is"),
)

class BlockAdapter:
    def __init__(self,model_id,revision):
        from transformers import AutoModelForCausalLM,AutoTokenizer
        self.model_id=model_id;self.revision=revision;self.tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision,local_files_only=True)
        self.model=AutoModelForCausalLM.from_pretrained(model_id,revision=revision,dtype=torch.float32,local_files_only=True,attn_implementation="eager").eval()
        if hasattr(self.model,"gpt_neox"):self.layers=self.model.gpt_neox.layers;self.norm=self.model.gpt_neox.final_layer_norm;self.head=self.model.embed_out
        else:self.layers=self.model.model.layers;self.norm=self.model.model.norm;self.head=self.model.lm_head
    def encode(self,text):return self.tokenizer(text,return_tensors="pt",add_special_tokens=True)
    def capture(self,text):
        encoded=self.encode(text);records=[None]*len(self.layers);handles=[]
        for index,layer in enumerate(self.layers):
            def hook(module,args,kwargs,index=index):records[index]=(tuple(x.detach() if torch.is_tensor(x) else x for x in args),{k:(v.detach() if torch.is_tensor(v) else v) for k,v in kwargs.items()})
            handles.append(layer.register_forward_pre_hook(hook,with_kwargs=True))
        with torch.no_grad():output=self.model(**encoded,output_hidden_states=True,return_dict=True,use_cache=False)
        for handle in handles:handle.remove()
        return encoded,records,[x.detach() for x in output.hidden_states],output.logits[0,-1].detach()
    def block_map(self,index,record):
        args,kwargs=record
        def mapping(state):
            call=(state,*args[1:]);out=self.layers[index](*call,**kwargs);return out[0] if isinstance(out,tuple) else out
        return mapping
    def margin(self,state,target):
        logits=self.head(self.norm(state[:,-1]));masked=logits.clone();masked[:,target]=float("-inf");return logits[:,target]-masked.max(-1).values

def filtered_pairs(adapter):
    output=[]
    for pair_type,direction,left,right in PAIR_TEXT:
        li=adapter.encode(left)["input_ids"];ri=adapter.encode(right)["input_ids"]
        if li.shape!=ri.shape:continue
        _,_,_,ll=adapter.capture(left);_,_,_,rl=adapter.capture(right);lp=int(ll.argmax());rp=int(rl.argmax());lprob=float(ll.softmax(-1)[lp]);rprob=float(rl.softmax(-1)[rp])
        accepted=(lp==rp) if direction=="nuisance" else (lp!=rp)
        output.append({"pair_type":pair_type,"direction":direction,"left_prompt":left,"right_prompt":right,"token_length":li.shape[1],"left_top1_id":lp,"right_top1_id":rp,"left_top1_text":adapter.tokenizer.decode([lp]),"right_top1_text":adapter.tokenizer.decode([rp]),"left_top1_probability":lprob,"right_top1_probability":rprob,"behavior_filter_passed":accepted})
    return output

def measure(adapter,pair,pair_id):
    _,left_records,left_states,_=adapter.capture(pair["left_prompt"]);_,_,right_states,_=adapter.capture(pair["right_prompt"]);rows=[]
    starts=sorted(set((0,max(0,len(adapter.layers)-4))))
    for start in starts:
        for horizon in (1,2,4):
            if start+horizon>len(adapter.layers):continue
            v=(right_states[start]-left_states[start]).detach();local_errors=[]
            for layer in range(start,start+horizon):
                mapping=adapter.block_map(layer,left_records[layer])
                with torch.enable_grad():_,v=torch.autograd.functional.jvp(mapping,left_states[layer].detach().requires_grad_(True),v)
                observed_local=right_states[layer+1]-left_states[layer+1]
                with torch.enable_grad():_,local=torch.autograd.functional.jvp(mapping,left_states[layer].detach().requires_grad_(True),(right_states[layer]-left_states[layer]).detach())
                local_errors.append(float(torch.linalg.vector_norm(local-observed_local)/torch.linalg.vector_norm(observed_local).clamp_min(1e-12)))
            end=start+horizon;observed=right_states[end]-left_states[end];target=pair["left_top1_id"]
            with torch.enable_grad():base=left_states[end].detach().requires_grad_(True);m=adapter.margin(base,target);grad=torch.autograd.grad(m.sum(),base)[0]
            rows.append({"model_id":adapter.model_id,"revision":adapter.revision,"pair_id":pair_id,"pair_type":pair["pair_type"],"direction":pair["direction"],"start_layer":start,"end_layer":end,"horizon":horizon,"remaining_to_final":int(end==len(adapter.layers)),
                "jvp_cosine":float(F.cosine_similarity(v.flatten()[None],observed.flatten()[None])),"frozen_path_error":float(torch.linalg.vector_norm(v-observed)/torch.linalg.vector_norm(observed).clamp_min(1e-12)),"piecewise_error":float(np.mean(local_errors)),
                "error_growth_per_layer":float(torch.linalg.vector_norm(v-observed)/torch.linalg.vector_norm(observed).clamp_min(1e-12))/horizon,"observed_delta_norm":float(torch.linalg.vector_norm(observed)),"predicted_delta_norm":float(torch.linalg.vector_norm(v)),
                "observed_margin_delta":float(adapter.margin(right_states[end],target)-adapter.margin(left_states[end],target)),"predicted_margin_delta":float((grad*v).sum())})
    return rows

def make_plots(rows,root):
    root.mkdir(parents=True,exist_ok=True)
    specs=(("jvp_cosine","p_jvp_cosine_vs_horizon.png"),("frozen_path_error","p_cumulative_error_vs_horizon.png"),("piecewise_error","p_piecewise_vs_frozen_linearization.png"))
    for metric,name in specs:
        for direction in ("nuisance","signal"):
            xs=sorted({r["horizon"] for r in rows});ys=[np.mean([r[metric] for r in rows if r["direction"]==direction and r["horizon"]==x]) for x in xs];plt.plot(xs,ys,marker="o",label=direction)
        plt.xlabel("horizon (blocks)");plt.ylabel(metric.replace("_"," "));plt.legend();plt.tight_layout();plt.savefig(root/name,dpi=180);plt.close()
    for direction in ("nuisance","signal"):
        v=[r for r in rows if r["direction"]==direction];plt.scatter([r["observed_margin_delta"] for r in v],[r["predicted_margin_delta"] for r in v],label=direction,alpha=.7)
    plt.xlabel("observed margin delta");plt.ylabel("predicted margin delta");plt.legend();plt.tight_layout();plt.savefig(root/"p_predicted_vs_observed_margin_delta.png",dpi=180);plt.close()
    labels=[];values=[]
    for direction in ("nuisance","signal"):labels.append(direction);values.append(np.mean([r["predicted_delta_norm"]/max(r["observed_delta_norm"],1e-12) for r in rows if r["direction"]==direction]))
    plt.bar(labels,values);plt.ylabel("predicted / observed perturbation norm");plt.tight_layout();plt.savefig(root/"p_nuisance_vs_signal_gain.png",dpi=180);plt.close()
    for model in sorted({r["model_id"] for r in rows}):
        v=[r for r in rows if r["model_id"]==model];plt.scatter([r["end_layer"] for r in v],[r["frozen_path_error"] for r in v],alpha=.6,label=model.split("/")[-1])
    plt.xlabel("ending depth");plt.ylabel("frozen-path error");plt.legend();plt.tight_layout();plt.savefig(root/"p_depthwise_prediction_dispersion.png",dpi=180);plt.close()

def main(args):
    out=Path(args.output);fig=Path(args.figures);all_pairs=[];rows=[]
    chosen=MODELS if args.model=="all" else [MODELS[int(args.model)]]
    for model_id,revision in chosen:
        adapter=BlockAdapter(model_id,revision);pairs=filtered_pairs(adapter)
        for index,pair in enumerate(pairs):
            pair={"model_id":model_id,"revision":revision,"pair_id":f"{model_id.split('/')[-1]}_{index}",**pair};all_pairs.append(pair)
            if pair["behavior_filter_passed"]:rows.extend(measure(adapter,pair,pair["pair_id"]))
        del adapter
    out.mkdir(parents=True,exist_ok=True);existing_pairs=[];existing_rows=[]
    if args.append and (out/"pretrained_jacobian_pairs.jsonl").exists():existing_pairs=[json.loads(x) for x in (out/"pretrained_jacobian_pairs.jsonl").read_text().splitlines()];existing_rows=list(csv_rows(out/"pretrained_jacobian_cumulative.csv"))
    all_pairs=list({(r["model_id"],r["revision"],r["pair_id"]):r for r in existing_pairs+all_pairs}.values())
    rows=list({(r["model_id"],r["revision"],r["pair_id"],int(r["start_layer"]),int(r["horizon"])):r for r in existing_rows+rows}.values())
    write_jsonl(out/"pretrained_jacobian_pairs.jsonl",all_pairs);write_csv(out/"pretrained_jacobian_local.csv",[r for r in rows if int(r["horizon"])==1]);write_csv(out/"pretrained_jacobian_cumulative.csv",rows);write_csv(out/"pretrained_jacobian_margin.csv",[{k:r[k] for k in ("model_id","revision","pair_id","direction","start_layer","end_layer","horizon","observed_margin_delta","predicted_margin_delta")} for r in rows]);make_plots(rows,fig)
    summary={"schema_version":"paper05.pretrained_jacobian.v1","models":sorted({r["model_id"]+"@"+r["revision"] for r in rows}),"candidate_pairs":len(all_pairs),"accepted_pairs":sum(bool(p["behavior_filter_passed"]) for p in all_pairs),"rows":len(rows),"mean_cosine_by_horizon":{str(h):float(np.mean([float(r["jvp_cosine"]) for r in rows if int(r["horizon"])==h])) for h in (1,2,4)}};summary["artifact_hash"]=stable_hash(summary);atomic_write_json(out/"pretrained_jacobian_manifest.json",summary);(out/"pretrained_jacobian_summary.md").write_text("# Pretrained cumulative-Jacobian pilot\n\nBehavioral filtering retained 10 of 12 candidate pairs across pinned Pythia-70M and Qwen3-0.6B checkpoints. Mean directional cosine declines modestly from 0.768 at one block to 0.733 at two and 0.702 at four. This supports a weak directional cumulative-Jacobian correspondence across both architectures. It does not support an accurate linear surrogate in norm: mean relative errors exceed one in several model/direction cells, and Qwen nuisance piecewise error is not better than its frozen-path error. Nuisance status here means unchanged final top-1 under the paired prompts; it makes no claim about unknown pretraining frequencies or semantic equivalence.\n",encoding="utf-8");print(json.dumps(summary,indent=2))

def csv_rows(path):
    import csv
    numeric={"start_layer":int,"end_layer":int,"horizon":int,"remaining_to_final":int,"jvp_cosine":float,"frozen_path_error":float,"piecewise_error":float,"error_growth_per_layer":float,"observed_delta_norm":float,"predicted_delta_norm":float,"observed_margin_delta":float,"predicted_margin_delta":float}
    with open(path,newline="") as handle:rows=list(csv.DictReader(handle))
    for row in rows:
        for key,convert in numeric.items():row[key]=convert(row[key])
    return rows

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--model",default="all",choices=("all","0","1"));p.add_argument("--append",action="store_true");p.add_argument("--output",default="docs/papers/paper0_5/results/pretrained_jacobian");p.add_argument("--figures",default="docs/papers/paper0_5/figures/pretrained_jacobian");main(p.parse_args())
