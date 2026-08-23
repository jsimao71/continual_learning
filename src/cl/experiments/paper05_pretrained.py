"""Pretrained replication, variance factorial, and causal mediation for Paper 0.5."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cl.analysis.equivalence import distribution_metrics, jensen_shannon_bits
from cl.common.artifacts import RunMetadata, atomic_write_json, stable_hash, write_csv, write_jsonl
from cl.common.metrics import bootstrap_ci


MODELS=(
    ("EleutherAI/pythia-70m-deduped","step0"),
    ("EleutherAI/pythia-70m-deduped","step1000"),
    ("EleutherAI/pythia-70m-deduped","step143000"),
    ("Qwen/Qwen3-0.6B","c1899de289a04d12100db370d81485cdf75e47ca"),
)
SEMANTICS={"number":("one","two"),"color":("red","blue")}
SYNTAXES=("alternation","mapping")


def probes():
    rows=[]
    nuisances=("", "Weather is mild. ", "A quiet library closes late. Music plays softly. ")
    for semantic,(left,right) in SEMANTICS.items():
        for syntax in SYNTAXES:
            for evidence in (1,2,4):
                for noise,text in enumerate(nuisances):
                    for realization in range(4):
                        prefix=(text+f"Note {realization}. ") if noise else ""
                        if syntax=="alternation": body=" ".join([left,right]*evidence+[left])
                        else: body="; ".join([f"{left} maps to {right}"]*evidence)+f"; {left} maps to"
                        rows.append({"family":f"{syntax}:{semantic}","syntax":syntax,"semantic":semantic,"evidence":evidence,
                                     "noise":noise*2,"realization":realization,"prompt":prefix+body,"target_text":" "+right})
    return rows


class Adapter:
    def __init__(self,model_id,revision,device):
        from transformers import AutoModelForCausalLM,AutoTokenizer
        self.tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision)
        self.model=AutoModelForCausalLM.from_pretrained(model_id,revision=revision,dtype=torch.float32).to(device).eval()
        self.device=device; self.model_id=model_id; self.revision=revision
        if hasattr(self.model,"gpt_neox"):
            self.layers=self.model.gpt_neox.layers; self.norm=self.model.gpt_neox.final_layer_norm; self.head=self.model.embed_out
        else:
            self.layers=self.model.model.layers; self.norm=self.model.model.norm; self.head=self.model.lm_head
        self.selected=sorted(set((0,len(self.layers)//2,len(self.layers)-1)))

    def target_id(self,text):
        ids=self.tokenizer(text,add_special_tokens=False).input_ids
        if len(ids)!=1: raise ValueError(f"target {text!r} is not one token for {self.model_id}: {ids}")
        return ids[0]

    def _module(self,layer,component):
        if component=="ff": return self.layers[layer].mlp
        return self.layers[layer].self_attn if hasattr(self.layers[layer],"self_attn") else self.layers[layer].attention

    @staticmethod
    def _tensor(output): return output[0] if isinstance(output,tuple) else output

    @staticmethod
    def _replace_output(output,tensor): return (tensor,*output[1:]) if isinstance(output,tuple) else tensor

    @contextmanager
    def intervention(self,layer,component,mode="none",vector=None):
        if mode=="none": yield; return
        def hook(module,args,output):
            value=self._tensor(output); changed=value.clone(); final=changed[:,-1]
            if mode=="zero": final.zero_()
            elif mode=="replace": final.copy_(vector.to(final))
            elif mode=="add":
                direction=vector.to(final); direction=direction/direction.norm().clamp_min(1e-12); final.add_(direction*final.norm(dim=-1,keepdim=True)*.5)
            elif mode=="project_remove":
                direction=vector.to(final); direction=direction/direction.norm().clamp_min(1e-12); final.sub_((final*direction).sum(-1,keepdim=True)*direction)
            else: raise ValueError(mode)
            return self._replace_output(output,changed)
        handle=self._module(layer,component).register_forward_hook(hook)
        try: yield
        finally: handle.remove()

    @torch.no_grad()
    def run(self,prompt,target,intervention=None,capture=False):
        encoded=self.tokenizer(prompt,return_tensors="pt").to(self.device); traces={}
        handles=[]
        if capture:
            for layer in self.selected:
                for component in ("sa","ff"):
                    def save(module,args,output,layer=layer,component=component): traces[(layer,component)]=self._tensor(output)[0,-1].detach().cpu()
                    handles.append(self._module(layer,component).register_forward_hook(save))
        context=self.intervention(**intervention) if intervention else self.intervention(0,"sa","none")
        with context: output=self.model(**encoded,output_hidden_states=True,return_dict=True,use_cache=False)
        for handle in handles: handle.remove()
        final_logits=output.logits[0,-1]; probability=torch.softmax(final_logits,-1); logprob=torch.log_softmax(final_logits,-1)[target]
        states={}
        if capture:
            for layer in self.selected:
                state=output.hidden_states[layer+1][0,-1]; logits=self.head(self.norm(state)); p=torch.softmax(logits,-1)
                states[layer]=(p.detach().cpu(),state.detach().cpu())
        return float(logprob),probability.detach().cpu(),states,traces


def aggregate_observations(rows):
    fields=("model_id","revision","family","syntax","semantic","evidence","noise","layer") ; groups=defaultdict(list)
    for row in rows: groups[tuple(row[f] for f in fields)].append(row)
    output=[]
    for key,values in sorted(groups.items()):
        p=np.asarray([v["probabilities"] for v in values]); centroid=p.mean(0); states=np.asarray([v["state"] for v in values]); covariance=np.cov(states,rowvar=False); eig=np.clip(np.linalg.eigvalsh(np.atleast_2d(covariance)),0,None)
        output.append({**dict(zip(fields,key)),"n":len(values),"mean_target_probability":float(np.mean([v["target_probability"] for v in values])),
                       "target_probability_variance":float(np.var([v["target_probability"] for v in values],ddof=1)),
                       "mean_js_dispersion":float(np.mean([jensen_shannon_bits(x,centroid) for x in p])),
                       "mean_entropy_bits":float(np.mean([v["entropy_bits"] for v in values])),
                       "residual_total_variance":float(np.trace(covariance)),"residual_effective_rank":float(eig.sum()**2/max(np.square(eig).sum(),1e-12)),
                       "centroid":centroid.tolist()})
    matched=defaultdict(list)
    for row in output: matched[(row["model_id"],row["revision"],row["evidence"],row["noise"],row["layer"])].append(row)
    for values in matched.values():
        for row in values:
            between=[jensen_shannon_bits(row["centroid"],other["centroid"]) for other in values if other["family"]!=row["family"]]
            row["between_family_js"]=float(np.mean(between)); row["snr_js_ratio"]=row["between_family_js"]/max(row["mean_js_dispersion"],1e-12)
    for row in output: del row["centroid"]
    return output


def family_inference(summary,causal):
    rows=[]
    for model,revision in sorted({(r["model_id"],r["revision"]) for r in summary}):
        values=[r for r in summary if r["model_id"]==model and r["revision"]==revision and r["evidence"]==2 and r["noise"]==4]
        first=min(r["layer"] for r in values); last=max(r["layer"] for r in values)
        by_family=defaultdict(dict)
        for r in values: by_family[r["family"]][r["layer"]]=r
        deltas=[v[last]["mean_js_dispersion"]-v[first]["mean_js_dispersion"] for v in by_family.values() if first in v and last in v]
        estimate,low,high=bootstrap_ci(deltas,samples=2000,seed=41)
        rows.append({"model_id":model,"revision":revision,"contrast":"last_minus_first_js_dispersion","n_families":len(deltas),"estimate":estimate,"ci_low":low,"ci_high":high})
    for model,revision in sorted({(r["model_id"],r["revision"]) for r in causal}):
        values=[r for r in causal if r["model_id"]==model and r["revision"]==revision]
        for mode in ("replace_equivalent","replace_syntax_mismatch","replace_semantic_mismatch","replace_nonequivalent","project_remove","add"):
            unit=defaultdict(list)
            for r in values:
                if r["mode"]==mode: unit[r["family"]].append(r["target_logprob_change"])
            family=[float(np.mean(v)) for v in unit.values()]
            if family:
                estimate,low,high=bootstrap_ci(family,samples=2000,seed=43)
                rows.append({"model_id":model,"revision":revision,"contrast":mode+"_target_logprob_change","n_families":len(family),"estimate":estimate,"ci_low":low,"ci_high":high})
    return rows


def plots(output,summary,causal):
    figures=output/"figures"; figures.mkdir(parents=True,exist_ok=True)
    for metric,name in (("mean_js_dispersion","pretrained_js_depth.png"),("snr_js_ratio","pretrained_snr_depth.png")):
        fig,ax=plt.subplots(figsize=(8,4.8))
        for model,revision in sorted({(r["model_id"],r["revision"]) for r in summary}):
            selected=[r for r in summary if r["model_id"]==model and r["revision"]==revision and r["evidence"]==2 and r["noise"]==4]; by=defaultdict(list)
            for r in selected: by[r["layer"]].append(r[metric])
            ax.plot(sorted(by),[np.median(by[x]) for x in sorted(by)],marker="o",label=f"{model.split('/')[-1]}@{revision[:10]}")
        ax.set_xlabel("selected depth"); ax.set_ylabel(metric); ax.legend(fontsize=7); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figures/name,dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.8)); grouped=defaultdict(list)
    for r in causal: grouped[(r["mode"],r["component"])].append(r["target_logprob_change"])
    labels=[f"{k[0]}/{k[1]}" for k in sorted(grouped)]; ax.bar(range(len(labels)),[np.mean(grouped[k]) for k in sorted(grouped)]); ax.set_xticks(range(len(labels)),labels,rotation=35,ha="right"); ax.axhline(0,color="black",lw=.7); ax.set_ylabel("target log-probability change"); fig.tight_layout(); fig.savefig(figures/"pretrained_causal_mediation.png",dpi=170); plt.close(fig)


def run(args):
    repo=Path(args.repo).resolve(); output=Path(args.output).resolve(); raw=output/"raw"; tables=output/"tables"; raw.mkdir(parents=True,exist_ok=True); tables.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device); observations=[]; causal=[]; metadata=[]
    all_probes=probes()
    for model_id,revision in MODELS:
        adapter=Adapter(model_id,revision,device); run_id=f"{model_id.split('/')[-1]}-{revision[:10]}"
        metadata.append(RunMetadata.capture(repo=repo,run_id=run_id,config={"models":MODELS,"probe_schema":"paper05.pretrained.crossed.v1"},model_id=model_id+"@"+revision,dataset_id="controlled-text-patterns-v1",seed=0,device=str(device),dtype="float32",data_hash=stable_hash(all_probes)).as_dict())
        captures=[]
        for probe in all_probes:
            target=adapter.target_id(probe["target_text"]); _,_,states,traces=adapter.run(probe["prompt"],target,capture=True)
            for layer,(probability,state) in states.items():
                metrics=distribution_metrics(probability.numpy(),target)
                observations.append({"model_id":model_id,"revision":revision,**probe,"layer":layer,"target_id":target,
                                     "probabilities":probability.numpy().astype(np.float32,copy=False),
                                     "state":state.numpy().astype(np.float32,copy=False),**metrics})
            if (probe["evidence"]==1 and probe["noise"]==0 and probe["realization"]<2) or (probe["evidence"]==2 and probe["noise"]==4 and probe["realization"]>=2): captures.append((probe,target,traces))
        # Identity/context/prefix-disjoint directions: weak clean realization 0/1 fit; stronger noisy 2/3 test.
        directions={}; donors={}
        for layer in adapter.selected:
            for component in ("sa","ff"):
                for family in sorted({p["family"] for p,_,_ in captures}):
                    train=[t[(layer,component)].numpy() for p,_,t in captures if p["family"]==family and p["evidence"]==1 and p["noise"]==0]
                    directions[(layer,component,family)]=torch.tensor(np.mean(train,axis=0))
                    donors[(layer,component,family)]=torch.tensor(train[0])
        for probe,target,traces in [x for x in captures if x[0]["evidence"]==2 and x[0]["noise"]==4]:
            base,_,_,_=adapter.run(probe["prompt"],target)
            same_semantic=next(f for f in sorted(donors_family for donors_family in {p["family"] for p,_,_ in captures}) if f!=probe["family"] and f.endswith(":"+probe["semantic"]))
            same_syntax=next(f for f in sorted({p["family"] for p,_,_ in captures}) if f!=probe["family"] and f.startswith(probe["syntax"]+":"))
            both_different=next(f for f in sorted({p["family"] for p,_,_ in captures}) if not f.startswith(probe["syntax"]+":") and not f.endswith(":"+probe["semantic"]))
            for layer in adapter.selected:
                for component in ("sa","ff"):
                    tests=(("zero",None),("replace_equivalent",donors[(layer,component,probe["family"])]),
                           ("replace_syntax_mismatch",donors[(layer,component,same_semantic)]),("replace_semantic_mismatch",donors[(layer,component,same_syntax)]),
                           ("replace_nonequivalent",donors[(layer,component,both_different)]),("project_remove",directions[(layer,component,probe["family"])]),("add",directions[(layer,component,probe["family"])]))
                    for label,vector in tests:
                        mode="replace" if label.startswith("replace") else label
                        changed,_,_,_=adapter.run(probe["prompt"],target,{"layer":layer,"component":component,"mode":mode,"vector":vector})
                        causal.append({"model_id":model_id,"revision":revision,"family":probe["family"],"syntax":probe["syntax"],"semantic":probe["semantic"],"realization":probe["realization"],"layer":layer,"component":component,"mode":label,"target_logprob_change":changed-base})
        del adapter.model; torch.mps.empty_cache() if hasattr(torch,"mps") and device.type=="mps" else None
    summary=aggregate_observations(observations); inference=family_inference(summary,causal)
    compact_observations=[{k:v for k,v in row.items() if k not in {"probabilities","state"}} | {
        "distribution_hash":stable_hash(np.asarray(row["probabilities"],dtype=np.float16).tobytes().hex()),
        "state_hash":stable_hash(np.asarray(row["state"],dtype=np.float16).tobytes().hex())} for row in observations]
    write_jsonl(raw/"pretrained_observations.jsonl",compact_observations); write_jsonl(raw/"pretrained_causal.jsonl",causal); write_jsonl(raw/"metadata.jsonl",metadata)
    write_csv(tables/"pretrained_variance.csv",summary); write_csv(tables/"pretrained_causal_mediation.csv",causal); write_csv(tables/"family_bootstrap_inference.csv",inference)
    plots(output,summary,causal); atomic_write_json(output/"manifest.json",{"schema_version":"paper05.pretrained.v1","models":MODELS,"identity_disjoint_direction_fit":True,"artifact_hash":stable_hash({"summary":summary,"causal":causal,"inference":inference})})
    print(json.dumps({"observations":len(observations),"causal":len(causal),"summary":len(summary)},indent=2))


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--repo",default="."); p.add_argument("--output",default="docs/papers/paper0_5/results/pretrained"); p.add_argument("--device",default="mps" if torch.backends.mps.is_available() else "cpu"); return p.parse_args()


if __name__=="__main__": run(parse_args())
