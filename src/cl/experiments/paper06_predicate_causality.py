"""Competence-gated local causality for Paper 0.6 v4 predicates and S3."""
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np,torch
from cl.common.artifacts import atomic_write_json,write_csv
from cl.common.model_adapter import Intervention,TinyTransformerLM
from cl.experiments.paper05_predictive_order_phase import resolve_device
from cl.semantic.predicates import DIST,PAD,QUERY,evaluation_matrix
from cl.semantic.v2 import rule_evaluation

def read(path):
    with path.open(newline="") as handle:return list(csv.DictReader(handle))
def model(config,depth,device,checkpoint,position="sinusoidal"):
    m=TinyTransformerLM(config["vocab_size"],config.get("max_length",config.get("sequence_length")),config["width"],depth,config["heads"],position_encoding=position);m.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True));return m.to(device).eval()
def tensor(row,device,tokens=None):return torch.tensor([tokens or row.tokens],device=device)
def stats(logits,target):
    p=logits.softmax(-1);chosen=logits[target];other=logits.clone();other[target]=float("-inf")
    return int(logits.argmax()==target),float(chosen-other.max()),p
def js(p,q):
    m=(p+q)/2;return float(.5*((p*(p.clamp_min(1e-12).log()-m.clamp_min(1e-12).log())).sum()+(q*(q.clamp_min(1e-12).log()-m.clamp_min(1e-12).log())).sum()))

def eligible_cells(root,threshold):
    rows=read(root/"s1_predicates/s1_predicate_competence.csv");groups=defaultdict(list)
    for r in rows:
        if r["evaluation_axis"]=="depth_path" and r["predicate"]=="isAncestor" and (int(r["total_depth"])>=8 or int(r["required_path"])>=4):groups[(int(r["model_depth"]),int(r["model_seed"]))].append(float(r["accuracy"]))
    return [(*key,float(np.mean(v)),min(v)) for key,v in groups.items() if min(v)>=threshold]

@torch.no_grad()
def predicate_causality(config,root,device,limit):
    tables=root/"tables";tables.mkdir(parents=True,exist_ok=True);cells=eligible_cells(root,config["competence_threshold"]);masking=[];decoding=[];attention=[];ablation=[];heads=[];replacement=[]
    for depth,seed,deep_mean,worst in cells:
        m=model(config,depth,device,root/"s1_predicates"/f"predicate_L{depth}_seed{seed}.pt");candidates=[r for r in evaluation_matrix(config,seed,4) if r.predicate=="isAncestor" and r.total_depth>=8];rows=[candidates[i] for i in np.linspace(0,len(candidates)-1,min(limit,len(candidates)),dtype=int)]
        for row in rows:
            x=tensor(row,device);base,trace=m(x,capture=True);base_logits=base[0,-1];base_ok,base_margin,base_p=stats(base_logits,row.target)
            states=[trace.layers[0].pre_sa]
            for block in trace.layers:states.extend((block.post_sa,block.post_block))
            for boundary,state in enumerate(states):
                z=m.diagnostic_logits(state[0,-1]);probs=z.softmax(-1);path_probs=[float(probs[node]) for node in row.node_path];decoding.append({"model_depth":depth,"model_seed":seed,"example_id":row.example_id,"total_depth":row.total_depth,"required_path":row.required_path,"boundary":boundary,"target_margin":stats(z,row.target)[1],"top_path_hop":int(np.argmax(path_probs)),"path_probabilities":";".join(f"{p:.7g}" for p in path_probs)})
            for layer,block in enumerate(trace.layers):
                weights=block.attention[0,:,-1].mean(0);masses=[float(weights[position]) for position in row.path_positions];attention.append({"model_depth":depth,"model_seed":seed,"example_id":row.example_id,"total_depth":row.total_depth,"required_path":row.required_path,"layer":layer,"preferred_hop":int(np.argmax(masses)),"preferred_mass":max(masses),"path_attention_mass":sum(masses),"off_path_mass":float(1-sum(masses))})
            positions={"parent":[row.path_positions[1]],"intermediate_candidate":[row.path_positions[min(row.required_path,row.total_depth)]],"root":[row.path_positions[-1]]}
            try:dist_start=row.tokens.index(DIST)+1;query_start=row.tokens.index(QUERY);positions["off_path_distractors"]=list(range(dist_start,query_start));positions["sibling_branch"]=positions["off_path_distractors"][:max(1,len(positions["off_path_distractors"])//2)]
            except ValueError:positions["off_path_distractors"]=[];positions["sibling_branch"]=[]
            for condition,pos in positions.items():
                changed=list(row.tokens)
                for index in pos:changed[index]=PAD
                z,_=m(tensor(row,device,tuple(changed)));ok,margin,p=stats(z[0,-1],row.target);masking.append({"model_depth":depth,"model_seed":seed,"example_id":row.example_id,"condition":condition,"total_depth":row.total_depth,"required_path":row.required_path,"baseline_correct":base_ok,"changed_correct":ok,"accuracy_delta":ok-base_ok,"margin_delta":margin-base_margin,"js_damage":js(base_p,p)})
            for layer in range(depth):
                for component in ("sa","ff"):
                    z,_=m(x,intervention=Intervention(layer,component,"zero"));ok,margin,p=stats(z[0,-1],row.target);ablation.append({"stage":"s1","model_depth":depth,"model_seed":seed,"example_id":row.example_id,"predicate":"isAncestor","layer":layer,"component":component,"mode":"zero","accuracy_delta":ok-base_ok,"margin_delta":margin-base_margin,"js_damage":js(base_p,p)})
                for head in range(config["heads"]):
                    z,_=m(x,intervention=Intervention(layer,"sa","head_zero",head=head));ok,margin,p=stats(z[0,-1],row.target);heads.append({"stage":"s1","model_depth":depth,"model_seed":seed,"example_id":row.example_id,"predicate":"isAncestor","required_path":row.required_path,"layer":layer,"head":head,"mode":"zero","accuracy_delta":ok-base_ok,"margin_delta":margin-base_margin,"js_damage":js(base_p,p)})
        # Mean replacement on equal-length batches.
        buckets=defaultdict(list)
        for r in rows:buckets[len(r.tokens)].append(r)
        for batch in [v for v in buckets.values() if len(v)>=2]:
            x=torch.tensor([r.tokens for r in batch],device=device);base,_=m(x)
            for layer in range(depth):
                for component in ("sa","ff"):
                    z,_=m(x,intervention=Intervention(layer,component,"mean"))
                    for i,r in enumerate(batch):
                        b_ok,b_margin,bp=stats(base[i,-1],r.target);ok,margin,p=stats(z[i,-1],r.target);ablation.append({"stage":"s1","model_depth":depth,"model_seed":seed,"example_id":r.example_id,"predicate":"isAncestor","layer":layer,"component":component,"mode":"mean","accuracy_delta":ok-b_ok,"margin_delta":margin-b_margin,"js_damage":js(bp,p)})
        # Role-matched cross-tree update/head replacement on equal-length rows.
        pairs=[]
        for recipient in rows:
            donor=next((r for r in candidates if len(r.tokens)==len(recipient.tokens) and r.tree_seed!=recipient.tree_seed and r.total_depth==recipient.total_depth and r.required_path==recipient.required_path),None)
            if donor is not None:pairs.append((recipient,donor))
        for recipient,donor in pairs[:max(2,limit//2)]:
            rx=tensor(recipient,device);dx=tensor(donor,device);base,_=m(rx);base_ok,base_margin,base_p=stats(base[0,-1],recipient.target);_,dt=m(dx,capture=True)
            for layer in range(depth):
                for component,update in (("sa",dt.layers[layer].delta_sa),("ff",dt.layers[layer].delta_ff)):
                    z,_=m(rx,intervention=Intervention(layer,component,"replace",update));ok,margin,p=stats(z[0,-1],recipient.target);replacement.append({"stage":"s1","model_depth":depth,"model_seed":seed,"recipient":recipient.example_id,"donor":donor.example_id,"donor_type":"same_predicate_cross_tree_same_role","layer":layer,"component":component,"accuracy_delta":ok-base_ok,"margin_damage":base_margin-margin,"js_damage":js(base_p,p)})
                for head in range(config["heads"]):
                    update=dt.layers[layer].head_outputs[:,head];z,_=m(rx,intervention=Intervention(layer,"sa","head_replace",update,head));ok,margin,p=stats(z[0,-1],recipient.target);heads.append({"stage":"s1","model_depth":depth,"model_seed":seed,"example_id":recipient.example_id,"predicate":"isAncestor","required_path":recipient.required_path,"layer":layer,"head":head,"mode":"role_matched_replace","accuracy_delta":ok-base_ok,"margin_delta":margin-base_margin,"js_damage":js(base_p,p)})
    blocked=[{"predicate":p,"status":"blocked_by_competence"} for p in ("parent","grandparent","ancestor_k","root")]
    write_csv(tables/"s1_intermediate_node_decoding.csv",decoding);write_csv(tables/"s1_attention_path.csv",attention);write_csv(tables/"s1_path_masking.csv",masking);write_csv(tables/"s1_causal_replacement.csv",replacement+blocked);write_csv(tables/"s1_sa_ff_ablation.csv",ablation+blocked);write_csv(tables/"s1_head_utility.csv",heads+blocked)
    return cells

@torch.no_grad()
def s3_causality(config,root,device,limit):
    tables=root/"tables";rows=rule_evaluation(config,"s3");replacement=[];ablation=[];heads=[]
    for seed in config["model_seeds"]:
        m=model(config,config["depth"],device,Path("docs/papers/paper0_6/results/v2/s3_compositional")/f"s3_seed{seed}.pt",position="learned")
        by_order={}
        for order in (1,2,3):
            candidates=[r for r in rows if r.predictive_order==order];by_order[order]=[candidates[i] for i in np.linspace(0,len(candidates)-1,min(limit,len(candidates)),dtype=int)]
        for order,batch in by_order.items():
            for row in batch:
                x=tensor(row,device);base,_=m(x);base_ok,base_margin,base_p=stats(base[0,-1],row.target)
                for layer in range(config["depth"]):
                    for component in ("sa","ff"):
                        z,_=m(x,intervention=Intervention(layer,component,"zero"));ok,margin,p=stats(z[0,-1],row.target);ablation.append({"model_seed":seed,"predictive_order":order,"example_id":row.example_id,"layer":layer,"component":component,"mode":"zero","accuracy_delta":ok-base_ok,"margin_delta":margin-base_margin,"js_damage":js(base_p,p)})
                    for head in range(config["heads"]):
                        z,_=m(x,intervention=Intervention(layer,"sa","head_zero",head=head));ok,margin,p=stats(z[0,-1],row.target);heads.append({"model_seed":seed,"predictive_order":order,"example_id":row.example_id,"layer":layer,"head":head,"accuracy_delta":ok-base_ok,"margin_delta":margin-base_margin,"js_damage":js(base_p,p)})
            groups=defaultdict(list)
            for r in batch:groups[len(r.tokens)].append(r)
            for same_length in [v for v in groups.values() if len(v)>=2]:
                x=torch.tensor([r.tokens for r in same_length],device=device);base,_=m(x)
                for layer in range(config["depth"]):
                    for component in ("sa","ff"):
                        z,_=m(x,intervention=Intervention(layer,component,"mean"))
                        for i,r in enumerate(same_length):
                            b_ok,b_margin,bp=stats(base[i,-1],r.target);ok,margin,p=stats(z[i,-1],r.target);ablation.append({"model_seed":seed,"predictive_order":order,"example_id":r.example_id,"layer":layer,"component":component,"mode":"mean","accuracy_delta":ok-b_ok,"margin_delta":margin-b_margin,"js_damage":js(bp,p)})
        order2=[r for r in rows if r.predictive_order==2];order3=by_order[3]
        for recipient in order3:
            semantic_value=int(recipient.class_id.rsplit("-",1)[1]);same=[r for r in order2 if len(r.tokens)==len(recipient.tokens) and int(r.class_id.rsplit("-",1)[1])==semantic_value];cross=[r for r in order2 if len(r.tokens)==len(recipient.tokens) and int(r.class_id.rsplit("-",1)[1])!=semantic_value]
            donor_sets=[("matched_order2",same[:1]),("unrelated_same_semantic_target",same[-1:]),("cross_target",cross[:1])]
            for donor_type,donors in donor_sets:
                if not donors:continue
                donor=donors[0]
                rx=tensor(recipient,device);dx=tensor(donor,device);base,_=m(rx);base_ok,base_margin,base_p=stats(base[0,-1],recipient.target);_,dt=m(dx,capture=True)
                for layer in range(config["depth"]):
                    for component,update in (("sa",dt.layers[layer].delta_sa),("ff",dt.layers[layer].delta_ff)):
                        z,_=m(rx,intervention=Intervention(layer,component,"replace",update));ok,margin,p=stats(z[0,-1],recipient.target);replacement.append({"model_seed":seed,"recipient":recipient.example_id,"donor":donor.example_id,"donor_type":donor_type,"layer":layer,"component":component,"accuracy_delta":ok-base_ok,"margin_damage":base_margin-margin,"js_damage":js(base_p,p)})
                    for head in range(config["heads"]):
                        update=dt.layers[layer].head_outputs[:,head];z,_=m(rx,intervention=Intervention(layer,"sa","head_replace",update,head));ok,margin,p=stats(z[0,-1],recipient.target);heads.append({"model_seed":seed,"predictive_order":3,"example_id":recipient.example_id,"layer":layer,"head":head,"mode":donor_type,"accuracy_delta":ok-base_ok,"margin_delta":margin-base_margin,"js_damage":js(base_p,p)})
    write_csv(tables/"s3_causal_replacement.csv",replacement);write_csv(tables/"s3_sa_ff_ablation.csv",ablation);write_csv(tables/"s3_head_utility.csv",heads)

def main(args):
    root=Path(args.output);pconfig=json.loads(Path(args.predicate_config).read_text());sconfig=json.loads(Path(args.s3_config).read_text());device=resolve_device(args.device or pconfig["device"]);cells=predicate_causality(pconfig,root,device,args.limit);s3_causality(sconfig,root,device,args.limit);atomic_write_json(root/"causal_manifest.json",{"eligible_local_isAncestor_cells":cells,"central_predicate_claim":"blocked_by_seed_instability","s2_status":"blocked_by_competence","examples_per_condition":args.limit})

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--predicate-config",default="configs/paper06/predicate_v4.json");p.add_argument("--s3-config",default="configs/paper06/semantic_v2.json");p.add_argument("--output",default="docs/papers/paper0_6/results/v4");p.add_argument("--device");p.add_argument("--limit",type=int,default=8);main(p.parse_args())
