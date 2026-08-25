"""Train-shallow/test-deep predicate generator for Paper 0.6 v4."""
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
import math,random
from cl.common.artifacts import stable_hash

PAD=0;PATH=7;DIST=8;QUERY=9
TEMPLATE=tuple(range(10,14));PREDICATE={"parent":14,"grandparent":15,"ancestor_k":16,"isAncestor":17,"root":18}
HOP=19;BRANCH=tuple(range(20,24));MODE={"arbitrary":24,"natural":25};BOOL_FALSE=26;BOOL_TRUE=27
TRAIN_LEAVES=tuple(range(32,104));TEST_LEAVES=tuple(range(104,176));RELATION_NODES=tuple(range(176,320));DISTRACTOR_POOL=tuple(range(320,376))

@dataclass(frozen=True)
class PredicateExample:
    tokens:tuple[int,...];target:int;example_id:str;split:str;predicate:str;tree_seed:int;model_seed:int
    total_depth:int;required_path:int;branching:int;distractors:int;template_id:int;position_mode:str;label_mode:str
    node_path:tuple[int,...];path_positions:tuple[int,...];query_node:int;candidate_node:int|None;positive:int|None

def _nodes(split:str,seed:int,count:int)->tuple[int,...]:
    leaves=TRAIN_LEAVES if split=="train" else TEST_LEAVES;rng=random.Random(seed);return (rng.choice(leaves),*rng.sample(RELATION_NODES,count-1))

def predicate_example(config:dict,*,split:str,predicate:str,total_depth:int,required_path:int|None,branching:int,distractors:int,template:int,position_mode:str,index:int,tree_seed:int,model_seed:int=0,label_mode:str="arbitrary",positive:int|None=None)->PredicateExample:
    if total_depth<1:raise ValueError("tree depth must be positive")
    seed=config["ontology_seed"]+tree_seed*1000003+index*101+total_depth*37+branching*17+template*7
    rng=random.Random(seed);path=_nodes(split,seed,total_depth+1);query_node=path[0]
    if predicate=="parent":hop=1
    elif predicate=="grandparent":hop=2
    elif predicate=="ancestor_k":hop=int(required_path or 1)
    elif predicate=="root":hop=total_depth
    elif predicate=="isAncestor":hop=int(required_path or min(1,total_depth))
    else:raise ValueError(predicate)
    if predicate!="isAncestor" and hop>total_depth:raise ValueError((predicate,hop,total_depth))
    offpath=[DISTRACTOR_POOL[(seed+i)%len(DISTRACTOR_POOL)] for i in range(distractors)]
    candidate=None
    if predicate=="isAncestor":
        positive=(index%2) if positive is None else positive
        if positive:candidate=path[min(hop,total_depth)];target=BOOL_TRUE
        else:
            negative_kind=index%4
            candidate=(offpath+[DISTRACTOR_POOL[(seed+999)%len(DISTRACTOR_POOL)]])[negative_kind%max(1,len(offpath)+1)];target=BOOL_FALSE
    else:target=path[hop]
    path_segment=[PATH,*path];distractor_segment=[DIST,*offpath] if offpath else []
    if template in (1,3):segments=[distractor_segment,path_segment]
    else:segments=[path_segment,distractor_segment]
    body=[TEMPLATE[template],MODE[label_mode],BRANCH[[1,2,4,8].index(branching)]]
    for segment in segments:body.extend(segment)
    query=[QUERY,PREDICATE[predicate],*([HOP]*hop if predicate=="ancestor_k" else []),query_node]
    if candidate is not None:query.append(candidate)
    if position_mode=="randomized":body=[PAD]*rng.randrange(9)+body
    tokens=tuple([*body,*query]);path_start=tokens.index(PATH)+1;path_positions=tuple(range(path_start,path_start+len(path)))
    if len(tokens)>config["max_length"]:raise ValueError((len(tokens),config["max_length"],predicate,total_depth,distractors))
    return PredicateExample(tokens,target,f"{split}:{predicate}:D{total_depth}:d{hop}:b{branching}:N{distractors}:t{template}:p{position_mode}:tree{tree_seed}:i{index}",split,predicate,tree_seed,model_seed,total_depth,hop,branching,distractors,template,position_mode,label_mode,path,path_positions,query_node,candidate,positive)

def training_batch(config:dict,rng:random.Random,model_seed:int)->list[PredicateExample]:
    predicate=rng.choice(config["predicates"]);depth=rng.choice(config["train_tree_depths"]);branching=rng.choice(config["branching_factors"]);distractors=rng.choice((0,4,16));template=rng.choice(config["templates"]);position=rng.choice(config["position_modes"]);tree_seed=rng.choice(config["tree_seeds"])
    if predicate=="grandparent" and depth<2:depth=2
    hop=rng.choice([k for k in config["train_hops"] if k<=depth]) if predicate in ("ancestor_k","isAncestor") else None
    return [predicate_example(config,split="train",predicate=predicate,total_depth=depth,required_path=hop,branching=branching,distractors=distractors,template=template,position_mode=position,index=rng.randrange(1_000_000),tree_seed=tree_seed,model_seed=model_seed) for _ in range(config["batch_size"])]

def evaluation_matrix(config:dict,model_seed:int,examples:int|None=None)->list[PredicateExample]:
    count=examples or config["evaluation_examples_per_cell"];rows=[];base={"split":"test","model_seed":model_seed,"label_mode":"arbitrary"}
    # Depth/path surface at baseline branching and nuisance.
    for predicate in config["predicates"]:
        for depth in config["test_tree_depths"]:
            hops=config["test_hops"] if predicate in ("ancestor_k","isAncestor") else [None]
            for hop in hops:
                if hop is not None and hop>depth:continue
                for tree_seed in config["tree_seeds"]:
                    for i in range(count):rows.append(predicate_example(config,**base,predicate=predicate,total_depth=depth,required_path=hop,branching=2,distractors=4,template=i%4,position_mode=config["position_modes"][(i//4)%2],index=i,tree_seed=tree_seed))
    # Independent branching and distractor controls at feasible fixed depths.
    for predicate in config["predicates"]:
        for branching in config["branching_factors"]:
            for tree_seed in config["tree_seeds"]:
                for i in range(count):rows.append(predicate_example(config,**base,predicate=predicate,total_depth=8,required_path=3 if predicate in ("ancestor_k","isAncestor") else None,branching=branching,distractors=4,template=i%4,position_mode=config["position_modes"][(i//4)%2],index=10_000+i,tree_seed=tree_seed))
        for distractors in config["distractor_counts"]:
            for tree_seed in config["tree_seeds"]:
                for i in range(count):rows.append(predicate_example(config,**base,predicate=predicate,total_depth=8,required_path=3 if predicate in ("ancestor_k","isAncestor") else None,branching=2,distractors=distractors,template=i%4,position_mode=config["position_modes"][(i//4)%2],index=20_000+i,tree_seed=tree_seed))
    return rows

def validate(config:dict)->dict:
    rng=random.Random(71);train=[r for _ in range(200) for r in training_batch({**config,"batch_size":64},rng,11)];test=evaluation_matrix(config,11,2);train_queries={r.query_node for r in train};test_queries={r.query_node for r in test};train_targets={r.target for r in train if r.predicate!="isAncestor"};test_targets={r.target for r in test if r.predicate!="isAncestor"};bools=Counter(r.target for r in test if r.predicate=="isAncestor")
    target_entropy=-sum((n/sum(bools.values()))*math.log2(n/sum(bools.values())) for n in bools.values())
    unseen_target_labels=test_targets-train_targets;passed=not(train_queries&test_queries) and not unseen_target_labels and target_entropy>.95 and all(r.target in r.tokens or r.predicate=="isAncestor" for r in test) and max(max(r.tokens) for r in test)<config["vocab_size"]
    return {"schema_version":"paper06.predicate_v4.validation.v2","passed":passed,"train_test_query_identity_overlap":len(train_queries&test_queries),"unseen_test_target_labels":len(unseen_target_labels),"topology_split":"disjoint generated paths; shared internal label vocabulary","isAncestor_target_entropy_bits":target_entropy,"max_sequence_length":max(len(r.tokens) for r in test),"predicates":sorted({r.predicate for r in test}),"tree_seeds":config["tree_seeds"],"example_hash":stable_hash([r.__dict__ for r in test])}
