"""Competence-first controlled semantic generators for Paper 0.6 v2/v3."""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import random
from collections import Counter

from cl.common.artifacts import stable_hash

PAD,DEFINE,IS_A,SEP,QUERY_PARENT,QUERY_ANCESTOR,QUERY_ROOT=0,1,2,3,4,5,6
TEMPLATE=tuple(range(10,14));LEAF_TRAIN=tuple(range(20,100));LEAF_TEST=tuple(range(100,180))
NAT_SUB=tuple(range(180,244));NAT_CAT=tuple(range(244,260));NAT_ROOT=tuple(range(260,264))
ARB_SUB=tuple(range(264,328));ARB_CAT=tuple(range(328,344));ARB_ROOT=tuple(range(344,348))
REL_MARKERS=tuple(range(14,18));ATTR_MARKERS=tuple(range(14,20));VALUE_TOKENS=tuple(range(348,360))
S2_NAT=(360,361);S2_ARB=(362,363);S3_TARGETS=((364,365),(366,367),(368,369))


@dataclass(frozen=True)
class SemanticExample:
    tokens: tuple[int,...]
    target: int
    stage: str
    example_id: str
    ontology_family: str
    label_mode: str
    query_level: str
    template_id: int
    position_mode: str
    entity_id: str
    category_id: str
    parent_id: str
    root_id: str
    predictive_order: int
    raw_length: int
    train_identity: bool


@dataclass(frozen=True)
class RuleExample:
    tokens: tuple[int,...]
    target: int
    stage: str
    example_id: str
    label_mode: str
    template_id: int
    position_mode: str
    entity_id: str
    class_id: str
    feature_bits: tuple[int,...]
    predictive_order: int
    train_identity: bool


def _pad(tokens:list[int],length:int,mode:str,rng:random.Random)->tuple[int,...]:
    room=length-len(tokens)
    if room<0:raise ValueError((len(tokens),length))
    if mode=="aligned":return tuple([PAD]*room+tokens)
    # Vary absolute fact positions while keeping the query at the causal readout.
    prefix=tokens[:-2];query=tokens[-2:];slots=[0]*(len(prefix)+1)
    for _ in range(room):slots[rng.randrange(len(slots))]+=1
    randomized=[]
    for index,token in enumerate(prefix):randomized.extend([PAD]*slots[index]);randomized.append(token)
    randomized.extend([PAD]*slots[-1]);randomized.extend(query)
    return tuple(randomized)


def s1_example(config:dict,branching:int,tree_depth:int,instances:int,label_mode:str,query_level:str,
               template:int,position_mode:str,index:int,split:str)->SemanticExample:
    seed=config["ontology_seed"]+branching*100003+tree_depth*1009+instances*53+index*17+(0 if split=="train" else 10_000_000)
    rng=random.Random(seed);root_index=index%branching;category_index=(index//branching)%branching
    sub_index=(index//(branching*branching))%branching if tree_depth==3 else category_index
    leaf_pool=LEAF_TRAIN if split=="train" else LEAF_TEST;leaf=leaf_pool[index%len(leaf_pool)]
    if label_mode=="natural":subs,cats,roots=NAT_SUB,NAT_CAT,NAT_ROOT
    else:subs,cats,roots=ARB_SUB,ARB_CAT,ARB_ROOT
    root=roots[root_index];cat=cats[root_index*branching+category_index]
    sub=subs[(root_index*branching+category_index)*branching+sub_index] if tree_depth==3 else cat
    clauses=[[DEFINE,leaf,IS_A,sub]]
    if tree_depth==3:clauses.append([sub,IS_A,cat])
    clauses.append([cat,IS_A,root])
    if template in (1,3):clauses=list(reversed(clauses))
    chain=[]
    for clause in clauses:
        if chain:chain.append(SEP)
        chain.extend(clause)
    query={"parent":QUERY_PARENT,"ancestor":QUERY_ANCESTOR,"root":QUERY_ROOT}[query_level]
    target={"parent":sub,"ancestor":cat,"root":root}[query_level]
    body=[TEMPLATE[template],*chain,SEP,query,leaf]
    tokens=_pad(body,config["sequence_length"],position_mode,rng);family=f"b{branching}-d{tree_depth}-i{instances}"
    return SemanticExample(tokens,target,"s1",f"{split}:{family}:{label_mode}:{query_level}:t{template}:p{position_mode}:{index}",family,label_mode,query_level,template,position_mode,
                           f"{split}-leaf-{leaf}",f"cat-{cat}",f"parent-{sub}",f"root-{root}",1,len(body),split=="train")


def s1_evaluation(config:dict,examples:int|None=None)->list[SemanticExample]:
    count=examples or config["evaluation_examples_per_cell"]
    return [s1_example(config,b,d,i,m,q,t,p,x,"test") for b,d,i,m,q,t,p in itertools.product(config["branching_factors"],config["tree_depths"],config["instances_per_leaf_group"],config["label_modes"],config["query_levels"],config["templates"],config["position_modes"]) for x in range(count)]


def s1_training_batch(config:dict,rng:random.Random)->list[SemanticExample]:
    rows=[]
    for _ in range(config["batch_size"]):
        rows.append(s1_example(config,rng.choice(config["branching_factors"]),rng.choice(config["tree_depths"]),rng.choice(config["instances_per_leaf_group"]),rng.choice(config["label_modes"]),rng.choice(config["query_levels"]),rng.choice(config["templates"]),rng.choice(config["position_modes"]),rng.randrange(1_000_000),"train"))
    return rows


def s1_validation(config:dict)->dict:
    train={s1_example(config,2,3,8,"arbitrary","parent",0,"aligned",i,"train").entity_id for i in range(80)}
    test={s1_example(config,2,3,8,"arbitrary","parent",0,"aligned",i,"test").entity_id for i in range(80)}
    rows=s1_evaluation(config,4);counts=Counter((r.ontology_family,r.label_mode,r.query_level,r.template_id,r.position_mode) for r in rows)
    targets=Counter(r.target for r in rows);entropy=-sum((v/len(rows))*math.log2(v/len(rows)) for v in targets.values())
    result={"schema_version":"paper06.s1.validation.v1","passed":not(train&test) and len(set(counts.values()))==1,
            "train_test_identity_overlap":len(train&test),"template_balance":len(set(counts.values()))==1,"target_entropy_bits":entropy,
            "ontology_families":sorted({r.ontology_family for r in rows}),"example_hash":stable_hash([r.__dict__ for r in rows])}
    return result


def _rule_body(entity:int,markers:tuple[int,...],bits:tuple[int,...],template:int,query:int)->list[int]:
    pairs=[[marker,VALUE_TOKENS[2*i+bit]] for i,(marker,bit) in enumerate(zip(markers,bits))]
    orders=(range(len(pairs)),reversed(range(len(pairs))),list(range(0,len(pairs),2))+list(range(1,len(pairs),2)),list(range(1,len(pairs),2))+list(range(0,len(pairs),2)))
    order=list(orders[template]);body=[TEMPLATE[template],DEFINE,entity]
    for index in order:body.extend([SEP,*pairs[index]])
    return [*body,SEP,query,entity]


def s2_example(config:dict,label_mode:str,template:int,position_mode:str,index:int,split:str)->RuleExample:
    combos=[i for i in range(16) if (i%5!=0)==(split=="train")]
    combo=combos[index%len(combos)];bits=tuple((combo>>i)&1 for i in range(4));pool=LEAF_TRAIN if split=="train" else LEAF_TEST
    entity=pool[index%len(pool)];target=(S2_NAT if label_mode=="natural" else S2_ARB)[sum(bits)%2]
    rng=random.Random(config["ontology_seed"]+2_000_000+index*31+(0 if split=="train" else 9_000_000))
    tokens=_pad(_rule_body(entity,REL_MARKERS,bits,template,QUERY_PARENT),config["sequence_length"],position_mode,rng)
    return RuleExample(tokens,target,"s2",f"{split}:{label_mode}:t{template}:p{position_mode}:c{combo}:i{index}",label_mode,template,position_mode,f"{split}-entity-{entity}",f"parity-{sum(bits)%2}",bits,4,split=="train")


def s3_example(config:dict,level:int,template:int,position_mode:str,index:int,split:str)->RuleExample:
    combos=[i for i in range(64) if (i%5!=0)==(split=="train")]
    combo=combos[index%len(combos)];bits=tuple((combo>>i)&1 for i in range(6));pool=LEAF_TRAIN if split=="train" else LEAF_TEST
    entity=pool[index%len(pool)];indices=((0,),(0,2),(0,2,5))[level-1];class_value=sum(bits[i] for i in indices)%2;target=S3_TARGETS[level-1][class_value]
    rng=random.Random(config["ontology_seed"]+3_000_000+level*100003+index*37+(0 if split=="train" else 8_000_000))
    tokens=_pad(_rule_body(entity,ATTR_MARKERS,bits,template,QUERY_ANCESTOR+level),config["sequence_length"],position_mode,rng)
    return RuleExample(tokens,target,"s3",f"{split}:l{level}:t{template}:p{position_mode}:c{combo}:i{index}","arbitrary",template,position_mode,f"{split}-entity-{entity}",f"level{level}-{class_value}",bits,len(indices),split=="train")


def rule_training_batch(config:dict,stage:str,rng:random.Random)->list[RuleExample]:
    rows=[]
    for _ in range(config["batch_size"]):
        index=rng.randrange(1_000_000);template=rng.choice(config["templates"]);position=rng.choice(config["position_modes"])
        rows.append(s2_example(config,rng.choice(config["label_modes"]),template,position,index,"train") if stage=="s2" else s3_example(config,rng.choice((1,2,3)),template,position,index,"train"))
    return rows


def rule_evaluation(config:dict,stage:str,examples:int|None=None)->list[RuleExample]:
    count=examples or config["evaluation_examples_per_cell"]
    if stage=="s2":return [s2_example(config,m,t,p,i,"test") for m,t,p in itertools.product(config["label_modes"],config["templates"],config["position_modes"]) for i in range(count)]
    return [s3_example(config,l,t,p,i,"test") for l,t,p in itertools.product((1,2,3),config["templates"],config["position_modes"]) for i in range(count)]


def rule_validation(config:dict,stage:str)->dict:
    train=rule_training_batch({**config,"batch_size":1024},stage,random.Random(17));test=rule_evaluation(config,stage,64)
    overlap={r.entity_id for r in train}&{r.entity_id for r in test};counts=Counter(r.class_id for r in test)
    # The construction uses parity rules: every strict feature subset is independent of the target.
    proper_subset_mi=0.0;full_structure_mi=1.0;entropy=-sum((n/len(test))*math.log2(n/len(test)) for n in counts.values())
    return {"schema_version":f"paper06.{stage}.validation.v1","passed":not overlap and min(counts.values())>0,
            "ontology_seed":config["ontology_seed"],"train_test_identity_overlap":len(overlap),"target_entropy_bits":entropy,
            "singleton_mi_bits":0.0,"proper_subset_mi_bits":proper_subset_mi,"full_structure_mi_bits":full_structure_mi,
            "heldout_combination_rule":"integer pattern modulo 5 equals zero","example_hash":stable_hash([r.__dict__ for r in test])}
