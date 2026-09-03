"""Leakage-audited relation stress generator for the Paper 0.6 extension.

This is deliberately separate from ``predicates.py`` so v4/v5 baselines remain
byte-for-byte reproducible.  Branching creates explicit off-path edges rather
than acting only as a metadata token.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict, dataclass
import math, random
from cl.common.artifacts import stable_hash

PAD, EDGE, DIST, QUERY, HOP = 0, 7, 8, 9, 10
TEMPLATE = tuple(range(11, 15))
PREDICATE = {p: 15+i for i,p in enumerate(("parent","grandparent","ancestor_k","isAncestor","root","sameParent","sameGrandparent","sameAncestorAtLevel_k"))}
BRANCH = {1:23, 2:24, 4:25, 8:26}; BOOL_FALSE, BOOL_TRUE = 27, 28
TRAIN_LEAVES=tuple(range(32,96)); TEST_LEAVES=tuple(range(96,160))
NODES=tuple(range(160,512)); DISTRACTORS=tuple(range(512,1024))
PAIRWISE={"sameParent":1,"sameGrandparent":2,"sameAncestorAtLevel_k":None}

@dataclass(frozen=True)
class StressExample:
    tokens: tuple[int,...]; target: int; example_id: str; split: str; predicate: str
    total_depth: int; required_path: int; branching: int; distractors: int
    template_id: int; position_mode: str; tree_seed: int; model_seed: int
    query_nodes: tuple[int,...]; paths: tuple[tuple[int,...],...]
    path_positions: tuple[int,...]; positive: int|None

def _sample_path(rng, split, depth):
    leaf=rng.choice(TRAIN_LEAVES if split=="train" else TEST_LEAVES)
    return (leaf,*rng.sample(NODES,depth))

def stress_example(config:dict, *, split:str, predicate:str, total_depth:int,
                   required_path:int|None, branching:int, distractors:int,
                   template:int, position_mode:str, index:int, tree_seed:int,
                   model_seed:int=0, positive:int|None=None)->StressExample:
    if predicate not in PREDICATE: raise ValueError(predicate)
    if total_depth < 1 or branching not in BRANCH: raise ValueError((total_depth,branching))
    hop = ({"parent":1,"grandparent":2,"root":total_depth}.get(predicate)
           or int(required_path or 1))
    if hop>total_depth: raise ValueError((predicate,hop,total_depth))
    seed=config["ontology_seed"]+tree_seed*1_000_003+index*101+total_depth*37+branching*17
    rng=random.Random(seed); left=_sample_path(rng,split,total_depth); paths=[left]
    pairwise=predicate in PAIRWISE
    if pairwise:
        positive=1-index%2 if positive is None else int(positive)
        # A positive pair shares the requested ancestor and all nodes above it;
        # a hard negative shares only the next higher ancestor when available;
        # at the root level it belongs to a disjoint tree.
        right=list(_sample_path(rng,split,total_depth)); shared=hop if positive else min(total_depth,hop+1)
        if not positive and hop==total_depth: shared=total_depth+1
        right[shared:]=left[shared:]; paths.append(tuple(right))
        target=BOOL_TRUE if positive else BOOL_FALSE
    elif predicate=="isAncestor":
        positive=1-index%2 if positive is None else int(positive)
        candidate=left[hop] if positive else rng.choice(DISTRACTORS)
        target=BOOL_TRUE if positive else BOOL_FALSE
    else: target=left[hop]
    edges=[]; edge_child_offsets=[]
    for path in paths:
        for level,(child,parent) in enumerate(zip(path,path[1:])):
            edge_child_offsets.append(len(edges)+1); edges.extend((EDGE,child,parent))
            # Explicit topology pressure: siblings point to the same parent.
            for j in range(branching-1): edges.extend((EDGE,NODES[(seed+level*11+j)%len(NODES)],parent))
    noise=[DISTRACTORS[(seed+i)%len(DISTRACTORS)] for i in range(distractors)]
    noise_segment=[DIST,*noise]
    noise_first=template in (1,3)
    segments=(noise_segment,edges) if noise_first else (edges,noise_segment)
    body=[TEMPLATE[template],BRANCH[branching],*[x for segment in segments for x in segment]]
    q=[QUERY,PREDICATE[predicate],*([HOP]*hop if predicate in ("ancestor_k","sameAncestorAtLevel_k") else []),left[0]]
    if pairwise:q.append(paths[1][0])
    elif predicate=="isAncestor":q.append(candidate)
    padding=rng.randrange(9) if position_mode=="randomized" else 0
    body=[PAD]*padding+body
    edge_start=padding+2+(len(noise_segment) if noise_first else 0)
    path_positions=tuple(edge_start+i for i in edge_child_offsets)
    tokens=tuple((*body,*q))
    if len(tokens)>config["max_length"]: raise ValueError((len(tokens),config["max_length"]))
    return StressExample(tokens,target,f"{split}:{predicate}:D{total_depth}:d{hop}:b{branching}:N{distractors}:t{template}:p{position_mode}:tree{tree_seed}:i{index}",split,predicate,total_depth,hop,branching,distractors,template,position_mode,tree_seed,model_seed,(left[0],)+(paths[1][0],) if pairwise else (left[0],),tuple(paths),path_positions,positive)

def stress_design(config:dict, model_seed:int, examples:int=2)->list[StressExample]:
    """Factorized controls: vary one of D/d, b, or N around a common anchor."""
    out=[]; common=dict(split="test",model_seed=model_seed)
    def presentation(i:int)->dict:
        templates=config.get("templates",[0]);positions=config.get("position_modes",["aligned"])
        return {"template":templates[i%len(templates)],
                "position_mode":positions[(i//len(templates))%len(positions)]}
    for predicate in config["predicates"]:
      for D in config["test_tree_depths"]:
       hops=config["test_hops"] if predicate in ("ancestor_k","isAncestor","sameAncestorAtLevel_k") else [None]
       for d in hops:
        if d is not None and d>D: continue
        for tree in config["tree_seeds"]:
         for i in range(examples):out.append(stress_example(config,**common,**presentation(i),predicate=predicate,total_depth=D,required_path=d,branching=2,distractors=4,index=i,tree_seed=tree))
      for b in config["branching_factors"]:
       for tree in config["tree_seeds"]:
        for i in range(examples):out.append(stress_example(config,**common,**presentation(i),predicate=predicate,total_depth=8,required_path=3,branching=b,distractors=4,index=10_000+i,tree_seed=tree))
      for N in config["distractor_counts"]:
       for tree in config["tree_seeds"]:
        for i in range(examples):out.append(stress_example(config,**common,**presentation(i),predicate=predicate,total_depth=8,required_path=3,branching=2,distractors=N,index=20_000+i,tree_seed=tree))
    return out

def training_batch(config:dict,rng:random.Random,model_seed:int,predicates:list[str]|None=None,
                   batch_size:int|None=None)->list[StressExample]:
    """Sample a balanced shallow-training batch under the v6 serialization."""
    allowed=predicates or config["predicates"];size=batch_size or config.get("batch_size",64)
    predicate=rng.choice(allowed);depth=rng.choice(config["train_tree_depths"])
    if predicate in ("grandparent","sameGrandparent") and depth<2:depth=2
    hops=[h for h in config["train_hops"] if h<=depth]
    hop=rng.choice(hops) if predicate in ("ancestor_k","isAncestor","sameAncestorAtLevel_k") else None
    branching=rng.choice(config["branching_factors"]);distractors=rng.choice((0,4,16))
    template=rng.choice(config["templates"]);position=rng.choice(config["position_modes"])
    tree_seed=rng.choice(config["tree_seeds"]);base=rng.randrange(1_000_000)
    rows=[]
    for i in range(size):
        positive=i%2 if predicate in (*PAIRWISE,"isAncestor") else None
        rows.append(stress_example(config,split="train",predicate=predicate,total_depth=depth,
            required_path=hop,branching=branching,distractors=distractors,template=template,
            position_mode=position,index=base+i,tree_seed=tree_seed,model_seed=model_seed,
            positive=positive))
    return rows

def axis(example:StressExample)->str:
    i=int(example.example_id.rsplit(":i",1)[1])
    return "distractors" if i>=20_000 else "branching" if i>=10_000 else "depth_path"

def validate_stress(config:dict)->dict:
    rows=stress_design(config,11,2); bools=Counter(r.target for r in rows if r.predicate in PAIRWISE or r.predicate=="isAncestor")
    entropy=-sum((n/sum(bools.values()))*math.log2(n/sum(bools.values())) for n in bools.values())
    factorized=all((r.branching==2 and r.distractors==4) if axis(r)=="depth_path" else (r.total_depth==8 and (r.distractors==4 if axis(r)=="branching" else r.branching==2)) for r in rows)
    passed=set(PREDICATE)=={r.predicate for r in rows} and entropy>.99 and factorized and max(map(len,(r.tokens for r in rows)))<=config["max_length"] and max(max(r.tokens) for r in rows)<config["vocab_size"]
    return {"schema_version":"paper06.stress_v6.validation.v1","passed":passed,"predicates":sorted(PREDICATE),"pairwise_target_entropy_bits":entropy,"factorized_controls":factorized,"maximum_sequence_length":max(map(len,(r.tokens for r in rows))),"explicit_topology_edges":True,"example_hash":stable_hash([asdict(r) for r in rows])}
