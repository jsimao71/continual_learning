"""CPU harness validation and planned matched M0--M4 Paper 0.85 comparison."""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
from cl.common.artifacts import atomic_write_json, stable_hash, write_csv
from cl.semantic.proof_harness import example_from_dict, oracle_harness
from cl.semantic.autoregressive_proofs import Implication,ProofExample
from cl.semantic.recurrence_chains import generate_chains,recurrence_pair_split

def load_examples(path:Path):
    return [example_from_dict(json.loads(line)) for line in path.read_text().splitlines()]

def matched_recurrence_examples(config):
    _,_,test_pairs=recurrence_pair_split(config["symbol_count"],config["pair_split_seed"],config["test_pair_fraction"])
    examples=[]
    for depth in config["test_depths"]:
        rows=generate_chains(test_pairs,depth,config["eval_per_depth"],9000+depth,"test")
        for index,row in enumerate(rows):
            rules=tuple(Implication(a,b) for a,b in zip(row.chain,row.chain[1:]))
            examples.append(ProofExample(f"matched:K{depth}:{index}","test","R1" if depth==2 else "R2",
                tuple(row.chain),rules,row.chain[-1],depth,0,1,False,"shared"))
    return examples

def main(args):
    config=json.loads(Path(args.config).read_text()); source=Path(config["paper085_source"])
    examples=matched_recurrence_examples(config) if config.get("matched_recurrence_stage1") else load_examples(source/"test.jsonl")
    if args.smoke: examples=examples[:24]
    oracle=[]; transitions=[]
    for example in examples:
        for strength in config["tool_strengths"]:
            summary,trace=oracle_harness(example,strength,max_steps=example.proof_depth+1)
            run_id=f"{example.example_id}:{strength}"
            prompt_tokens=4*example.proof_depth+4 if config.get("matched_recurrence_stage1") else len(example.prompt().split()); total=prompt_tokens+summary["generated_action_tokens"]
            oracle.append({"run_id":run_id,"example_id":example.example_id,"stage":example.stage,
                "depth":example.proof_depth,"distractors":example.distractors,"branching":example.branching,
                "shuffled":example.shuffled,"machine":"M4","tool_strength":strength,"result_type":"reference_oracle",
                "prompt_tokens":prompt_tokens,"total_context_tokens":total,**summary})
            transitions += [{"run_id":run_id,"tool_strength":strength,**row} for row in trace]
    grid=config["smoke_grid"] if args.smoke else config["architecture_grid"]
    plan=[]
    supervision={"M0":"final_answer","M1":"free_ar","M2":"derivation","M3":"tool_action","M4":"tool_action"}
    for L,W,H,T,seed,machine in itertools.product(grid["layers"],grid["widths"],grid["heads"],grid["budgets"],
                                                  config["model_seeds"],config["machines"]):
        if W%H: continue
        plan.append({"layers":L,"width":W,"heads":H,"budget":T,"seed":seed,"machine":machine,
            "supervision":supervision[machine],"dataset_hash":config["paper085_dataset_hash"],
            "status":"planned_accelerator","learned_model_result":False})
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    write_csv(output/"paper085_machine_ladder.csv",plan)
    write_csv(output/"paper085_tool_strength.csv",oracle)
    write_csv(output/"paper085_transition_errors.csv",transitions)
    write_csv(output/"paper085_execution_cost.csv",({k:r[k] for k in ("run_id","depth","machine","tool_strength",
        "model_forwards","generated_action_tokens","tool_calls","prompt_tokens","total_context_tokens","result_type")} for r in oracle))
    atomic_write_json(output/"paper085_comparison_manifest.json",{"schema_version":config["schema_version"],
        "smoke":args.smoke,"paper085_dataset_hash":config["paper085_dataset_hash"],"matched_test_examples":len(examples),
        "reference_runs":len(oracle),"planned_accelerator_runs":len(plan),
        "all_reference_harness_runs_correct":all(r["correct"] for r in oracle),"learned_model_results":False,
        "oracle_controller":True,"dataset_fingerprint":stable_hash([{"id":e.example_id,"chain":e.chain,
            "rules":[(r.lhs,r.rhs) for r in e.rules],"depth":e.proof_depth} for e in examples])})
    print(json.dumps({"examples":len(examples),"reference_runs":len(oracle),"planned_runs":len(plan)},indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/paper09/paper085_comparison_v1.json")
    p.add_argument("--output",default="docs/papers/paper0_9/results/paper085_comparison"); p.add_argument("--smoke",action="store_true")
    main(p.parse_args())
