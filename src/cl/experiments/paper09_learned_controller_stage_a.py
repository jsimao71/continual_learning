"""Paper 0.9 v3 Stage A: exact nested 2x/4x budget continuation."""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from cl.common.artifacts import atomic_write_json, write_csv
from cl.experiments.paper05_predictive_order_phase import resolve_device
from cl.experiments.paper09_learned_controller_v1 import evaluate_m3, evaluate_m4, train
from cl.semantic.recurrence_chains import generate_chains, recurrence_pair_split


def read_csv(path):
    import csv
    if not path.exists(): return []
    with path.open(newline="", encoding="utf-8") as handle: return list(csv.DictReader(handle))


def summarize(rows, threshold=.95):
    grouped=defaultdict(list)
    for row in rows: grouped[(int(row["snapshot_updates"]),row["machine"],int(row["seed"]),int(row["depth"]))].append(row)
    seed_depth=[]
    for (updates,machine,seed,depth),values in sorted(grouped.items()):
        mean=lambda key:float(np.mean([float(v[key]) for v in values]))
        record={"snapshot_updates":updates,"machine":machine,"seed":seed,"depth":depth,"n":len(values),
                "final_accuracy":mean("final_correct"),"invalid_call_rate":mean("invalid_call")}
        if machine=="M3":
            record.update(one_call_coverage=mean("one_call_edge_coverage"),selected_edge_validity=mean("selected_edge_valid"),
                          post_tool_answer_accuracy=mean("post_tool_answer_correct"))
            record["seed_pass"]=int(record["final_accuracy"]>=threshold and record["one_call_coverage"]>=threshold
                                    and record["post_tool_answer_accuracy"]>=threshold)
        else:
            record.update(per_transition_accuracy=mean("per_transition_accuracy"),exact_trajectory_accuracy=mean("exact_trajectory_correct"),
                          termination_accuracy=mean("termination_correct"))
            record["seed_pass"]=int(record["final_accuracy"]>=threshold and record["per_transition_accuracy"]>=threshold
                                    and record["exact_trajectory_accuracy"]>=threshold and record["termination_accuracy"]>=threshold)
        seed_depth.append(record)
    gates=[]
    for updates in sorted({r["snapshot_updates"] for r in seed_depth}):
      for machine in ("M3","M4"):
        values=[r for r in seed_depth if r["snapshot_updates"]==updates and r["machine"]==machine]
        frontier=0
        for depth in sorted({r["depth"] for r in values}):
            cells=[r for r in values if r["depth"]==depth]
            if depth==frontier+1 and len(cells)==3 and all(r["seed_pass"] for r in cells):frontier=depth
            else:break
        k4=[r for r in values if r["depth"]==4]
        gates.append({"snapshot_updates":updates,"machine":machine,"contiguous_frontier":frontier,
                      "all_seed_K4":int(len(k4)==3 and all(r["seed_pass"] for r in k4)),
                      "worst_seed_K4_final":min((r["final_accuracy"] for r in k4),default=0.0)})
    return seed_depth,gates


def main(args=None):
    parser=argparse.ArgumentParser();parser.add_argument("--config",default="configs/paper09/learned_controller_v3_staged.json")
    parser.add_argument("--output");parser.add_argument("--device",default="mps");parser.add_argument("--resume",action="store_true")
    ns=parser.parse_args(args);plan=json.loads(Path(ns.config).read_text());common=plan["common"];dataset=plan["dataset"]
    output=Path(ns.output or plan["stages"]["A"]["output"]);output.mkdir(parents=True,exist_ok=True)
    device=resolve_device(ns.device);_,train_pairs,test_pairs=recurrence_pair_split(dataset["symbol_count"],dataset["pair_split_seed"],dataset["test_pair_fraction"])
    cfg={"train_depths":dataset["train_depth_default"],"max_length":common["max_length"],"model":{**common["baseline"],"mlp_ratio":common["mlp_ratio"]},
         "learning_rate":common["learning_rate"],"log_every":50,"checkpoint_every":common["checkpoint_every"]}
    cfg["model"]={key:cfg["model"][key] for key in ("layers","width","heads","mlp_ratio")}
    raw=read_csv(output/"stage_a_raw.csv") if ns.resume else [];loss_rows=read_csv(output/"stage_a_loss.csv") if ns.resume else []
    done={(int(r["snapshot_updates"]),r["machine"],int(r["seed"])) for r in raw}
    targets=plan["stages"]["A"]["snapshot_updates"]
    for machine in common["machines"]:
      for seed in common["model_seeds"]:
        working=output/"checkpoints"/f"{machine}_baseline_seed{seed}.pt"
        for target in targets:
          key=(target,machine,seed)
          if key in done: continue
          model,losses=train(machine,seed,cfg,device,working,train_pairs,target,common["batch_size"])
          snapshot=output/"snapshots"/f"{machine}_baseline_seed{seed}_step{target}.pt";snapshot.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(working,snapshot)
          for depth in dataset["test_depths"]:
            examples=generate_chains(test_pairs,depth,dataset["eval_per_depth_per_seed"],seed*1000+depth,"test")
            metrics=evaluate_m3(model,examples,device) if machine=="M3" else evaluate_m4(model,examples,device,common["max_extra_steps"])
            raw.extend({"snapshot_updates":target,"machine":machine,"seed":seed,"depth":depth,"example_id":i,**row} for i,row in enumerate(metrics))
          loss_rows=[r for r in loss_rows if not (r["machine"]==machine and int(r["seed"])==seed)]
          loss_rows.extend({"machine":machine,"seed":seed,**row} for row in losses)
          write_csv(output/"stage_a_raw.csv",raw);write_csv(output/"stage_a_loss.csv",loss_rows)
          seed_depth,gates=summarize(raw,plan["gates"]["competence_threshold"]);write_csv(output/"stage_a_by_seed_depth.csv",seed_depth);write_csv(output/"stage_a_gates.csv",gates)
          print(f"{machine} seed={seed} snapshot={target} complete",flush=True)
    seed_depth,gates=summarize(raw,plan["gates"]["competence_threshold"]);write_csv(output/"stage_a_by_seed_depth.csv",seed_depth);write_csv(output/"stage_a_gates.csv",gates)
    complete=len({(int(r["snapshot_updates"]),r["machine"],int(r["seed"])) for r in raw})==len(targets)*len(common["machines"])*len(common["model_seeds"])
    atomic_write_json(output/"stage_a_manifest.json",{"schema_version":"paper09.learned_controller.stage_a.v1","device":str(device),"exact_resume":True,
      "snapshots":targets,"completed":complete,"raw_rows":len(raw),"gates":gates,"later_stages_launched":False})


if __name__=="__main__":main()
