"""Join observed Paper 0.85 baselines to pending Paper 0.9 machine cells."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from cl.common.artifacts import atomic_write_json,write_csv

def read(path):
    with Path(path).open(newline="",encoding="utf-8") as h:return list(csv.DictReader(h))

def main(args):
    source=Path(args.paper085);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    frontiers=read(source/"one_token_vs_multitoken.csv");mapping={"O0":"M0","O1":"M1","O2":"M2","O3":"M2"}
    observed=[{"machine":mapping[r["condition"]],"paper085_condition":r["condition"],"supervision":{
        "O0":"final_only","O1":"free_trace","O2":"structured_no_state_labels","O3":"full_teacher_forced"}[r["condition"]],
        "measured_frontier":int(r["measured_frontier"]),"K4_acquisition_probability":float(r["acquisition_probability_at_K4"]),
        "result_status":"observed_learned_model","recurrence_gain_vs_M0":int(r["recurrence_gain_vs_O0"])} for r in frontiers]
    pending=[{"machine":m,"paper085_condition":"","supervision":"tool_action","measured_frontier":"","K4_acquisition_probability":"",
        "result_status":"planned_accelerator","recurrence_gain_vs_M0":""} for m in ("M3","M4")]
    write_csv(out/"paper085_machine_ladder_status.csv",observed+pending)
    atomic_write_json(out/"paper085_status_manifest.json",{"schema_version":"paper09.paper085_status_v1",
        "observed_machines":["M0","M1","M2"],"pending_machines":["M3","M4"],"M0_frontier":3,"M1_frontier":3,
        "recurrence_gain":0,"learned_M3_M4_results":False,"reference_controller_results_are_learned_results":False})
    print(json.dumps({"observed":len(observed),"pending":len(pending)},indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--paper085",default="docs/papers/paper0_85/results/recurrence_stage1_v2")
    p.add_argument("--output",default="docs/papers/paper0_9/results/paper085_comparison_v2");main(p.parse_args())
