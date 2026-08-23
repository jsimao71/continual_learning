"""Audit whether Paper 0.7's learned-category prerequisite is satisfied."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from cl.common.artifacts import atomic_write_json, stable_hash


def run(args):
    paper05=json.loads(Path(args.paper05_manifest).read_text(encoding="utf-8"))
    with Path(args.competence).open(newline="",encoding="utf-8") as handle: competence=list(csv.DictReader(handle))
    passed=[row for row in competence if row["competent"].lower()=="true"]
    decision={
        "schema_version":"paper07.eligibility.v1",
        "paper05_predictive_equivalence_available":paper05.get("schema_version")=="paper05.results.v2",
        "paper06_competent_runs":len(passed),
        "paper06_total_runs":len(competence),
        "paper06_threshold":float(competence[0]["threshold"]) if competence else None,
        "paper06_accuracy_range":[min(float(r["heldout_accuracy"]) for r in competence),max(float(r["heldout_accuracy"]) for r in competence)] if competence else [],
        "eligible":bool(passed) and paper05.get("schema_version")=="paper05.results.v2",
        "decision":"blocked: do not run semantic type-dispatch, composition, or rule-learning experiments" if not passed else "eligible for preregistered E1--E6",
    }
    decision["source_hash"]=stable_hash({"paper05":paper05,"paper06":competence})
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True); atomic_write_json(output,decision)
    print(json.dumps(decision,indent=2))


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--paper05-manifest",default="docs/papers/paper0_5/results/manifest.json")
    p.add_argument("--competence",default="docs/papers/paper0_6/results/tables/competence_gate.csv")
    p.add_argument("--output",default="docs/papers/paper0_7/results/eligibility_decision.json"); return p.parse_args()


if __name__=="__main__": run(parse_args())
