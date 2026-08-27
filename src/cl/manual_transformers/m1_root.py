"""M1 root traversal by autoregressive reuse of one parent lookup circuit."""
from __future__ import annotations

from itertools import permutations
from .m1_recurrence import build_transition_model, generate


NODES = ("R", "A", "B", "C")
SYMBOLS = NODES + ("STOP",)


def build_root_variants():
    return {name: build_transition_model(SYMBOLS, name) for name in ("sa_only", "ff_only", "sa_ff")}


def legal_cases():
    # Every ordering of the three non-root identities and every start node:
    # 6 chains x 4 starts, spanning zero through three parent hops before R.
    for order in permutations(("A", "B", "C")):
        mapping = {"R": "STOP", order[0]: "R", order[1]: order[0], order[2]: order[1]}
        for start in ("R",) + order:
            expected, current = [], start
            while True:
                current = mapping[current]; expected.append(current)
                if current == "STOP": break
            yield mapping, start, expected


def evaluate(model, vocab):
    rows, step_rows = [], []
    for case_id, (mapping, start, expected) in enumerate(legal_cases()):
        generated, steps = generate(model, vocab, mapping, start, "STOP")
        rows.append({"case_id": case_id, "mapping": ";".join(f"{a}->{b}" for a,b in mapping.items()), "start": start,
                     "depth_to_root": len(expected)-1, "target_trajectory": " ".join(expected), "predicted_trajectory": " ".join(generated),
                     "correct": generated == expected, "steps_expected": len(expected), "steps_observed": len(generated),
                     "minimum_logit_margin": min(s["logit_margin"] for s in steps)})
        for s in steps:
            step_rows.append({k:v for k,v in s.items() if k != "trace"} | {"case_id": case_id})
    return rows, step_rows
