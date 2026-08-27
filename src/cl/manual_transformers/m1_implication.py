"""M1 implication-chain recurrence via the shared transition circuit."""
from __future__ import annotations

from itertools import permutations
from .m1_recurrence import build_transition_model, generate


ATOMS = ("A", "B", "C", "D")
SYMBOLS = ATOMS + ("STOP",)


def build_implication_variants():
    return {name: build_transition_model(SYMBOLS, name) for name in ("sa_only", "ff_only", "sa_ff")}


def legal_cases():
    # All 4! directed chains and every possible fact/start: 96 trajectories.
    for order in permutations(ATOMS):
        mapping = {order[i]: order[i+1] for i in range(3)} | {order[3]: "STOP"}
        for start_index, start in enumerate(order):
            yield mapping, start, list(order[start_index+1:]) + ["STOP"]


def evaluate(model, vocab):
    rows, step_rows = [], []
    for case_id, (mapping, start, expected) in enumerate(legal_cases()):
        generated, steps = generate(model, vocab, mapping, start, "STOP")
        rows.append({"case_id": case_id, "rules": ";".join(f"{a}->{b}" for a,b in mapping.items()), "fact": start,
                     "chain_steps": len(expected), "target_trajectory": " ".join(expected), "predicted_trajectory": " ".join(generated),
                     "correct": generated == expected, "steps_expected": len(expected), "steps_observed": len(generated),
                     "minimum_logit_margin": min(s["logit_margin"] for s in steps)})
        for s in steps:
            step_rows.append({k:v for k,v in s.items() if k != "trace"} | {"case_id": case_id})
    return rows, step_rows
