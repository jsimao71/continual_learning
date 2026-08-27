"""Generate all Milestone-1 fixed-weight witness artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .artifacts import write_matrix, write_model, write_rows, write_trace
from .m0_successor import TOKEN as S_TOKEN, build_successor_variants, evaluate as eval_successor
from .m0_lookup import TOKEN as L_TOKEN, ALPHA as L_ALPHA, build_lookup_variants, evaluate as eval_lookup
from .m0_grandparent import TOKEN as G_TOKEN, ALPHA as G_ALPHA, build_grandparent_variants, evaluate as eval_grandparent


EXPECTED = {
    "successor": {"sa_only": False, "ff_only": True, "sa_ff": True},
    "pair_lookup": {"sa_only": True, "ff_only": False, "sa_ff": True},
    "grandparent": {"sa_only": True, "ff_only": False, "sa_ff": True, "sa_only_1layer": False},
}


def canonical(task, model):
    if task == "successor":
        tokens, target = ["B"], "C"; ids = [S_TOKEN[x] for x in tokens]
    elif task == "pair_lookup":
        tokens, target = ["PAIR_A_B", "PAIR_B_C", "PAIR_C_A", "QUERY_B"], "C"; ids = [L_TOKEN[x] for x in tokens]
    else:
        tokens, target = ["EDGE_A_R", "EDGE_B_A", "EDGE_C_B", "QUERY_C"], "A"; ids = [G_TOKEN[x] for x in tokens]
    prediction, margin, trace = model.predict(ids)
    return tokens, target, prediction, margin, trace


def construction_text(task: str) -> str:
    if task == "successor":
        return """# FF-only successor construction

This is a fixed-weight existence witness, not a minimality or acquisition claim.  The legal domain is A->B, B->C, C->D.  `d_model=4`, `L=1`, `H=0` effectively for the primary FF-only topology; the recorded attention matrices are zero.  Embeddings and unembedding are the four-dimensional identity.  With row-vector convention, `W1=I`, ReLU is identity on the one-hot legal inputs, and `W2=M-I`, where row `i` of `M` is the successor one-hot.  The identity residual therefore cancels and leaves exactly `M e_i`.  LayerNorm and dropout are omitted; the positional matrix is zero.

Canonical input B has embedding `[0,1,0,0]`, zero Q/K/scores/V/head update, FF preactivation `[0,1,0,0]`, and post-FF/logits `[0,0,1,0]`.  Its exact output margin is 1.  SA-only is the identity and fails all three successor cases; SA+FF passes because its SA branch is explicitly zero.
"""
    if task == "pair_lookup":
        p = np.exp(L_ALPHA) / (np.exp(L_ALPHA) + 3)
        return f"""# SA-only associative pair lookup construction

This fixed-weight witness represents each contextual pair as one token carrying a key and value; it does not claim a minimal serialization.  `d_model=9`, `L=1`, `H=1`, `d_head=3`; subspaces are `[pair key(3)|query identity(3)|transported output(3)]`.  The final query token has identity only in the query subspace.  `W_Q` maps it to `alpha*sqrt(3)e_q`, `W_K` reads pair keys, `W_V` reads pair values, and `W_O` writes the selected value into the output subspace.  We use finite `alpha={L_ALPHA:g}`, causal softmax, identity residuals, zero positions, no LayerNorm/dropout, and no FF update.

For three pair tokens plus the query, the desired score is {L_ALPHA:g} and all other final-row scores are 0.  Thus desired probability is `exp(16)/(exp(16)+3)={p:.12f}`; leakage is finite, not hard attention.  Across all 6 bijections of A/B/C and all 3 queries, the correct output logit has a strictly positive margin.  FF-only cannot communicate the contextual map and reaches only tie-breaking chance; SA+FF passes with an explicit zero FF.
"""
    p = np.exp(G_ALPHA) / (np.exp(G_ALPHA) + 3)
    return f"""# Two-layer SA grandparent construction

This is a bounded two-hop existence witness on the complete declared legal domain B->R and C->A in the chain R<-A<-B<-C.  Each edge is one contextual token carrying child key and parent value.  `d_model=20`, `L=2`, `H=1`, `d_head=4`; subspaces are `[edge key(4)|initial query(4)|edge parent value(4)|first-hop state(4)|second-hop output(4)]`.  Layer 1 matches the queried child and writes its parent into first-hop state.  Layer 2 queries from that transported state, matches the corresponding parent edge, and writes the grandparent into output.  Edge keys/values occupy invariant residual subspaces.

Both layers use finite `alpha={G_ALPHA:g}`, causal softmax, identity residuals, zero positions, no LayerNorm/dropout, and zero FF.  Layer-1 desired probability is `exp(16)/(exp(16)+3)={p:.12f}`.  The second layer receives a slightly leaky state but retains a near-unit attention and output-logit margin, recorded exactly in the CSV traces.  Two-layer SA-only and SA+FF pass.  FF-only cannot access contextual edges; a one-layer SA control transports only the parent and has no output support, so both fail.  This demonstrates bounded composition under this encoding, not closure, minimality, or SGD acquisition.
"""


def plot_results(root: Path, summaries: list[dict], raw_by_task: dict[str, dict[str, list[dict]]]) -> None:
    labels = ["SA only", "FF only", "SA+FF"]
    tasks = ["successor", "pair_lookup", "grandparent"]
    x = np.arange(3); width = .25
    plt.figure(figsize=(7.6, 3.6))
    for i, task in enumerate(tasks):
        values = [next(r for r in summaries if r["task"] == task and r["topology"] == topology)["accuracy"] for topology in ("sa_only", "ff_only", "sa_ff")]
        plt.bar(x + (i-1)*width, values, width, label=task)
    plt.xticks(x, labels); plt.ylabel("Exact legal-domain accuracy"); plt.ylim(0, 1.08); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(root / "manual_sa_vs_ff.png", dpi=180); plt.close()

    rows = raw_by_task["grandparent"]["sa_only"]
    q = [r["sequence"].split()[-1] for r in rows]
    first = [r["layer1_parent_probability"] for r in rows]
    second = [r["layer2_grandparent_probability"] for r in rows]
    x = np.arange(len(q));
    plt.figure(figsize=(5.8, 3.4)); plt.bar(x-.18, first, .36, label="layer 1 parent"); plt.bar(x+.18, second, .36, label="layer 2 grandparent")
    plt.xticks(x, q); plt.ylim(0, 1.05); plt.ylabel("Attention probability"); plt.legend(); plt.tight_layout()
    plt.savefig(root / "manual_two_hop_attention.png", dpi=180); plt.close()


def main(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    builders = {
        "successor": (build_successor_variants, eval_successor),
        "pair_lookup": (build_lookup_variants, eval_lookup),
        "grandparent": (build_grandparent_variants, eval_grandparent),
    }
    summaries, architectures, necessity = [], [], []
    raw_by_task = {}
    for task, (builder, evaluator) in builders.items():
        task_root = output / task; task_root.mkdir(exist_ok=True)
        (task_root / "construction.md").write_text(construction_text(task))
        raw_by_task[task] = {}
        for topology, model in builder().items():
            variant_root = task_root / topology
            rows = evaluator(model); raw_by_task[task][topology] = rows
            write_rows(variant_root / "legal_domain_results.csv", rows)
            write_model(variant_root, model)
            tokens, target, prediction, canonical_margin, trace = canonical(task, model)
            write_trace(variant_root, trace)
            write_rows(variant_root / "canonical_example.csv", [{
                "tokens": " ".join(tokens), "target": target, "prediction": prediction,
                "correct": target == prediction, "logit_margin": canonical_margin,
            }])
            accuracy = sum(bool(r["correct"]) for r in rows) / len(rows)
            expected = EXPECTED[task][topology]
            passed = accuracy == 1.0
            summaries.append({
                "task": task, "topology": topology, "legal_cases": len(rows), "correct_cases": int(accuracy*len(rows)),
                "accuracy": accuracy, "expected_full_pass": expected, "observed_full_pass": passed,
                "expectation_met": passed == expected, "minimum_logit_margin": min(float(r["logit_margin"]) for r in rows),
                "claim_scope": "fixed_weight_constructive_representability_only",
            })
            architectures.append({
                "task": task, "topology": topology, "controller": "M0/C0", "data_location": "weights" if task == "successor" else "context",
                "tool": "none", "layers": len(model.layers), "heads": 1 if any(layer.use_sa for layer in model.layers) else 0,
                "d_model": model.embeddings.shape[1], "d_head": model.layers[0].attention.W_Q.shape[1],
                "layernorm": model.layernorm, "residual": model.residual, "finite_softmax": True,
            })
        primary = {"successor": "ff_only", "pair_lookup": "sa_only", "grandparent": "sa_only"}[task]
        necessity.append({
            "task": task, "primary_topology": primary,
            "sa_needed_under_encoding": task != "successor", "ff_needed_under_encoding": task == "successor",
            "recurrence": False, "external_iteration": False,
            "control_result": ";".join(f"{name}:{'pass' if all(r['correct'] for r in rows) else 'fail'}" for name, rows in raw_by_task[task].items()),
        })
    write_rows(output / "manual_witness_summary.csv", summaries)
    write_rows(output / "manual_architectures.csv", architectures)
    write_rows(output / "manual_component_necessity.csv", necessity)
    plot_results(output, summaries, raw_by_task)
    if not all(bool(r["expectation_met"]) for r in summaries):
        raise RuntimeError("A witness/control outcome violated its declared expectation")
    print(f"Milestone 1 PASS: {sum(r['legal_cases'] for r in summaries)} variant-cases; {len(summaries)} topology cells")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=Path("docs/papers/paper0_1/results/manual_witnesses"))
    main(parser.parse_args().output)
