"""Generate Milestone-2 autoregressive fixed-weight witness artifacts."""
from __future__ import annotations

import argparse, csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from .artifacts import write_model, write_rows, write_trace
from .m1_recurrence import ALPHA, generate
from .m1_root import build_root_variants, evaluate as eval_root
from .m1_implication import build_implication_variants, evaluate as eval_implication


EXPECTED = {task: {"sa_only": True, "ff_only": False, "sa_ff": True} for task in ("root_recurrence", "unary_chain_recurrence")}


def _read(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_union(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    write_rows(path, [{field: row.get(field, "") for field in fields} for row in rows])


def _construction(task: str) -> str:
    semantics = "parent facts" if task == "root_recurrence" else "unary transition records"
    domain = "all 6 orderings of three non-root nodes x all 4 starts (24 trajectories; root depths 0--3)" if task == "root_recurrence" else "all 24 orderings of four symbols x all 4 starts (96 trajectories; chain lengths 1--4)"
    return f"""# {task.replace('_', ' ').title()} construction

This is a deterministic fixed-weight representability witness; no weight is trained.  A single local transition circuit is called once per generated token by an external autoregressive model loop (not a tool loop).  Atomic `MAP_X_Y` tokens store {semantics} in separate key/value subspaces, while `STATE_X` tokens place the latest generated symbol in a query-only subspace.  The model has `L=1`, `H=1`, `d_model=15`, `d_head=5`, identity residuals, zero positions, finite causal softmax, no LayerNorm/dropout, and a zero FF update.

At every step, `W_Q` reads the current state, `W_K` reads map keys, `W_V` reads map values, and `W_O` writes the selected successor into the output subspace.  The predicted symbol is appended as the next `STATE` token and the same matrices are reused.  The selector score is finite (`alpha={ALPHA:g}`); as prior generated states accumulate they contribute zero-score leakage.  At the longest declared step, the desired probability is `exp(16)/(exp(16)+7)=0.9999992122543974`, and the minimum signed target margin is `0.9999990997193113`.  Each step records both the winner--runner-up margin and target-minus-best-other margin rather than treating attention as hard.

The exhaustive declared domain is {domain}.  `STOP` is an explicit terminal value.  SA-only and SA+FF must reproduce every complete trajectory by correct argmax with positive signed target margin; FF-only cannot read the contextual map and must fail the full-domain gate.  Reversed and fixed-shuffle fact orders are tracked controls.  Excluded domains include cycles, missing, duplicated, or conflicting records, larger vocabularies, non-atomic serialization, and lengths beyond the declared positional budget.  These controls are encoding-relative.  The witness establishes bounded tested-depth autoregressive reuse, not minimality, unbounded closure, robustness outside the domain, or SGD acquisition.
"""


def _canonical(task, model, vocab):
    if task == "root_recurrence":
        mapping, start, expected = {"R":"STOP", "A":"R", "B":"A", "C":"B"}, "C", ["B","A","R","STOP"]
    else:
        mapping, start, expected = {"A":"B", "B":"C", "C":"D", "D":"STOP"}, "A", ["B","C","D","STOP"]
    generated, steps = generate(model, vocab, mapping, start, "STOP")
    return mapping, start, expected, generated, steps


def main(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    tasks = {
        "root_recurrence": (build_root_variants, eval_root),
        "unary_chain_recurrence": (build_implication_variants, eval_implication),
    }
    summaries, architectures, necessity = [], [], []
    depth_plot = {}
    for task, (builder, evaluator) in tasks.items():
        task_root = output / task; task_root.mkdir(exist_ok=True)
        (task_root / "construction.md").write_text(_construction(task))
        depth_plot[task] = {}
        for topology, (model, vocab) in builder().items():
            variant_root = task_root / topology
            rows, step_rows = evaluator(model, vocab)
            write_rows(variant_root / "legal_domain_results.csv", rows)
            write_rows(variant_root / "generation_steps.csv", step_rows)
            write_model(variant_root, model)
            mapping, start, expected, generated, steps = _canonical(task, model, vocab)
            write_rows(variant_root / "canonical_example.csv", [{
                "mapping": ";".join(f"{a}->{b}" for a,b in mapping.items()), "start": start,
                "target_trajectory": " ".join(expected), "predicted_trajectory": " ".join(generated),
                "correct": generated == expected, "generation_steps": len(generated),
            }])
            canonical_rows = []
            for step in steps:
                step_root = variant_root / "canonical_steps" / f"step_{step['step']:02d}"
                write_trace(step_root, step["trace"])
                canonical_rows.append({k:v for k,v in step.items() if k != "trace"})
            write_rows(variant_root / "canonical_generation_steps.csv", canonical_rows)
            correct = sum(r["correct"] for r in rows); passed = correct == len(rows)
            expected_pass = EXPECTED[task][topology]
            summaries.append({
                "task": task, "topology": topology, "legal_cases": len(rows), "correct_cases": correct,
                "accuracy": correct/len(rows), "expected_full_pass": expected_pass, "observed_full_pass": passed,
                "expectation_met": passed == expected_pass,
                "minimum_logit_margin": min(float(s["winner_runner_up_margin"]) for s in step_rows),
                "minimum_signed_target_margin": min(float(s["signed_target_margin"]) for s in step_rows),
                "claim_scope": "fixed_weight_bounded_autoregressive_representability_only",
            })
            architectures.append({
                "task": task, "topology": topology, "controller": "M1 witness / C1 protocol", "machine_stage": "M1",
                "controller_protocol": "C1_autoregressive", "data_location": "context", "tool": "none",
                "layers": 1, "heads": 1 if topology != "ff_only" else 0, "d_model": model.embeddings.shape[1],
                "d_head": model.layers[0].attention.W_Q.shape[1], "layernorm": model.layernorm,
                "residual": model.residual, "finite_softmax": True,
                "execution_protocol": "autoregressive_model_loop", "external_tool_iteration": False,
            })
            depth_plot[task][topology] = rows
        # Context-order controls use every legal case under reversed and one
        # deterministic shuffled ordering; semantic predictions must be invariant.
        serialization_rows = []
        primary_model, primary_vocab = builder()["sa_only"]
        from .m1_root import legal_cases as root_cases
        from .m1_implication import legal_cases as chain_cases
        cases = list(root_cases() if task == "root_recurrence" else chain_cases())
        for case_id, (mapping, start, expected) in enumerate(cases):
            keys = tuple(mapping)
            orders = {"reversed": tuple(reversed(keys)), "fixed_shuffle": keys[1::2] + keys[0::2]}
            for condition, order in orders.items():
                generated, steps = generate(primary_model, primary_vocab, mapping, start, "STOP", context_order=order)
                serialization_rows.append({"case_id": case_id, "condition": condition, "start": start,
                    "context_order": ";".join(order), "target_trajectory": " ".join(expected),
                    "predicted_trajectory": " ".join(generated), "correct": generated == expected,
                    "minimum_signed_target_margin": min(s["signed_target_margin"] for s in steps)})
        assert all(r["correct"] for r in serialization_rows)
        write_rows(task_root / "serialization_controls.csv", serialization_rows)
        necessity.append({
            "task": task, "primary_topology": "sa_only", "sa_needed_under_encoding": True,
            "ff_needed_under_encoding": False, "recurrence": True, "external_iteration": False,
            "control_result": "sa_only:pass;ff_only:fail;sa_ff:pass",
        })

    write_rows(output / "manual_autoregressive_summary.csv", summaries)
    write_rows(output / "manual_autoregressive_architectures.csv", architectures)
    write_rows(output / "manual_autoregressive_component_necessity.csv", necessity)
    # Preserve Milestone-1 rows while making the required cross-milestone tables cumulative.
    _write_union(output / "manual_witness_summary.csv", _read(output / "manual_witness_summary.csv")[:10] + summaries)
    _write_union(output / "manual_architectures.csv", _read(output / "manual_architectures.csv")[:10] + architectures)
    _write_union(output / "manual_component_necessity.csv", _read(output / "manual_component_necessity.csv")[:3] + necessity)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=True)
    for ax, task in zip(axes, tasks):
        primary = depth_plot[task]["sa_only"]
        depths = sorted({int(r["steps_expected"]) for r in primary})
        for topology, marker, linestyle in (("sa_only", "o", "-"), ("ff_only", "s", "--"), ("sa_ff", "^", ":")):
            rows = depth_plot[task][topology]
            values = [np.mean([bool(r["correct"]) for r in rows if int(r["steps_expected"]) == depth]) for depth in depths]
            ax.plot(depths, values, marker=marker, linestyle=linestyle, label=topology.replace("_", " "))
        ax.set_title(task.replace("_recurrence", "").replace("_", " ")); ax.set_xlabel("Generated transition steps")
        ax.set_xticks(depths); ax.set_ylim(-.04, 1.04); ax.grid(alpha=.25)
    axes[0].set_ylabel("Argmax-correct trajectory accuracy"); axes[1].legend(fontsize=8)
    fig.suptitle("One local transition circuit reused across tested depths")
    fig.tight_layout(); fig.savefig(output / "manual_autoregressive_depth.png", dpi=180); plt.close(fig)

    if not all(r["expectation_met"] for r in summaries):
        raise RuntimeError("A Milestone-2 witness/control violated its expected outcome")
    print(f"Milestone 2 PASS: {sum(r['legal_cases'] for r in summaries)} variant-trajectories; {len(summaries)} topology cells")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=Path("docs/papers/paper0_1/results/manual_witnesses")); main(parser.parse_args().output)
