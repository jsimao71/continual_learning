# Paper 1 natural replication v2: post-run manuscript integration

Use this checklist only after the complete-only analyzer succeeds. It intentionally
contains no result placeholders and authorizes no persistent-learning experiment by
itself.

## Authoritative inputs

Run:

```bash
PYTHONPATH=src python -m cl.experiments.paper1_natural_replication_analyze \
  --config configs/paper1/natural_replication_v2.json \
  --root docs/papers/paper1/results/natural_replication_v2 \
  --output docs/papers/paper1/results/natural_replication_v2_analysis
```

The analyzer must reject incomplete seeds, changed preregistration, raw-count drift,
identity leakage, or an incomplete selector grid. Before using any number, require:

- `analysis_manifest.json`: `complete_only=true`, `completed_seeds=3`, and three
  successful seed audits;
- `gate_decision.json`: the only authoritative stop/go decision;
- `tables/identity_overlap_audit.csv`: inspect per-dataset sampling-seed overlap and
  unique-identity totals;
- exact config hash agreement with `configs/paper1/natural_replication_v2.json`;
- 24 validation and 24 test identities per dataset and sampling seed, 12 candidates,
  all 10 selectors, and all three native K/V budgets (64/128/192 tokens).

The primary sampling unit for inference is the unique held-out identity. If an
identity recurs across dataset-sampling seeds, average its repeated measurements
before the paired bootstrap; never count it as an independent replicate.

## Output-to-paper map

| Analyzer output | Manuscript destination | Permitted claim |
|---|---|---|
| `tables/natural_replication_results.tex` | Replace or follow the current natural-results table in `paper1.tex` | Three-seed bridge-minus-base paired answer-logprob deltas and identity-bootstrap intervals only. |
| `figures/replication_paired_frontier.png` | Natural Candidate-Level Intervention / Results | The full preregistered 64/128/192-token paired frontier; do not show only a favorable budget. |
| `tables/quality_frontier.csv` | Results and, if useful, a generated appendix table | Absolute answer-logprob means for all selectors and budgets; comparisons remain matched-budget. |
| `tables/paired_unique_identity.csv` | Results prose and replication table provenance | Unique-identity effect sizes, intervals, and sample counts. This is the source for frontier-gate language. |
| `tables/causal_incremental_r2.csv` | Candidate-removal paragraph | Seedwise incremental held-out R2 summary and interval. Do not substitute rank correlation for the registered R2 gate. |
| `tables/identity_overlap_audit.csv` | Methods / Reproducibility | Actual unique identities and any cross-seed reuse after validation/test separation. |
| `tables/natural_replication_macros.tex` | Optional manuscript input | Gate and unique-identity counts only; verify macros against the decision JSON before build. |
| `gate_decision.json` | Decision Gate and Conclusion | `persistent_learning_gate` controls stop/go wording. No other table may override it. |
| `analysis_manifest.json` | Reproducibility and provenance | Completion, hashes, pooled-row counts, and source-audit provenance. |

## Required manuscript edits after analysis

1. **Abstract.** Retain the original one-seed diagnostic as prior evidence, then add
   the replication outcome. State three dataset-sampling seeds and unique-identity
   inference. Use “passes” only if `persistent_learning_gate=1`; otherwise state the
   registered null and continued stop decision.
2. **Natural Candidate-Level Intervention.** Preserve the unchanged protocol:
   Qwen3-0.6B at revision `c1899de289a04d12100db370d81485cdf75e47ca`, 12
   candidates, native K/V materialization, the same selectors/features/endpoints, and
   64/128/192-token budgets. Update sample counts from the overlap audit rather than
   multiplying nominal seed counts.
3. **Results.** Keep the original 8+8 one-seed findings identifiable as the pilot.
   Insert the generated replication table and paired-frontier figure. Report all six
   dataset-by-budget bridge comparisons, including unfavorable or inconclusive cells.
4. **Removal utility.** Report surface-controls versus surface-plus-structure
   incremental held-out R2 from `causal_incremental_r2.csv`, including seed
   consistency. Candidate removal is a causal diagnostic; evidence recall is not a
   replacement endpoint.
5. **Series-Level Interpretation.** Do not import Paper 0.6 v7 geometry as a feature:
   its cue-free predicate acquisition gate failed. Keep Paper 0.5 functional features
   labeled proxies wherever candidate-conditioned distributions were not retained.
6. **Decision Gate and Falsification.** Apply the registered disjunction exactly:
   frontier evidence must pass on both datasets at one or more preregistered budgets,
   or causal incremental R2 must be positive on both datasets with at least two of
   three positive seeds. If neither branch passes, persistent prototypes/adapters,
   override tests, and rollback experiments remain gated stops.
7. **Limitations and Conclusion.** Replace the pilot-only “one seed/eight held-out”
   limitation with observed unique-identity counts, while retaining one-model,
   fixed-candidate, evidence-label, answer-likelihood, and research-system limits.

## Claim-discipline checks

- Distinguish the controlled weak-bridge mechanism result, the original natural pilot,
  and the larger replication in every summary paragraph.
- A positive mean with an interval crossing zero is directional, not reproducible.
- A gate pass establishes bounded frozen-selector utility, not continual learning,
  symbolic rule discovery, or equivalence-aware consolidation.
- Report native K/V tokens/bytes and quality together; do not describe a quality gain
  without its matched materialization budget.
- Keep oracle ordering diagnostic and exclude it from learned/operational claims.
- Preserve negative combined-selector and removal-prediction results if reproduced.

## Final validation

Run the analyzer test and manuscript build, then inspect the generated table and every
page containing the new table/figure:

```bash
PYTHONPATH=src pytest -q tests/test_paper1_natural_replication.py \
  tests/test_paper1_natural_replication_analyze.py
```

Confirm that all prose numbers occur in an authoritative CSV/JSON or generated TeX
table, `git diff --check` passes, the PDF has no undefined references or overflow
warnings, and the tracked PDF was rebuilt from the edited source.
