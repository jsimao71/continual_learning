# Instruction-to-artifact experiment audit

Audit date: 2026-09-07. The machine-readable companion is
[`experiment_instruction_artifact_audit.csv`](experiment_instruction_artifact_audit.csv).
It covers every repository `AGENTS*.md` plus `agents-status.md`. Rows consolidate
overlapping instructions when later patches refine an earlier requirement. `Complete`
means that an authoritative tracked artifact supports the bounded requirement; it does
not imply that every suggested future extension was run.

## Status vocabulary

- `complete`: required bounded experiment and artifact exist.
- `complete_with_scope_limit` / `complete_gated_null`: execution is complete, with a
  limitation or null that constrains the claim.
- `gated-stop`: a registered prerequisite failed, so downstream work is not currently
  required and must not be silently promoted.
- `pending`: required planned work with no completed empirical artifact.
- `aspirational_optional`: explicitly optional or later research, not a completion gap.
- `optional-future`: status used for those optional/aspirational rows in the CSV.

## Findings

The two experiments that remained active at the original audit boundary are now
complete:

1. **Paper 0.9 staged A--E resource study: completed bounded null.** All five stages
   ran with their registered three-seed comparisons. Depth, width/head count, data
   diversity, and K=4 training exposure did not produce a repeatable M4 procedure;
   the complete analysis and paper are tracked in commit `934585e`.
2. **Paper 1 larger natural replication: completed gated null.** The unchanged
   frozen intervention ran across three sampling seeds, yielding 71 unique HotpotQA
   and 63 unique QASPER test identities. All six bridge-minus-base point estimates
   are positive, but every identity-bootstrap interval crosses zero; incremental
   causal-utility R2 is negative for every seed on both datasets. The authoritative
   decision is `persistent_learning_gate=0`, tracked in commit `b34c49a`.

Paper 0.8's learned D5 structured-rule experiment has since completed as a three-seed
gated null. All 12 planned cells and 51,456 evaluation rows are tracked, but no seed
passes; downstream D5 mechanism work therefore remains stopped.

One additional required experimental gap was visible at audit time and is now closed:

- **Paper 0.6 cue-free compositional predicate frontier: completed gated null.** The completed v6
  `isAncestor` stress run is valid as a presentation/lookup robustness frontier, but
  its own manuscript and instruction completion criteria note that the positive
  endpoint is present among path records. Consequently its measured `d_max=16` cannot
  establish iterative ancestor composition or `L_min(d)`. The corrected v7 generator
  matches candidate membership and endpoint roles across labels. All nine CPU models
  completed, but none of L2/L4/L8 was competent even at trained depth one. The
  compositional acquisition prerequisite fails, so `L_min(d)` is not estimable and
  deeper scaling is a registered gated stop.

This result is separate from the already complete v6 grid and closes the gap with a
narrow corrected frontier rather than a broad rescue sweep.

## Important non-gaps

- Paper 0.7 deeper P3--P6/F3--F9 and mechanism work is stopped by the failed F2
  three-seed gate; proceeding would violate its competence ordering.
- Paper 0.85's later mechanism and architecture branches are stopped because the v2
  data interventions did not move the recurrence frontier beyond three.
- Paper 0.6 S2 mechanism work is blocked by failed held-out competence.
- Paper 1 persistent consolidation, override-after-learning, and rollback studies are
  stopped because the completed larger natural gate did not pass.
- Paper 0.5 E7, extra Paper 0.1 M2--M5 witnesses, Paper 0.8 D3-B/C, and pretrained
  appendices are optional or aspirational rather than missing requirements.

## Audit boundary

The matrix treats tracked manifests, raw/aggregate tables, built papers, and scoped
commits as evidence. It does not treat an instruction document, a plan-only manifest,
a validated generator, or prose describing a future run as an empirical result.
Conversely, it does not label gate-dependent downstream work as missing after the gate
has legitimately failed.
