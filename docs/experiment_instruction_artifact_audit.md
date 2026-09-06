# Instruction-to-artifact experiment audit

Audit date: 2026-09-06. The machine-readable companion is
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

The three already identified active/deferred experiments remain genuine requirements:

1. Paper 0.9's bounded staged A--E resource study. Its v3 artifact is plan-only.
2. Paper 1's larger unchanged natural bridge-versus-base replication. Its frozen
   three-seed design is committed but unrun.
3. Paper 0.8's learned D5 structured-rule experiment. Exact generator/posterior
   validation is complete, but there is no model result.

One additional required experimental gap is visible:

- **Paper 0.6 needs a cue-free compositional predicate frontier.** The completed v6
  `isAncestor` stress run is valid as a presentation/lookup robustness frontier, but
  its own manuscript and instruction completion criteria note that the positive
  endpoint is present among path records. Consequently its measured `d_max=16` cannot
  establish iterative ancestor composition or `L_min(d)`. A corrected generator must
  remove that endpoint-membership cue before those required questions can be answered.

This is separate from rerunning the already complete v6 grid. It should be designed as
a narrow corrected frontier, not a broad rescue sweep.

## Important non-gaps

- Paper 0.7 deeper P3--P6/F3--F9 and mechanism work is stopped by the failed F2
  three-seed gate; proceeding would violate its competence ordering.
- Paper 0.85's later mechanism and architecture branches are stopped because the v2
  data interventions did not move the recurrence frontier beyond three.
- Paper 0.6 S2 mechanism work is blocked by failed held-out competence.
- Paper 1 persistent consolidation, override-after-learning, and rollback studies are
  conditional on the larger natural gate and remain forbidden before it passes.
- Paper 0.5 E7, extra Paper 0.1 M2--M5 witnesses, Paper 0.8 D3-B/C, and pretrained
  appendices are optional or aspirational rather than missing requirements.

## Audit boundary

The matrix treats tracked manifests, raw/aggregate tables, built papers, and scoped
commits as evidence. It does not treat an instruction document, a plan-only manifest,
a validated generator, or prose describing a future run as an empirical result.
Conversely, it does not label gate-dependent downstream work as missing after the gate
has legitimately failed.
