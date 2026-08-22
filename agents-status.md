# Codex Handoff Status

Last updated: 2026-08-22

Repository: `D:\git\rd\continual_learning`

Remote: `https://github.com/jsimao71/continual_learning.git`

Branch: `main`

## Purpose

This file is the continuation checkpoint for another Codex instance or machine. Read it together with `README.md`, `docs/AGENTS.md`, `docs/AGENTS0_5.md`, and `docs/AGENTS0_6.md` before changing experiments or claims. The agent instruction files are authoritative when this summary is incomplete.

## Current research conclusion

The four-paper series now tells one evidence-backed, deliberately narrow story:

1. Recurrent internal structure is observable, but observation is not causal evidence.
2. Component responsibility changes with the transformation, context, intervention, and requested abstraction level.
3. Attention-motif recurrence and representational geometry can disagree with causal utility.
4. Weak bridge structure can improve frozen sparse selection when the mechanism is known.
5. Persistent learning is justified only after held-out natural-task causal utility or a matched-budget quality/materialization frontier improvement.

Do **not** summarize the result as "motifs become symbols" or as a monotone abstraction ladder. Neural Structural Consolidation remains a falsifiable research program; general structural consolidation is not yet established.

## Completed and pushed work

The following commits are on `origin/main`, in execution order:

| Commit | Work |
|---|---|
| `631c806` | Extended Paper 0 position/research plan. |
| `a6032aa` | Implemented Paper 0.5 n-gram experiments, artifacts, tests, paper update, and PDF. |
| `f5d6f81` | Implemented Paper 0.6 abstraction-mapping experiments, artifacts, tests, paper update, and PDF. |
| `ae562ed` | Implemented Paper 1 frozen structural-control experiments, artifacts, tests, paper update, and PDF. |
| `04fa28c` | Aligned all four papers with the measured results and unified the series-level interpretation. |

The commit that adds this handoff file follows those commits.

## Paper and artifact status

### Paper 0: position and evidence ladder

- Source: `docs/papers/paper0/position_plan.tex`
- Built paper: `docs/papers/paper0/position_plan.pdf`
- Status: updated from a forward-looking plan into a position paper plus evidence update.
- Current decision: the controlled stages are informative, but online consolidation remains blocked pending a natural candidate-level intervention.

### Paper 0.5: common n-grams as causal probes

- Source: `docs/papers/paper0_5/paper_0_5.tex`
- Built paper: `docs/papers/paper0_5/paper_0_5_preview.pdf`
- Result summary: `docs/papers/paper0_5/results/summary.md`
- Manifest: `docs/papers/paper0_5/results/manifest.json`
- Machine-readable rows, generated tables, and figures: `docs/papers/paper0_5/results/`

Key measured results:

- Atlas: 21,630 prefix rows; final component analysis: 1,920 rows.
- Familiar low-entropy zero ablation: FFN `0.478` vs SA `0.301` nats.
- Context-introduced mappings: FFN `0.827` vs SA `0.305`.
- Override/repair: SA `1.497` vs FFN `0.261`.
- Attention-motif specificity: `0.2282`.
- Motif specificity vs SA causal contribution: Spearman `-0.0093`.
- Exploratory controlled regression: `R^2 = 0.0730`.

Interpretation: n-gram computation is distributed and transformation-dependent. Stable attention motifs are not a sufficient causal or learning signal. E7 was correctly stopped.

Reproduction command:

```powershell
python -m cl.experiments.paper05_ngram --steps 160 --checkpoints 0 40 100 160 --seeds 11,23 --device cpu
```

### Paper 0.6: abstraction-layer mapping

- Source: `docs/papers/paper0_6/paper0_6.tex`
- Built paper: `docs/papers/paper0_6/paper0_6.pdf`
- Result summary: `docs/papers/paper0_6/results/summary.md`
- Manifest and reuse map: `docs/papers/paper0_6/results/manifest.json`, `docs/papers/paper0_6/results/reuse_map.md`
- Machine-readable rows, generated tables, and figures: `docs/papers/paper0_6/results/`

Key measured results:

- Mean normalized hierarchy separation: `0.1100`.
- Hierarchy-distance/activation-distance RSA: `0.1941`.
- Tree-neighbor recovery: `0.4774` vs permuted-parent control `0.2535`.
- Cross-template cosine: `0.9597`.
- Parent query zero ablation: FFN `1.117` vs SA `0.211` nats.
- Root query zero ablation: SA `0.987` vs FFN `0.671`.
- Semantic attention-motif specificity: `-0.0121`.

Interpretation: weak hierarchy geometry survives the matched control, but it is task-conditioned, distributed, present early, and not evidence for a literal stored taxonomy or monotone layer-to-abstraction ladder.

Reproduction command:

```powershell
python -m cl.experiments.paper06_abstraction --steps 160 --checkpoints 0 40 100 160 --seeds 11,23 --device cpu
```

### Paper 1: transient structure as sparse control

- Source: `docs/papers/paper1/paper1.tex`
- Built paper: `docs/papers/paper1/paper1.pdf`
- Result summary: `docs/papers/paper1/results/summary.md`
- Manifest and reuse map: `docs/papers/paper1/results/manifest.json`, `docs/papers/paper1/results/reuse_map.md`
- Machine-readable rows, generated tables, and figures: `docs/papers/paper1/results/`

Key measured results:

- Controlled suite: five seeds and 600 held-out examples.
- At four chunks, base quality/path completion: `0.2971 / 0.0683`.
- Bridge-preserving selection: `0.6268 / 0.4800`.
- Combined structural selector: `1.0000 / 1.0000` in the deliberately diagnostic generator.
- Controlled held-out causal-utility Spearman: `0.4860` to `0.5281` after structural features.
- Frozen-Qwen audit: 84 identity-disjoint 2WikiMultiHopQA/MuSiQue examples.
- Natural audit Spearman: `-0.0273` to `-0.0048`; corresponding `R^2` values are negative.
- Fixed-power and entropy-adaptive monotone sharpening reproduce exact base top-k membership.

Interpretation: explicit bridge preservation can shift a controlled matched-budget frontier. The perfect combined-selector result is a generator-specific diagnostic ceiling, not an external-validity claim. The retained pretrained summaries are observationally null after surface controls and do not include candidate removal or task-generation outcomes.

Reproduction command:

```powershell
python -m cl.experiments.paper1_structural_control
```

## Implementation map

- Shared infrastructure: `src/cl/common/`
  - provenance/artifacts: `artifacts.py`
  - hooks and trace capture: `hooks.py`
  - shared metrics: `metrics.py`
  - inspectable model adapter: `model_adapter.py`
- Paper 0.5 data and controls: `src/cl/ngram/`
- Paper 0.6 hierarchy construction: `src/cl/semantic/`
- Shared analyses: `src/cl/analysis/`
- Paper 1 NSC feature/graph/selectors: `src/cl/nsc/`
- Reproducible entry points: `src/cl/experiments/`
- Tests: `tests/`

Keep experiment-specific code under `src/cl/`. Put only reusable, architecture-independent infrastructure in `src/cl/common/`. Reuse existing modules instead of creating parallel loaders, hooks, metrics, or artifact writers.

## Current decision gate and exact next work

Paper 2 prototype/adaptor consolidation is **blocked**. Do not implement an online learner merely to continue the sequence.

The next decisive experiment is a natural candidate-level frozen-Qwen study on HotpotQA and QASPER:

1. Regenerate candidate-level model features rather than relying on the retained example-level graph summaries.
2. Preserve identity-disjoint validation/test splits and record exact model/tokenizer revision, dataset IDs, seed, resolved configuration, package versions, and candidate-cache hash.
3. Retain candidate IDs, surface scores, lexical/semantic controls, compact layer/head graph summaries, and actual task/evidence labels.
4. Perform real matched-budget candidate selection and K/V materialization; do not report accounting-equivalent bytes as measured native latency.
5. Compare B0--B3 and S1--S5 using identical candidate sets and exact token/byte budgets.
6. Measure task quality, evidence recall, complete-path recovery, materialized tokens/bytes, selection overhead, end-to-end latency, and peak memory.
7. On a feasible diagnostic subset, compute candidate removal utility
   `U_i = Q(C) - Q(C \ {c_i})`.
8. Fit identity-disjoint held-out models using surface controls first, then surface plus structural features. Retain calibration/correlation and negative `R^2`; the main decision is the quality/materialization frontier.
9. Run both HotpotQA and QASPER. A result limited to the synthetic bridge generator is not sufficient.

The gate passes only if at least one result is reproducible on natural held-out data:

- structural features predict candidate causal utility beyond base and lexical/semantic controls; or
- a structural selector improves the matched-budget task-quality/materialization Pareto frontier.

If both fail, record the null result and keep Paper 2 stopped. Do not respond by adding an unconstrained learner or tuning on the test split.

## Scientific guardrails

- Keep observation, intervention, and learning separate.
- Attention weights, motifs, probe accuracy, RSA, and geometry are not mechanistic proof by themselves.
- Keep signed effects; negative updates and override/repair cases are substantive evidence.
- Control generic layer importance, lexical/token frequency, tokenization, n-gram predictability, template, and context sensitivity.
- Use the scientific unit (n-gram, semantic item/branch, or example), not correlated token occurrences.
- Compare selectors at exact candidate and materialization budgets.
- Weak edges may be bridges; low magnitude is not automatically noise.
- Preserve failed hypotheses and null results in artifacts and prose.
- Do not claim standard training ignores attention; the narrower claim concerns explicit accumulation of recurrent sample-level structure during deployment.

## Validation state

After the cross-paper narrative update:

```text
python -m pytest -q       -> 17 passed
python -m compileall -q src tests -> passed
```

All four PDFs built successfully and all 33 rendered pages were visually inspected. No clipping, overlap, malformed glyphs, or unreadable figures/tables were found. Rebuild and visually inspect affected PDFs whenever their TeX sources change.

## Workspace hygiene

At the time this handoff was written, this machine had two unrelated local changes:

- `.gitignore` modified by the user;
- `tmp/` untracked build/render artifacts.

They are not part of the research commits and must not be staged, reverted, deleted, or moved as part of paper work. A fresh clone on another machine will not contain those local changes. Always inspect `git status --short --branch` before editing and stage files explicitly.

## Startup checklist for the next Codex instance

1. Clone/pull `origin/main` and confirm the expected handoff commit is present.
2. Read this file, `README.md`, and all three `docs/AGENTS*.md` instruction files completely.
3. Inspect `git status --short --branch`; preserve unrelated changes.
4. Read the three result summaries and manifests before interpreting or regenerating experiments.
5. Run `python -m pytest -q` before starting a new experiment family.
6. Continue with the natural candidate-level gate above; do not start Paper 2 first.
7. Update result summaries, decisions, paper text, this handoff file, and generated PDFs from committed machine-readable artifacts.
8. Build, render, and inspect changed papers; commit and push coherent milestones.
