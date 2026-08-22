# Continual Learning from Transformer Computation

This repository studies whether recurrent structures observed inside Transformer inference can be measured causally and, only after passing explicit evidence gates, consolidated into persistent neural state.

## Research sequence

- **Paper 0** defines Neural Structural Consolidation (NSC), its evidence hierarchy, persistence timescales, controls, and stop/go criteria.
- **Paper 0.5** uses token n-gram continuation statistics to map recurrent prediction relations across self-attention, feed-forward layers, residual depth, interventions, and training checkpoints.
- **Paper 0.6** maps natural and synthetic semantic hierarchies across depth while controlling lexical and Paper 0.5 surface statistics.
- **Paper 1** tests transient graph statistics as frozen-model sparse-control signals before any online learner is introduced.

The project separates four evidential levels: observation, held-out prediction, causal intervention, and consolidation. Attention weights, probe accuracy, logit-lens alignment, or geometry alone are not treated as mechanistic proof.

## Layout

```text
docs/
  AGENTS*.md          experiment-specific scientific constraints
  papers/             LaTeX sources, PDFs, and result summaries
src/cl/
  common/             shared artifacts, metrics, hooks, and model adapters
  ngram/              Paper 0.5 atlas and controls
  semantic/           Paper 0.6 hierarchy and controls
  analysis/           shared mechanistic analyses
  experiments/        reproducible experiment entry points
tests/                unit and integration tests
```

Paper-specific code lives under `src/cl/`; only reusable, architecture-independent utilities belong in `src/cl/common/`. Machine-readable artifacts are the source of paper tables and figures.

## Reproducibility contract

Each run records the git commit, command, resolved configuration, model/tokenizer and data identifiers, seed, device/dtype, package versions, timestamp, and atlas or hierarchy hash. Statistical aggregation uses the scientific unit (for example, n-gram or semantic branch), not correlated token occurrences.

Paper 0.5 controlled run:

```powershell
python -m cl.experiments.paper05_ngram --steps 160 --checkpoints 0 40 100 160 --seeds 11,23 --device cpu
```

Its exact artifacts and retained negative results are documented in `docs/papers/paper0_5/results/summary.md`. Paper 0.6 reuses the same model adapter, trace locations, interventions, metric ontology, and artifact schema.

Paper 0.6 controlled run:

```powershell
python -m cl.experiments.paper06_abstraction --steps 160 --checkpoints 0 40 100 160 --seeds 11,23 --device cpu
```

The semantic results, reuse map, hierarchy controls, and artifact manifest are under `docs/papers/paper0_6/results/`.

Paper 1 frozen structural-control pilot:

```powershell
python -m cl.experiments.paper1_structural_control
```

This runs the matched-budget controlled bridge suite and the residualized frozen-Qwen graph audit. Results and the explicit no-go decision for online consolidation are under `docs/papers/paper1/results/`.

Paper 1 natural candidate-level gate:

```bash
PYTHONPATH=src python -m cl.experiments.paper1_natural_gate
```

This evaluates frozen Qwen3-0.6B on identity-disjoint HotpotQA and QASPER candidates with actual native K/V prefill, exact token budgets, answer-likelihood frontiers, and a diagnostic removal-utility subset. Its committed artifacts are under `docs/papers/paper1/results/natural/`. Favorable bridge means remain statistically unresolved, so Paper 2 is still blocked.
