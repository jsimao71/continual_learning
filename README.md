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

The runnable commands and exact artifact paths are maintained in each paper's result summary as implementations are added.
