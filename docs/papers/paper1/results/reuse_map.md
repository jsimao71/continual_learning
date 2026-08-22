# Paper 1 reuse map

## Reused inside this repository

- `cl.common.artifacts`: atomic JSON, JSONL/CSV writing, stable hashes, and run provenance.
- `cl.common.metrics`: example-level bootstrap intervals and Spearman correlation.
- Paper 0.5/0.6 conventions: immutable candidate identities, explicit scientific units, retained negative results, and generated paper tables/figures.

## Reused from PRA by interface or artifact

- The opt-in configuration and compact-trace boundary follow `pdattention`'s routing/materialization separation; no attention implementation was copied or forked.
- The frozen-Qwen audit imports the retained `layerwise_graph_rows.csv` artifact and records its SHA-256. The deliberately untracked 7.8 GB tensor cache is not reconstructed.
- Qwen-equivalent K/V accounting uses the recorded eight K/V heads, native head width 128, K and V tensors, and float16 payload size.

## Paper-specific extensions under `src/cl`

- `cl.nsc`: compact structural features, deterministic graph operations, opt-in selector configuration, B0--B3/S1--S5/O1 selection, and the weak-bridge generator.
- `cl.analysis.pretrained_graph_audit`: identity-disjoint surface-control versus structural prediction on retained pretrained graph summaries.
- `cl.experiments.paper1_structural_control`: five-seed budget sweeps, causal-utility diagnostics, provenance, tables, and figures.

The experiment does not mutate the PRA repository and does not introduce online learning.
