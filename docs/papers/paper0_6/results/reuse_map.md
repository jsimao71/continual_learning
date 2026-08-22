# Paper 0.6 reuse map

Paper 0.6 extends Paper 0.5 rather than creating a parallel instrumentation stack.

| Paper 0.5 module/artifact | Paper 0.6 use | Extension |
|---|---|---|
| `cl.common.model_adapter.TinyTransformerLM` | Same two causal decoder settings and exact tensor locations | No semantic-specific model fork |
| `cl.common.hooks` | Exact capture parity, residual identities, compact final-position traces | None |
| `cl.analysis.component_contrib` | Signed SA/FFN progress; zero, mean, matched replacement, head, and FFN interventions | Semantic item/family/abstraction labels join after measurement |
| `cl.analysis.attention_motifs` | Row-normalized motif vectors and matched-control specificity | Semantic equivalence-class grouping |
| `cl.common.metrics` | Update geometry, cosine, bootstrap CI, Spearman/RSA, CKA, effective rank, invariance, onset/persistence/re-entry | Hierarchy dispersion and neighbor recovery in `cl.analysis.hierarchy_geometry` |
| `cl.common.artifacts` | Atomic JSON, JSONL/CSV, stable hashes, complete run metadata | Hierarchy version/hash and Paper 0.5-control hash |
| Paper 0.5 controlled atlas | Frequency/entropy schema and naming discipline | Balanced semantic corpus supplies true local `training_frequency`; the transferable control field remains `reference_corpus_frequency` |
| Paper 0.5 checkpoints | Establish checkpoint conventions and E8 availability | Paper 0.6 records matching 0/40/100/160 semantic-training checkpoints |

Semantic-specific code is limited to hierarchy construction, controlled templates, hierarchy geometry, and the Paper 0.6 runner under `src/cl/`.
