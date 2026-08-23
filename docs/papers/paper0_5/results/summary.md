# Paper 0.5 controlled experiment summary

## What was run

- Two model settings x 2 seeds on the controlled n-gram corpus.
- E1--E6: atlas, SA/FFN diagnostics, zero ablations, motif controls, stored-vs-context mapping, override/repair, and checkpoint dynamics.
- Atlas entries: 21630; component aggregate rows: 40.

## Main pilot observations

- Final training loss range: 2.4357--2.5646.
- Stored/context aggregate mean causal drop across SA cells: 0.3029; across FFN cells: 0.6525.
- Mean attention motif specificity over matched controls: 0.2282.
- Motif-specificity versus SA causal-drop Spearman correlation: -0.0093 over 18 relation-layer units.
- Exploratory unit-level regression R2: 0.0730 (controlled synthetic factors; not a population estimate).
- Externally defined within-family distributions have mean JS divergence 0.0278 bits versus 0.4938 bits for nonequivalent controls (top-5 overlap 0.878 versus 0.645).
- Layer-0 SA equivalent replacement changes the output by 0.0277 JS bits, versus 0.4358 for nonequivalent replacement; the associated target-probability changes are +0.0031 and -0.3520.
- Entropy is not monotone: mean pre-SA/post-SA/post-block entropy is 0.069/0.046/0.083 bits for deterministic familiar probes and 0.848/1.158/1.959 for context-introduced probes.

## Interpretation and failures

These are controlled local-model results, not evidence about pretrained-model training frequency. Corpus frequency is known here because the model is trained locally. Signed logit-lens progress is retained as diagnostic only; component results include zero, matched-mean, matched-replacement, selective-head, and FFN-layer interventions, but generic component importance remains a competing explanation. Motif similarity is compared with matched controls and shows no relation to SA causal contribution. Pretrained replication and path patching remain required before a strong mechanistic claim.

## Exact artifacts

- `raw/atlas.jsonl`
- `raw/components.jsonl`
- `raw/motifs.jsonl`
- `raw/override.jsonl`
- `raw/entropy_trajectories.jsonl`
- `raw/equivalence_families.jsonl`
- `raw/equivalence_patches.jsonl`
- `tables/component_summary.csv`
- `tables/stored_vs_context.csv`
- `tables/regression.csv`
- `tables/motif_causal_association.csv`
- `tables/training_dynamics.csv`
- `tables/predictive_equivalence.csv`
- `tables/equivalence_patching.csv`
- `tables/common_update_components.csv`
- `figures/*.png`

## Next falsifiable question

Do the SA/FFN causal fractions and override trajectories replicate in a small pretrained checkpoint series after matching reference-corpus frequency, entropy, tokenization, and generic layer importance?
