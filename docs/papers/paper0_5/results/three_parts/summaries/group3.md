# Group 3 — Head contributions

One trained four-layer, four-head controlled model supplies 16 complete layer/head cells. Each cell has zero ablation, batch-mean replacement, equivalent-family replacement, and mismatched-family replacement; all 24 within-layer head pairs are also ablated.

Mean target-logprob drops are 0.0316 nats for zero ablation, 0.0108 for mean replacement, 0.0006 for equivalent replacement, and 0.0194 for mismatched replacement. Thus naturally occurring same-family head outputs are close to interchangeable in this controlled model, while mismatched outputs are materially less compatible.

Stability and discrimination utilities are not identical. Zero ablation increases within-family JS by 0.000040 bits on average and reduces between-family JS by 0.000393; mismatched replacement has the largest average discrimination damage (0.001439 bits).

Pair interactions have mixed signs: median `Gamma` is +0.0050 nats, ranging from −0.0116 to +0.0187. Individual effects therefore do not add linearly. Median pairwise correlation between head-output norms across nuisance realizations is 0.002, which rejects a simple globally correlated-head account but does not prove statistical independence of the full vectors.

Attention motif specificity again fails to predict causal target utility: Spearman `rho=-0.0059` across 16 layer/head cells, closely reproducing the prior relation/layer null of −0.009. Visual regularity remains a negative control, not a causal attribution method.

This run meets the per-head smoke gate but does not establish reproducibility across seeds, pattern-length recruitment, or head-count scaling. The single point in the head-count figure is explicitly a baseline, not a scaling law; those confirmatory extensions remain open.
