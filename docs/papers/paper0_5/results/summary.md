# Paper 0.5 experiment summary

## Scope

- Controlled E1--E6: two architectures x two seeds, 21,630 atlas entries and 1,920 component rows.
- Conditional variance factorial: depth x weak/pattern/full prefix x 0/2/4 nuisance, with irrelevant, answer-changing, shuffled-continuation, and random-checkpoint controls.
- Pretrained replication: Pythia-70M-deduped at steps 0/1,000/143,000 and pinned Qwen3-0.6B; four crossed syntax x semantic families.
- Causal transfer directions fit on weak clean identities and evaluated on stronger noisy held-out identities.

## Results

- Controlled pattern/four-noise JS dispersion falls 0.0650 -> 0.0307 -> 0.0106 bits; last-minus-first family/run contrast is -0.0474, 95% CI [-0.0778, -0.0217].
- Within-example entropy rises while across-realization dispersion falls, establishing that these quantities are distinct.
- Equivalent-minus-nonequivalent replacement is 1.047 nats at Pythia step 1,000 [0.545, 1.455], 1.206 at step 143,000 [0.224, 2.187], and 0.599 for Qwen [0.299, 0.899]. Step 0 does not separate reliably.
- Common-direction removal is harmful in trained Pythia and Qwen; common-direction addition is not reliably beneficial.
- Pretrained variance refinement is conditional: final Pythia has a small negative last-minus-first JS contrast, while Qwen increases.
- Motif specificity remains unrelated to SA causal utility (Spearman -0.009).

## Decision

The pretrained causal-mechanism gate passes for functional equivalence, but the online-learning gate does not. Addition provides no benefit and Paper 1's natural held-out utility gate remains negative. E7 stays stopped.

## Primary artifacts

- `tables/variance_by_depth_prefix_noise.csv`
- `tables/variance_family_bootstrap.csv`
- `pretrained/tables/pretrained_variance.csv`
- `pretrained/tables/pretrained_causal_mediation.csv`
- `pretrained/tables/family_bootstrap_inference.csv`
- `pretrained/tables/causal_family_contrasts.csv`
- `pretrained/tables/factorial_regression.csv`
- split `raw/variance_*.jsonl` scientific-unit artifacts
- `pretrained/raw/*.jsonl`
- controlled and pretrained `figures/*.png`

## Next falsifiable question

Does the equivalent-transfer advantage survive a larger family inventory, natural corpus frequency/tokenization matching, and path-specific mediation while remaining useful under a matched-budget natural outcome gate?
