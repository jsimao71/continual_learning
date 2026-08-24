# Group 1 — Layer composition and predictive stability

The legacy pilot is reproduced from its committed fixed-depth aggregates: with pattern evidence and four distractors, post-block target-probability variance is 0.0347, 0.0106, and 0.0109 across layers 0–2, while JS dispersion is 0.0650, 0.0307, and 0.0106 bits.

The new controlled matrix covers three generators, nine predictive families, three model seeds, three nuisance counts, four layers, and twelve realizations per cell. All three generators reach 100% final top-1 accuracy under eight nuisances, so contraction is not a low-signal artifact.

At eight nuisances, first-to-last within-family JS changes are:

- contiguous n-gram: −0.0244 bits, family/seed bootstrap 95% CI [−0.0414, −0.0098];
- skip-gram: −0.0415, CI [−0.0791, −0.0129];
- binary functor: −0.0402, CI [−0.0738, −0.0143].

Only 7/27 family-seed trajectories contract monotonically. Eleven expand and then contract; the remainder are oscillatory, late-contracting, or non-contracting. Depth therefore improves final predictive-class organization reliably without acting as a monotone denoiser at every block.

The local block linearization is informative but incomplete. Median JVP/observed-delta cosine is 0.984 for nuisance directions and 0.928 for answer-changing signal directions. Median prediction-space relative error is 0.179 and 0.311, respectively. Local Jacobians describe nuisance propagation more accurately than the larger signal changes, but they do not replace the composed nonlinear trajectory.

## Correctness-aware follow-up

The dense nine-boundary trace rules out stable-but-wrong contraction. Under eight nuisances, mean correct-target margin changes from −32.02 to +10.51 logits, target probability from approximately $1.4\times10^{-10}$ to 0.9998, and top-1 fraction from 0 to 1. Margin variance increases from 0 to 1.89 rather than contracting, so the supported result is signal growth faster than fluctuation growth. Mean first/stable decision depths are 3.41/3.88 sublayer boundaries, with 0.47 mean settling delay.

Layer updates add a median cumulative +40.39-logit signal. Median summed off-diagonal update covariance is −2.12, reducing median cumulative fluctuation variance to 0.51; this supports cross-layer cancellation/repair. No scaling law is supported: stable-depth fits have at most $R^2=0.013$, and positive-margin SNR fits at most 0.010. See `../group1_summary.md` for the complete thirteen-question audit.
