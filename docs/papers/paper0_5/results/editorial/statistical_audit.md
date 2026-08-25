# Paper 0.5 statistical audit

| Audit item | Finding | Status / action |
|---|---|---|
| Inferential unit | Held-out predictive family, model seed, and architecture cell—not raw token rows. | Pass; stated in appendix. |
| Pseudoreplication | Dense per-token artifacts are descriptive inputs, not independent inferential replicates. | Pass; keep family/seed aggregation explicit. |
| Confidence intervals | Dataset V2 uses family/seed bootstrap intervals; Group 2.5 uses architecture-cell holdouts and paired mask contrasts; bridge donors are identity matched. | Pass for reported intervals. |
| Competence filtering | Internal timing, reuse, head, and mechanism claims exclude architecture/order strata below 0.80. Failed cells remain in artifacts. | Pass. |
| Threshold provenance | 0.80 primary and 0.90 secondary are fixed analysis gates. Available records do not justify calling them preregistered. | Corrected manuscript wording. |
| Phase-grid replication | Final architecture surface uses seed 11 only. | Material limitation; labeled in setup, results, discussion, and limitations. |
| Raw-length regression | $R^2=0.016$ is descriptive and does not establish a scaling law. | Pass; explicitly described as a raw-length null. |
| Scaling fits | Constant, linear, log, power, threshold, and saturating descriptions do not yield a defensible universal law. | Pass; no scaling exponent promoted. |
| Multiple comparisons | The paper makes a small set of theory-driven contrasts, but the full phase surface and head scans are exploratory. | Treat head scans and phase-shape details as descriptive; require replication for confirmatory claims. |
| Covariance identity | $\operatorname{Var}(\sum_l\epsilon_l)=\sum_l\sigma_l^2+2\sum_{l<j}\operatorname{Cov}(\epsilon_l,\epsilon_j)$. | Verified. Reported summed off-diagonal covariance is now labeled as already including the factor of two. |
| Margin interpretation | Positive correct-target margin is exactly top-1 correctness, but SNR alone could mislead when variance is small. | Pass; manuscript reports signal and variance and triangulates with probability, JS, masking, and replacement. |
| RNN fairness | Data, vocabulary, target entropy, batching, optimizer family, 1,200 updates, and 12.288M tokens match; parameters and architecture-specific tuning do not match exactly. | Acceptable controlled comparison with disclosed limitation; not an architecture leaderboard. |
| Initialization and seeds | Recurrent comparison uses the tracked model-seed protocol; phase surface remains one-seed. | Partial; disclose rather than generalize. |
| Post-hoc exclusions | Negative and chance cells remain machine-readable; exclusions are competence based. | Pass. |

## Quantitative claim cross-check

- Phase grid: 14 models; depth set 2/4/8/12/16/24; widths 32/64 where feasible; 400-update base and selected 2x/4x runs; one seed.
- Order-three rescue at depth 8/width 64: 0.634 → 0.927 → 0.972.
- Best order 4/6/8 accuracies: 0.507/0.387/0.391; no minimum-depth law.
- Recurrent parameters: Transformer 40,896; vanilla RNN 16,640; GRU 33,280; LSTM 41,600.
- Recurrent exposure: 1,200 updates and 12,288,000 training tokens for every architecture.
- Selective nuisance masking: accuracy remains 1.000 and margin variance falls 1.864 → 1.109 (40.5%); signal masking gives 0.249 accuracy.
- Frozen cumulative JVP: directional cosine 0.782/0.552/0.335 at horizons 1/2/4; relative error rises to 1.047.

## Statistical conclusion

The evidence supports bounded within-regime contrasts and falsifications. It does not support a universal scaling curve, a minimum-depth law, an independent causal effect of architectural depth, or universal architecture superiority. The highest-value future statistical improvement is multi-seed replication of the complete depth–width–training surface.
