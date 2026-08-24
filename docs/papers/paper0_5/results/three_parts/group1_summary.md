# Group 1 correctness-aware summary

This addendum uses nine predictive families, three seeds, nuisance counts 0/4/8, twelve realizations, and nine ordered sublayer boundaries (initial pre-SA followed by post-SA and post-block boundaries for four layers).

1. **First top-1 depth.** Across successful trajectories, the correct target first becomes top-1 at mean sublayer depth 3.00, 3.08, and 3.41 for nuisance 0, 4, and 8.
2. **Stable top-1 depth.** It remains top-1 from mean depth 3.40, 3.48, and 3.88. Empty sentinels—not the final layer—encode never-correct cases.
3. **Nuisance and decision depth.** Stable depth rises by 0.48 boundaries from nuisance 0 to 8. The association is weak: the best candidate fit has only `R2=0.013`, so this is a descriptive trend, not a scaling law.
4. **Correct-target margin.** At nuisance 8, mean margin rises from −32.02 logits at the initial boundary to +10.51 at the final boundary. All three generators change sign and finish positive.
5. **Margin fluctuation.** Margin variance does not decrease: it starts at exactly zero because all initial prediction-position states are identical and finishes at 1.89 for nuisance 8. The strict absolute-noise-contraction hypothesis is falsified in this setup.
6. **Margin SNR.** Final median family/seed SNR is +10.24 at nuisance 8. Its improvement comes from a roughly 42.5-logit signal gain, not denominator collapse; variance grows while signal grows much faster.
7. **Monotonicity and reversals.** Mean correctness reversal counts are 1.52, 1.37, and 1.38. Mean settling delay is 0.48, 0.40, and 0.47 boundaries. Correctness formation is commonly non-monotone.
8. **Entropy.** Entropy and margin have only `r=0.081` across aggregate cells. Entropy can increase while the margin changes from negative to positive; it is not a correctness order parameter.
9. **Common layer update.** Mean margin update is +5.12 logits and 73.7% of family/seed/noise transition cells have positive mean updates. Median cumulative signal is +40.39 logits.
10. **Cross-layer fluctuations.** Median summed update variance is 3.55, while median summed off-diagonal covariance is −2.12; median actual cumulative fluctuation variance is therefore only 0.51. Negative cross-layer covariance indicates cancellation/repair rather than independent layer noise.
11. **Scaling evidence.** No convincing scaling law is found. Stable-depth fits explain at most 1.3% of variance. Positive-margin SNR fits explain at most 1.0%, with logarithmic and linear BIC nearly tied.
12. **Best functional forms.** Linear has the lowest BIC for stable depth versus nuisance, but its advantage is tiny and predictive error remains 1.65 boundaries. Logarithmic has the lowest BIC for SNR versus depth, only 0.79 below linear, and has `R2=0.006`. Both are labelled empirical non-results.
13. **Generalization.** The correct margin becomes positive and top-1 fraction reaches 1.0 under nuisance 8 for contiguous n-grams, skip-grams, and binary functors across all three seeds. Architecture-scale generalization is not tested by this run.

The supported statement is narrower than absolute fluctuation contraction:

> Depth increases correct-target signal much faster than nuisance-conditioned margin fluctuation, producing reliable positive margins and stable top-1 predictions. Prediction-space JS contracts, while absolute margin variance need not.
