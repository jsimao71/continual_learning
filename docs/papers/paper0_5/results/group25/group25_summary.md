# Experiment 2.5 summary: SA window and nuisance-conditioned variance

1. **Reachable nuisance and variance.** A universal reachability transition is not observed. Full attention has mean final variance 1.890 and accuracy 0.999; truncated full-trained models often lose signal and have much larger variance.
2. **Threshold.** The graph criterion $Lw\ge s$ is necessary but not sufficient; training adaptation controls the functional threshold.
3. **Signal-only regime.** Inference-only signal-reachable truncation is not successful (accuracy 0.411). Locally trained restricted models reach 1.000 accuracy with variance 1.110.
4. **Immediate SA change.** Across the full intervention grid, mean SA margin-variance change is 4.811; this average is dominated by failing truncated evaluations and is not evidence of nuisance import alone.
5. **FF change.** Mean FF variance change is 4.490; SA is not uniquely dominant.
6. **Normalization.** Mean normalization-associated change is -0.655, smaller in magnitude than SA/FF changes.
7. **Attention shifts.** Nuisance attention mass is measured and enters the attribution models, but attention mass alone does not establish causal variance import.
8. **Competition.** Competitor switches explain part of failure under truncation; they do not explain the successful selective-mask contrast.
9. **Repair.** Median valid repair fraction from cross-layer update covariance is 0.812; negative covariance provides partial later-layer cancellation where positive.
10. **Selective masking.** Removing nuisance preserves accuracy and increases mean margin by 0.261, while reducing nuisance-conditioned variance by 43.9%. Removing signal reduces accuracy to 0.252.
11. **Broader exposure.** Full exposure is best for full-trained models; generic ``less is more'' is falsified. Selective exclusion, not indiscriminate restriction, is beneficial.
12. **Depth tolerance.** Depth interacts with truncated-window competence, but three depth points do not support a scaling law.
13. **Largest mechanism.** The nonlinear forest achieves held-out $R^2=0.959$. Mechanism importance is reported descriptively in `group25_variance_models.csv`; the causal masking contrast is stronger evidence than attribution ranking.

Conclusion: SA can transport nuisance that measurably perturbs the margin, as shown by selective masking, but the broad variance surface is governed jointly by access, learned routing, nonlinear amplification, FF processing, candidate competition, and repair. Do not write that SA simply ``injects noise.''
