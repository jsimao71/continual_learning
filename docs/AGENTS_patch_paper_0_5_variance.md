# Core correction
Variance reduction is measured **across token-sequence samples conditional on the same underlying pattern**, not across layers. Depth is the refinement axis.

For samples `X_j ~ X|P`, measure at depth `l`:
- residual/sample dispersion `Var[R_l(X)|P]` in task-relevant directions;
- target-logit/probability/margin variance across samples;
- full prediction dispersion, e.g. mean JS divergence from the conditional mean distribution;
- within-example entropy separately.

Keep distinct:
1. within-example entropy `H[p_l(Y|X)]`;
2. across-realization dispersion `Var/JS[p_l(.|X)|P]`;
3. signal-to-noise or between-pattern / within-pattern separation.

The variance hypothesis concerns primarily (2) and (3). A model may retain high genuine entropy while becoming invariant to irrelevant noise.

Depth does not create iid samples. Test it as **effective evidence/sample amplification**: successive state-conditioned transformations repeatedly extract, combine and denoise evidence already present in the sequence:
`r_(l+1)=r_l+F_l(r_l,context)`.
Use boosting/iterative-estimation language only as a functional analogy.

# Patch — Paper 0.5: n-grams

For each focal n-gram/prefix P create many realizations:
`N_left(j) + P + N_right(j)`, varying irrelevant lexical/semantic distractors, position, distance and natural corpus context. Keep contradictory context as a separate repair condition.

At every layer and pre/post-SA/MLP point estimate across realizations:
- mean and variance of correct-continuation probability;
- target-logit and margin variance;
- mean JS prediction dispersion;
- task-relevant residual covariance/effective rank;
- mean entropy and surprisal.

Primary prediction: with useful depth, mean pattern signal rises or stabilizes while nuisance-conditioned sample variance falls, so pattern SNR/separation rises. Do not require layerwise monotonicity.

Extend the existing depth × prefix experiment to:
`metric(depth, informative-prefix-length, noise-level)`.
Test whether depth partially compensates for fewer/weaker observations, becomes more useful with greater nuisance noise, and saturates earlier with strong lexical evidence.

Controls: different focal patterns with matched noise, shuffled continuations, frequency matching, random/undertrained checkpoints, irrelevant vs answer-changing context.

Required figures:
1. across-sample target-probability variance vs depth;
2. JS prediction dispersion vs depth;
3. pattern SNR vs depth;
4. depth × prefix × noise surfaces;
5. within-example entropy beside across-sample dispersion;
6. SA vs MLP contribution to invariance.

Interpretation: Paper 0.5 tests whether depth makes learned lexical patterns increasingly invariant to irrelevant sequence variation while preserving/increasing predictive signal.
