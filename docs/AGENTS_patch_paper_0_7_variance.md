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

# Patch — Paper 0.7: functors + learned categories

Inherit Paper 0.6's competence/category-learning gate. Do not call behavior type-sensitive unless the semantic categories were behaviorally learned.

For functor/template F and learned category C construct many realizations `X_j = realization_j(F(x))`, varying argument identity, irrelevant context, paraphrase/surface syntax, position/distance and distractors.

Estimate:
- `Var[Delta R_l(X)|F,C]`;
- prediction JS/variance conditional on F,C;
- within-(F,C) covariance;
- between-functor and between-category separation;
- rule-consistent continuation mass;
- target surprisal/margin.

Core prediction: a reusable fuzzy rule yields decreasing nuisance-conditioned variance relative to between-rule/category structure, while remaining sensitive to legitimate F, C and exception differences.

Use a crossed design with factors:
- functor/syntax F;
- semantic category C;
- entity x within C;
- nuisance context N.

Estimate F, C, F×C, entity-within-C and nuisance effects. The F×C interaction is central evidence for soft type-sensitive dispatch.

Generalization cells:
- seen F × seen entity;
- seen F × unseen category member;
- paraphrased F × seen/unseen member;
- composed `G(F(x))`;
- exception/override cases.

## Depth × evidence × noise
Vary supporting demonstrations, noise/distractors and depth. Produce surfaces for rule accuracy, conditional dispersion and SNR. Test whether more depth and more supporting examples partially substitute, interact, or saturate.

Growing-context core:
`F(a)->u, F(b)->u, ..., query F(c)->?`
Vary number/quality/distance of demonstrations. Measure whether more demonstrations and/or depth reduce nuisance-conditioned query variance.

Entropy remains complementary: a fuzzy rule may support several valid outputs, giving high within-example entropy but low across-realization variance and high rule-consistent class mass.

Causal patch/ablation must test effects on both mean rule signal and conditional sample variance.

Required figures:
1. within-(F,C) transformation variance vs depth;
2. between-rule/category to within-rule variance ratio;
3. F×C interaction vs depth;
4. depth × examples × noise surfaces;
5. held-out substitution invariance;
6. entropy vs across-realization dispersion;
7. causal intervention effects on mean and variance.

Interpretation: a rule-like transformation is a statistically stable, causally relevant transformation family increasingly invariant to nuisance-varying realizations while sensitive to functor, learned type and legitimate exceptions.
