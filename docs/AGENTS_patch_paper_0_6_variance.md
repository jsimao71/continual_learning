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

# Patch — Paper 0.6: learned semantic categories

Hard prerequisite: categories must be learned through text completion. Generator labels are evaluation metadata only. Do not interpret category residuals until the custom model passes a held-out completion-competence threshold; use earlier checkpoints as controls.

For learned category C sample:
- same entity, varying nuisance context;
- different entities in C;
- different categories under matched contexts.

Estimate by depth:
- within-entity/context variance;
- within-category variance;
- between-category variation;
- covariance spectrum/effective rank;
- centroid separation;
- output-distribution dispersion;
- category-consistent continuation mass.

Primary prediction: after competence emerges, **between-category signal / within-category nuisance variance** increases with useful depth. Use Fisher/ANOVA-like ratios. Total hidden-state variance need not decrease.

Repeat at learned hierarchy levels: instance → subtype → category → supercategory. Test which invariances emerge at each depth without assuming a monotonic layer-to-abstraction mapping. Lexical identity may remain represented while nuisance variance contracts and category-stable components emerge.

Keep output entropy separate from across-context dispersion: a broad semantic category may remain high-entropy while becoming highly invariant to nuisance context.

Controls: shuffled memberships, frequency-matched pseudo-categories, lexical leakage, matched contexts, incompetent/undertrained checkpoints.

Required figures:
1. within- vs between-category variance across depth;
2. Fisher-style separation ratio;
3. nuisance-context vs semantic-member variance decomposition;
4. JS dispersion vs within-example entropy;
5. hierarchy-level invariance curves;
6. competence vs emergence of semantic variance reduction.
