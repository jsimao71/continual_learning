# AGENTS.md — Paper 0.5 Extension: Predictive Equivalence and Sequential Invariants

## Mission
Preserve the completed Paper 0.5 results. Add a second experimental phase testing whether n-grams are a special case of predictive equivalence and whether residual-layer composition amplifies common sequential/syntactic structure relative to nuisance variation.

Do not replace the established findings: SA/FFN responsibility is transformation-dependent, override is strongly SA-dependent, and attention-motif specificity does not predict SA causal contribution.

## Shared invariant ontology
Distinguish:
- representational invariance: task-relevant hidden geometry survives nuisance transformations;
- dynamical invariance: SA/FFN/residual updates share direction, subspace, or transformation role;
- functional invariance: matched interventions produce equivalent outcome effects;
- structural-role invariance: coordinates may differ while relational role is preserved.

Recurrence, magnitude, motif similarity, geometry, or probe accuracy alone are not consolidation evidence. Do not interpret variance reduction as necessarily decreasing raw hidden-state variance. Test increasing dominance of a common task-relevant component relative to within-class variation.

## H1 — Predictive equivalence classes
For contexts x_i, define x_i ~P x_j externally when their continuation distributions are sufficiently similar under preregistered metrics (JS distance, top-k overlap, entropy, log-odds/margin).

The main construction is n-grams sharing a suffix with different prefixes and increasing total context length.

Never define equivalence from hidden states and then claim hidden-state recovery of it.

## H2 — Contextual refinement
A suffix s defines C(s). Adding prefix information refines it:

C(s) >= C(p1 s) >= C(p2 p1 s) ...

Measure marginal information from each added prefix. Separate:
- irrelevant prefix: continuation distribution changes little;
- informative prefix: continuation uncertainty/distribution changes materially and splits the class.

Relate external information gain to target progress, output entropy/margin, residual alignment, SA/FFN causal contribution, and representation separation.

## H3 — Sequential/syntactic dominance
Cross rather than confound syntax and semantics:
- same suffix/different grammatical prefix;
- same syntax/different semantics;
- lexical substitutions preserving syntax;
- semantic paraphrases changing surface sequence;
- shuffled/order controls.

Estimate layer/location effects for sequential/syntactic, semantic, lexical, template, and instance factors. Cross-paper prediction: Paper 0.5 should show relatively stronger sequential/syntactic effects than Paper 0.6, not exclusive syntactic representation.

## H4 — Residual common-component amplification
For equivalence class E estimate class-conditioned common SA, FFN and block updates.

Measure mean-update norm, common-component explained energy/variance, angular concentration, shared-subspace rank, within-vs-between update distance, cross-prefix transfer, and evolution across depth.

Hypothesis: repeated partially aligned transformations increase the relative signal of common predictive structure. Do not claim literal averaging of independent samples.

## H5 — Confidence coupling
Test whether common-component dominance tracks continuation entropy, target probability, target-vs-competitor margin, and calibration where meaningful.

Correlation is descriptive until causal intervention confirms it.

## H6 — Motif null localizes the invariant downstream
Test whether different attention configurations can generate equivalent residual effects:
- motif similarity vs residual-update equivalence;
- motif similarity vs shared-subspace similarity;
- motif similarity vs functional equivalence.

A key possible result is low attention similarity with high residual/functional equivalence.

## H7 — Bridge toward P(x,y)
Treat prefix/suffix prediction as a special case of a general relation. Exploratory progression:
1. exact n-gram;
2. varying-prefix predictive equivalence;
3. lexical substitution;
4. non-adjacent P(x,y);
5. variable-renaming controls.

Export stages 4–5 for a relational-invariance/ESR bridge rather than bloating 0.5.

## Required causal tests
Patch between predictively equivalent contexts and matched nonequivalent contexts; project out candidate common subspaces; add/replace common update directions; measure changes to the continuation distribution, not only one target token.

## Required outputs
Add figures/tables for nested suffix families; conditional information gain; equivalent-vs-nonequivalent geometry; syntactic/sequential-vs-semantic effects; common residual-update strength; confidence coupling; motif-vs-residual/functional equivalence; and direct 0.5-vs-0.6 comparison.

## Statistical discipline
Scientific unit = equivalence class/suffix family, not occurrence. Bootstrap families. Use identity-disjoint splits for fitted analyses. Correct layer/head searches and preserve signed effects.

## Definition of done
Predictive equivalence is externally defined/versioned; nested context families are reproducible; syntactic and semantic factors are crossed; common residual structure and confidence are measured; causal tests distinguish equivalent from nonequivalent classes; Paper 0.6 comparison uses shared metrics; nulls remain visible.
