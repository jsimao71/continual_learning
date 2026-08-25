# AGENTS.md — Paper 0.6 Extension: Semantic Equivalence and Multiscale Abstraction

## Mission
Preserve the completed Paper 0.6 results: hierarchy geometry exceeds permutation but is weak; cross-template similarity is high; hierarchy is present early; SA/FFN responsibility changes with requested level; semantic motif specificity is negative.

Add a second phase testing abstraction as relative amplification of structure common across semantic equivalence classes rather than a monotone layer ladder.

## Shared invariant ontology
Distinguish representational, dynamical, functional, and structural-role invariance. Raw variance need not monotonically fall; measure common task-relevant structure relative to instance/template variation.

## H1 — Nested semantic equivalence
At abstraction level k, x ~k y when they share the relevant category/role. Classes are nested:

C_instance subset C_parent subset C_superclass ...

Moving upward enlarges the equivalence class by discarding distinctions. Do not infer a literal symbolic taxonomy.

## H2 — Common semantic component
Decompose representation statistically into instance, category, template/syntax, task/request, and residual factors. Test whether category-relevant common structure strengthens relative to within-category nuisance variation at particular layers/locations.

Use within/between dispersion, RSA, neighbor recovery, shared-subspace metrics, and causal transfer.

## H3 — Multiscale coexistence
Replace the naive early=specific/late=abstract hypothesis with: multiple abstraction scales may coexist in one residual state and task/query selects their functional weighting.

Test parent, root, and intermediate targets on the same leaves. Measure simultaneous geometry, causal contribution, scale-subspace overlap/orthogonality, and whether selecting one scale suppresses or merely downweights another.

## H4 — Semantic-vs-syntactic variance/effect decomposition
Cross semantic category, abstraction depth, instance, template/syntax, natural-vs-arbitrary labels, and requested target level.

Use the same analysis as Paper 0.5. Prediction: 0.6 has relatively stronger semantic/category effects; 0.5 has relatively stronger sequential/syntactic effects. Neither must be pure.

## H5 — Residual common-component amplification
For semantic class C estimate common SA, FFN, and block-update directions/subspaces.

Measure explained energy/variance, angular concentration, effective rank, within-vs-between update distance, parent/root subspace overlap, natural/synthetic transfer, and cross-template transfer.

A low-dimensional component is a candidate dynamical invariant, not a literal Lie generator.

## H6 — Confidence follows requested abstraction
Relate common semantic structure to target probability, target-vs-competitor margin, entropy, and sibling/ancestor confusion. Test whether depth increases confidence in the requested semantic resolution rather than globally increasing abstraction.

## H7 — Explain the layer-0 hierarchy signal
Separate hierarchy already encoded in embeddings/training, hierarchy induced by prompt/context before the first block, and later redistribution among scales. Compare embedding/input, post-SA, post-FFN, post-block. Arbitrary-label in-context hierarchies are especially important.

## H8 — Motif null localizes semantic invariance elsewhere
Compare attention-motif, hidden-state, residual-update, and functional equivalence. Test whether distinct attention patterns map to the same semantic residual effect.

## H9 — Typed relations and P(X,Y)
Extend carefully from nouns to actions and relations, then typed relation instances P(x:Cx,y:Cy) and variable-renaming controls. Export successful designs to a dedicated relational-invariance/ESR bridge paper.

## Duality with Paper 0.5
0.5 contextual refinement adds information and tends to split predictive classes.
0.6 semantic abstraction removes distinctions and enlarges semantic classes.

Use common metrics to compare these opposite transformations.

## Required causal tests
Sibling patching; cross-category matched patching; parent/root subspace projection; add/replace semantic common directions; cross-template transfer. Strong claims require predicted changes in requested category behavior.

## Required outputs
Layer x semantic-factor profile; multiscale coexistence; common semantic residual strength; category-vs-instance/template variation; confidence by requested level; motif-vs-residual-vs-functional equivalence; natural-vs-synthetic transfer; direct 0.5-vs-0.6 comparison.

## Definition of done
Common-vs-nuisance semantic structure is decomposed; multiple abstraction levels are tested on the same leaves; common residual components receive causal tests; confidence is linked to requested level; layer-0 signal is localized; all invariant types are compared; metrics align directly with Paper 0.5.
