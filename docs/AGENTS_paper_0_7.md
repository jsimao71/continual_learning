# AGENTS.md --- Paper 0.7: Functors, Learned Categories, and Fuzzy Rule-Like Computation

## Mission

Test whether competent text-completion transformers develop reusable,
substitution-tolerant syntactic/relational transformations over learned
semantic categories: an emergent fuzzy symbolic computational regime.

Do not assume literal symbols, rules, unification, or an interpreter.
Require behavioral, representational, and causal evidence.

## Dependencies

-   Paper 0.5: n-gram continuation, prefix dependence, layerwise
    entropy/surprisal.
-   Paper 0.6: semantic categories/hierarchies learned through
    completion.
-   Residual program: coherence, superposition, variance reduction,
    complementary layer/head contributions.

Paper 0.6 is a hard methodological dependency: category structure must
be learned, not injected.

## Competence gate

Residual analysis in a custom model is interpretable only after the
model is competent at held-out text completion.

Run checkpoints before, near, and after the competence threshold.
Category labels from a synthetic generator may be used for scoring only;
do not expose them as privileged training signals in the main condition.

## Core hypotheses

1.  Soft functor invariance: equivalent syntactic/relational frames
    induce partially invariant residual transformations across argument
    substitutions.
2.  Learned type-sensitive dispatch: the same frame induces different
    continuation/transformation families for arguments belonging to
    different learned semantic categories.
3.  Category generalization: behavior transfers to unseen members and
    unseen member×functor combinations.
4.  State-conditioned rewriting: Delta_l = F_l(r_l); layers are
    complementary causal stages, not independent estimators.
5.  Predictive uncertainty is progressively reorganized toward
    rule-consistent equivalence classes, without requiring monotonic
    entropy reduction.
6.  Compositions of functors produce sequentially conditioned
    transformations.

## Dataset

Create a controlled natural-language text-completion world with entities
whose categories and hierarchy are statistically learnable from ordinary
completion examples.

Include unary frames F(x), binary relations R(x,y), nested/composed
forms G(F(x)), exceptions, competing defaults, and paraphrased
realizations.

Formal F/R notation is analysis notation; do not make the main task
trivially symbolic.

Use splits withholding: - entity×functor combinations; - category
members; - paraphrases; - compositions; - longer chains; -
category-boundary/exception cases.

## Experiments

E1 --- N-gram → functor: replace fixed elements of common templates with
substitutable arguments and test emergence of invariant transformation
structure.

E2 --- Soft types: after competence is established, test whether
held-out members of a learned category produce similar
functor-conditioned trajectories.

E3 --- Type-sensitive dispatch: same syntax, different learned
categories.

E4 --- Novel argument generalization: learn F(a), F(b), F(c), withhold
F(d), then test d across many categories.

E5 --- Composition: G(F(x)), R(F(x),y), and chains. Test whether depth
shows sequential transformation rather than memorized whole-string
mapping.

E6 --- Exceptions/competition: contradict frequent/default mappings and
separate early priming from later correction.

## Measurements

Behavior: - completion cross-entropy; - accuracy where defined; - target
probability/rank; - systematic/substitution generalization; -
compositional generalization; - exception/counterfactual sensitivity.

Layerwise output: - raw logit-lens entropy; - normalized vocabulary
entropy; - target surprisal; - rule-consistent category probability
mass; - pre/post-SA and pre/post-MLP deltas.

Residual geometry: - transformation cosine similarity across
substitutions; - within/between-category similarity; -
aligned/orthogonal energy; - covariance/effective dimension; -
head/layer complementarity; - paraphrase invariance; - trajectory
similarity.

Causal: - head/layer skip and ablation; - activation patching across
equivalent functor instances; - category-state patching; - norm-matched
random controls; - counterfactual category substitution.

## Critical comparisons

Always include: - same syntax / different category; - different syntax /
same category; - same semantic relation / paraphrased syntax; - seen
entity / unseen entity; - competent / undertrained checkpoint.

These separate semantic priming, syntactic priming, lexical
memorization, and genuine reusable transformation behavior.

## Interpretation threshold

Do not call a pattern rule-like from attention maps, probes, or
clustering alone. Strong evidence requires convergence of: 1. held-out
behavioral generalization; 2. substitution/paraphrase invariance; 3.
dependence on learned semantic category; 4. coherent residual/entropy
trajectories; 5. causal transfer/disruption.

A valid negative result is that apparent rule behavior collapses under
held-out combinations or is explained by lexical frequency.

## Main conceptual target

Evaluate the functional hypothesis:

> Continuous transformer computation can organize into approximate
> objects, soft semantic types, relations, and reusable
> state-conditioned transformations. Growing context can instantiate
> temporary facts and patterns on which this fuzzy rule-like machinery
> operates.

Keep this as an empirical hypothesis, not a philosophical assertion.
