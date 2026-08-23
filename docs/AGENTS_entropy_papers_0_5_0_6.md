# AGENTS.md --- Entropy Additions for Papers 0.5 and 0.6

## Goal

Add layerwise predictive uncertainty as a shared observable connecting
residual geometry/variance reduction to actual next-token prediction.

## Shared instrumentation

At every focal position capture residuals pre-block, post-SA, post-MLP,
and post-block. Decode with the model final norm and unembedding:

    z_l = W_U LN_f(r_l)
    p_l = softmax(z_l)

Measure: - entropy H_l = -sum p log2 p; - normalized vocabulary entropy
H_l/log2(\|V\|); - effective vocabulary fraction 2\^H_l/\|V\|; - target
surprisal -log2 p_l(y\*); - target probability and rank; - deltas across
SA, MLP, and complete layers; - cumulative residual trajectory.

Do not assume monotonic entropy reduction. A useful layer may broaden,
redirect, prune, or sharpen predictions.

Relate these measurements to existing residual metrics: update
alignment, aligned/orthogonal energy, covariance/effective dimension,
head/layer complementarity, causal skip utility, and output margin.
Treat layers as causally conditioned:

    r_(l+1) = r_l + Delta_l
    Delta_l = F_l(r_l)

not as independent estimators.

## Paper 0.5 --- n-grams

For common n-grams vary both transformer depth l and available prefix
length k. Plot H(l,k), normalized entropy, target surprisal, probability
and rank. Compare frequent/rare, deterministic/ambiguous, minimal focal
sequence/added context, and override cases where context contradicts a
common continuation.

Key test: does increasing lexical evidence make the correct continuation
recoverable earlier in depth, and do later layers refine/repair/align
it?

## Paper 0.6 --- semantic categories

Critical methodological constraint: categories must be learned by the
model. Generator category labels are evaluation metadata, not privileged
model inputs.

Internal residual/category analysis is meaningful only after the model
is demonstrably competent at the underlying held-out text-completion
task. Establish a completion-performance gate before interpreting
semantic geometry.

Protocol: 1. Train/select checkpoints spanning undertrained to
competent/converged. 2. Establish held-out completion competence. 3.
Independently test whether categories/hierarchies are recoverable
behaviorally and geometrically. 4. Only then interpret abstraction and
entropy trajectories.

Measure global entropy plus probability mass over controlled lexical
realization sets at different abstraction levels. Test non-monotonic
abstraction, coexistence and re-entry rather than assuming a fixed
layer-to-abstraction map.

## Controls

Use shuffled continuations/categories, frequency-matched unrelated
words, random and undertrained models, label/permutation controls,
norm-matched perturbations, context controls, and multiple seeds.

## Deliverables

Update both papers' hypotheses, methods, plots and tables. Explicitly
distinguish representation-space variance from prediction-space entropy.
Paper 0.5 remains the lexical laboratory; Paper 0.6 establishes learned
semantic structure. Paper 0.7 combines learned categories with
functor/rule-like behavior.
