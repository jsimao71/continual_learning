# AGENTS.md Addendum — Paper 0.5
## Dataset V2: Increasing Structural Complexity + Pretrained Cumulative-Jacobian Integration

### Mission

Upgrade Paper 0.5 in two ways:

1. Replace the current easy synthetic predictive families with a progressively harder controlled generator ladder that removes single-token shortcuts and introduces structured nuisance.
2. Integrate pretrained models more directly into the layer-composition theory by testing a weaker but meaningful cumulative-Jacobian prediction, without relying on unknown pretrained n-gram frequencies.

The paper remains about **layer composition and predictive stability**. Do not turn this iteration into a continual-learning study.

---

# PART A — CONTROLLED DATASET V2

## A1. Scientific Goal

The current generator is acceptable as a first proof that depth can reduce sensitivity to random prefix variation, but it is too easy for stronger claims because:

- some individual predictive tokens uniquely identify the target;
- skip-gram filler tokens can identify the relation;
- nuisance tokens come from a distinct vocabulary range;
- the three generator families reuse the same underlying mappings;
- the prediction position is fixed;
- distractors are mostly lexical background variation rather than structured interference.

Dataset V2 must test the stronger hypothesis:

> Depth improves the ratio of predictive signal to structured nuisance even when the target cannot be inferred from any single token and distractors come from the same distribution as predictive content.

---

## A2. Core Generator Contract

Every sample must record:

```text
generator_family
predictive_family_id
surface_identity_id
target_token
pattern_tokens
pattern_length
dependency_span
predictive_arity
nuisance_tokens
nuisance_count
nuisance_type
nuisance_difficulty
competing_pattern_count
answer_changing_context
continuation_entropy
single_token_target_MI
subset_target_MI
full_pattern_target_MI
position_mode
train_frequency
split
generator_seed
```

Also retain a machine-readable symbolic description of the generating rule:

```text
rule_signature
rule_inputs
rule_output
```

Example:

```text
rule_signature = "latin_square_mod4"
rule_inputs = [A2, B3]
rule_output = Y1
```

---

# A3. Mandatory No-Shortcut Constraints

For a pattern \(X_1,\ldots,X_n\) predicting \(Y\), the generator should satisfy approximately:

\[
I(X_i;Y)\approx 0
\]

for each individual token where feasible, while:

\[
I(X_1,\ldots,X_n;Y) \gg 0.
\]

For two-token deterministic relations, use a balanced Latin-square/XOR-like construction such that:

\[
H(Y\mid A)>0,\qquad
H(Y\mid B)>0,
\]

but:

\[
H(Y\mid A,B)=0.
\]

Example with four values:

\[
Y=(i+j)\bmod 4.
\]

Do not allow a unique token ID to identify the relation or target.

Add generator validation tests that estimate empirical:

```text
I(single_token; target)
I(pair; target)
I(full_pattern; target)
```

and fail if singleton predictability exceeds a configured threshold.

---

# A4. Complexity Ladder

Implement these levels in order.

## Level 0 — Current Easy Control

Preserve current:

```text
(A,B) -> Y
```

with relation-specific tokens and random-prefix nuisance.

Purpose:

- regression test;
- reproduce existing results;
- provide an easy baseline.

Do not use this level for strong theoretical claims.

---

## Level 1 — Balanced Pair Relation

Use:

```text
(A_i, B_j) -> Y_f(i,j)
```

with balanced marginals so neither \(A_i\) nor \(B_j\) alone identifies \(Y\).

Suggested mappings:

```text
modular addition
XOR-like table
random balanced Latin square
```

Primary criterion:

```text
single-token target prediction ~= chance
pair target prediction = deterministic
```

---

## Level 2 — Balanced Contiguous n-gram

Extend to:

```text
(X1, X2, ..., Xn) -> Y
```

for:

```text
n = [2,3,4,6,8]
```

Construct target functions where small subsets are deliberately ambiguous.

Possible mappings:

```text
modular sum
parity
lookup table with balanced marginals
hierarchical composition
```

Measure:

\[
I(X_S;Y)
\]

for subsets \(S\) of increasing size.

Goal:

> quantify how predictive information becomes available as more of the pattern is integrated.

---

## Level 3 — Skip / Sparse Relation

Use the same balanced relation but distribute relevant tokens across fillers:

```text
A _ _ B -> Y
A _ C _ _ B -> Y
```

Crucially, filler identity must not identify the predictive family.

Use fillers sampled from a pool shared across all families.

Cross dependency span:

```text
s = [2,4,8,16,32]
```

---

## Level 4 — Functor / Structured Template

Use symbolic-like templates:

```text
F(A,B) -> Y
G(A,B) -> Y'
```

where:

- `F` selects a transformation rule;
- `A,B` are arguments;
- no single field fully identifies \(Y\);
- multiple functors can operate on the same arguments.

Examples:

```text
F_add(A,B) -> Y_(A+B mod K)
F_xor(A,B) -> Y_(A xor B)
F_perm(A,B) -> table lookup
```

This tests compositional rule conditioning.

---

## Level 5 — Nested n vs n+k Override

Construct:

```text
B C -> Y1
A B C -> Y2
```

such that the short pattern is genuinely predictive in one distribution but the longer pattern overrides it.

Also include:

```text
B C -> Y1
A1 B C -> Y2
A2 B C -> Y3
```

Measure:

```text
first layer favoring short-pattern target
first layer favoring long-pattern target
stable override layer
top1 reversals
```

This is a high-value dataset for studying non-monotonic refinement.

---

## Level 6 — Compositional Multi-Step Rule

Examples:

```text
U = F(A,B)
Y = G(U,C)
```

or sequence form:

```text
F A B ... G C -> Y
```

The model must integrate two learned subrelations.

Record:

```text
composition_depth
intermediate_latent_class
```

This tests whether deeper networks show staged construction.

---

## Level 7 — Hierarchical Rule Family

Construct predictive classes with coarse-to-fine structure:

```text
coarse category -> candidate set
subpattern -> final Y
```

Example:

```text
A determines family {Y1,Y2}
B determines which member
```

or:

```text
F determines rule family
A,B determine instance
```

Measure whether layer depth first separates coarse classes, then fine classes.

---

# A5. Nuisance Difficulty Ladder

Nuisance must be independently controlled from pattern complexity.

## N0 — Constant Neutral Context

Fixed padding/neutral tokens.

Purpose:

- baseline;
- no nuisance variation.

## N1 — Separate-Vocabulary Random Nuisance

Current style:

```text
random tokens from nuisance-only vocabulary
```

Purpose:

- easiest perturbation baseline.

## N2 — Same-Vocabulary Random Nuisance

Distractors sampled from the same token pool used for predictive variables.

Purpose:

- prevents trivial vocabulary-range filtering.

## N3 — Partial-Pattern Distractors

Insert fragments of valid learned patterns:

```text
A2
B3
A1 _
_ B4
```

but ensure they do not change the target.

Purpose:

- realistic representational interference.

## N4 — Complete Competing Patterns

Insert full valid relations:

```text
A2 B3
A4 B1
...
[target-designated A1 B2]
```

Only one marked/structurally designated relation determines the answer.

Purpose:

- tests selective stabilization among valid alternatives.

## N5 — Near-Target Competing Patterns

Distractors should preferentially imply the strongest incorrect competitors.

Purpose:

- maximize decision-boundary interference.

## N6 — Structured Answer-Changing Context

Matched control where context legitimately changes the correct target.

This is **not nuisance**.

Purpose:

- verify that the model does not simply learn context invariance.

Always keep:

```text
answer_changing_context = true
```

for these rows.

---

# A6. Position Controls

Run both:

```text
position_mode = aligned
position_mode = randomized
```

Aligned condition preserves current control.

Randomized condition varies absolute position while preserving relative predictive structure.

The stability effect should survive randomized position if it is structural rather than positional memorization.

---

# A7. Shared vs Disjoint Relation Families

Run two regimes.

## Shared-relation regime

Same underlying mapping appears in:

```text
contiguous
skip
functor
```

Purpose:

> test whether the same predictive relation survives different surface realization formats.

## Disjoint-relation regime

Each generator family receives its own independently generated relation table.

Purpose:

> test whether the layer-composition effect independently replicates across rule families.

Do not conflate the two.

---

# A8. Dataset Matrix

Start with a tractable confirmatory core:

```text
pattern_levels = [1,2,3,5]
nuisance_levels = [N0,N2,N3,N4]
nuisance_count = [0,2,4,8,16]
pattern_length = [2,3,4,6]
model_seeds = [11,23,37]
```

Only expand to Levels 6–7 after the core pipeline is validated.

Avoid full Cartesian explosion. Use staged sweeps.

---

# A9. Required Validation Tests

Before training, automatically verify:

1. singleton token target predictability is near chance;
2. configured full pattern predicts target at expected entropy;
3. nuisance variables satisfy:
   \[
   I(D;Y\mid G)\approx 0;
   \]
4. answer-changing controls do change target;
5. filler identity is not predictive of target;
6. train/test surface identities are disjoint;
7. randomized position does not leak target;
8. competing distractor count is independent of target;
9. target frequencies are balanced unless intentionally varied.

Create:

```text
dataset_v2_validation.json
dataset_v2_information_checks.csv
dataset_v2_examples.md
```

Do not run expensive experiments if validation fails.

---

# A10. Primary Measurements

Retain all current Paper 0.5 metrics plus the correctness-aware addendum.

At every layer:

```text
target_probability
target_rank
target_margin
top1_correct
output_entropy
within_family_JS
between_family_JS
R
covariance_trace
effective_rank
margin_SNR
```

Also compute:

```text
first_top1_layer
stable_top1_layer
settling_delay
top1_reversal_count
```

For nested patterns additionally record:

```text
short_target_probability
long_target_probability
short_target_margin
long_target_margin
override_crossing_layer
stable_override_layer
```

---

# A11. Dataset V2 Scaling Questions

Test whether:

\[
L_Y^{stable}
\]

scales with:

```text
pattern length n
dependency span s
nuisance count k
nuisance difficulty level
number of competing patterns
composition depth
number of predictive alternatives
```

Candidate empirical forms:

\[
L^\star \propto n^\alpha,
\]

\[
L^\star \propto k^\beta,
\]

\[
L^\star \propto (s/w)^\gamma,
\]

and saturating/logarithmic alternatives.

Do not force power laws.

---

# PART B — PRETRAINED MODELS VIA WEAKER CUMULATIVE-JACOBIAN TESTS

## B1. Rationale

For pretrained models, the true training distribution and true predictive pattern family are unknown. Therefore we should not claim:

```text
this n-gram was learned with frequency f
```

or:

```text
this nuisance variable is guaranteed independent of Y in pretraining
```

Instead test a weaker architectural/dynamical prediction:

> For paired prompts whose final prediction is empirically stable to a controlled perturbation, does the cumulative Jacobian product predict attenuation of that perturbation through depth?

This integrates pretrained models with the overall theory without pretending we know the underlying corpus statistics.

---

# B2. Pretrained Pair Construction

Use controlled prompt pairs constructed at inference time.

## Type P1 — Semantically Irrelevant Prefix Variation

Example structure:

```text
d1 + query
d2 + query
```

where `d1` and `d2` are unrelated benign prefixes and the model's final top prediction/answer is unchanged.

Only retain pairs satisfying a behavioral stability filter such as:

```text
same final top1 target
target probability above threshold
small final answer disagreement
```

This makes nuisance status empirical rather than corpus-assumed.

## Type P2 — Paraphrastic Surface Variation

Different wording, same intended prediction.

Use carefully controlled short prompts.

## Type P3 — Structured Distractor Prefix

Insert unrelated but syntactically plausible material.

## Type P4 — Answer-Changing Matched Control

Change a decisive fact/context so the expected prediction changes.

These are signal directions, not nuisance.

---

# B3. Local Jacobian Product

For two prompt realizations \(x\) and \(x'\), define:

\[
\delta r_\ell = r_\ell(x')-r_\ell(x).
\]

For each block:

\[
\delta r_{\ell+1}\approx J_\ell\delta r_\ell.
\]

Compute JVPs rather than dense Jacobians.

For the actual local trajectory:

\[
\widehat{\delta r}_{\ell+1}
=
J_\ell \delta r_\ell.
\]

Record:

```text
jvp_cosine
relative_norm_error
prediction_space_relative_error
```

---

# B4. Cumulative Jacobian Product

The stronger weak-form test is cumulative propagation.

Starting from a perturbation at depth \(a\):

\[
\widehat{\delta r}_b
=
J_{b-1}J_{b-2}\cdots J_a\,\delta r_a.
\]

Do not materialize matrices.

Implement repeated JVP composition:

```python
v = delta_r_a
for layer in range(a, b):
    v = JVP(F_layer, r_layer, v)
```

Compare with observed:

\[
\delta r_b.
\]

Evaluate for horizons:

```text
1 layer
2 layers
4 layers
remaining-to-final
```

Because linearization error compounds, report performance as a function of horizon.

---

# B5. Piecewise Re-Linearization Control

Compare cumulative frozen-path prediction against piecewise re-linearization.

## Frozen-path cumulative

Always propagate around the left/reference trajectory.

## Piecewise local

At each layer, use the actual current paired states and re-estimate the local perturbation.

The gap measures accumulated nonlinearity.

Report:

```text
frozen_path_error
piecewise_error
error_growth_per_layer
```

---

# B6. Decision-Space Projection

The most important pretrained quantity is not full residual norm.

For a stable final target \(Y\), define target margin:

\[
m_\ell =
z_\ell(Y)-\max_{j\neq Y}z_\ell(j).
\]

For perturbation \(\delta r_\ell\), compute approximate decision effect:

\[
\delta m_\ell
\approx
\nabla m_\ell^\top\delta r_\ell.
\]

Then ask whether cumulative Jacobian propagation predicts the observed reduction/increase of nuisance effect on the target margin.

Record:

```text
observed_margin_delta
predicted_margin_delta
margin_delta_error
sign_agreement
```

This is more directly relevant than raw hidden-space contraction.

---

# B7. Nuisance-vs-Signal Comparison

For empirically stable nuisance pairs:

\[
\delta r^N
\]

and matched answer-changing pairs:

\[
\delta r^S,
\]

compare cumulative gains:

\[
G^N_{a\to b}
=
\frac{
\|J_{b:a}\delta r_a^N\|
}{
\|\delta r_a^N\|
},
\]

\[
G^S_{a\to b}
=
\frac{
\|J_{b:a}\delta r_a^S\|
}{
\|\delta r_a^S\|
}.
\]

Primary weak hypothesis:

\[
G^N_{a\to L}
<
G^S_{a\to L}
\]

in prediction-relevant geometry for behaviorally stable nuisance pairs.

Do not require this in raw residual norm.

---

# B8. Singular-Value Proxy Without Dense SVD

Do not compute full Jacobian SVD for large pretrained models.

Estimate directional gains using:

```text
actual nuisance directions
actual signal directions
random isotropic directions
top empirical covariance directions
```

Optionally use power iteration/JVP-VJP if feasible for approximate extremal singular values of selected blocks.

But directional empirical gains are primary.

---

# B9. Pretrained Models

Use small/medium inspectable models first:

```text
Pythia-70M / 160M if available
Qwen3-0.6B
one second architecture family if practical
```

Do not expand to large models until the measurement pipeline is validated.

Pin exact revisions.

---

# B10. Required Pretrained Outputs

Generate:

```text
pretrained_jacobian_pairs.jsonl
pretrained_jacobian_local.csv
pretrained_jacobian_cumulative.csv
pretrained_jacobian_margin.csv
pretrained_jacobian_summary.md
```

Required figures:

```text
p_jvp_cosine_vs_horizon.png
p_cumulative_error_vs_horizon.png
p_nuisance_vs_signal_gain.png
p_predicted_vs_observed_margin_delta.png
p_piecewise_vs_frozen_linearization.png
p_depthwise_prediction_dispersion.png
```

---

# B11. Interpretation Rules

A positive pretrained result would support:

> The same local/cumulative dynamical mechanism observed in controlled models is detectable in pretrained networks for empirically stable perturbations.

It would **not** prove:

- the model learned a specific n-gram frequency;
- the perturbation is nuisance relative to the unknown pretraining distribution;
- the model has a universal denoising objective;
- the Jacobian product fully explains the network.

The correct claim is architectural/dynamical, not corpus-statistical.

---

# B12. Falsification

The pretrained cumulative-Jacobian hypothesis is weakened if:

- local JVPs fail even for small perturbations;
- cumulative error explodes immediately;
- nuisance and signal directional gains are indistinguishable;
- predicted target-margin changes do not match observed changes;
- only raw residual contraction is seen while prediction-space effects disagree;
- results are architecture-specific and do not replicate;
- apparent nuisance attenuation disappears after behavioral matching.

---

# PART C — PAPER INTEGRATION

## C1. Revised Evidence Hierarchy

The paper should present evidence in this order:

### Tier 1 — Controlled causal theory

Dataset V2 with exact known:

```text
G
D
Y
pattern complexity
nuisance status
```

This supports the strongest claims.

### Tier 2 — Controlled dynamical mechanism

JVP/covariance/cumulative transformation analysis inside the custom model.

### Tier 3 — Pretrained architectural external validity

Behaviorally matched perturbation pairs + cumulative Jacobian tests.

This supports the weaker statement that similar dynamical attenuation exists in pretrained models.

### Tier 4 — Legacy pretrained equivalence/manifold experiments

Keep the previous replacement/subspace results in appendix context.

---

## C2. Suggested Paper Wording

The theory claim should be framed approximately as:

> In controlled models, Transformer depth composes state-dependent transformations that can increase correct predictive signal relative to nuisance-conditioned fluctuation. Local Jacobians explain small perturbation propagation, while cumulative products capture part of the depthwise stabilization. In pretrained models, where the generating distribution is unknown, behaviorally matched perturbation pairs provide a weaker external-validity test of the same dynamical mechanism.

---

# PART D — IMPLEMENTATION ORDER

Run in this order:

1. Preserve/reproduce current Group 1.
2. Implement Dataset V2 validation.
3. Run Level 1 balanced-pair + N0/N2.
4. Add N3/N4 structured distractors.
5. Add balanced n-gram length sweep.
6. Add nested n vs n+k override.
7. Recompute correctness/order-parameter metrics.
8. Run custom-model local JVP + cumulative-JVP tests.
9. Build pretrained pair filter.
10. Run pretrained local JVP tests.
11. Run cumulative horizon tests.
12. Integrate pretrained results into main theory section only if they pass.

Do not proceed to higher complexity if lower-level validation or learning fails.

---

# Minimum Completion Gate

This addendum is complete only when:

- no-shortcut validation passes for Dataset V2;
- at least two balanced predictive generators learn to high final accuracy;
- at least three nuisance difficulty levels are tested;
- stable decision depth and margin SNR are captured;
- nested override produces a measurable short-to-long target competition trajectory;
- custom cumulative JVP results exist for at least two horizons;
- pretrained cumulative-Jacobian results exist for at least two model checkpoints/architectures or are explicitly negative;
- all generated artifacts are reproducible from manifests;
- the paper distinguishes controlled strong claims from pretrained weak claims.

---

# Final Scientific Target

The strongest supported result would be:

\[
\boxed{
\text{Depth increases predictive order by selectively attenuating
structured nuisance relative to target-relevant distinctions,
and this effect is partly predictable from the composition of
local layer Jacobians.}
}
\]

The controlled synthetic models establish the causal/statistical statement.

The pretrained models test only the weaker architectural statement:

\[
\boxed{
\text{behaviorally stable perturbations are preferentially attenuated
through cumulative layer dynamics in prediction-relevant geometry.}
}
\]
