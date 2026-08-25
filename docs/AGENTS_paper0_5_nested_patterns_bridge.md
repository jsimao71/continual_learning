# AGENTS.md — Paper 0.5 Nested-Pattern Representation Bridge
## Shared Core, Contextual Extension, and Override Across n, n+1, and n+k Patterns

### Mission

Add a focused bridge experiment between Paper 0.5 (predictive layer composition) and Paper 0.6 (semantic/abstraction structure).

The experiment asks:

> When a shorter predictive pattern is embedded inside a longer pattern, how much of the shorter pattern's internal computation is preserved, reused, reinforced, or overridden?

Study controlled families such as:

\[
ABC \to Y
\]

versus:

\[
XABC \to Y
\]

\[
ABCX \to Y
\]

and:

\[
XABC \to Z.
\]

The goal is not just behavioral accuracy. Measure the relation between the internal representations and transformations of the short and long patterns across depth.

## 1. Core Scientific Questions

1. Does a longer pattern that preserves the same target reuse the same residual/update representation as the shorter pattern?
2. Does supportive extra context reinforce an existing predictive representation or create a distinct one?
3. In override cases, do short and long patterns initially share a trajectory and diverge only after contextual information is integrated?
4. Are the same heads causally important for the shared core pattern?
5. Are additional heads recruited specifically for contextual refinement or override?
6. Is representational reuse stronger at the level of block updates than raw residual states?
7. Is causal substitutability stronger for nested same-target patterns than for matched nonequivalent controls?
8. Can these relations be summarized as a hierarchy of predictive equivalence classes suitable for Paper 0.6?

## 2. Controlled Pattern Families

Use Dataset V2 no-shortcut construction.

No singleton or proper subset should uniquely determine the target unless explicitly part of a controlled short-rule condition.

### Family A — Irrelevant Extension

Short:
\[
ABC \to Y
\]

Long:
\[
XABC \to Y.
\]

Require:
\[
Y \perp X \mid ABC.
\]

Also test suffix extension:
\[
ABCX \to Y.
\]

Purpose: test invariance under causally irrelevant extension.

### Family B — Supportive / Redundant Extension

Short:
\[
ABC \to Y.
\]

Long:
\[
XABC \to Y
\]

where \(X\) is correlated with the same predictive class but is not necessary for correctness.

Require:
\[
I(X;Y)>0
\]

but:
\[
H(Y\mid ABC)=0.
\]

Purpose: test whether additional compatible context reinforces the same internal representation.

### Family C — Refining Extension

Short rule:
\[
BC \to Y_1.
\]

Longer rule:
\[
ABC \to Y_2.
\]

The shorter rule is valid when \(A\) is absent.

Purpose: test hierarchical refinement of a valid shorter predictive class.

### Family D — Override Extension

Short:
\[
ABC \to Y_1.
\]

Long:
\[
XABC \to Y_2.
\]

Require \(Y_1 \neq Y_2\).

The shorter relation must be well learned independently.

Purpose: test whether the long pattern initially inherits the shorter trajectory and later diverges.

### Family E — Multi-Level Nested Hierarchy

Construct:
\[
C \to Y_0,
\]
\[
BC \to Y_1,
\]
\[
ABC \to Y_2,
\]
\[
XABC \to Y_3.
\]

Use balanced conditional rules so every added context variable genuinely partitions the previous predictive class.

Purpose: test progressively finer predictive equivalence classes.

## 3. Required Controls

For every nested pair include:

- Same-target unrelated control: \(DEF \to Y\)
- Matched nonequivalent control: \(ABD \to Z\)
- Surface perturbation control: same predictive relation with different nuisance realization
- Position control: aligned and randomized absolute positions
- Random initialization control

## 4. Depthwise State Capture

At every boundary record:

```text
residual_pre_sa
sa_update
residual_post_sa
ff_update
residual_post_block
target_logits
target_probability
target_rank
target_margin
top1_token
```

For selected runs retain:

```text
per_head_output
attention_weights
Q
K
V
```

## 5. Residual Similarity

For short pattern \(g_n\) and long pattern \(g_{n+k}\), define:

\[
C_\ell^{res}
=
\cos\left(
r_\ell(g_n),
r_\ell(g_{n+k})
\right).
\]

Also compute centered similarity after subtracting family means.

Report:

```text
raw_residual_cosine
centered_residual_cosine
CKA_or_linear_similarity
L2_distance
```

Compare:

```text
nested_same_target
same_target_unrelated
nested_override
nonequivalent_control
```

## 6. Update Similarity

Primary metric:

\[
\Delta r_\ell
=
r_{\ell+1}-r_\ell.
\]

Measure:

\[
C_\ell^{update}
=
\cos\left(
\Delta r_\ell(g_n),
\Delta r_\ell(g_{n+k})
\right).
\]

Separately measure SA and FF update similarity:

\[
C_\ell^{SA}
=
\cos\left(
\Delta r_\ell^{SA}(g_n),
\Delta r_\ell^{SA}(g_{n+k})
\right),
\]

\[
C_\ell^{FF}
=
\cos\left(
\Delta r_\ell^{FF}(g_n),
\Delta r_\ell^{FF}(g_{n+k})
\right).
\]

Hypothesis: same-target nested patterns should show greater update similarity than matched nonequivalent controls.

## 7. Representation Retention Curve

Define:

\[
\rho_\ell(n,n+k)=C_\ell^{update}.
\]

Also define normalized retention:

\[
\rho_\ell^{norm}
=
\frac{
C_\ell^{nested}-C_\ell^{nonequiv}
}{
1-C_\ell^{nonequiv}+\epsilon
}.
\]

Interpretation:

```text
rho ~ 1   -> strong reuse of short-pattern computation
rho ~ 0   -> no extra nested similarity beyond control
rho < 0   -> long pattern actively diverges from short pattern
```

## 8. Core + Context Decomposition

Test whether:

\[
r_\ell(g_{n+k})
\approx
\alpha_\ell r_\ell(g_n)+c_\ell.
\]

Fit:

\[
\alpha_\ell
=
\frac{
r_\ell(g_{n+k})^\top r_\ell(g_n)
}{
\|r_\ell(g_n)\|^2+\epsilon
}.
\]

Define contextual residual:

\[
c_\ell
=
r_\ell(g_{n+k})
-
\alpha_\ell r_\ell(g_n).
\]

Measure:

```text
alpha_depth
context_component_norm
context_component_target_alignment
```

Treat this as descriptive, not unique.

## 9. Predictive-Subspace Retention

For target \(Y\):

\[
v_{\ell,Y}
=
\nabla_{r_\ell}
\left[
z_\ell(Y)-z_\ell(c^*)
\right].
\]

Measure target alignment for short and long updates.

For override cases measure both short and long targets:

```text
short_target_alignment
long_target_alignment
alignment_crossing_layer
```

## 10. Causal Update Replacement

For target long example \(g_{n+k}\), replace the update at layer \(\ell\) with the update from:

1. embedded short pattern \(g_n\)
2. unrelated same-target pattern
3. matched nonequivalent pattern

Measure:

```text
target_logprob_delta
target_margin_delta
JS_to_intact
final_top1_change
```

Primary hypothesis for same-target nesting:

\[
\text{damage(short nested donor)}
<
\text{damage(unrelated same-target donor)}
<
\text{damage(nonequivalent donor)}.
\]

For override conditions, short-pattern donor replacement should become increasingly harmful after the divergence layer.

## 11. Divergence Layer

Define representational divergence layer \(L_{div}\) from nested-similarity drop relative to controls.

Define behavioral divergence layer:

\[
L_{Y_2>Y_1}
=
\min\{\ell:z_\ell(Y_2)>z_\ell(Y_1)\}.
\]

Compare whether representational divergence precedes, coincides with, or follows behavioral override.

## 12. Head-Level Reuse

For every head \(h\) at layer \(\ell\), compare:

\[
\Delta r_{\ell h}(g_n)
\]

and:

\[
\Delta r_{\ell h}(g_{n+k}).
\]

Compute:

```text
head_output_cosine
head_output_norm_ratio
head_target_alignment
```

Build a layer × head × nesting-depth matrix.

## 13. Head Causal Overlap

Obtain causal head utility vectors:

\[
U^{(n)}_{\ell h}
\]

and:

\[
U^{(n+k)}_{\ell h}.
\]

Compare:

```text
Spearman utility correlation
top-k head overlap
Jaccard overlap
rank-biased overlap
```

Hypotheses:

- same-target extension: high overlap
- supportive extension: core heads preserved + extra supportive heads
- override: early overlap high, later overlap falls as override-specific heads become important

## 14. Head Role Categories

Classify heads only with causal evidence:

```text
core_pattern_head
context_support_head
override_head
nuisance_sensitive_head
distributed/no_stable_role
```

## 15. Attention Topology as Secondary Evidence

Measure whether the same heads attend to the same core token relations across \(n\) and \(n+k\), but preserve the Paper 0.5 motif null.

Correlate motif reuse with causal head reuse; do not assume the correlation is positive.

## 16. Nested Hierarchy / Equivalence-Class Analysis

For:

\[
C \subset BC \subset ABC \subset XABC
\]

construct a similarity matrix at each depth using:

```text
residual similarity
update similarity
causal replacement similarity
head-utility similarity
```

Test whether geometry reflects nesting.

## 17. Bridge to Paper 0.6

The controlled bridge claim should be narrow:

> Transformers can preserve and refine internal predictive structure across nested contexts, providing a controlled precursor to studying semantic abstraction hierarchies.

Do not claim semantic abstraction from lexical nesting alone.

## 18. Required Experimental Matrix

At minimum:

```text
relation_types:
  - irrelevant_extension
  - supportive_extension
  - refining_extension
  - override_extension
  - multi_level_hierarchy

base_lengths:
  - 2
  - 3
  - 4

extension_lengths:
  - +1
  - +2
  - +4

seeds:
  - 3 minimum

positions:
  - aligned
  - randomized
```

Use enough independent relation families for family-level bootstrap intervals.

## 19. Required Figures

Generate:

```text
nested_residual_similarity_vs_depth.png
nested_update_similarity_vs_depth.png
nested_sa_ff_similarity_vs_depth.png
nested_retention_vs_depth.png
nested_core_context_decomposition.png
nested_override_target_crossing.png
nested_replacement_damage_vs_depth.png
nested_head_similarity_matrix.png
nested_head_utility_overlap.png
nested_divergence_vs_behavioral_override.png
nested_hierarchy_similarity_matrix.png
```

## 20. Required Tables

Generate:

```text
nested_pair_metrics.csv
nested_update_similarity.csv
nested_replacement_results.csv
nested_divergence_layers.csv
nested_head_similarity.csv
nested_head_causal_overlap.csv
nested_hierarchy_metrics.csv
```

## 21. Summary Questions

Create:

```text
docs/papers/paper0_5/results/nested_patterns/nested_summary.md
```

Answer:

1. Are raw residuals similar between \(n\) and \(n+k\) nested patterns?
2. Are block updates more similar than residual states?
3. Are SA or FF updates more strongly reused?
4. Does same-target nesting show more similarity than unrelated same-target controls?
5. Does causal update replacement support nested representation reuse?
6. Is there a measurable divergence layer in override cases?
7. Does representational divergence align with target-margin crossing?
8. Are the same heads causally important across \(n\) and \(n+k\)?
9. Are extra heads recruited for support/refinement?
10. Are distinct override heads recruited when the target changes?
11. Does head causal overlap fall at the same depth as residual/update divergence?
12. Does a multi-level hierarchy produce structured similarity geometry?
13. Which level—residual, update, head output, or causal utility—best preserves nested structure?
14. Is the effect robust across seeds and randomized positions?
15. What exact result should be carried into Paper 0.6?

## 22. Falsification Criteria

The reusable-core hypothesis is weakened if:

- same-target nested patterns are no more similar than unrelated same-target controls
- embedded-short update replacement is no better than nonequivalent replacement
- head causal overlap is no higher for nested same-target pairs
- override divergence occurs immediately with no shared early trajectory
- nested geometry is unstable across seeds
- absolute position explains the similarity
- similarity exists at random initialization and does not strengthen with training

The Paper 0.6 bridge is weakened if no metric shows systematic relation between nesting depth and internal computation.

## 23. Interpretation Rules

Do not equate high cosine with identical representation.

Prefer causal replacement and matched controls.

Do not call a head a core-pattern head without causal evidence.

Distinguish:

```text
surface similarity
target identity
predictive-rule nesting
causal representation reuse
```

## 24. Implementation Order

1. Build irrelevant-extension and override generators.
2. Validate no-shortcut/MI constraints.
3. Add residual and block-update similarity.
4. Add SA/FF similarity.
5. Add matched same-target and nonequivalent controls.
6. Run causal update replacement.
7. Add divergence-layer / target-crossing analysis.
8. Add per-head output similarity.
9. Add causal head-utility overlap.
10. Add supportive extension.
11. Add multi-level hierarchy.
12. Integrate a short bridge section into Paper 0.5.
13. Carry resulting hierarchy metrics into Paper 0.6 planning.

## Final Scientific Target

\[
\boxed{
\text{Longer predictive patterns reuse a measurable core computation from
their embedded shorter patterns when the target is preserved, while
context-dependent refinement recruits additional transformations and
override cases diverge only when the longer context changes the predictive
class.}
\]
