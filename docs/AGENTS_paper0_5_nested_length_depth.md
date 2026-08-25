# AGENTS.md — Paper 0.5 Nested Length-vs-Depth Experiment
## Does Longer Predictive Structure Require More Transformer Depth?

### Mission
Run a controlled experiment to determine whether increasing predictive-pattern length increases the depth required for the correct target to emerge and stabilize.

Distinguish raw sequence length from predictive integration complexity.

Primary observables:
- first top-1 layer
- stable top-1 layer
- settling delay
- minimum architecture depth required for competence
- target-margin growth by layer
- nested-computation reuse
- causal head recruitment

Central question:

L_stable = f(pattern length, predictive necessity, dependency span, nuisance)

### Hypotheses

1. Raw-length hypothesis:
Longer patterns require more settling depth.

2. Predictive-necessity hypothesis:
Depth depends more on how many tokens are actually required to determine Y than on raw token count.

3. Reuse hypothesis:
Redundant/supportive same-target extensions reuse an existing short-pattern computation and therefore add little or no settling depth.

4. Integration-cost hypothesis:
Necessary higher-order patterns require later stable decisions and/or deeper architectures.

5. Head-recruitment hypothesis:
Longer necessary patterns recruit additional causal heads or recruit them later.

### Pattern ladder
Use Dataset V2 no-shortcut construction with nested lengths:
n = [2, 3, 4, 6, 8]

Create matched families sharing a core.

#### Regime A — Redundant extension
AB -> Y
ABC -> Y
ABCD -> Y

Extra tokens are conditionally irrelevant:
Y independent of extra tokens given the short core.

Purpose: test raw length alone.

#### Regime B — Supportive extension
AB -> Y
ABC -> Y
ABCD -> Y

Extra tokens provide compatible evidence but are not necessary.

Purpose: test whether added supporting evidence accelerates, leaves unchanged, or delays stabilization.

#### Regime C — Necessary extension
Y = f(X1,...,Xn)

Use balanced modular/parity/lookup constructions.

Require proper subsets to have low target information while the full pattern is deterministic or near-deterministic.

Purpose: test true predictive-integration cost.

#### Optional Regime D — Hierarchical refinement
X1X2 -> Y2
X1X2X3 -> Y3
X1X2X3X4 -> Y4

Interpret only if all lengths clear the competence gate.

### Competence gate
For every length/regime/seed/architecture:
- preferred final held-out accuracy >= 0.80
- minimum descriptive threshold >= 0.70
- near-chance cells excluded from positive mechanistic claims

Always report exclusions.

### No-shortcut validation
For every length/regime compute:
- I(single token; Y)
- max I(proper subset; Y)
- I(full pattern; Y)

For necessary extensions:
- proper-subset MI below configured threshold
- full-pattern MI high

Generate:
nested_length_mi_validation.csv
nested_length_dataset_validation.json

### Primary layerwise metrics
At every sublayer boundary record:
- target logit
- target probability
- target rank
- target margin
- top-1 correctness
- strongest competitor
- entropy

Compute:
L_first = first layer where Y becomes top-1
L_stable = first layer after which Y remains top-1
settling_delay = L_stable - L_first
top1_reversal_count

### Matched nested depth comparison
For each shared-core family:
AB
ABC
ABCD
ABCDEF

Compute paired:
Delta L_stable(n,n+k) = L_stable(long) - L_stable(short)

This paired quantity is primary.

Bootstrap over predictive families and seeds.

### Two different depth questions

A. Internal decision depth:
Within one trained model, how many layers are used before Y stabilizes?

B. Architectural minimum depth:
Train models with layers = [2,4,6,8,12].
Define:
L_min(n) = minimum architecture depth achieving target competence.

Do not conflate these.

### Margin-growth analysis
For each n:
m_l(n) = z_l(Y) - z_l(strongest competitor)

Measure:
- mean margin by depth
- early margin slope
- late margin slope
- layer of maximum margin gain
- per-layer delta margin

Test whether necessary longer patterns shift major margin growth later.

### Scaling analysis
Candidate dependent variables:
- first-top1 depth
- stable-top1 depth
- architectural minimum depth
- final margin
- margin-growth rate

Candidate predictors:
- raw length n
- minimum sufficient predictive order p*
- dependency span s
- number of indispensable tokens
- nuisance count
- conditional target entropy

Compare:
- constant/null
- linear
- logarithmic
- power law
- piecewise linear
- saturating

Report:
R2, AIC/BIC, held-out error, confidence intervals, fit range, seed stability, architecture stability.

Do not claim a power law unless it clearly beats alternatives.

### Predictive-order metric
Define p* as the minimum subset size sufficient to predict Y below a chosen conditional-entropy threshold.

Test whether:
L_stable = f(p*)
fits better than:
L_stable = f(n)

This is the central theory comparison.

### Span controls
Separate token count from distance.

Compare:
- same length, different span
- different length, similar span

Examples:
ABCD contiguous
A _ B _ C _ D sparse

Analyze raw length, span, predictive order separately.

### Nuisance controls
Repeat selected cells with nuisance counts:
k = [0,2,4,8]

Fit a grouped model such as:
L_stable = beta0 + beta_n*n + beta_p*p* + beta_s*span + beta_k*k + interactions

Primary grouping:
- predictive family
- seed
- architecture

### Representation-reuse link
Reuse the nested-pattern bridge metrics:
- block-update cosine
- SA-update cosine
- FF-update cosine
- causal replacement damage

Test:
high reuse -> smaller added stable-depth cost

Compare redundant/supportive vs necessary extensions.

### Head recruitment
For every length:
- per-head causal utility
- top-k head set
- utility Spearman correlation across lengths
- Jaccard overlap
- number of newly recruited heads
- last newly recruited layer

Test whether predictive order increases causal head recruitment or recruitment depth.

### SA vs FF timing
Measure per-layer target-margin contribution from SA and FF.

Test:
- whether SA contribution shifts later with longer span
- whether FF reuse remains high for redundant same-target extensions
- whether necessary higher-order integration requires additional SA-mediated contextual combination before FF stabilization

### Required figures
nested_length_first_top1.png
nested_length_stable_top1.png
nested_length_settling_delay.png
depth_length_accuracy_phase.png
nested_length_margin_trajectory.png
nested_length_margin_gain_layer.png
predictive_order_vs_stable_depth.png
raw_length_vs_predictive_order_fit.png
nested_length_update_reuse.png
nested_length_head_overlap.png
nested_length_new_head_recruitment.png
nested_length_span_nuisance_partial_effects.png

### Required tables
nested_length_metrics.csv
nested_length_decision_depth.csv
nested_length_accuracy_by_model_depth.csv
nested_length_scaling_fits.csv
nested_length_predictive_order.csv
nested_length_update_reuse.csv
nested_length_head_recruitment.csv
nested_length_partial_effects.csv

### Required summary
Create:
docs/papers/paper0_5/results/nested_length_depth/nested_length_summary.md

Answer:
1. Does raw pattern length increase first-top1 depth?
2. Does raw length increase stable-top1 depth?
3. Do longer patterns require deeper architectures to learn?
4. Is any relation linear/log/power/saturating or unsupported?
5. Do redundant extensions require additional depth?
6. Do supportive extensions accelerate or delay stabilization?
7. Do necessary extensions require additional depth?
8. Does predictive order explain depth better than raw length?
9. Does span explain depth after predictive order is controlled?
10. Does nuisance explain more variance than length?
11. Does stronger computation reuse predict smaller added depth?
12. Are new heads recruited as predictive order increases?
13. Do SA/FF timing profiles change with length?
14. Are results stable across seeds and architecture depths?
15. What exact result should be carried into Paper 0.6?

### Falsification criteria
- Raw-length hypothesis is falsified if stable decision depth is flat across n after competence filtering.
- Predictive-order hypothesis is weakened if p* does not explain depth better than raw n.
- Reuse hypothesis is weakened if redundant/supportive extensions show the same depth penalty as necessary extensions.
- Architectural-depth hypothesis is weakened if L_min does not rise with predictive complexity.

Do not interpret failed near-chance models as evidence that more depth is needed unless deeper models actually recover competence.

### Interpretation rules
Do not write "longer n-grams need more layers" unless matched nested comparisons and architecture-depth sweeps support it.

Prefer "higher predictive integration order requires more depth" only if predictive-order metrics outperform raw length.

Keep separate:
- sequence length
- dependency span
- predictive order
- nuisance count

### Implementation order
1. Build redundant/supportive/necessary matched nested ladders.
2. Validate MI/no-shortcut constraints.
3. Run one competent 8-layer smoke model.
4. Capture first/stable decision depth.
5. Add architecture sweep [2,4,6,8,12].
6. Fit raw-length curves.
7. Compute predictive-order metric.
8. Compare raw length vs predictive order.
9. Add nested-computation reuse metrics.
10. Add head recruitment.
11. Add nuisance and span controls.
12. Integrate focused results into Paper 0.5.
13. Carry the supported variable—or a clean null—into Paper 0.6.

### Final scientific target

Strong positive result:
Transformer depth tracks predictive integration complexity rather than raw token length: redundant same-target extensions reuse an existing computation with little additional settling depth, while genuinely higher-order patterns require later stable decisions and/or deeper architectures.

Clean null result:
If competent, matched no-shortcut patterns of increasing length settle at the same depth, then raw n-gram length is not a primary determinant of required Transformer depth.
