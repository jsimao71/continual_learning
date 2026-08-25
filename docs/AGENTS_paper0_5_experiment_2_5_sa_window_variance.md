# AGENTS.md — Paper 0.5 Experiment 2.5
## SA Window × Predictive Span × Nuisance Distance

### Mission
Test whether self-attention (SA) is a major source of nuisance-conditioned predictive variance by transporting realization-specific context into the prediction position, and compare that effect against other causes: nonlinear amplification, normalization, candidate competition, and later-layer repair.

The core controlled surface is:

\[
\boxed{s_G \times s_N \times w}
\]

where:
- \(s_G\): predictive dependency span,
- \(s_N\): nuisance distance from the prediction position,
- \(w\): SA window.

Do not assume that SA is the dominant source of variance. Measure relative effect sizes.

---

## 1. Main Hypotheses

### H2.5.1 — SA transports both signal and nuisance
If nuisance is outside the causal window, it should not affect the prediction position. Once it becomes reachable, post-SA nuisance-conditioned variance may rise.

### H2.5.2 — Signal-preserving restricted window
The critical regime is:

\[
s_G \le w < s_N.
\]

Here the model can access all predictive evidence but not distant nuisance. If SA transport is a major source of fluctuation, this regime should preserve signal while reducing nuisance variance relative to full attention.

### H2.5.3 — Sublayer variance injection
At each block measure:

\[
r_{\ell,\mathrm{preSA}}
\to
r_{\ell,\mathrm{postSA}}
\to
r_{\ell,\mathrm{postFF}}.
\]

Define:

\[
\Delta N_\ell^{SA}
=
N_{\ell,\mathrm{postSA}}^2-
N_{\ell,\mathrm{preSA}}^2,
\]

\[
\Delta N_\ell^{FF}
=
N_{\ell,\mathrm{postFF}}^2-
N_{\ell,\mathrm{postSA}}^2.
\]

A positive \(\Delta N_\ell^{SA}\) after nuisance becomes reachable supports SA-mediated variance import.

### H2.5.4 — Relative, not absolute, stabilization
Use:

\[
S_\ell=E_d[m_\ell],
\qquad
N_\ell^2=\operatorname{Var}_d[m_\ell],
\qquad
\Gamma_\ell=\frac{S_\ell}{\sqrt{N_\ell^2+\epsilon}}.
\]

The main success criterion is growing correct-target signal and SNR, not necessarily decreasing absolute variance.

---

## 2. Dataset Requirements

Use Dataset V2 no-shortcut rules. The target must not be identifiable from a single token.

Preferred base rule:

\[
Y=f(A,B)
\]

using balanced Latin-square/modular mappings such that:

\[
I(A;Y)\approx 0,\qquad I(B;Y)\approx 0,
\]

while:

\[
H(Y\mid A,B)=0.
\]

Nuisance must satisfy:

\[
I(D;Y\mid G)=0.
\]

Use train/test surface-identity separation.

---

## 3. Three-Variable Grid

### Predictive span
\[
s_G\in\{1,2,4,8,16\}.
\]

### Nuisance distance
\[
s_N\in\{2,4,8,16,32\}.
\]

### SA window
\[
w\in\{1,2,4,8,16,\mathrm{full}\}.
\]

Use depths:

\[
L\in\{2,4,6,8\}.
\]

Start with a sparse designed grid; do not launch the full Cartesian product immediately.

Tag every cell as:

```text
signal_unreachable
signal_only_reachable
signal_and_nuisance_reachable
nuisance_reachable_before_full_signal
full_attention
```

---

## 4. Nuisance Difficulty Levels

Run at least:

### N1 — Same-vocabulary random nuisance
Distractors sampled from the same token pool as predictive inputs.

### N2 — Partial-pattern nuisance
Distractors contain fragments of valid predictive patterns.

### N3 — Complete competing-pattern nuisance
Distractors contain full learned relations that are not the designated target relation.

Optional:

### N4 — Near-target competitor nuisance
Distractors preferentially support the strongest incorrect target.

---

## 5. Required Measurements

At every:

```text
initial
pre-SA
post-SA
post-FF/post-block
```

record:

```text
target_logit
target_probability
target_rank
target_margin
strongest_competitor
top1_correct
output_entropy
within_family_JS
between_family_JS
margin_variance
margin_SNR
residual_covariance_trace
```

For selected runs also retain:

```text
SA_update
FF_update
per_head_output
attention_weights
pre_norm_state
post_norm_state
```

---

## 6. Primary Surfaces

Generate:

\[
S_L(s_G,s_N,w),
\]

\[
N_L^2(s_G,s_N,w),
\]

\[
\Gamma_L(s_G,s_N,w),
\]

\[
P(\mathrm{top1}=Y\mid s_G,s_N,w),
\]

\[
L_Y^{stable}(s_G,s_N,w).
\]

Required figures:

```text
g25_signal_surface.png
g25_margin_variance_surface.png
g25_margin_snr_surface.png
g25_accuracy_surface.png
g25_stable_decision_depth_surface.png
g25_signal_only_vs_full_attention.png
```

---

## 7. Direct SA Transport Test

For every layer plot:

\[
\Delta N_\ell^{SA}
\]

against window and nuisance distance.

Key prediction:

> SA-induced variance should show a reachability transition when nuisance enters the causal cone.

Also compare:

\[
\Delta N_\ell^{SA}
\quad\text{vs}\quad
\Delta N_\ell^{FF}.
\]

Required figures:

```text
g25_delta_variance_sa_vs_window.png
g25_delta_variance_sa_vs_nuisance_distance.png
g25_sa_reachability_threshold.png
g25_sa_vs_ff_variance_change.png
```

---

## 8. Alternative Explanations for Variance

### A. Nonlinear signal amplification
Absolute variance can rise simply because the whole decision variable is amplified.

Measure:

\[
CV_\ell=
\frac{\sqrt{N_\ell^2}}{|S_\ell|+\epsilon}.
\]

If absolute variance rises while \(CV_\ell\) falls, signal amplification is a strong explanation.

### B. Normalization coupling
Measure variance before and after normalization:

\[
\Delta N_\ell^{Norm}
=
N_{\ell,\mathrm{postNorm}}^2-
N_{\ell,\mathrm{preNorm}}^2.
\]

### C. Attention-weight modulation
Measure:

\[
A_{\mathrm{signal},\ell}
=
\sum_{j\in G}a_{tj},
\qquad
A_{\mathrm{nuis},\ell}
=
\sum_{j\in D}a_{tj}.
\]

Track whether nuisance changes attention mass on relevant tokens, not only imported nuisance values.

### D. Candidate competition
Record:

```text
competitor_entropy
competitor_switch_rate
top1_reversal_count
margin_variance_conditional_on_competitor
```

### E. Later-layer repair
For margin updates:

\[
a_{\ell i}
=
\mu_\ell^\Delta+\epsilon_{\ell i}^\Delta.
\]

Compute cross-layer covariance:

\[
C_{\ell k}
=
\operatorname{Cov}(\epsilon_\ell^\Delta,\epsilon_k^\Delta).
\]

Define:

```text
repair_fraction =
1 - actual_cumulative_variance / summed_marginal_variance
```

when valid.

---

## 9. Variance Attribution Model

Fit descriptive models of layerwise margin variance using predictors:

```text
SA nuisance attention mass
SA signal attention mass
window size
nuisance distance
predictive span
post-SA residual norm
normalization variance change
competitor switch rate
mean target margin
layer index
```

Use:
- mixed-effects linear model,
- regularized regression,
- tree model as nonlinear descriptive control.

Primary inferential units are predictive family, model seed, and architecture cell.

Report partial explanatory strength for grouped mechanisms:

```text
transport
normalization
competition
signal amplification
repair
```

---

## 10. Causal Window Intervention

Where possible use the same trained model and change effective window at inference time:

```text
trained full / evaluated full
trained full / truncated window
trained local / evaluated local
```

This separates immediate information-exposure effects from training adaptation.

Strong evidence for nuisance transport would be:

> a full-attention-trained model preserves target signal when distant nuisance is masked, while nuisance-conditioned variance falls.

---

## 11. Selective Nuisance Masking

Under full attention selectively block:

```text
nuisance positions only
signal positions only
matched random positions
```

Measure:

\[
\Delta S,\quad
\Delta N^2,\quad
\Delta\Gamma,\quad
\Delta\mathrm{accuracy}.
\]

Desired diagnostic pattern:

```text
mask nuisance:
    signal preserved
    variance decreases

mask signal:
    signal collapses
    accuracy falls
```

This is stronger evidence than window manipulation alone.

---

## 12. Distance-Specific Nuisance Arrival

Place equal-size nuisance bands at controlled distances:

```text
near
middle
far
```

Estimate the first layer at which each band measurably affects target margin.

Compare the observed arrival depth with causal reachability from \(w\) and \(L\).

Generate:

```text
g25_nuisance_arrival_by_distance.png
```

---

## 13. Interaction With Depth / Information Exposure

Test whether deeper models tolerate wider information exposure.

Define an empirical tolerated exposure budget:

\[
w^\star(L)
\]

or nuisance-access budget:

\[
k^\star(L)
\]

as the largest exposure satisfying thresholds such as:

```text
accuracy >= threshold
margin_SNR >= threshold
```

Do not claim a scaling law unless enough independent depth points exist.

This is the controlled bridge to PRA bounded materialization.

---

## 14. PRA / “Less Is More” Connection

Keep this as a theory bridge, not a primary claim.

Interpret SA window as a controlled information-exposure budget.

PRA materialization budget is an analogous external mechanism.

Shared hypothesis:

\[
\text{more accessible information}
\not\Rightarrow
\text{better prediction}.
\]

Test whether a signal-preserving restricted window can outperform full exposure in:

```text
margin_SNR
stable decision depth
accuracy
within-family_JS
```

---

## 15. Required Result Files

Generate:

```text
group25_surface_metrics.csv
group25_sublayer_variance.csv
group25_sa_ff_variance_attribution.csv
group25_attention_mass.csv
group25_competitor_dynamics.csv
group25_repair_covariance.csv
group25_variance_models.csv
group25_window_interventions.csv
group25_summary.md
```

---

## 16. Summary Questions

`group25_summary.md` must answer:

1. Does margin variance increase when nuisance becomes SA-reachable?
2. Is the threshold controlled by \(w\), \(s_N\), and depth?
3. Is there a regime \(s_G\le w<s_N\) with high signal and lower variance?
4. How much variance is injected immediately by SA?
5. How much is added or removed by FF?
6. How much is attributable to normalization?
7. How much is associated with attention-weight shifts on predictive tokens?
8. How much is associated with competitor switching?
9. How much later fluctuation is cancelled by negative cross-layer covariance?
10. Does inference-time nuisance masking reduce variance without damaging target signal?
11. Does broader information exposure ever reduce correctness or margin SNR?
12. Does depth increase tolerance to wider exposure?
13. Which mechanism explains the largest share of variance?

---

## 17. Falsification Criteria

The SA-transport hypothesis is weakened if:

- variance does not change when nuisance crosses the reachability boundary;
- masking nuisance positions does not reduce variance;
- post-SA variance does not rise relative to pre-SA;
- window size has little effect after signal is reachable;
- normalization or nonlinear amplification explains most variance;
- competitor switching explains nearly all variance;
- later-layer repair is absent.

The “less is more” bridge is weakened if:

- signal-preserving restricted windows never outperform full exposure;
- nuisance access raises variance but has no effect on correctness, SNR, or settling;
- deeper models show no interaction with tolerated exposure.

---

## 18. Interpretation Rules

Do not write “SA injects noise” unless causal masking/window tests support it.

Prefer:

> SA transports realization-specific context, which can increase decision-variable variance.

Separate:

```text
absolute variance
relative fluctuation
decision-boundary risk
functional output divergence
```

The main success variables are margin SNR and correctness, not variance reduction alone.

---

## 19. Implementation Order

1. Validate Dataset V2 balanced rules.
2. Add \(s_G,s_N,w\) metadata and reachability checks.
3. Run same-vocabulary nuisance.
4. Add pre-SA/post-SA/post-FF variance tracing.
5. Compare signal-only-reachable vs full-access regimes.
6. Add selective nuisance masking.
7. Add partial-pattern nuisance.
8. Add complete competing patterns.
9. Add attention-mass and competitor analyses.
10. Add normalization control.
11. Add cross-layer repair covariance.
12. Fit variance-attribution models.
13. Add depth × tolerated-exposure analysis.
14. Integrate Experiment 2.5 between Investigations II and III.

---

## Final Scientific Target

The strongest supported result would be:

\[
\boxed{
\text{SA expands the information available at the prediction state,
including both task-relevant and nuisance components. Layer composition
then amplifies common predictive signal and partially repairs
realization-specific deviations. Restricting information exposure can
therefore improve predictive signal-to-fluctuation ratio when excluded
context is causally irrelevant.}
}
