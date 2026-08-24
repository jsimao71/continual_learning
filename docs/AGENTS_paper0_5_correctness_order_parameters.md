# AGENTS.md Addendum — Paper 0.5
## Correct-Prediction Order Parameters, Signal-to-Noise Across Depth, and Empirical Scaling Laws

### Scope

Extend Experiment Group 1 of Paper 0.5 so that predictive stability is not measured only as lower nuisance-conditioned dispersion. The analysis must also establish that the model is converging toward the **correct** target \(Y\).

The central question is:

> Does depth increase the mean evidence for the correct predictive class while reducing the nuisance-induced fluctuation of that evidence?

This addendum is mandatory for the next Paper 0.5 iteration.

Do not use low dispersion alone as evidence of successful stabilization. A random or collapsed model can have very low variance while predicting the wrong answer everywhere.

## 1. Correctness-Aware Layer Metrics

For every evaluation example, every sublayer boundary, and every depth level, record:

```text
target_token
target_logit
target_probability
target_rank
strongest_competitor_token
strongest_competitor_logit
target_margin
top1_token
top1_probability
prediction_correct
output_entropy
full_output_distribution_hash
```

Define target margin:

\[
m_\ell(x)=z_\ell(Y)-\max_{j\neq Y}z_\ell(j).
\]

Positive target margin means \(Y\) is top-1.

For every nuisance realization \(d\) of predictive family \(g\), compute:

```text
first_top1_layer
stable_top1_layer
top1_reversal_count
final_correct
```

Definitions:

\[
L_Y^{first}=\min\{\ell:\operatorname{rank}_\ell(Y)=1\},
\]

\[
L_Y^{stable}=\min\{\ell:\operatorname{rank}_j(Y)=1\;\forall j\ge\ell\}.
\]

If \(Y\) never becomes top-1, encode these as missing/sentinel values, never as the final layer.

Define settling delay:

\[
\Delta L_Y=L_Y^{stable}-L_Y^{first}.
\]

## 2. Required Depth Trajectories

For every predictive family and nuisance level, aggregate across realizations:

```text
mean_target_probability
var_target_probability
mean_target_logprob
mean_target_margin
var_target_margin
mean_target_rank
fraction_top1
fraction_stably_top1
mean_output_entropy
within_family_JS
between_family_JS
```

The core 2D trajectory is:

\[
(E_d[p_\ell(Y)],\operatorname{Var}_d[p_\ell(Y)]).
\]

Interpret quadrants as:

```text
low signal, high dispersion   -> unformed / noisy
low signal, low dispersion    -> collapse / uniformly wrong risk
high signal, high dispersion  -> correct but fragile
high signal, low dispersion   -> stable correct prediction
```

## 3. Primary Order Parameters

### 3.1 Correct-Target Signal

\[
S_\ell=E_d[m_\ell(g,d)].
\]

Also retain:

\[
P_\ell=E_d[p_\ell(Y)].
\]

Do not use probability alone because saturation near 0 or 1 can hide useful logit-space structure.

### 3.2 Fluctuation Amplitude

\[
N_\ell^2=\operatorname{Var}_d[m_\ell(g,d)].
\]

Also record:

```text
var_target_probability
within_family_JS
representation_covariance_trace
```

### 3.3 Margin Signal-to-Noise Ratio

\[
\Gamma_\ell=\frac{S_\ell}{\sqrt{N_\ell^2+\epsilon}}.
\]

Also report robust SNR:

\[
\Gamma_\ell^{robust}=
\frac{\operatorname{median}_d(m_\ell)}
{1.4826\,\operatorname{MAD}_d(m_\ell)+\epsilon}.
\]

Interpretation:

```text
Gamma < 0    -> mean prediction favors a competitor
Gamma ~ 0    -> unresolved
Gamma > 0    -> correct target dominates on average
large Gamma  -> correct target dominates reliably across nuisance
```

Always report numerator and denominator separately.

### 3.4 Predictive-Class Separation

Preserve:

\[
R_\ell=
\frac{D_{\mathrm{between}}(\ell)}
{D_{\mathrm{within}}(\ell)+\epsilon}.
\]

This measures class organization but not correctness. Always report it jointly with \(S_\ell\), \(\Gamma_\ell\), and top-1 fraction.

### 3.5 Representation-Space Order Parameter

Preserve:

\[
\chi_\ell=
\frac{q_\ell}{m_\ell^{class}+\epsilon},
\]

with

\[
q_\ell=E_g[\operatorname{Tr}\Sigma_{\ell,g}]
\]

and

\[
m_\ell^{class}=E_{g\neq g'}\|\mu_{\ell,g}-\mu_{\ell,g'}\|^2.
\]

Use distinct code names for target margin and class separation.

## 4. Signal-to-Noise Decomposition Across Layers

For each layer:

\[
r_{\ell+1}=r_\ell+\Delta r_\ell.
\]

Measure target-directed progress:

\[
a_{\ell,i}=m_{\ell+1,i}-m_{\ell,i}.
\]

Across nuisance realizations:

\[
a_{\ell,i}=\mu_\ell^\Delta+\epsilon_{\ell,i}^\Delta.
\]

Compute:

\[
\mu_\ell^\Delta=E_i[a_{\ell,i}]
\]

and

\[
(\sigma_\ell^\Delta)^2=\operatorname{Var}_i[a_{\ell,i}].
\]

Report:

```text
mean_margin_update
var_margin_update
margin_update_snr
fraction_positive_margin_update
```

with

\[
\Gamma_\ell^\Delta=
\frac{\mu_\ell^\Delta}{\sigma_\ell^\Delta+\epsilon}.
\]

Also compute cumulative signal:

\[
S_L^{cum}=\sum_{\ell=0}^{L-1}\mu_\ell^\Delta.
\]

Measure cross-layer fluctuation covariance:

\[
C_{\ell k}=
\operatorname{Cov}(\epsilon_\ell^\Delta,\epsilon_k^\Delta).
\]

Then actual cumulative fluctuation is:

\[
(N_L^{cum})^2=
\sum_\ell(\sigma_\ell^\Delta)^2+
2\sum_{\ell<k}C_{\ell k}.
\]

This is required for the statistical-mechanics interpretation.

## 5. Correctness Transition / Decision Depth

For every predictive family and nuisance level, estimate distributions of:

```text
first_top1_layer
stable_top1_layer
settling_delay
top1_reversal_count
```

Report:

\[
E[L_Y^{first}],\qquad
E[L_Y^{stable}],\qquad
E[\Delta L_Y].
\]

Primary hypothesis:

\[
E[L_Y^{stable}\mid k]
\]

should increase with nuisance count \(k\), pattern complexity, or dependency span if more depth is needed to overcome noise.

Also measure whether deeper models reduce final failure rate for examples whose stable decision layer exceeds the depth of shallower models.

## 6. Empirical Scaling Laws

Candidate dependent variables:

```text
stable_top1_layer
first_top1_layer
final_margin_snr
final_within_JS
depth_to_R_threshold
depth_to_margin_threshold
depth_to_stable_top1_fraction
```

Candidate explanatory variables:

```text
nuisance_count k
dependency_span s
attention_window w
pattern_length n
num_competing_classes C
continuation_entropy H
model_depth L
model_width d_model
num_heads H_heads
training_frequency f
```

Test hypotheses such as:

\[
L^\star\propto k^\alpha,
\qquad
L^\star\propto(s/w)^\beta,
\qquad
\Gamma_L\propto L^\gamma.
\]

Compare at minimum:

```text
linear
logarithmic
power law
exponential saturation
rational saturation
```

For every fit report:

```text
fit_parameters
parameter_CI
R2
AIC_or_BIC
heldout_error
fit_range
number_of_independent_cells
```

A claimed power law requires:

1. stable exponent across seeds;
2. stable exponent across at least two architecture scales;
3. clear improvement over simpler alternatives;
4. no dependence on a tiny fitting range;
5. residual inspection.

Otherwise label it an empirical trend.

## 7. Layer-by-Layer Entropy Analysis

Continue recording:

\[
H_\ell=-\sum_y p_\ell(y)\log p_\ell(y).
\]

But do not define refinement as entropy reduction.

Plot entropy jointly with:

```text
target_probability
target_margin
margin_snr
within_family_JS
fraction_top1
```

Required interpretation:

- \(p(Y)\uparrow\) with \(H\uparrow\) is possible;
- target margin can improve while entropy increases;
- lower nuisance dispersion and higher within-example entropy can coexist.

## 8. Jacobian / Directional Follow-Up

For nuisance pair:

\[
\delta r_\ell^N=r_\ell(g,d_1)-r_\ell(g,d_2).
\]

For signal pair:

\[
\delta r_\ell^S=r_\ell(g_1,d)-r_\ell(g_2,d).
\]

Compute:

\[
c_\ell^N=
\frac{\|J_\ell\delta r_\ell^N\|}
{\|\delta r_\ell^N\|},
\]

\[
c_\ell^S=
\frac{\|J_\ell\delta r_\ell^S\|}
{\|\delta r_\ell^S\|}.
\]

Also project onto target-margin gradient:

\[
v_\ell=
\nabla_{r_\ell}
[z_\ell(Y)-\max_{j\neq Y}z_\ell(j)].
\]

Measure nuisance and signal sensitivity in decision space.

Do not require every layer to contract nuisance; cumulative anisotropy is the stronger target.

## 9. Required Figures

Add:

```text
g1_target_probability_vs_depth.png
g1_target_rank_vs_depth.png
g1_target_margin_vs_depth.png
g1_fraction_top1_vs_depth.png
g1_first_vs_stable_top1_layer.png
g1_signal_vs_fluctuation_trajectory.png
g1_margin_snr_vs_depth.png
g1_margin_update_snr_vs_depth.png
g1_entropy_vs_margin_snr.png
g1_signal_dispersion_phase_plane.png
g1_decision_depth_vs_nuisance.png
g1_decision_depth_scaling_fits.png
g1_snr_scaling_fits.png
g1_layer_update_covariance.png
```

## 10. Required Tables

Generate:

```text
group1_correctness_order_parameters.csv
group1_decision_depth.csv
group1_layer_signal_noise.csv
group1_scaling_law_fits.csv
group1_entropy_correctness_joint.csv
```

## 11. Summary Questions

Update `group1_summary.md` to answer:

1. At what depth does the correct target first become top-1?
2. At what depth does it remain top-1?
3. Does nuisance increase decision depth?
4. Does mean correct-target margin increase with depth?
5. Does nuisance-induced margin variance decrease with depth?
6. Does margin SNR increase with depth?
7. Are improvements monotonic or do they show reversals?
8. Does entropy track correctness, or move independently?
9. Does the common layer update toward \(Y\) dominate fluctuation?
10. Are layer-update fluctuations correlated across depth?
11. Is there evidence for empirical scaling of decision depth or SNR?
12. Which candidate functional form fits best?
13. Does the effect generalize across generators and architectures?

## 12. Falsification Criteria

The new hypothesis is weakened or falsified if:

- target margin does not improve despite lower nuisance dispersion;
- the model converges to the same wrong answer across nuisance realizations;
- fraction top-1 does not increase with depth;
- stable decision depth is unrelated to nuisance/task complexity;
- margin SNR does not improve with depth;
- SNR gains come only from denominator collapse while signal remains near zero;
- layer-update means are near zero and no cumulative target-directed signal exists;
- cumulative fluctuation grows as fast as or faster than signal;
- claimed power-law exponents are unstable across seeds/architectures;
- simpler saturation/log models fit as well as a claimed power law.

## 13. Priority Order

Implement in this order:

1. target probability/rank/margin across depth;
2. first/stable top-1 layers;
3. margin mean/variance/SNR;
4. signal-dispersion phase plane;
5. layer-update signal/noise decomposition;
6. cross-layer update covariance;
7. nuisance-dependent decision-depth curves;
8. empirical scaling-law fits;
9. Jacobian decision-direction follow-up.

## 14. Final Scientific Target

The strongest result would be evidence that:

\[
\boxed{
\text{depth increases correct predictive signal while reducing
its nuisance-conditioned relative fluctuation}
}
\]

so that:

\[
S_\ell\uparrow,\qquad
\Gamma_\ell\uparrow,\qquad
P(\operatorname{rank}_\ell(Y)=1)\uparrow,
\]

while:

\[
N_\ell^2\downarrow
\]

or at least grows more slowly than signal.

A successful scaling result would identify how the depth required to reach stable prediction depends on nuisance, dependency span, architecture, and pattern complexity.
