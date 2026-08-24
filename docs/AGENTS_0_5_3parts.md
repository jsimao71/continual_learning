# AGENTS.md — Paper 0.5: Layer Composition and Predictive Stability

## Mission
Run three linked experimental programs:
1. Layer Composition and Predictive Stability
2. Self-Attention Transport: Window Size vs Predictive Span
3. Head-Level Contributions to Stability and Pattern Discrimination

The primary scientific goal is to determine whether and how Transformer depth reorganizes noisy input realizations into increasingly stable predictive equivalence classes.

Do not turn this iteration into a continual-learning project. Do not make broad SA-vs-MLP claims unless they are needed to explain the layer-composition results. Custom Transformers with exact synthetic training distributions are the primary instruments; pretrained models are optional external-validity checks only after controlled experiments are complete.

## Scientific boundary
The paper is not primarily about online learning, memorizing attention motifs, proving “attention retrieves, MLP memorizes,” natural-language n-gram frequency estimation, or pretrained-model claims from unknown training distributions.

The paper is about inputs `x=(g,d)` with target `Y` satisfying `Y ⟂ d | g`, and how the depth trajectory changes within-family variation, between-family separation, target stability, sensitivity to answer-changing context, nuisance covariance, causal reachability, and head contribution.

## First action: inspect existing Paper 0.5
1. Read `docs/papers/paper0_5/paper_0_5.tex`.
2. Locate current experiment modules, expected near `src/cl/experiments/paper05_ngram.py`, `src/cl/experiments/paper05_pretrained.py`, and `src/cl/analysis/paper05_inference.py`.
3. Locate current results/tables/figures/manifests under `docs/papers/paper0_5/results/`.
4. Run relevant tests.
5. Preserve reproducible prior artifacts.
6. Do not delete previous pilot results; move superseded analyses to clearly named legacy/prior-pilot artifacts if needed.

The current paper reports a useful pilot under pattern evidence + four distractors: target-probability variance about `0.0347 -> 0.0106 -> 0.0109`, JS dispersion about `0.0650 -> 0.0307 -> 0.0106` bits, and increasing within-example entropy. Reproduce before generalizing.

# Shared infrastructure

## Controlled pattern generator
Create a reusable generator. Each example records at least:
`generator_family, predictive_family_id, surface_identity_id, target_token, pattern_tokens, pattern_length, dependency_span, nuisance_tokens, nuisance_count, nuisance_type, nuisance_seed, answer_changing_context, continuation_entropy, train_frequency, split, generator_seed`.

Implement at minimum:
- `contiguous_ngram`
- `skip_gram`
- `prefix_suffix`
- `binary_functor`
- `template_rule`
- `compositional_rule`
- `hierarchical_rule`
- `nested_override`

Explicitly distinguish nuisance variables, answer-changing variables, surface identity, and predictive family. Use train/test identity separation.

## Model matrix
Support tiny (2–4 layers, width 32–64, 2–4 heads), small (6–8 layers, width 96–192, 4–8 heads), and deeper (10–12 layers, width 192–384, 6–12 heads). Start small. Use at least 3 seeds for confirmatory results; 5 preferred for key scaling laws if compute permits.

## Instrumentation
At every layer capture `residual_pre_sa`, `sa_update`, `residual_post_sa`, `ff_update`, `residual_post_block`, `output_logits`, `output_distribution`; optionally per-head attention/output and Q/K/V. Add tests for trace parity, residual identity, intervention locality, deterministic generator replay, metadata consistency, and attention-window mask correctness.

# Shared metrics
Prediction-space: target probability/logprob/margin/rank, full-distribution JS, within-family JS, between-family JS. Define `R_l = D_between/(D_within+eps)`.

Representation-space: family centroid, covariance trace, effective rank, within-family cosine/L2, between-centroid distance. Define `q_l = E Tr Sigma`, `m_l = E ||mu_g-mu_g'||^2`, `chi_l=q_l/(m_l+eps)`.

Information-style metrics: predictive-family decodability/information, nuisance-identity decodability conditional on family. Do not overclaim mutual information from weak estimators.

# GROUP 1 — Layer Composition and Predictive Stability

## Goal
Establish whether depth increases predictive-class organization under controlled nuisance and whether the mechanism is monotonic contraction, expansion/reconvergence, or another distributed process.

## Factorial
Cross generator family, pattern length/span, nuisance count `[0,2,4,8,16]`, nuisance similarity `[low, medium, high]`, continuation entropy `[deterministic, low, high]`, number of competing classes, depth, width, seed. Start with a tractable orthogonal subset.

## E1.1 Reproduce pilot
Regenerate prior controlled depth/noise results. Output `group1_reproduction.csv` and manifest.

## E1.2 Generalize across pattern families
Run contiguous, skip, functor, compositional, hierarchical. Primary question: does `R_l` improve across structurally different predictive families?

## E1.3 Signal vs nuisance directions
Paired examples: same `g`, different `d` for nuisance perturbation; different `g`, same `d` for signal perturbation. Compute finite-difference `delta_r_nuis` and `delta_r_sig` across depth and their output effects.

## E1.4 Local Jacobian test
For small models use JVPs to compare `J_l delta_r_l` with observed `delta_r_{l+1}`. Report cosine, relative norm error, and prediction-space error separately for nuisance and signal. Do not form dense Jacobians unless tiny.

## E1.5 Non-monotonicity classification
Classify trajectories: `monotone_contraction`, `expansion_then_contraction`, `oscillatory`, `late_contraction`, `no_contraction`, `signal_collapse`. Temporary expansion is not failure if later `R_l` improves.

## E1.6 Scaling/capacity boundary
Estimate minimum depth `L*` satisfying task/stability criteria. Fit simple alternatives versus nuisance count, pattern complexity, continuation entropy, classes. Do not force power laws.

### Required figures
`g1_within_between_vs_depth.png`, `g1_R_vs_depth.png`, `g1_covariance_trace_vs_depth.png`, `g1_entropy_vs_across_realization_JS.png`, `g1_signal_vs_nuisance_direction.png`, `g1_jvp_prediction_quality.png`, `g1_nonmonotonic_examples.png`, `g1_depth_noise_phase_map.png`.

### Summary must answer
Does stability improve? Is it monotonic? Does it generalize beyond n-grams? Are nuisance directions preferentially contracted? Do local Jacobians explain it? What controls required depth?

# GROUP 2 — SA Transport: Window Size vs Pattern Span

## Goal
Test whether contextual transport through SA constrains when predictive stability can emerge.

Use controlled patterns `A ... B -> Y` and nested `B C -> Y1`, `A ... B C -> Y2` where longer context genuinely changes the answer.

## Grid
At minimum span `s=[1,2,4,8,16,32]`, window `w=[1,2,4,8,16,full]`, depth `L=[1,2,4,8,12]` with architecture-compatible subsets.

## E2.1 Reachability sanity
Verify the mask makes source positions unreachable before expected depth. Build graph-based causal reachability from the mask.

## E2.2 Transport delay
Measure accuracy, target logprob, `R_l`, source-token/predictive-class decodability at prediction position, and source-token causal patch effect. Estimate first functional arrival depth.

## E2.3 Scaling law
Test collapse against `s/w`; compare `L* ~ s/w`, `L* ~ (s/w)^alpha`, and `L* ~ log s`. Do not assume `Lw ~ s`.

## E2.4 Local-pattern control
If the target is fully determined inside the retained window and longer context is irrelevant, shrinking the window should have little effect.

## E2.5 Nested override
Track where long-range context changes the trajectory from the short-pattern answer to the correct longer-pattern answer. Record SA/FF effects only as explanatory measurements.

### Required figures
`g2_span_window_accuracy_phase.png`, `g2_span_window_R_phase.png`, `g2_transport_delay.png`, `g2_Lstar_vs_span_over_window.png`, `g2_local_pattern_control.png`, `g2_nested_override_trajectory.png`.

### Summary must answer
Is reachability necessary? Does transport approximately scale with depth×window? Where does that law break? Are local predictions insensitive to truncating irrelevant long context? At what depth does the longer pattern override the shorter one?

# GROUP 3 — Head Contributions to Stability and Discrimination

## Goal
Determine how heads implement the SA contribution, especially for nested `n` vs `n+k` distinctions.

## E3.1 Per-head causal utility
Run zero ablation, mean ablation, equivalent-family replacement, mismatched-family replacement. Measure impact on `within_JS`, `between_JS`, `R`, target logprob, target margin. Keep stability and discrimination utility separate.

## E3.2 Pattern-length recruitment
For `g_n -> Y1` and `g_n+k -> Y2`, map causal head utility by layer/head/pattern length.

## E3.3 Redundancy/complementarity
Compare individual utility, sum of individual utilities, and actual subset utility. Compute `Gamma(H)=U(H)-sum_h U(h)`. Start pairwise.

## E3.4 Head covariance
Across nuisance realizations measure covariance/correlation of head output contributions. Test whether independent useful heads stabilize more than correlated heads. Do not assume `1/H` scaling.

## E3.5 Motif vs causal utility
Repeat motif specificity and correlate with head causal utility. Prior pilot had Spearman `rho ~ -0.009`; treat this as a negative-control benchmark.

## E3.6 Head-count scaling
Train matched-capacity models with different head counts. Measure final `R`, within/between JS, nuisance robustness, redundancy, and specialization stability across seeds.

### Required figures
`g3_head_stability_utility.png`, `g3_head_discrimination_utility.png`, `g3_head_by_pattern_length.png`, `g3_head_pair_interactions.png`, `g3_head_output_covariance.png`, `g3_motif_vs_causal_utility.png`, `g3_head_count_scaling.png`.

### Summary must answer
Are head roles reproducible? Which reduce nuisance vs distinguish classes? Does larger span recruit different heads? Redundant or complementary? Does motif specificity predict causal utility? Does head count improve stability under matched capacity?

# Cross-group integration
Generate `paper05_integrated_summary.md`, `paper05_integrated_metrics.csv`, and `paper05_integrated_manifest.json`.

Follow this order: establish system-level effect; explain non-monotonicity; explain SA transport; explain head implementation; score statistical, information-theoretic, dynamical-systems, and statistical-mechanical views as supported/partially supported/unsupported/not tested with concrete metrics.

# Statistical requirements
Bootstrap over predictive families, model seeds, and architecture cells where applicable. For key comparisons report effect size, 95% CI, number of predictive families, seeds, and architectures. Correct multiple comparisons for head scans.

# Anti-storytelling rules
1. Never call lower dispersion better if target signal is absent.
2. Never interpret answer-changing context as nuisance.
3. Never equate lower entropy with stability.
4. Never equate visual attention regularity with causal utility.
5. Never assume monotonicity.
6. Never claim a phase transition without finite-size/scaling evidence.
7. Never claim universal SA/FF roles from one generator.
8. Never use pretrained unknown frequency as ground truth.
9. Preserve failed hypotheses and negative controls.
10. Do not proceed to continual-learning experiments.

# Suggested CLI
Prefer one entrypoint:
```bash
PYTHONPATH=src python -m cl.experiments.paper05_layer_composition --group 1 --config configs/paper05/group1.yaml
PYTHONPATH=src python -m cl.experiments.paper05_layer_composition --group 2 --config configs/paper05/group2.yaml
PYTHONPATH=src python -m cl.experiments.paper05_layer_composition --group 3 --config configs/paper05/group3.yaml
PYTHONPATH=src python -m cl.analysis.paper05_layer_composition --all
```
If preserving current modules is cleaner, add orchestration rather than rewriting working code.

# Recommended artifact layout
```text
docs/papers/paper0_5/
  paper_0_5.tex
  AGENTS.md
  results/
    manifests/
    raw/group1/
    raw/group2/
    raw/group3/
    aggregates/group1/
    aggregates/group2/
    aggregates/group3/
    figures/
    tables/
    summaries/
```

# Minimum completion gate
All three groups must run end-to-end on smoke configs; Group 1 must cover at least 3 generator families; Group 2 at least 4 span/window regimes; Group 3 complete per-head causal metrics on one trained multi-head model; manifests must capture seeds/config/checkpoints; summaries and figures regenerate from raw artifacts; tests pass; `.tex` results placeholders are populated only with regenerated numbers; negative results remain visible.

# Stop conditions
Stop and report rather than invent interpretation if trace changes logits materially, generator metadata fail to reproduce targets, nuisance correlates with target, attention masks violate the intended window, the model fails the base task, low dispersion occurs only in low-signal states, head ablation breaks tensor semantics, or the requested grid exceeds available compute.

When compute is limited prioritize: Group 1 stability, Group 2 transport, Group 3 heads, deeper scaling, pretrained external validity.

# Final scientific question
Does Transformer depth compose state-dependent residual transformations that organize diverse surface realizations into stable predictive equivalence classes by suppressing nuisance-sensitive variation relative to answer-changing variation, and can this system-level effect be explained by contextual transport through self-attention and distributed contributions across heads?
