# AGENTS.md — Paper 0.5 Final Polish and Reviewer Audit
## Scope Calibration, Related Work, Novelty Positioning, and Submission Readiness

### Mission
Perform a final submission-quality pass on Paper 0.5 without adding major new experiments.

Goals:
1. sharpen scope and claims;
2. fix small exposition gaps identified by external review;
3. strengthen related-work positioning and novelty differentiation;
4. foreground the non-monotonic optimization/depth result;
5. clarify the GRU result and its implications;
6. perform an adversarial reviewer-style audit for alternative explanations, statistical weaknesses, and overclaiming;
7. preserve the current ~10-page concise structure unless a change is clearly justified.

Do not reopen the experimental program unless the review uncovers a concrete validity problem.

## 1. Verify macro resolution
The paper imports `docs/papers/paper0_5/results/final_experiment_macros.tex`.
Verify that `\PhaseAbstract`, `\RNNAbstract`, `\PhaseResult`, `\RNNResult`, `\PhaseStatus`, and `\RNNStatus` all resolve in the built PDF and no raw macro names appear. Do not remove the macro system merely because a source-only reviewer missed the include.

## 2. Scope calibration
Retain the broad opening question, but explicitly state near the start of the introduction that the evidence comes from a controlled synthetic micro-mechanistic regime.

Suggested sentence:
> We study this question in a synthetic micro-mechanistic regime where predictive structure, nuisance, and information accessibility can be independently controlled; our claims concern these controlled dynamics rather than universal behavior of large pretrained language models.

## 3. Explain predictive order intuitively
Immediately after
\[
p^*=\min_A |A|\quad\text{s.t.}\quad H(Y\mid X_A)\le\delta
\]
add:
> Intuitively, \(p^*\) is the smallest number of jointly available input variables sufficient to determine the target to the chosen uncertainty tolerance.

Keep raw length \(n\), span \(s\), predictive order \(p^*\), and nuisance \(k\) distinct.

## 4. Tighten JVP interpretation
Do not claim the cumulative-JVP failure falsifies Neural ODE views in general.

Preferred wording:
> The result cautions against treating Transformer depth as a single globally valid tangent flow: local linear response is informative, but the relevant tangent geometry changes substantially along the trajectory.

Keep the supported local relation
\[
\delta r_{l+1}\approx J_l\delta r_l
\]
and the falsified global replacement
\[
\delta r_L\approx J_{L-1}\cdots J_0\delta r_0.
\]

## 5. Foreground the phase-grid result
Near the phase figure explicitly state:
- tested depth/width range;
- training budgets;
- competence threshold;
- one-seed limitation if still applicable.

Highlight the non-monotonic depth result:
- depth 8 good;
- depths 12/16 poor;
- depth 24 recovers.

Interpret explicitly:
> Architectural depth is not an independently monotonic resource under fixed training conditions.

Also foreground order-3 training rescue (0.634 -> 0.927 -> 0.972) as evidence that optimization shifts the competence frontier.

## 6. Add local RNN context
Near the RNN figure include exact parameter counts and equal training budget:
> The matched models contain approximately 40.9k Transformer, 16.6k vanilla-RNN, 33.3k GRU, and 41.6k LSTM parameters; all receive 1,200 updates and 12.288M training tokens.

State that parameter matching is approximate.

## 7. Strengthen the GRU interpretation
Add a short discussion paragraph:
> The GRU result also shows that predictive refinement is not Transformer-specific. Gated recurrence can realize the controlled integration task extremely effectively, indicating that the broader computational principle may be iterative state transformation rather than self-attention itself.

Clarify:
- vanilla RNN shows a transport limitation;
- GRU does not;
- LSTM is strong;
- Transformer is not universally superior on these generators.

## 8. Related-work expansion and novelty audit
Review closest literature in:
- Transformer depth, pruning, early exit, LayerDrop, Mixture-of-Depths, intermediate predictions;
- mechanistic interpretability: residual stream, logit/tuned lens, induction heads, activation patching, causal tracing;
- SA vs FFN: associative/key-value memory views;
- context interference: irrelevant context, lost-in-the-middle, retrieval noise, attention dilution;
- residual/Jacobian dynamics and iterative refinement;
- representation geometry and information-theoretic views;
- RNN vs attention, long-range dependency, gated recurrence, modern recurrent/state-space alternatives.

For each close paper record:
- manipulated variable;
- metric;
- whether correctness is gated;
- whether nuisance/length/order/span are separated;
- whether interventions are causal;
- whether local or cumulative dynamics are studied.

Create `docs/papers/paper0_5/results/editorial/related_work_matrix.md`.

## 9. Novelty statement
Do not claim novelty as “Transformers refine predictions with depth.”

Preferred positioning:
> Prior work separately studies intermediate predictions, depth allocation, context interference, residual dynamics, and recurrence. We jointly control predictive order, raw length, dependency span, nuisance, reachability, correctness, and architecture, allowing several common refinement interpretations to be directly falsified.

Emphasize the separating results:
\[
\text{length}\neq\text{predictive order},
\]
\[
\text{reachability}\neq\text{functional routing},
\]
\[
\text{stability}\neq\text{variance contraction},
\]
\[
\text{local linearity}\neq\text{global linear composition}.
\]

## 10. Adversarial reviewer audit
Simulate four reviewers:
- mechanistic interpretability;
- learning theory/optimization;
- architecture/sequence modeling;
- statistics/experimental methodology.

For each identify:
- strongest objection;
- whether already answered;
- missing control if any;
- overclaiming language;
- likely score-impacting issue;
- whether rebuttal is sufficient.

Create `docs/papers/paper0_5/results/editorial/adversarial_review.md`.

## 11. Alternative-explanation audit
Audit at least:

### Optimization artifact
Could non-monotonic depth effects be training instability? Acknowledge this and use training-budget rescue to show optimization matters.

### Synthetic-task artifact
Keep scope synthetic and pretrained evidence weak.

### Margin artifact
Tie margin/SNR to top-1 correctness, target probability, JS, masking, and replacement; never rely on SNR alone.

### Covariance arithmetic artifact
Verify
\[
\operatorname{Var}(\sum_l\epsilon_l)=\sum_l\sigma_l^2+2\sum_{l<j}\operatorname{Cov}(\epsilon_l,\epsilon_j)
\]
and explicitly label whether the reported covariance sum already includes the factor of two.

### Head overinterpretation
Keep motif/head-role claims narrow and causal.

### RNN fairness
Audit parameters, tuning budget, training tokens, initialization, batching, optimizer, hidden dimensions.

## 12. Statistical audit
Verify:
- family/seed are inferential units;
- raw token rows are not treated as independent;
- confidence intervals use correct aggregation;
- descriptive R² is not a scaling-law claim;
- failed competence cells are excluded from mechanistic claims;
- one-seed phase results are labeled;
- no post-hoc threshold is presented as preregistered.

Create `docs/papers/paper0_5/results/editorial/statistical_audit.md`.

## 13. Figure readability audit
Render the final PDF and inspect at normal print-equivalent size. Check:
- labels;
- legends;
- clipping;
- caption self-containment;
- dense two-panel layouts.

Pay special attention to phase diagram, RNN curves, signal/fluctuation plot, JVP plot, and selective masking plot. If a `.48\linewidth` panel is too dense, make it full width or move a secondary panel to the appendix.

## 14. Main-text figure context
For each main figure ensure nearby prose states:
- generator/dataset;
- model family;
- independent variable;
- competence gate;
- seeds where relevant;
- what the figure establishes.

## 15. Prose tightening
Standardize around:
- predictive signal;
- fluctuation;
- predictive refinement;
- predictive order;
- information exposure;
- repair;
- functional routing.

Definition:
> Predictive refinement is the depthwise process by which correct-target evidence becomes increasingly dominant over realization-dependent fluctuation and competing alternatives.

Remove duplicated caveats and experiment-history prose.

## 16. Abstract final pass
The abstract should:
1. state synthetic controlled scope;
2. state central question;
3. report strongest positive result;
4. report strongest falsifications;
5. mention selective masking;
6. mention raw-length null / predictive-order unresolved result;
7. mention RNN result in one sentence;
8. end with the predictive-refinement account.

Avoid too many numbers.

## 17. Introduction final pass
Ensure page 1 contains:
- broad question;
- synthetic micro-mechanistic scope;
- predictive-refinement definition;
- variable separation \(n,s,k,p^*\);
- 4–5 contributions;
- no implied universal LLM claim.

## 18. Discussion final pass
Add three implications:

### Predictive refinement is architecture-independent in principle
GRU shows the broader mechanism can appear in gated recurrence.

### Information exposure is selective, not monotonic
More context is neither always better nor always worse.

### Depth is not a monotonic resource
Within-model updates and across-model architectural depth both show non-monotonic effects.

State explicitly:
> More layers do not guarantee better competence under fixed optimization.

## 19. Limitations
Ensure limitations include:
- synthetic generators;
- small models;
- high-order optimization failure;
- one-seed phase surface if still true;
- approximate RNN parameter matching;
- limited recurrent families;
- no state-space comparison;
- no universal scaling law;
- weak pretrained external validity;
- no semantic abstraction claim in Paper 0.5.

## 20. Submission-readiness checklist
Do not declare final until:
1. macros verified in PDF;
2. scope sentence added;
3. intuitive p* sentence added;
4. JVP/ODE wording softened;
5. phase context added;
6. RNN parameter/training-budget sentence added;
7. GRU interpretation added;
8. related-work matrix completed;
9. novelty paragraph revised;
10. adversarial review completed;
11. alternative-explanation audit completed;
12. statistical audit completed;
13. figure readability checked;
14. main paper stays about 9–11 pages;
15. appendices contain details;
16. citations resolve;
17. tests pass;
18. TeX compiles without warnings;
19. PDF visually inspected;
20. claim table matches the prose.

## Final editorial target
The paper should read as:
\[
\boxed{\text{one theory}\rightarrow\text{controlled variable separation}\rightarrow\text{four empirical claims}\rightarrow\text{explicit falsifications}\rightarrow\text{bounded interpretation}.}
\]

Final positioning:
> Paper 0.5 does not claim to discover that depth refines predictions. It provides a controlled micro-mechanistic framework that separates predictive order, raw length, span, nuisance, reachability, correctness, and architecture, and uses that separation to reject monotonic denoising, raw-length scaling, reachability sufficiency, globally linear composition, and broad recurrent disadvantage while supporting nonlinear predictive refinement with correlated repair and selective information exposure.
