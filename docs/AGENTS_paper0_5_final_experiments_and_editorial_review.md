# AGENTS.md — Paper 0.5 Final Experimental + Editorial Iteration
## Predictive-Order Phase Diagram, RNN Comparison, and Major Manuscript Consolidation

### Mission
Complete the final scientific and editorial iteration of Paper 0.5.

This iteration has three linked goals:

1. Finish the missing controlled phase diagram:
   - predictive integration order × model depth × model width/capacity;
   - separate raw sequence length, dependency span, nuisance, and predictive order;
   - determine whether higher-order dependencies specifically require greater depth.

2. Add a matched recurrent baseline:
   - train RNN/GRU/LSTM models on the same controlled generators;
   - compare sensitivity to sequence length, dependency span, nuisance, and predictive order;
   - test whether attention decouples information-access distance from refinement depth more effectively than recurrence.

3. Perform a major editorial consolidation:
   - reduce Paper 0.5 to one central theory;
   - rewrite abstract, introduction, prose, theory, mathematics, related work, results organization, discussion, conclusion;
   - move secondary analyses, legacy pilots, pretrained work, detailed head scans, derivations, and exhaustive controls to appendices;
   - target a concise main paper with clear supported/falsified/unresolved claims.

Do not add unrelated experiments.

# PART I — PREDICTIVE-ORDER PHASE DIAGRAM

## 1. Scientific question
Current evidence shows:
- raw target-preserving pattern length barely affects stable decision depth;
- nuisance increases stabilization delay;
- necessary higher-order patterns become difficult;
- predictive order is a stronger descriptive candidate than raw length;
- high-order cells currently fail competence.

Unresolved:
> Does higher predictive integration order specifically require greater model depth, or merely more capacity/training?

## 2. Grid
Predictive order:
p* = [1,2,3,4,6,8]

Model depth:
L = [2,4,6,8,12,16,24]

Width:
d_model = [32,64,128,256] or compute-feasible matched subset.

Training budget:
base, 2x, 4x steps for selected difficult cells.

Hold head dimension approximately stable where practical.

## 3. Generator
Use no-shortcut necessary-pattern generator.

Require:
- prohibited proper subsets have near-zero target MI;
- full required pattern determines target;
- balanced target classes;
- no positional/token shortcut.

Generate enough independent families for family-level bootstrap.

## 4. Separate raw length from predictive order
For each p*, use multiple surface lengths n >= p*.

Examples:
- p*=2 with n=[2,4,8]
- p*=4 with n=[4,6,8]
- p*=6 with n=[6,8,10]

Extra tokens must be redundant, neutral nuisance, or supportive-but-nonessential.

Estimate:
L_min = f(p*, n)

## 5. Separate dependency span
For matched (p*, n), vary:
contiguous / moderate span / large span.

Record span s separately.

## 6. Nuisance control
Repeat selected cells with nuisance_count=[0,2,4,8].

Use same-vocabulary nuisance.

## 7. Competence surface
For every cell record held-out accuracy.

Create:
A(L, W, p*)

Primary figures:
- phase_depth_width_predictive_order.png
- phase_depth_predictive_order_by_width.png
- phase_width_predictive_order_by_depth.png
- accuracy_vs_depth_by_predictive_order.png

## 8. Minimum required depth and width
Primary threshold:
tau=0.80
Secondary:
tau=0.90

Define:
L_min(p*,W)=minimum L reaching competence.
W_min(p*,L)=minimum width reaching competence.

This is the decisive depth-vs-capacity test.

## 9. Internal decision depth
For competent cells measure:
- first top-1 layer
- stable top-1 layer
- settling delay
- reversals
- margin trajectory

Do not interpret failed cells mechanistically.

## 10. Explanatory models
Fit grouped models with predictors:
- predictive_order
- raw_length
- dependency_span
- nuisance_count
- model_depth
- model_width
- training_steps

Key interactions:
- predictive_order × depth
- predictive_order × width
- predictive_order × training budget

Compare whether p*×L is stronger than p*×W.

## 11. Scaling
Compare:
- constant
- linear
- logarithmic
- power
- piecewise threshold
- logistic competence frontier
- saturating

Prefer a competence boundary over a power-law claim if that fits better.

## 12. Required outputs
Tables:
- phase_grid_results.csv
- phase_min_depth.csv
- phase_min_width.csv
- phase_internal_decision_depth.csv
- phase_model_fits.csv

Summary:
docs/papers/paper0_5/results/predictive_order_phase/summary.md

# PART II — MATCHED RNN COMPARISON

## 13. Scientific question
Test:
> Does self-attention decouple information-access distance from refinement depth more effectively than conventional recurrence?

Do not claim universal Transformer superiority.

## 14. Models
Train:
- vanilla RNN
- GRU
- LSTM

If constrained, prioritize GRU and LSTM.

Match approximately on:
- parameter count
- embedding/hidden size
- training tokens
- optimizer
- training steps
- vocabulary
- target entropy

Report exact parameter counts.

## 15. Transformer baseline
Use the custom Transformer family from Paper 0.5 with approximately matched parameter budgets.

## 16. Controlled datasets

### R1 — Fixed predictive order, increasing raw length
Keep p*=2 fixed.
Use n=[2,4,8,16,32].

Question:
Does recurrent competence degrade with sequence length while Transformer competence remains more stable?

### R2 — Fixed predictive order, increasing dependency span
Use span=[1,2,4,8,16,32,64].

Question:
Is recurrent performance more sensitive to transport distance?

### R3 — Increasing predictive order
Use p*=[1,2,3,4,6,8] at controlled span.

Question:
Do both architectures require more capacity when actual integration complexity rises?

### R4 — Nuisance burden
Use nuisance=[0,2,4,8,16].

Question:
Which architecture is more sensitive to irrelevant context?

## 17. RNN internal metrics
At every time step capture:
- target probability if decoded
- target margin if decoded
- top1
- hidden norm
- hidden pair distance

For nuisance pairs track delta h_t.

## 18. Compare axes carefully
RNN computation is sequential in time t.
Transformer sequence positions are parallel within each layer and refinement is over layer l.

Do not equate one RNN time step with one Transformer layer.

Compare:
- length sensitivity
- span sensitivity
- predictive-order sensitivity
- nuisance sensitivity

## 19. Hypotheses
H-RNN1: at fixed p*, RNN accuracy declines more with raw length.
H-RNN2: at fixed p*, RNN accuracy declines more with span.
H-RNN3: both may require capacity as p* rises.
H-RNN4: Transformer depth should be less tied to traversal distance.

## 20. Fairness controls
Control:
- parameters
- training token budget
- hyperparameter search budget
- dataset seed
- optimizer family where appropriate

Permit architecture-specific tuning under equal tuning budget.

## 21. Optional recurrent theory
For:
h_t = F(h_{t-1}, x_t)

a perturbation at i reaches T through a Jacobian product whose path length grows with T-i.

Use as motivation only, not a proof of expressivity separation.

## 22. Required outputs
Figures:
- rnn_transformer_accuracy_vs_length.png
- rnn_transformer_accuracy_vs_span.png
- rnn_transformer_accuracy_vs_predictive_order.png
- rnn_transformer_accuracy_vs_nuisance.png
- rnn_hidden_signal_vs_time.png
- transformer_margin_vs_depth_matched.png
- rnn_transformer_parameter_matched_summary.png

Tables:
- rnn_transformer_results.csv
- rnn_transformer_parameter_match.csv
- rnn_transformer_model_fits.csv

Summary:
docs/papers/paper0_5/results/rnn_comparison/summary.md

# PART III — MAJOR EDITORIAL REVIEW

## 23. Central thesis
Rewrite around one thesis:

> Transformer depth performs nonlinear predictive refinement over accessible contextual information: correct predictive signal grows relative to realization-dependent fluctuation and competing alternatives, while required refinement depends more on predictive integration/discrimination than on raw context length.

Do not present a chronology of experiments.

## 24. Main-paper structure
Target 9–11 pages main text plus appendices.

Suggested structure:
1. Introduction (~1 page)
2. Related Work (~1–1.5 pages)
3. Controlled Framework and Theory (~1.5–2 pages)
4. Experimental Setup (~1 page)
5. Results (~3–4 pages)
6. Discussion and Limitations (~1 page)
7. Conclusion (~0.3–0.5 page)

## 25. Abstract rewrite
Report actual findings:
- controlled predictive structure/nuisance/access;
- correct-target signal growth;
- failure of monotonic denoising;
- negative cross-layer repair;
- local but not global JVP validity;
- selective nuisance exclusion;
- raw-length null;
- predictive-order result/status;
- RNN result if confirmatory.

Do not write an aspirational abstract.

## 26. Introduction rewrite
Lead with:
> Once relevant context is accessible, what computation does Transformer depth perform?

Motivate:
information access != predictive refinement.

End with 4–5 explicit contributions.

## 27. Consolidate the math
Main model:

r_{l+1}=r_l+F_l(r_l)

Correct-target margin:
m_{li}=z_{li}(Y)-max_{y!=Y}z_{li}(y)

Decompose:
m_{li}=S_l+eta_{li}

S_l=E[m_{li}]
N_l^2=Var(m_{li})
Gamma_l=S_l/sqrt(N_l^2+epsilon)

Layer update:
Delta m_{li}=mu_l+epsilon_{li}

Cumulative:
m_{Li}=m_{0i}+sum_l mu_l+sum_l epsilon_{li}

Variance:
Var(sum_l epsilon_l)
= sum_l sigma_l^2 + 2 sum_{l<k} Cov(epsilon_l,epsilon_k)

Use this as the central derivation for repair/cancellation.

## 28. Define predictive order formally
Define p* as minimum sufficient subset size satisfying a conditional-entropy threshold.

Distinguish early and consistently:
- raw length n
- dependency span s
- predictive order p*
- nuisance count k

## 29. JVP section
Main text only:
delta r_{l+1}=J_l delta r_l + O(||delta r_l||^2)

State:
- local nuisance JVP works well;
- cumulative product degrades with horizon;
- globally linear composition is falsified.

Move full covariance propagation, SVD/eigenvalue material, and detailed diagnostics to appendix.

## 30. Organize results by claims

### Result 1 — Depth builds correct predictive contrast
Combine:
- margin growth
- target probability
- top1
- SNR
- entropy independence
- non-monotonicity

### Result 2 — Refinement includes correlated repair and is globally nonlinear
Combine:
- negative cross-layer covariance
- repair fraction
- local JVP
- cumulative JVP failure

### Result 3 — Information exposure changes refinement burden
Combine:
- SA reachability
- nuisance delay
- selective masking
- window restriction failure
- selective exclusion

### Result 4 — Raw context length is not the controlling variable
Combine:
- redundant/supportive length null
- predictive-order phase diagram
- nested reuse
- RNN comparison

Head analysis should support these claims rather than become a standalone story.

## 31. Hypothesis table
Add a compact table with:
- hypothesis
- status: supported / falsified / unresolved
- key evidence

Include:
- monotonic nuisance suppression — falsified
- entropy reduction — falsified
- raw length controls settling depth — not supported
- reachability sufficient — falsified
- global Jacobian product — falsified
- correct signal grows relative to fluctuation — supported
- later repair — supported
- selective nuisance exclusion — supported
- predictive order controls required depth — update from phase diagram
- stable head motifs explain utility — not supported
- RNN more length/span sensitive — update from RNN experiment

## 32. Related work
Build a proper conceptual section covering:

### Transformer depth and adaptive computation
- layer dropping/pruning
- early exit
- LayerDrop
- Mixture-of-Depths
- adaptive depth/token routing
- intermediate predictions

### Mechanistic interpretability
- residual stream
- logit lens / tuned lens
- induction heads
- activation patching / causal tracing
- head specialization

### SA vs FFN
- FFNs as key-value memories
- attention retrieval vs MLP memorization
- associative-memory views

### Context interference
- irrelevant-context degradation
- lost-in-the-middle
- retrieval noise
- attention dilution
- distractor sensitivity

### Dynamics
- ResNets as dynamical systems / ODEs
- Jacobian/stability analyses
- iterative refinement

### Representation geometry / information
- class separation
- anisotropy
- neural collapse where relevant
- information-bottleneck interpretations

### RNN vs attention
- path length
- long-range dependency
- gradient propagation
- parallelism
- modern recurrent/state-space alternatives

Novelty statement should be modest:
Prior work studies these separately; this paper jointly controls predictive order, surface length, nuisance, reachability, correctness, and architecture.

## 33. Appendix plan

Appendix A — generator definitions and MI audits
Appendix B — training/configuration
Appendix C — statistical details and failed fits
Appendix D — mathematical derivations/JVP
Appendix E — full head analyses
Appendix F — nested-pattern details
Appendix G — predictive-order phase grid
Appendix H — RNN comparison details
Appendix I — pretrained Pythia/Qwen
Appendix J — legacy pilot

## 34. Main figure budget
Target about 6–8 main figures/panels:
1. conceptual setup/generator
2. signal-fluctuation trajectory
3. repair + local/global JVP
4. information exposure/selective masking
5. predictive-order phase diagram
6. Transformer-vs-RNN length/span comparison
7. optional nested reuse
8. optional summary hypothesis figure

Move the rest to appendix.

## 35. Prose style
Standardize around:
- predictive signal
- fluctuation
- predictive refinement
- predictive order
- information exposure
- repair

Operational definition:
Predictive refinement is the depthwise process by which correct-target evidence becomes increasingly dominant over realization-dependent fluctuation and competing alternatives.

Avoid gratuitous synonym switching among denoising/stability/organization/discrimination/contrast/refinement.

## 36. Claim discipline
Use:
supported / falsified / unresolved

Do not say:
"depth tracks predictive order"
unless the phase diagram supports it.

Do not say:
"Transformers are more scalable than RNNs"
from one synthetic benchmark.

Prefer:
"under matched controlled tasks, recurrent baselines show stronger sensitivity to sequence distance" if actually observed.

## 37. Pretrained placement
Keep all pretrained details in appendices.

Main text may include one concise external-validity sentence only.

## 38. Discussion
Focus on:

A. Information access vs refinement
Attention makes evidence available; depth transforms predictive consequences.

B. Selective information exposure
More context is not inherently better or worse. Selective exclusion can reduce refinement burden while arbitrary restriction can break learned routing.

C. Architecture
If supported by RNN results:
attention may decouple transport distance from refinement depth more effectively than conventional recurrence.

## 39. Limitations
Explicitly include:
- synthetic generators
- small custom models
- limited semantic complexity
- high-order competence challenges
- architecture-specific findings
- no universal scaling law unless established
- weak pretrained external validity
- RNN comparison excludes modern state-space/recurrent-attention families

## 40. Conclusion
Target a compact final conclusion:
Transformer depth is not well described as monotonic denoising or computation proportional to context length. In controlled tasks, depth performs nonlinear predictive refinement: correct evidence grows relative to fluctuation, later layers repair earlier deviations, and computational demand is governed by structured predictive distinctions rather than raw token extent.

Adjust based on final phase/RNN results.

# REQUIRED FINAL DELIVERABLES

Generate/update:
- docs/papers/paper0_5/paper_0_5_v3.tex
- docs/papers/paper0_5/paper_0_5_v3.pdf
- docs/papers/paper0_5/appendix.tex
- docs/papers/paper0_5/results/predictive_order_phase/summary.md
- docs/papers/paper0_5/results/rnn_comparison/summary.md
- docs/papers/paper0_5/results/editorial_claim_table.md

If repository convention requires preserving v2, keep it untouched and create v3.

# COMPLETION GATE

Do not call complete until:
1. high-order competence retried over depth × width × training budget;
2. clear competence frontier or documented null exists;
3. RNN/GRU/LSTM comparison runs on matched length/span/order controls;
4. conclusions separate raw length from predictive order;
5. abstract is rewritten from actual findings;
6. related work is expanded and current;
7. math is consolidated around signal/fluctuation/repair;
8. full JVP derivations move to appendix;
9. main paper is reduced toward 9–11 pages where practical;
10. pretrained work is appendix-level;
11. hypothesis table reports supported/falsified/unresolved outcomes;
12. figures/tables regenerate from tracked artifacts;
13. all tests pass;
14. PDFs build without warnings.

# Final Scientific Target

Strong result:
Transformer depth performs nonlinear predictive refinement whose cost is better explained by predictive integration/discrimination than by raw context length. Attention reduces transport distance, while recurrent baselines remain more sensitive to sequential span under matched conditions.

Fallback narrow result:
Raw context length is not a primary determinant of Transformer settling depth; predictive refinement is governed by structured signal, fluctuation, information exposure, and learned nonlinear composition.
