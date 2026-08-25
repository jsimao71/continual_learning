# Paper 0.5 related-work and novelty matrix

This matrix records the closest methodological neighbors and the distinction made in Paper 0.5. “Correctness gate” means that internal/mechanistic interpretation is explicitly conditioned on held-out behavioral competence, not merely that a paper reports task accuracy somewhere.

| Work | Manipulated variable | Primary metric | Correctness gated? | Separates nuisance / length / order / span? | Causal intervention? | Dynamics | Relation to Paper 0.5 |
|---|---|---|---|---|---|---|---|
| [LayerDrop (Fan et al., 2020)](https://arxiv.org/abs/1909.11556) | Structured layer dropout / inference depth | Accuracy versus retained layers | No explicit mechanism gate | No | Layer removal, behaviorally causal | Whole-network depth | Establishes depth robustness and pruning; does not separate predictive variables or analyze margin repair. |
| [Mixture-of-Depths (Raposo et al., 2024)](https://arxiv.org/abs/2404.02258) | Token-wise dynamic compute allocation | Quality, FLOPs, speed | No explicit mechanism gate | No | Learned routing changes computation | Dynamic layer use | Allocates depth efficiently; Paper 0.5 instead diagnoses what controlled predictive burden remains after access. |
| [Learning to Skip (Wang et al., 2023)](https://arxiv.org/abs/2311.15436) | Learned layer skipping | LM quality versus compute | No explicit mechanism gate | No | Skip policy changes computation | Whole-network depth | Addresses efficient depth allocation, not predictive-order identification or repair covariance. |
| [Tuned Lens (Belrose et al., 2023)](https://arxiv.org/abs/2303.08112) | Intermediate-state decoding | Calibrated intermediate predictions | Behavioral decoding, not a competence gate for mechanisms | No | Primarily diagnostic | Local layer trajectory | Motivates intermediate logits; Paper 0.5 adds controlled generators, correctness margins, fluctuation, and causal tests. |
| [Induction Heads (Olsson et al., 2022)](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) | Training time, head circuits, repeated-token patterns | Loss and circuit diagnostics | Behavior connected, but not the same cellwise gate | No | Ablation / circuit evidence | Training and layer-local | Close mechanistic precedent; Paper 0.5 finds that repeated motifs need not predict causal head utility. |
| [ROME / causal tracing (Meng et al., 2022)](https://arxiv.org/abs/2202.05262) | Corruption, restoration, weight editing | Probability recovery / edit success | No held-out generator gate | No | Yes | Layer-local | Motivates causal restoration; Paper 0.5 uses matched donors and competence filtering rather than equating localization with storage. |
| [Activation-patching best practices (Zhang & Nanda, 2024)](https://arxiv.org/abs/2309.16042) | Patch metric and corruption choice | Localization quality / patch effects | Task dependent | No | Yes | Layer-local | Supports the audit requirement that baseline, metric, and donor choice bound mechanistic claims. |
| [Transformer FFNs as key--value memories (Geva et al., 2021)](https://arxiv.org/abs/2012.14913) | Layer and FF memory cells | Pattern association and vocabulary promotion | No explicit competence gate | No | Mostly diagnostic; targeted analysis | Layer-local and across depth | Motivates SA/FF hypotheses; Paper 0.5 measures both component progress and causal replacement without assuming fixed roles. |
| [Lost in the Middle (Liu et al., 2024)](https://arxiv.org/abs/2307.03172) | Relevant-information position / context length | QA and retrieval accuracy | Behavioral evaluation | Partly position/length; not predictive order | No component intervention | End behavior | Documents position sensitivity; Paper 0.5 separates formal reachability, span, routing, nuisance, and selective masking. |
| [LLMs distracted by irrelevant context (Shi et al., 2023)](https://arxiv.org/abs/2302.00093) | Relevant versus irrelevant problem text | Reasoning accuracy | Behavioral evaluation | Irrelevance manipulated, but not order/span jointly | Prompt interventions | End behavior | Closest nuisance comparison; Paper 0.5 uses known conditional irrelevance and key-selective causal masks. |
| [Neural ODE (Chen et al., 2018)](https://arxiv.org/abs/1806.07366) | Continuous-depth parameterization | Likelihood, accuracy, solver cost | No | No | Architecture change | Global continuous flow | Motivates a flow analogy, which Paper 0.5 narrows rather than generally rejects. |
| [Do ResNets discretize Neural ODEs? (Sander et al., 2022)](https://arxiv.org/abs/2205.14612) | Residual depth and discretization | Representation/trajectory diagnostics | No | No | Comparative architectural analysis | Local and global | Closest residual-dynamics critique; Paper 0.5 directly contrasts local JVP validity with failed frozen cumulative composition. |
| [LSTM (Hochreiter & Schmidhuber, 1997)](https://www.bioinf.jku.at/publications/older/2604.pdf) and [GRU (Cho et al., 2014)](https://arxiv.org/abs/1406.1078) | Recurrent gating | Sequence-model likelihood / task quality | No Paper-0.5-style gate | Not jointly | Architectural comparison | Recurrent time | Supply gated recurrent baselines; the GRU result shows controlled refinement is not attention-specific. |
| [Mamba (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752) | Input-selective state-space recurrence | Language/modality quality, throughput | No Paper-0.5-style gate | Long sequence, not the same controlled axes | Architectural comparison | Recurrent state evolution | Important untested alternative; excluded from any Transformer-versus-recurrence generalization. |

## Novelty position

Paper 0.5 does not claim that Transformers are the first models to refine predictions with depth. Prior work separately studies intermediate predictions, depth allocation, context interference, residual dynamics, and recurrence. The contribution is their joint controlled separation under a behavioral competence gate:

\[
\text{length}\neq\text{predictive order},\qquad
\text{reachability}\neq\text{functional routing},
\]
\[
\text{stability}\neq\text{variance contraction},\qquad
\text{local linearity}\neq\text{global linear composition}.
\]

This design permits direct falsification of monotonic denoising, raw-length scaling, reachability sufficiency, globally frozen tangent composition, and broad recurrent disadvantage while retaining the narrower nonlinear-refinement account.
