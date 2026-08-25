# Paper 0.5 claim audit

| Hypothesis | Status | Key evidence and boundary |
|---|---|---|
| Correct predictive signal grows relative to realization-dependent fluctuation | supported | Correct margin moves from strongly negative to positive while final nuisance-conditioned variation remains finite; median margin SNR is positive. |
| Layers monotonically suppress nuisance | falsified | Only 7/27 primary trajectories contract monotonically; expansion-then-contraction, oscillation, and late contraction are common. |
| Predictive refinement reduces entropy | falsified | Entropy can increase while the correct margin crosses zero and predictive-family JS contracts; margin–entropy correlation is 0.081. |
| Later layers repair earlier realization-specific deviations | supported | Off-diagonal margin-update covariance is negative and the selective-exposure grid has median repair fraction 0.812. |
| A frozen global Jacobian product explains depthwise composition | falsified | Controlled JVP cosine declines from 0.782 at one block to 0.335 at four, while relative error rises above one. Local directional validity survives. |
| Reachability is sufficient for learned transport | falsified | A graph-reachable local-window cell reaches only 0.078 accuracy; locally trained and full-attention controls show that routing must be learned. |
| Selective nuisance exclusion reduces refinement burden | supported | Nuisance-key masking preserves 1.000 accuracy and lowers margin variance by 40.5%; signal masking collapses accuracy to 0.249. |
| Raw target-preserving length controls settling depth | not supported | Redundant/supportive lengths through eight remain competent without a consistent stable-depth delay; descriptive raw-length fit has $R^2=0.016$. |
| Predictive order controls minimum required depth | unresolved | Orders 1–2 reach competence at base budget and order 3 only at 2×/4× for depth 8/width 64, but orders 4/6/8 fail and depth is non-monotonic. Width and optimization cannot be separated into a minimum-depth law. |
| Architectural depth is independently monotonic under fixed training | falsified in the one-seed phase grid | Order-two competence is perfect at depth 8/width 64, poor at depths 12/16, and recovers at depth 24; replication is required beyond this controlled surface. |
| Optimization shifts the observed competence frontier | supported locally | At depth 8/width 64, order-three accuracy improves from 0.634 to 0.927/0.972 at 2×/4× training. |
| Stable attention-head motifs explain causal utility | not supported | Motif specificity and causal contribution are essentially uncorrelated in two controlled scans; head-pair effects are non-additive. |
| Conventional recurrence is more length/span sensitive than attention | falsified broadly; vanilla-RNN contrast supported | Transformer stays perfect across tested length/span and vanilla RNN degrades, but GRU is perfect and LSTM nearly perfect. Gated recurrence also outperforms the Transformer at high predictive order. |
| Controlled equivalence establishes natural semantic abstraction | falsified | Pretrained and on-manifold controls weaken label-specific replacement; all pretrained evidence remains appendix-level. |
