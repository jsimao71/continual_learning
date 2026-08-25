# Paper 0.5 adversarial reviewer audit

## Reviewer 1 — mechanistic interpretability

- **Strongest objection:** Margin trajectories and residual geometry are diagnostic, not mechanisms; replacement can be off-manifold, and head importance depends on the intervention baseline.
- **Already answered:** The paper separates diagnostic progress from zero/mean/replacement ablation, uses matched donors, reports selective key masks, and rejects motif recurrence as a proxy for causal utility. Failed competence strata are excluded from positive mechanism claims.
- **Missing control:** A broader on-manifold donor inventory and path-specific mediation would strengthen the target-preserving bridge. This is not necessary for the central signal/repair claim and is retained as future work.
- **Overclaiming corrected:** “Head roles” is restricted to task-conditioned causal summaries; no stable semantic circuit or fixed SA/FF division is claimed.
- **Likely score impact:** Moderate. A reviewer may want more causal localization, but the paper’s central theory is trajectory-level and already contains causal masking/replacement evidence.
- **Rebuttal sufficiency:** Sufficient if the scope remains micro-mechanistic and the diagnostic/causal boundary stays explicit.

## Reviewer 2 — learning theory and optimization

- **Strongest objection:** The non-monotonic depth surface may be an optimization artifact rather than a property of depth or predictive order.
- **Already answered:** The paper now treats this as part of the result, not a nuisance to explain away. At depth 8/width 64, order-three accuracy moves from 0.634 to 0.927/0.972 with 2x/4x training, proving that optimization shifts the observed frontier. Depth 8 succeeds, depths 12/16 fail, and depth 24 recovers under the fixed base conditions.
- **Missing control:** Multiple seeds for the complete phase surface and architecture-specific schedules would distinguish systematic interactions from seed instability.
- **Overclaiming corrected:** No minimum-depth law, phase transition, scaling exponent, or independently causal depth effect is claimed.
- **Likely score impact:** High. The one-seed phase grid is the clearest statistical limitation.
- **Rebuttal sufficiency:** Sufficient for a bounded negative conclusion—depth is not independently monotonic under these conditions—but not for a universal optimization theory.

## Reviewer 3 — architecture and sequence modeling

- **Strongest objection:** The recurrent comparison is not exactly parameter matched and may favor or disadvantage architectures through shared rather than architecture-specific tuning.
- **Already answered:** Exact counts are disclosed (40.9k Transformer, 16.6k RNN, 33.3k GRU, 41.6k LSTM); all models receive 1,200 updates and 12.288M tokens with matched generator, batching, optimizer family, and target entropy. The manuscript calls matching approximate.
- **Missing control:** Wider hyperparameter searches, bidirectional/recurrent-attention baselines, and selective state-space models such as Mamba.
- **Overclaiming corrected:** Only vanilla-RNN transport sensitivity is supported. Broad recurrent disadvantage and universal Transformer superiority are falsified; GRU demonstrates architecture-independent refinement in principle.
- **Likely score impact:** Moderate if presented as an architecture tournament; low if retained as a controlled falsification of the broad claim.
- **Rebuttal sufficiency:** Sufficient for the reported controlled axes, not for general sequence-model ranking.

## Reviewer 4 — statistics and experimental methodology

- **Strongest objection:** Dense token-level rows could create pseudoreplication, phase results use one seed, and descriptive fits could be mistaken for inferential scaling laws.
- **Already answered:** Family/seed/architecture cells are the inferential units; Dataset V2 uses family/seed bootstrap intervals; selective masks are paired; failed competence cells remain recorded but do not support mechanisms; the raw-length $R^2=0.016$ is explicitly descriptive.
- **Missing control:** Replicated phase-grid seeds and multiplicity-aware confirmatory tests for any future broad architecture law.
- **Overclaiming corrected:** The 0.80/0.90 thresholds are described as fixed analysis gates, not preregistered; no power law is promoted.
- **Likely score impact:** Moderate to high because the phase surface is prominent.
- **Rebuttal sufficiency:** Sufficient for the four bounded empirical claims and explicit falsifications; insufficient for a universal order-to-depth scaling claim, which the paper does not make.

## Alternative-explanation audit

| Alternative explanation | Audit result | Manuscript treatment |
|---|---|---|
| Optimization instability creates non-monotonic depth | Plausible and not eliminated. Training-budget rescue proves optimization matters. | Foregrounded; depth is not called an independent monotonic resource. |
| Synthetic-task artifact | Likely limits external validity, but enables exact no-shortcut and nuisance controls. | Scope sentence and limitations bound claims; pretrained evidence remains weak/appendix-level. |
| Margin/SNR artifact | Not sufficient alone. | Margin is tied to exact top-1 correctness and triangulated with target probability, JS, masking, and replacement. Signal and variance are always reported beside SNR. |
| Covariance arithmetic artifact | Formula verified: the cross term is doubled. | “Summed off-diagonal covariance” is explicitly defined as including the factor of two. |
| Head-role overinterpretation | Motifs and causal utility can diverge; pair effects are non-additive. | Head claims remain narrow, causal, and task-conditioned. |
| RNN unfairness | Parameters differ and shared tuning is not exhaustive; tokens/updates/data/batching/optimizer family are matched. | Exact counts, approximate matching, and missing modern alternatives are disclosed. |

## Overall recommendation

The strongest defensible paper is a controlled falsification-and-framework contribution, not a universal depth law. The revised framing is internally consistent: one theory (predictive refinement), controlled variable separation, four empirical claims, explicit falsifications, and bounded interpretation. The remaining score-limiting issue is replication of the one-seed architecture surface, which should be acknowledged rather than repaired with a post-hoc claim.
