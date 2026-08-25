# AGENTS.md — Paper 0.8: Inception of In-Context Learning

## Mission
Find the smallest training distribution and smallest Transformer in which a frozen model uses context to promote a target that is low-ranked without context to rank 1. Do not scale until the minimal mechanism is understood.

## Core criterion
Require rank(Y|X) >> 1 but rank(Y|D_context,X)=1, with no direct X->Y training leakage.

## Primary example
Train only local orders:
A->B, B->C
1->2, 2->3

Initially never train C->3 or complete paired A1/B2/C3 mappings.

Test:
A 1
B 2
C ?

Measure whether 3 is contextually promoted.

## Primary metric
G_ICL(Y)=margin(Y|context,X)-margin(Y|X)

Also record context-free/contextual rank, target probability, top1, first layer entering top-k, and first/stable top1 layer.

## Dataset ladder
D0 random/no structure.
D1 isolated local successor chains.
D2 many isomorphic independent chains.
D3 minimal paired/alignment exposure only if D1-D2 fail.
D4 episode-specific random permutations preventing global mapping memorization.
D5 richer arbitrary/compositional relations only after D1-D4 are understood.

Add exactly one structural ingredient at each stage.

## Dataset phase variables
Sweep number of relation families, examples per family, task/mapping entropy, chain length, symbol reuse, template variation, and position variation. Goal: minimal sufficient statistical structure, not just minimal sample count.

## Tiny-model grid
layers=[1,2,3,4,6]
d_model=[8,16,32,64]
heads=[1,2] where feasible
Use >=3 seeds near the capability boundary.

## Mandatory controls
1. no demonstrations
2. correct demonstrations
3. shuffled outputs
4. wrong-chain demonstrations
5. same-length irrelevant context
6. order/reversal perturbation
7. direct X->Y leakage audit
8. random initialization
9. held-out symbols
10. held-out relation families

A positive ICL claim requires correct demonstrations to selectively outperform matched controls.

## Leakage audit
Verify:
- C->3 never occurs in the relevant training regime
- no token/position uniquely predicts 3
- no complete cross-domain mapping is recoverable from weights alone
- context-free C keeps 3 low-ranked
Store exact counts and MI/conditional statistics.

## Layerwise capture
At every boundary:
- residual_pre_SA
- SA_update
- residual_post_SA
- FF_update
- residual_post_FF
- target logit/rank/margin
- top-k candidates
- entropy

Compute contextual delta trajectories relative to matched context-free runs.

## Mechanism questions
1. When does 3 first gain probability mass?
2. Does SA make 3 a candidate?
3. Does FF make an SA-created candidate top1?
4. Is a relational/successor state transported rather than token 3 itself?
5. Are the two chains aligned by particular heads?
6. Does the mechanism transfer across symbol families?
7. Is the mechanism already present below threshold but too weak, or does a qualitatively new circuit appear?

## Attention and FF analysis
Measure query attention to A,B,1,2 and separators, but do not infer causality from motifs.
Measure SA and FF target-margin increments separately.
Test FF update similarity across isomorphic chain families and causal role-matched replacement.

## Causal replacement
Patch from:
- same relational role/different symbols
- correct demo
- shuffled demo
- wrong chain
- unrelated matched-norm state

Measure margin damage, JS damage, rank change, top1 flip.

## Acquisition-threshold experiment
Find matched D- and D+ conditions immediately below/above ICL emergence.

Compare:
- head utility
- SA/FF updates
- residual target projection
- relation-role similarity
- cross-layer covariance
- weight changes

This before/after comparison is a primary contribution.

## Minimality deletion tests
After finding a positive generator remove one factor at a time:
- one relation family
- one edge type
- positional regularity
- symbol overlap
- one layer
- one head
- width

If ICL survives, simplify again.

## Required artifacts
Create:
docs/papers/paper0_8/
  paper0_8.tex
  results/generator_validation/
  results/phase/
  results/mechanism/
  results/causality/
  results/summaries/
  figures/

Tables:
- icl_generator_validation.csv
- icl_phase_grid.csv
- icl_rank_promotion.csv
- icl_layerwise_margin.csv
- icl_sa_ff_contributions.csv
- icl_head_utility.csv
- icl_causal_replacement.csv
- icl_minimality_deletions.csv

Figures:
- icl_context_free_vs_context_rank.png
- icl_dataset_phase_boundary.png
- icl_model_phase_boundary.png
- icl_target_margin_vs_depth.png
- icl_sa_ff_promotion_vs_depth.png
- icl_head_utility.png
- icl_before_after_acquisition.png
- icl_causal_replacement.png

## Summary questions
Create results/summaries/inception_summary.md answering:
1. What is the smallest dataset structure producing ICL?
2. What is the smallest competent model?
3. Is local order alone sufficient?
4. If not, what next ingredient is necessary?
5. Does task diversity/mapping entropy matter more than raw example count?
6. Where does the target first gain mass?
7. Which sublayer makes it top1?
8. Which heads are causal?
9. Does FF encode reusable successor/order transformations?
10. Do role-matched states transfer across symbol families?
11. What changes at the acquisition threshold?
12. Can the mechanism be stated without regression/Bayesian/gradient-descent metaphors?

## Interpretation rules
Do not call generic next-token completion ICL.
Do not infer ICL if the answer is already high-ranked without context.
Do not infer mechanism from attention patterns alone.
Do not invoke high-level algorithmic analogies unless required by causal evidence.
Prefer the smallest physical mechanism consistent with measurements.

## Scaling policy
Scale only after the minimal positive regime is understood.

## Completion gate
Do not call the paper complete until:
- a leakage-free positive ICL regime exists, or the tested minimal ladders give a documented negative result;
- the minimal dataset/model boundary is mapped;
- correct vs shuffled/wrong/irrelevant controls are complete;
- context-free target remains low-ranked;
- layerwise candidate promotion is captured;
- SA/FF and head causal interventions are run on the minimal positive regime;
- D-/D+ acquisition threshold is compared;
- minimality deletion tests are complete;
- >=3 seeds support central claims;
- all artifacts are tracked and tests pass.

## Final scientific target
Identify the inception of ICL: the smallest statistical structure and neural architecture that lets context create a rank-1 prediction not supported by the frozen weights alone, and causally explain where the new probability mass comes from.
