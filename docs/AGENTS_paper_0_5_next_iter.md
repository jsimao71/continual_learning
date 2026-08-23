# AGENTS.md — Paper 0.5 Next Iteration

## Mission
Advance Paper 0.5 from a controlled n-gram probe into a broader pretrained-mechanism study centered on **transferable functional equivalence in residual computation**.

Current strongest claim:

> After training, transformer sublayer updates become partially interchangeable within externally defined functional families, while equivalent replacements are substantially less harmful than nonequivalent replacements across held-out identities and harder noise/evidence regimes.

Do not center the paper on universal depthwise variance reduction. Conditional nuisance refinement remains important, but pretrained results show it is model- and regime-dependent.

## Frozen baseline
Preserve as established:
- relation-specific SA/FFN computation;
- intervention-dependent component responsibility;
- equivalent-donor replacement much less disruptive than nonequivalent replacement;
- attention motif specificity does not predict SA causal utility;
- entropy trajectories are non-monotone;
- controlled models show strong nuisance-conditioned dispersion reduction with depth;
- random initialization does not show the trained causal-equivalence effect;
- trained Pythia and Qwen show preferential equivalent transfer;
- common-direction removal is harmful in trained models, while addition is not reliably beneficial;
- pretrained depthwise dispersion reduction is not universal;
- online consolidation remains gated.

Interpret removal-without-addition as evidence for a **state-conditioned contributory subspace**, not a literal reusable rule vector.

## Revised thesis
Paper 0.5 should argue:

> Transformer training creates partially transferable, family-conditioned sublayer computations that can be causally exchanged across equivalent inputs. Depth can additionally improve nuisance invariance in some regimes, but such refinement is conditional rather than universal. Functional equivalence is therefore a stronger mechanistic signature than attention-motif stability or raw variance contraction.

Retain:

    r_(l+1) = r_l + Delta_l
    Delta_l = F_l(r_l)

Layers are sequentially conditioned, not independent estimators.

## Goal 1 — Expand pretrained family coverage
The current pretrained study has too few independent relation families for a broad claim.

Expand families across:
- lexical mappings;
- syntactic alternations/templates;
- semantic attribute mappings;
- number and color domains;
- entity/relation mappings;
- argument-structure templates;
- simple relational substitutions;
- controlled compositional templates;
- natural-language n-gram families where reliable matching is possible.

The **family** is a primary scientific unit. Do not treat many realizations from one family as independent evidence.

Use enough families for stable family-clustered bootstrap intervals and, if justified, mixed-effects analysis.

## Goal 2 — Syntax × semantics factorization
Make this a primary factorial experiment.

Factors:

    syntax S
    semantic domain C
    identity x
    nuisance N

For target family `(S,C)`, use donor conditions:
1. same syntax + same semantics;
2. same syntax + different semantics;
3. different syntax + same semantics;
4. different syntax + different semantics.

Run:
- SA replacement;
- FFN replacement;
- complete sublayer replacement;
- zero ablation;
- common-direction removal;
- common-direction projection/addition.

Questions:
- Is causal transfer mainly syntactic?
- mainly semantic?
- does `S × C` matter?
- does the balance change by component or depth?

Use family-clustered bootstrap as primary unless family count is large enough for stable interaction models.

## Goal 3 — Composition
This is the highest-value new mechanistic experiment.

Study:

    F(x)
    G(x)
    G(F(x))

or controlled natural-language equivalents.

Test three levels:

### Behavioral composition
Can the model solve held-out `G(F(x))` combinations?

### Representational composition
Do composed trajectories contain recoverable F-like and G-like components/subspaces?

### Causal composition
Can F- or G-family updates be patched into the composed computation and selectively alter the corresponding stage?

Strong evidence requires:
- validated F transfer;
- validated G transfer;
- held-out composition behavior;
- stage-sensitive causal effects;
- incompatible-donor controls.

Do not infer operator algebra from cosine similarity alone.

## Goal 4 — Stronger on-manifold intervention controls
Whole-vector replacement and projection may be off-manifold.

Add:
- nearest-neighbor donors in residual space;
- residual-norm matched donors;
- baseline-logit matched donors;
- naturally occurring equivalent donors;
- interpolation between intact and donor states;
- low-rank subspace replacement;
- matched same-layer/same-component donor pools;
- path-specific patching where feasible.

Always compare:
- intact;
- zero;
- mean/matched mean;
- equivalent;
- same-syntax/different-semantics;
- same-semantics/different-syntax;
- fully nonequivalent;
- nearest-neighbor nonequivalent;
- norm-matched random perturbation.

The goal is to rule out “equivalent donors are simply less off-manifold.”

## Goal 5 — Common subspace analysis
For each family/layer/component:
1. fit common directions/subspaces on train identities only;
2. measure explained variance/effective rank;
3. remove them on held-out identities;
4. add them to compatible and incompatible states;
5. transfer across paraphrases and related families;
6. compare rank-1 vs low-rank models.

Questions:
- Is family computation one-dimensional or subspace-valued?
- Does dimensionality differ between SA and FFN?
- Does it change over training?
- Does paraphrase preserve the same subspace?
- Is the subspace useful only in a compatible state?

Preferred interpretation when removal hurts but addition fails:

> the component is necessary/contributory in context but not sufficient independently.

## Goal 6 — Reframe variance as conditional refinement
Keep separate:

A. within-example entropy:

    H[p_l(Y|X)]

B. across-realization dispersion:

    Dispersion[p_l(.|X) | P]

C. between-pattern / within-pattern separation

D. causal functional-transfer utility

Do not claim universal denoising.

Classify families/models as:
- dispersion contraction;
- expansion;
- non-monotone;
- early saturation;
- late repair.

Test predictors:
- baseline entropy;
- target margin;
- family type;
- model/checkpoint;
- number of demonstrations;
- distractor count;
- component;
- override requirement.

Question:

> Under what conditions does additional depth improve nuisance invariance?

## Goal 7 — Evidence count × depth × noise
Add an explicit number-of-supporting-examples axis.

For relation family F:

    F(a) -> y
    F(b) -> y
    ...
    query F(c)

vary:
- demonstrations: 0, 1, 2, 4, 8 where feasible;
- distractor count/type;
- evidence distance;
- depth.

Produce:

    quality(depth,evidence,noise)
    JS_dispersion(depth,evidence,noise)
    transfer_utility(depth,evidence,noise)
    separation(depth,evidence,noise)

Test whether depth and evidence:
- partially substitute;
- interact;
- saturate;
- differ by model/family.

Use **effective evidence utilization**, not literal iid sample averaging.

## Goal 8 — Training emergence
Use dense Pythia checkpoints.

Track:
- equivalent vs nonequivalent transfer;
- syntax/semantic mismatch effects;
- common-subspace removal;
- subspace rank;
- nuisance dispersion;
- entropy;
- target margin;
- SA/FFN responsibility.

Questions:
- When does functional equivalence emerge?
- Does it appear before or after behavioral competence?
- Does it sharpen gradually or in phases?
- Does it move between SA and FFN?

Do not assume monotone SA-to-FFN compilation.

## Goal 9 — Frequency/familiarity as explanatory variables
For corpus-derived families record:
- reference frequency;
- continuation entropy;
- PMI/association;
- tokenization length;
- lexical overlap;
- context dependence.

Test whether these predict:
- causal transfer magnitude;
- common-subspace rank;
- SA/FFN balance;
- nuisance refinement;
- override sensitivity.

Treat the previous low explanatory R² as a baseline.

## Goal 10 — Expand override/repair
Construct cases where a frequent/local mapping supports `y_default` but wider context requires `y_override`.

Measure:
- early default activation;
- later suppression;
- equivalent transfer of override-supporting computation;
- interference from default-family donors;
- effect of common-subspace removal;
- entropy expansion during reconsideration.

This is critical for continual learning: any future consolidation mechanism must preserve override behavior.

## Statistics
Primary rules:
- family/relation identity is the independent scientific unit where appropriate;
- use family-clustered bootstrap confidence intervals;
- use identity-disjoint fit/evaluation;
- use permutation controls;
- fixed train/validation/test splits;
- no family-specific tuning on test.

Report effect sizes, intervals, family-level scatter, sign consistency, and failures.

Avoid row-level significance tests over correlated intervention rows.

## Models
Minimum:
- Pythia-70M-deduped across dense checkpoints;
- pinned Qwen3-0.6B;
- one additional architecture if feasible.

Preferred third architecture:
- small Llama-family;
- Gemma-family;
- or another hookable small decoder.

Pin all revisions.

## Required figures
1. Family-level causal transfer: equivalent vs syntax mismatch vs semantic mismatch vs fully nonequivalent.
2. Syntax × semantics interaction by component/layer.
3. Composition: F, G, G(F(x)).
4. Common-subspace rank plus removal/addition effects.
5. Training emergence of causal transfer.
6. Depth × evidence × noise surfaces.
7. Family-specific contraction/expansion/non-monotone variance regimes.
8. Override/repair trajectories.
9. On-manifold donor-control comparison.

## Required tables
### A — Pretrained family matrix
Model/checkpoint, independent families, syntax domains, semantic domains, realizations, evidence counts, noise conditions.

### B — Causal transfer
Equivalent-minus-nonequivalent, equivalent-minus-syntax-mismatch, equivalent-minus-semantic-mismatch, removal effect, addition effect.

### C — Composition
F behavior, G behavior, composed held-out behavior, causal transfer.

### D — Variance/entropy
Keep separate: entropy, JS dispersion, target-probability variance, between/within separation, causal transfer.

## Evidence ladder
### Observation
Common direction, geometric similarity, attention motif.

### Functional prediction
Family membership predicts held-out output/transfer similarity.

### Causal transfer
Equivalent replacement preserves behavior better than mismatches; removal selectively harms.

### Composition
Validated family computations causally participate in a held-out composed transformation.

Only after the composition level and broad replication should stronger phrases such as **reusable transformation family** or **soft operator-like behavior** be used.

Do not call Paper 0.5 symbolic-rule execution.

## Relation to Papers 0.6 and 0.7
Paper 0.5 should establish:
- functional equivalence;
- transferable residual computation;
- conditional nuisance invariance;
- state/component dependence;
- primitive composition if supported.

Paper 0.6 remains responsible for **behaviorally learned semantic categories**.

Paper 0.7 combines:
- Paper 0.5 transformations;
- Paper 0.6 learned semantic types;
- syntax × semantic interactions;
- type-sensitive dispatch and composition.

Do not let generator-defined categories in Paper 0.5 substitute for learned semantic types.

## Export to residual/gating research
Retain per-example/per-layer labels sufficient for later gating work:
- true causal skip utility;
- equivalent-transfer utility;
- nuisance-dispersion improvement;
- common-subspace removal effect;
- repair requirement.

The gating line should learn to predict:

    expected marginal functional refinement

rather than relying only on entropy, residual norm, or attention concentration.

## Continual-learning gate
E7 remains stopped.

Do not implement persistent prototypes/adapters merely because functional equivalence is stronger.

Persistent learning still requires:
- reproducible natural future utility;
- matched-budget gains;
- override preservation;
- contamination/rollback controls;
- improvement beyond ordinary token/hidden-state baselines.

## Falsification criteria
Accept as valid outcomes:
1. Equivalent transfer weakens when family breadth increases.
2. Syntax explains most apparent semantic transfer.
3. Semantic mismatch is no worse than equivalent transfer.
4. Common structure is high-rank/distributed.
5. Composition fails despite individual transfer.
6. Nuisance refinement remains architecture-specific.
7. Family effects are highly heterogeneous.
8. Functional equivalence exists without a stable geometric subspace.
9. On-manifold controls reduce current intervention effects.
10. Continual-learning utility remains absent despite mechanistic equivalence.

## Implementation order
1. Expand pretrained families.
2. Run syntax/semantic factorized causal transfer.
3. Fit/test common low-rank subspaces and on-manifold controls.
4. Replicate across dense Pythia checkpoints.
5. Implement F/G/composition experiments.
6. Add evidence-count × depth × noise.
7. Expand override/repair.
8. Rewrite Paper 0.5 around functional equivalence first, conditional variance second.

## Preferred terminology
Use:
- functional equivalence;
- family-conditioned computation;
- causal transfer;
- transferable residual update;
- state-conditioned transformation;
- conditional nuisance invariance;
- effective evidence utilization;
- common residual subspace;
- override/repair;
- composition candidate.

Avoid:
- literal rule vector;
- independent layer samples;
- universal denoising;
- fixed “attention retrieves / MLP memorizes” claims;
- symbolic rule execution;
- consolidation based on recurrence alone.

## Target end-state
A strong next revision should be able to state:

> Across a broader pretrained family matrix, trained transformers exhibit causal functional equivalence: sublayer computation transfers preferentially among externally equivalent relation instances, with syntax and semantics contributing separable and/or interacting structure. Common subspaces are causally relevant but state-dependent. Conditional nuisance refinement occurs in identifiable regimes rather than universally. Primitive composition either emerges under held-out tests or is cleanly falsified. These findings establish transferable transformation families as a mechanistic precursor for later semantic-type and fuzzy-rule studies, while online consolidation remains gated on independent natural utility.
