# AGENTS.md — Paper 0.6: Hierarchical Semantic Abstraction as a Causal Probe of Transformer Computation

## Mission
Implement Paper 0.6 as the semantic/hierarchical counterpart to Paper 0.5. Paper 0.5 asks how recurrent token-prefix -> continuation relations are represented across SA, MLP, residual depth, and training time. Paper 0.6 asks the parallel question for instance -> category -> superclass, action, relation, and abstraction transformations.

This is a mechanistic measurement paper first. Do not turn it into a generic probing project or a broad theory-of-abstraction paper.

## Hard dependency / reuse from Paper 0.5
Reuse rather than duplicate Paper 0.5 infrastructure wherever possible:
- `common/hooks.py`, model adapter, stats, IO and config conventions;
- residual before SA `r_l`, SA update `delta_sa_l`, post-SA `r'_l`, MLP update `delta_ff_l`, block output `r_{l+1}`;
- optional per-head, Q/K/V, post-RoPE, OV and MLP-intermediate capture;
- exact hook parity tests;
- descriptive logit-space progress conventions;
- zero, mean/matched-control, activation replacement, head and MLP ablations;
- attention motif representations and controls;
- residual/mechanistic trajectory schemas;
- bootstrap/mixed-effects/multiple-comparison conventions;
- reproducibility metadata and machine-readable result rules.

Before implementation, inventory the actual Paper 0.5 repository and write `results/reuse_map.md` mapping reused modules and any required extensions.

## Paper boundary
### Mandatory
- E1 semantic hierarchy atlas + controls;
- E2 layer-wise hierarchy geometry;
- E3 SA-vs-MLP contribution and causal intervention;
- E4 semantic attention-motif invariance;
- E5 natural-vs-synthetic hierarchy;
- E6 contextual resolution / override / repair;
- E7 cross-domain replication over nouns, actions and relations;
- E8 checkpoint/training-time development on at least one model family if Paper 0.5 checkpoint infrastructure is available.

### Optional
- E9 micro/macro coarse-graining pilot.

### Out of scope
- full continual-learning implementation;
- claiming exact Lie-group structure;
- treating linear probe accuracy as primary evidence;
- large ontology engineering;
- PRA as a dependency.

## Repository layout
Prefer Paper 0.5's existing layout. Add only semantic-specific modules, e.g.:
```text
semantic/
  atlas.py
  hierarchy.py
  templates.py
  synthetic.py
  controls.py
analysis/
  hierarchy_geometry.py
  semantic_motifs.py
  semantic_trajectories.py
  # reuse component_contrib.py, causal.py, checkpoint_dynamics.py
experiments/
  e1_semantic_atlas.py
  e2_hierarchy_geometry.py
  e3_components.py
  e4_semantic_motifs.py
  e5_natural_vs_synthetic.py
  e6_resolution_repair.py
  e7_cross_domain.py
  e8_training_dynamics.py
  e9_micro_macro.py
```
Do not create parallel loaders/hooks/stats code if Paper 0.5 already provides it.

## Semantic hierarchy atlas
Each item must record:
- item/leaf ID and decoded label;
- semantic family;
- hierarchy path, parent IDs, depth and sibling set;
- known pairwise tree distance;
- template/context ID;
- tokenizer IDs and token count;
- lexical/reference-corpus frequency where available;
- Paper 0.5 n-gram frequency, continuation entropy/predictability where applicable;
- split and hierarchy version/hash.

Required strata:
- instance -> category -> superclass;
- sibling substitutions at equal depth;
- actions: e.g. crawl/walk -> locomotion -> movement;
- relations: specific -> broader relation class;
- same lexical item under contexts requiring different resolution;
- natural hierarchy;
- arbitrary-label synthetic hierarchy introduced by definitions/examples;
- permuted/random hierarchy negative control;
- matched frequency/n-gram-statistics subsets.

For opaque pretrained models, call corpus counts `reference_corpus_frequency`, never `training_frequency`.

## Controlled language
Do not assume arbitrary hypernym substitution preserves proposition truth. Use:
1. category membership/definition prompts;
2. controlled entailment pairs;
3. semantically licensed templates;
4. synthetic worlds with exact hierarchy;
5. arbitrary-label in-context hierarchies.

Use multiple paraphrases/templates. A result tied to one template is insufficient.

## Primary geometry metrics
For every layer/sub-layer compute:
- within-class dispersion;
- matched between-class dispersion;
- normalized separation;
- activation-distance matrix;
- correlation/RSA with known hierarchy distance;
- tree/neighborhood recovery;
- invariant onset;
- persistence length;
- disappearance/re-entry;
- residual-update cosine/subspace similarity;
- shared-update explained variance and rank.

The main summary should support `I(layer, abstraction_level, transformation_family)`.

## SA / MLP decomposition
Use Paper 0.5's exact conventions:
- pre-SA -> post-SA;
- post-SA -> post-MLP;
- block input -> block output.

Where a task supplies a target answer/token, reuse Paper 0.5 diagnostic `U_sa`, `U_ff` and causal `C(component, layer, x, y)` quantities. Keep signed effects. Negative updates are important for contextual resolution/repair.

Do not assume attention retrieves relations and MLP stores concepts. Test the division of labor.

## Semantic attention motifs
Extend Paper 0.5 motif machinery from exact n-grams to semantic equivalence classes. Compare:
- raw attention;
- row-normalized motifs;
- cosine/Frobenius/JS metrics;
- thresholded or top-k attention graphs;
- QK similarity;
- OV/output directions;
- residual-update directions.

Required controls mirror Paper 0.5:
- same lexical tokens with changed semantic relation where possible;
- same syntactic form but cross-category substitution;
- position-matched unrelated examples;
- lexical substitution preserving category/relation;
- synthetic hierarchy introduced only in context.

Motif stability must be related to causal contribution, not just visual similarity.

## Mandatory Paper 0.5 statistical controls
Semantic hierarchy effects must be tested after controlling for:
- token/lexical frequency;
- token length/tokenization;
- reference-corpus n-gram frequency;
- continuation entropy;
- next-token predictability/top probability;
- sentence template/syntax;
- context sensitivity.

Use matching/stratification and, where appropriate, mixed-effects regression or residualization. A semantic effect that disappears under these controls must be reported as such.

## Semantic transformation trajectories
Reuse Paper 0.5's trajectory pipeline and augment it with hierarchy metrics. Candidate post-hoc classes:
- early categorization;
- progressive abstraction;
- distributed SA-MLP composition;
- late contextual resolution;
- override/repair;
- multi-scale coexistence.

Do not force clusters to match these labels. Test robustness to feature normalization and clustering method.

## E6 contextual resolution / repair
Construct contexts where the same entity/event must be processed at coarse versus fine semantic resolution. Track across depth:
- fine and coarse category scores/geometry;
- target answer probabilities where available;
- SA and MLP signed updates;
- causal effects of candidate promoting/suppressing components;
- geometry between early coarse updates and later fine-resolution updates.

Reuse Paper 0.5 repair labels where meaningful: opposing, orthogonal/complementary, projection/removal, replacement in another subspace.

## Causal stage
Primary claims cannot rely only on probes/RSA. Reuse:
1. zero ablation;
2. mean/matched-control ablation;
3. activation replacement/patching;
4. selective head ablation;
5. selective MLP-layer ablation.

Add when justified:
- sibling vs non-sibling patching;
- add/remove candidate category directions;
- project out candidate invariant subspaces;
- cross-template transfer of discovered directions.

Always control generic layer importance.

## Training-time development
If Paper 0.5 E6/checkpoint code is available, use the same model/checkpoint series where possible. Track whether hierarchy structure:
- emerges first as lexical/local association;
- becomes increasingly relational/context-invariant;
- remains distributed;
- shows changing SA/MLP responsibility.

Do not assume monotonic abstraction or a fixed compilation direction.

## Statistics
Unit of inference should normally be semantic item, hierarchy branch, or controlled transformation—not raw token occurrence.
Use bootstrap CIs, seed/model replication, effect sizes, mixed-effects models where useful, and multiple-comparison correction for layer/head searches.

## Minimum figures
1. concept diagram: hierarchy transformation -> SA/MLP/residual trajectory;
2. semantic atlas/hierarchy design;
3. layer x abstraction-level invariance heatmap;
4. hierarchy-distance vs activation-distance correspondence by layer;
5. within vs between dispersion;
6. SA/MLP signed and causal contribution curves;
7. semantic attention-motif stability vs controls;
8. invariant onset/persistence/re-entry;
9. contextual resolution/repair aggregate;
10. natural vs synthetic hierarchy;
11. n-gram-controlled vs uncontrolled semantic effects;
12. training dynamics if E8 is run.

## Required tests
In addition to Paper 0.5 shared tests:
- hierarchy paths/tree distances against brute-force reference;
- sibling/parent/depth metadata consistency;
- deterministic synthetic hierarchy generation;
- template semantic labels and token/span alignment;
- invariance metrics on toy data with known expected hierarchy;
- random/permuted hierarchy gives chance-like recovery;
- result joins with Paper 0.5 n-gram statistics are deterministic;
- aggregation uses semantic item/branch as configured unit.

## Interpretation guardrails
Do not equate:
- probe decodability with invariant computation;
- cosine similarity with semantic identity;
- attention weight with causal importance;
- hierarchy correlation with exact symbolic taxonomy storage;
- shared residual direction with a literal Lie generator;
- semantic abstraction with thermodynamic coarse-graining;
- later depth with higher abstraction by definition.

Prefer claims of approximate, transformation-conditioned invariance.

## Falsification checklist
Preserve/report negative results if:
- hierarchy geometry vanishes after Paper 0.5 surface-statistical controls;
- synthetic hierarchy is not recovered above random/permuted controls;
- probes succeed while causal/geometric evidence fails;
- semantic motifs fail to generalize across lexical substitutions;
- abstraction depth has no systematic layer relation;
- residual commonality is weak/non-transferable;
- Paper 0.5 n-gram statistics explain the apparent semantic hierarchy better than hierarchy variables.

## Definition of done
Paper 0.6 is ready for result-driven drafting when:
- semantic atlas and controls are reproducible;
- Paper 0.5 infrastructure is reused and documented;
- at least two model settings have layer-wise hierarchy/SA/MLP measurements;
- hierarchy effects are tested against lexical and n-gram controls;
- natural and synthetic hierarchies are compared;
- semantic motif stability is tested against matched controls;
- contextual resolution/repair is analyzed;
- primary claims include causal interventions;
- main figures regenerate from saved data;
- negative results are retained;
- the paper makes a useful claim without E9.
