# AGENTS.md — Paper 0.5: Common N-Grams as Causal Probes of Transformer Computation

## Mission

Implement the experiments for **Paper 0.5 of the Continuous Self-Attention Learning line**.

The scientific objective is to determine how recurrent token-prefix -> next-token relations are represented across:

1. self-attention (SA),
2. feed-forward / MLP layers,
3. the residual stream,
4. model depth and training time.

This is a **mechanistic measurement paper first**. Do not turn it prematurely into a large continual-learning implementation project. The later continuous-learning papers depend on this work to define what stable relation should be learned and what signal should drive that learning.

## Core research question

For a token n-gram prefix `g = (x[t-n+1], ..., x[t])` with empirical continuation distribution `P(y | g)`, where and how does the model implement the transformation toward likely continuation `y`?

Possible mechanisms:

- direct/compiled MLP association;
- attention-dominant contextual retrieval;
- distributed SA -> MLP -> later SA/MLP circuit;
- early lexical prediction followed by later contextual override/repair.

Do not assume one mechanism is universally correct.

## Paper boundary

### Mandatory for Paper 0.5

- E1 corpus/synthetic n-gram atlas;
- E2 SA-vs-MLP layer-wise contribution analysis;
- E3 attention motif stability;
- E4 stored-vs-context-introduced relations;
- E5 contextual override / repair;
- E6 training-checkpoint emergence on at least one small model family.

### Optional / bridge experiment

- E7 tiny frozen-base adapter or prototype learner driven by recurring relation statistics.

### Explicitly out of scope

- full production continual learning;
- large replay buffers / long-horizon memory management;
- catastrophic-forgetting solution as a main contribution;
- RL-based adaptation;
- PRA integration as a dependency;
- distributed training/productization work.

Do not expand scope without a documented reason in `results/decisions.md`.

---

# 1. Repository layout

Prefer the following structure unless the existing repository already has equivalent modules:

```text
common/
  hooks.py
  model_adapter.py
  stats.py
  io.py
ngram/
  atlas.py
  controls.py
  synthetic.py
  sampling.py
analysis/
  component_contrib.py
  attention_motifs.py
  residual_trajectories.py
  causal.py
  clustering.py
  checkpoint_dynamics.py
experiments/
  e1_atlas.py
  e2_components.py
  e3_motifs.py
  e4_stored_vs_context.py
  e5_override_repair.py
  e6_training_dynamics.py
  e7_online_pilot.py
configs/
  *.yaml
results/
  raw/
  tables/
  figures/
  decisions.md
  summary.md
tests/
```

Reuse existing project `common/` utilities where sensible. Avoid duplicate model loaders, config loaders, seed helpers, or result writers if they already exist.

---

# 2. Reproducibility rules

Every experiment output must record:

- git commit;
- command line;
- config file and effective config after overrides;
- model identifier and revision/checkpoint;
- tokenizer identifier/revision;
- dataset/corpus identifier and split;
- random seed;
- device and dtype;
- package versions;
- timestamp;
- exact n-gram atlas version/hash.

Prefer JSONL or Parquet for per-example/per-ngram results and CSV for small aggregate tables.

No figure should contain values that cannot be regenerated from a saved machine-readable result file.

---

# 3. N-gram atlas

## 3.1 Required statistics

For each token prefix `g` and candidate continuation `y`, compute at minimum:

- `n`;
- prefix token IDs;
- decoded prefix for inspection;
- prefix frequency `f(g)`;
- continuation counts;
- `P(y | g)`;
- continuation entropy `H(Y | g)`;
- top continuation probability;
- PMI or log-odds-like association statistic;
- number of distinct continuations;
- source corpus/split.

Build atlas for at least `n in {1,2,3,4,5,8}` where corpus size allows.

## 3.2 Required strata

Generate reproducible samples for:

- high-frequency + low-entropy;
- high-frequency + high-entropy;
- low-frequency + low-entropy;
- matched-frequency groups with different entropy;
- matched-entropy groups with different frequency;
- same recent suffix but different earlier-prefix continuation;
- lexical substitution controls;
- context-override examples;
- synthetic unseen/context-introduced relations.

Avoid cherry-picking famous phrases as the primary evidence. Human-readable examples are for illustration only; main results must use stratified samples.

## 3.3 Leakage / corpus caveat

For pretrained models we generally do not know exact training-set occurrence counts. Corpus frequency is therefore a **proxy** unless the model has a documented training corpus.

Label variables clearly as:

- `reference_corpus_frequency`, not `training_frequency`, for opaque pretrained models;
- true `training_frequency` only for models we train ourselves or models with exact accessible training data.

This distinction must appear in plots and prose.

---

# 4. Model instrumentation

Create a model adapter exposing a common interface for each transformer block.

At minimum capture:

- residual before attention `r_l`;
- attention output/update `delta_sa_l`;
- residual after attention `r'_l`;
- MLP output/update `delta_ff_l`;
- block output `r_{l+1}`;
- attention probabilities where available;
- Q/K/V tensors when enabled;
- per-head output where feasible;
- MLP intermediate activation when enabled.

Instrumentation must be optional and memory-aware. A default experiment should not retain all tensors for all tokens if only the final prefix position is required.

Add exact parity tests verifying that hooks do not change logits when interventions are disabled.

---

# 5. Descriptive contribution metrics

For target continuation `y`, record layer-wise diagnostics such as:

- target logit;
- target log probability;
- target rank;
- top-k candidate list;
- change across SA update;
- change across MLP update.

Example diagnostic quantities:

```text
U_sa(l, y) = logit_y(post_sa) - logit_y(pre_sa)
U_ff(l, y) = logit_y(post_ff) - logit_y(post_sa)
```

Use model-appropriate normalization before unembedding. Label these as diagnostic/logit-lens quantities, not causal effects.

Also compute normalized component contribution ratios and concentration across layers, but retain signed values. Negative contributions are scientifically important, especially for override/repair cases.

---

# 6. Causal intervention experiments

Descriptive activation/logit measurements are insufficient for primary claims.

Implement at least:

1. zero ablation;
2. mean/matched-control ablation;
3. activation replacement/patching from a matched non-ngram context;
4. selective head ablation for candidate heads;
5. selective MLP-layer ablation.

Where computationally feasible add path patching or an equivalent causal mediation experiment.

Primary causal outcome:

```text
C(component, layer, g, y) =
    log P_intact(y | context)
  - log P_intervened(y | context)
```

Always include controls for generic layer importance. A layer being important for language modeling is not evidence that it stores a specific n-gram relation.

---

# 7. Attention motif analysis

For repeated occurrences of an n-gram, extract an aligned attention submatrix covering the prefix plus a controlled surrounding window.

Compare motifs using multiple representations:

- raw attention weights;
- row-normalized matrices;
- cosine/Frobenius similarity;
- Jensen-Shannon distance over rows;
- thresholded/adaptive attention graph;
- top-k edge sets;
- optional community/connected-component summaries;
- QK similarity matrices;
- OV/output directions.

Do not claim a stable ``attention signature'' from visual heatmaps alone.

Required controls:

- same tokens in different order;
- same final token with unrelated prefix;
- same recent suffix but different earlier prefix;
- position-matched unrelated sequence;
- lexical substitution preserving broad syntactic form;
- synthetic relation where mapping is introduced only in context.

Measure whether motif stability predicts **causal contribution**. This is more important than motif stability by itself.

---

# 8. Residual trajectory analysis

Represent each `(g, y)` using a trajectory over depth containing at least:

```text
[U_sa_1, U_ff_1, ..., U_sa_L, U_ff_L,
 C_sa_1, C_ff_1, ..., C_sa_L, C_ff_L]
```

Optionally add:

- residual/update cosine similarities;
- update norms;
- subspace angles;
- rank trajectory;
- entropy of next-token distribution;
- attention motif stability features.

Cluster trajectories only after establishing that results are robust to feature normalization and clustering method.

Candidate labels such as `early_lookup`, `progressive_composition`, `retrieval_copy`, `late_disambiguation`, and `override_repair` are hypotheses. Do not force clusters to match these labels.

Use unsupervised clustering plus post-hoc interpretation and supervised prediction of mechanistic features from corpus statistics.

---

# 9. Stored versus context-introduced experiment

Build a controlled factorial dataset approximating:

| Relation | Familiar/stored | Introduced in context |
|---|---|---|
| Exact lexical mapping | common n-gram | arbitrary `A -> B` mapping |
| Abstract relation | reusable schema | in-context rule/pattern |

The objective is to test whether:

- familiar lexical associations become more MLP-heavy;
- novel context-introduced mappings become more attention-heavy;
- intermediate cases use distributed circuits.

Treat this as an empirical hypothesis, not an assumption.

---

# 10. Override and repair experiment

Create or mine contexts where a high-probability local n-gram continuation is contradicted by wider context.

For each example track at every layer:

- probability/logit of the habitual continuation;
- probability/logit of the contextually correct continuation;
- SA and MLP updates for both;
- causal effect of candidate suppressing/promoting components;
- cosine relation between early habitual update and later corrective update.

Classify repair geometry:

- directly opposing;
- orthogonal/complementary;
- projection/removal of a subcomponent;
- replacement via a different subspace.

This experiment should export its metrics in a form reusable by the residual-stream paper line.

---

# 11. Training-time development

Use at least one model family with dense checkpoints (e.g. an existing checkpoint series) and, preferably, one tiny model trained locally on a controlled corpus.

For selected n-grams compute the same metrics at training steps `t`.

Test competing trajectories:

### Compilation hypothesis

```text
diffuse/contextual -> stable motif -> localized MLP association
```

### Early-MLP hypothesis

```text
simple MLP association emerges first -> later attention adds context sensitivity
```

### Persistent-distribution hypothesis

```text
relation remains distributed across depth throughout training
```

Do not discard evidence for any of the three.

For locally trained models, vary controlled n-gram frequency and entropy independently when possible.

---

# 12. Statistical analysis

The unit of inference should normally be the **n-gram**, not individual token occurrences.

Use:

- bootstrap CIs over n-grams;
- seed/model replication;
- mixed-effects regression where useful;
- effect sizes in addition to p-values;
- multiple-comparison correction for head-level searches.

Primary predictors:

- log frequency;
- continuation entropy;
- top continuation probability;
- PMI/association strength;
- n-gram length;
- context sensitivity / override status.

Primary responses:

- SA/MLP causal contribution ratio;
- layer of maximal causal contribution;
- contribution concentration across depth;
- attention motif stability;
- residual trajectory class/features.

---

# 13. E7 online-learning pilot — optional

Only start after E1-E6 produce stable signals.

Base model must remain frozen.

Compare small learners under matched parameter/update budgets:

- token/continuation-only adapter baseline;
- hidden-state-conditioned adapter;
- SA-statistic-conditioned adapter;
- prototype/local-rule learner;
- random/matched-statistics control.

A useful result is not merely lower loss. Measure whether the learned mechanism:

- reduces repeated SA dependence for the learned relation;
- reproduces the target residual direction or continuation distribution;
- preserves contextual override;
- avoids harming matched unrelated prefixes.

This is the bridge to the next paper, not the primary Paper 0.5 contribution.

---

# 14. Minimum figures

Produce at least:

1. **Concept diagram:** prefix relation -> SA/MLP/residual trajectory -> possible compilation.
2. **Atlas plot:** frequency vs continuation entropy with sampled strata.
3. **Layer contribution plot:** SA and MLP signed contributions across depth for multiple strata.
4. **Causal contribution plot:** matched intervention effects.
5. **Attention motif stability plot:** exact n-gram vs controls.
6. **Trajectory map:** clustering or low-dimensional projection of mechanistic trajectories.
7. **Override/repair example + aggregate:** habitual vs correct continuation across layers.
8. **Training dynamics:** component contribution vs checkpoint/training step.

Optional: E7 online-learning result.

Never rely only on a few illustrative heatmaps.

---

# 15. Required tables

At minimum:

- datasets/corpora and atlas sizes;
- model configurations;
- n-gram stratum definitions;
- aggregate SA/MLP causal contribution by stratum;
- regression/model linking corpus statistics to mechanism metrics;
- training-dynamics comparison;
- ablation/control summary.

---

# 16. Tests

Add unit/integration tests for:

- n-gram counts against a brute-force reference;
- entropy/probability calculations;
- token alignment and occurrence extraction;
- hook parity with instrumentation disabled;
- residual identity: `post_sa ~= pre_sa + delta_sa`, architecture permitting;
- block identity: `post_block ~= post_sa + delta_ff`, architecture permitting;
- causal intervention actually changes only intended component;
- attention-window alignment;
- deterministic sampling under fixed seed;
- serialization round trips;
- aggregation treats n-gram as primary unit where configured.

Run tests before every committed experimental change.

---

# 17. Interpretation guardrails

Do not equate:

- attention weight with causal importance;
- logit-lens alignment with computation;
- corpus frequency with true training frequency for opaque models;
- one neuron with a complete stored n-gram;
- a visually stable heatmap with an invariant circuit;
- an MLP contribution with proof that the relation is ``stored only in the MLP''.

Prefer statements such as:

> Under matched causal interventions, high-frequency low-entropy prefixes show a larger fraction of target-continuation causal effect in early MLP updates than matched high-entropy prefixes.

Avoid statements such as:

> MLPs store language while attention only retrieves it.

---

# 18. Falsification checklist

The project must preserve and report negative results if:

- motif stability vanishes under controls;
- frequency is unrelated to MLP localization;
- causal results disagree with logit-lens results;
- trajectories do not cluster robustly;
- training does not show a compilation trend;
- SA-derived statistics do not improve the E7 learner over hidden-state/token baselines.

A null result can still constrain the continuous-learning line and should not trigger repeated tuning until a preferred hypothesis appears.

---

# 19. Relationship to adjacent papers

## Residual-stream papers

Export common result schemas for:

- per-layer residual states;
- SA/MLP updates;
- update norms and cosine geometry;
- target logit/probability trajectories;
- repair/override labels.

These data should be reusable without making Paper 0.5 depend on the residual paper implementation.

## Continuous SA learning papers

Paper 0.5 determines:

- what recurring structures are stable;
- whether they are causal;
- where they appear;
- whether they correlate with later compiled mappings;
- which signals should condition online learning.

Future papers may then test local prototype learning, adapters, local learning rules, or backprop-based adaptation.

## PRA

PRA is not a dependency. If useful, later reuse motif/graph metrics for query-facet or retrieval-selection research. Do not introduce PRA-specific assumptions into the core experiments.

---

# 20. Execution order

Implement in this order unless blocked:

1. atlas builder + tests;
2. model instrumentation + parity tests;
3. descriptive per-layer contribution pipeline;
4. causal ablation pipeline;
5. E1/E2 baseline runs;
6. motif stability + controls;
7. stored-vs-context synthetic controls;
8. override/repair dataset and analysis;
9. training-checkpoint dynamics;
10. only then evaluate E7.

After each stage update `results/summary.md` with:

- what was run;
- exact artifacts;
- main numbers;
- failures/null results;
- next falsifiable question.

---

# Definition of done for Paper 0.5

Paper 0.5 is ready for drafting around results when all of the following are true:

- n-gram atlas and controls are reproducible;
- at least two model settings have layer-wise SA/MLP measurements;
- primary claims use causal intervention, not attention visualization alone;
- attention motif stability is tested against matched controls;
- stored-vs-context-introduced relations are compared;
- override/repair is analyzed both illustratively and in aggregate;
- at least one checkpoint series or controlled training run tests developmental hypotheses;
- all main figures/tables regenerate from saved data;
- negative results and failed hypotheses are retained;
- the paper can make a useful claim without requiring E7 to succeed.
