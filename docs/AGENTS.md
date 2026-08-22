# AGENTS.md — Neural Structural Consolidation

## Mission

Extend the current PRA research codebase with a disciplined experimental layer for **Neural Structural Consolidation (NSC)**. The first milestone is *not* to train new model weights. It is to test whether transient attention/retrieval graph statistics improve sparse selection and reduce PRA materialization cost at matched task quality.

Preserve all existing PRA behavior and tests. New behavior must be opt-in.

## Scientific constraints

1. Do not claim that standard training “ignores attention structure.” Gradients already pass through attention. The hypothesis is that **sample-level recurrent relational structure is usually not explicitly accumulated as persistent reusable state during deployment**.
2. Keep three stages separate:
   - observation: collect structural statistics;
   - intervention: use them for frozen-model selection/sharpening;
   - learning: prototypes/adapters, deferred until Paper 1 passes its decision gate.
3. Never report a structural metric as mechanistic evidence without a causal or predictive utility test.
4. Weak attention edges may be bridge evidence. Do not equate low weight with noise.
5. Primary systems metric is **quality versus materialized KV tokens/bytes**, not raw accuracy alone.
6. All new selectors require matched-budget comparisons against tuned score-only baselines.

## Architecture

Create a small shared NSC package under the existing common/shared research code location, adapting names to the repository rather than duplicating infrastructure.

Suggested modules:

```text
common/nsc/
  __init__.py
  config.py
  trace.py
  features.py
  graph.py
  selectors.py
  metrics.py
  causal.py
  prototypes.py        # scaffold only for Paper 2
```

Integrate through existing PRA routing/materialization interfaces. Do not fork the attention implementation merely to collect metrics.

## Configuration

Add namespaced configuration with safe defaults:

```yaml
nsc:
  enabled: false
  trace:
    enabled: false
    level: candidate
    retain_full_attention: false
  graph:
    aggregation: layer_sum
    threshold: null
    community: label_propagation
  selector:
    mode: base
    gamma: 1.0
    bridge_budget_fraction: 0.0
  features:
    entropy: true
    persistence: true
    agreement: true
    community: true
    bridge: true
```

All fields must be CLI-overridable through the existing config/CLI mechanism. Record the fully resolved configuration in every artifact.

## Trace format

Prefer compact aggregated traces. Full attention matrices are diagnostic-only because O(N^2) trace storage can invalidate systems measurements.

Each example trace should include:

- run ID, git commit, model ID/revision;
- dataset/split/example ID;
- seed and resolved config;
- candidate IDs and lengths;
- base semantic/lexical/hybrid scores;
- selected/materialized set;
- per-layer/head compact attention summaries;
- entropy/concentration;
- cross-layer persistence;
- cross-head/layer agreement;
- graph/community/bridge features;
- task/evidence metrics;
- latency and memory measurements.

Use JSONL/Parquet or existing artifact conventions. Make schema version explicit.

## Paper 1 conditions

Implement:

- `base_topk`
- `budget_tuned_topk`
- `power_sharpen`
- `entropy_adaptive_sharpen`
- `persistence`
- `agreement`
- `community`
- `bridge_preserving`
- `combined_structural`
- `oracle_evidence` diagnostic only where labels exist

A selector must accept the same candidate set and materialization budget. Discovery and selection must remain separable.

## Attention sharpening

Baseline:

```python
p2 = normalize(p.clamp_min(eps).pow(gamma))
```

or equivalently score-temperature scaling where mathematically appropriate.

Test fixed gamma sweeps and entropy-adaptive gamma. Do not silently alter model-native attention in the first experiment if the intervention is intended only for PRA candidate selection. Keep `retrieval_score_sharpening` and `model_attention_sharpening` separate configuration modes.

## Graph features

Start with cheap GPU-friendly operations.

Candidate-level graph first:
- aggregate token/query-to-chunk relations;
- layer/head sums or rank-normalized aggregates;
- thresholded edge persistence;
- cross-head/layer agreement;
- connected components / label propagation;
- inexpensive bridge/cross-community scores.

Do not add NetworkX to hot paths. PyTorch tensor operations should be the default for benchmarked code. CPU graph libraries may be used for offline validation only.

## Causal utility

For manageable diagnostic subsets estimate candidate utility:

```text
U_i = quality(M) - quality(M \ {i})
```

or an evidence/routed-loss analogue.

Compare prediction of `U_i` from:
1. base score only;
2. base + lexical/semantic controls;
3. base + NSC structural features.

Use held-out examples. Report calibration/AUC/correlation as appropriate, but the decisive result is whether selection shifts the quality-budget frontier.

## Datasets

First reuse current PRA assets and splits:

- HotpotQA;
- QASPER;
- existing synthetic long-context tasks.

Add synthetic bridge tests with controlled:
- number of hops;
- distractor count;
- lexical overlap;
- bridge-edge strength;
- candidate count;
- chunk size.

Do not redesign datasets before smoke tests establish that instrumentation is correct.

## Model order

1. current low-cost Qwen3-0.6B PRA/HF path;
2. one larger model only after selectors and metrics are stable;
3. additional architectures only for generalization evidence.

Preserve GQA and exact disabled-PRA parity guarantees already used by the HF integration work.

## Evaluation outputs

Every main run should produce machine-readable results and plots for:

- task quality vs materialized tokens;
- task quality vs materialized KV bytes;
- quality vs latency;
- quality vs peak GPU memory;
- evidence recall vs budget;
- structural feature distributions;
- structural-feature prediction of causal utility;
- bridge failure cases;
- layer/head persistence heatmaps where useful.

Plot all conditions across a budget sweep. Do not select a single favorable k.

## Required ablations

- semantic vs lexical vs hybrid discovery;
- raw score vs rank-normalized score;
- fixed vs entropy-adaptive sharpening;
- persistence alone;
- agreement alone;
- community alone;
- bridge preservation alone;
- combined selector;
- layer ranges;
- head subsets;
- graph thresholds;
- chunk size/overlap;
- candidate-set size.

Where RoPE/rebinding configuration affects the experiment, log and ablate it rather than changing defaults.

## Tests

Add unit tests for:
- feature formulas on hand-constructed matrices;
- sharpening normalization and monotonicity;
- gamma=1 identity;
- graph aggregation;
- deterministic community labels up to canonical relabeling;
- bridge-score toy graphs;
- budget enforcement;
- selector determinism;
- trace serialization round-trip;
- disabled-NSC parity.

Add integration tests proving:
- existing PRA output is unchanged with `nsc.enabled=false`;
- no materialization-budget violation;
- trace-disabled hot path does not retain full attention matrices;
- CPU and CUDA selector results agree within tolerance.

## Performance discipline

Instrumentation overhead must be measured. Report routing/feature time separately from model forward and KV materialization. If graph computation costs more than materialization savings, record that as a negative systems result rather than hiding it.

Use vectorized PyTorch (`scatter_reduce_`, segmented reductions, tensor sorting/top-k) before Python loops.

## Paper 1 decision gate

Do not implement online prototype/adaptor learning as a mainline feature until results show at least one of:

1. structural features predict causal candidate utility beyond base retrieval scores on held-out data; or
2. a structural selector reproducibly shifts the quality/materialization Pareto frontier.

If neither holds, document the negative result and investigate why before proceeding.

## Paper 2 scaffold

`prototypes.py` may define interfaces only:

```python
class StructuralPrototypeStore(Protocol):
    def match(self, features): ...
    def update(self, features, outcome=None): ...
    def state_dict(self): ...
    def load_state_dict(self, state): ...
```

Future candidates include EMA/competitive prototypes, Oja/Hebbian updates, hierarchical prototypes, and outcome-gated updates. Do not bake one learning rule into Paper 1 architecture.

## Relationship to gated residuals and modal LLM

Keep NSC feature APIs generic enough that future callers can supply:
- residual state `r_l`;
- residual update `delta_l`;
- goal/task state `Z`;
- mode (`encode`, `generate`, `review`, `validate`);
- causal skip utility.

Do not introduce those architecture changes into Paper 1. The first paper must remain a clean PRA sparse-control test.

## Documentation

Update README/results docs after each experiment family. Record negative findings. Keep `.tex` tables/figures generated from committed machine-readable artifacts where practical. Never manually copy favorable numbers into the paper without provenance.

## Definition of done for Paper 1 implementation

- all existing tests pass;
- NSC-disabled parity passes;
- B0–B3 and S1–S5 implemented;
- controlled bridge synthetic suite implemented;
- multi-seed HotpotQA/QASPER runs reproducible;
- budget Pareto artifacts generated deterministically;
- instrumentation overhead measured;
- causal-utility diagnostic completed on a feasible subset;
- paper tables/plots regenerate from artifacts;
- limitations and failed hypotheses are retained in results.
