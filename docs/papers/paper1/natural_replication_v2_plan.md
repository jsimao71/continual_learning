# Paper 1 natural replication v2: frozen preregistration

## Scope

This is a larger replication of the existing frozen-Qwen bridge-versus-base protocol,
not a selector redesign. The model remains `Qwen/Qwen3-0.6B` at revision
`c1899de289a04d12100db370d81485cdf75e47ca`. Candidate extraction, the eight frozen
candidate features, all ten selectors, 12-candidate sets, 32-token chunks, native K/V
materialization, and 64/128/192-token budgets are unchanged.

The replication uses dataset-sampling seeds 20260906, 20260917, and 20260929. Each
seed samples 24 identity-disjoint validation and 24 test identities from each of
HotpotQA and QASPER. QASPER is split at paper identity and HotpotQA at example identity;
test identities are never used to fit the combined selector. Candidate-removal utility
expands from one to six validation and six test identities per dataset and sampling
seed. Each completed seed is immutable and independently resumable.

## Analysis fixed before execution

The primary endpoint remains the paired bridge-preserving minus base-top-k difference
in answer-token mean log probability at each native K/V budget. Seed-level results are
reported separately. Because independent sampling seeds can repeat a frozen test
identity, the pooled bridge analysis averages duplicate evaluations within identity
before an identity-level paired bootstrap; repeated identities are not counted as new
scientific units. Identity overlap and effective unique counts are mandatory outputs.

The combined selector remains validation-fitted with the existing feature columns and
ridge penalty. Its test result is reported per sampling seed without tuning. Removal
utility retains the existing surface-controls and surface-plus-structure models; their
held-out R2, RMSE, and Spearman values are aggregated across the three sampling seeds.
The causal-prediction branch passes only if the mean seedwise structural-minus-surface
R2 is positive and at least two of three seedwise differences are positive on each
dataset. The frontier branch passes only if the unique-identity bootstrap lower bound
is above zero on each dataset at one or more of its preregistered budgets.
No feature, selector, threshold, budget, or endpoint may be changed in response to test
results. The oracle selector remains diagnostic.

The gate passes only if the unchanged structural intervention reproducibly improves a
natural matched-budget frontier, or structural features improve held-out causal-utility
prediction beyond surface controls on both datasets. Otherwise persistent learning and
Paper 2 remain stopped. Directional means without uncertainty support do not pass.

## Execution and resources

Each seed writes to `results/natural_replication_v2/seeds/seed-<seed>` and receives a
completion record only after its manifest and required tables validate. Aggregation is
safe to rerun and writes only the replication root. Expected accelerator time is about
2--2.5 hours, based on 554.7 seconds for 236 unique selector evaluations in the
8-identity diagnostic plus the sixfold removal expansion. Expected tracked tabular and
trace storage is 10--15 MB; no model checkpoint is produced.
