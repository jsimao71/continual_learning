# Paper 1 frozen structural-control pilot

## Scope

- Controlled bridge suite: 120 test examples x 5 seeds.
- Conditions B0--B3 and S1--S5 use identical candidate sets and exact chunk budgets; O1 is diagnostic.
- Frozen-Qwen audit: retained layerwise summaries for 84 identity-disjoint 2WikiMultiHopQA/MuSiQue examples.
- A follow-up frozen-Qwen candidate intervention uses eight validation and eight test identities per dataset, twelve 32-token candidates, and measured native K/V budgets of 64/128/192 tokens.
- No model weights were trained and no online prototype/adaptor was implemented.

## Main measured results

- At four chunks, base quality/path completion is 0.2971/0.0683.
- Bridge preservation reaches 0.6268/0.4800; combined structural selection reaches 1.0000/1.0000.
- Structural features raise held-out causal-utility Spearman from 0.4860 to 0.5281 in the controlled suite.
- In the pretrained observational audit, identity-disjoint Spearman changes from -0.0273 to -0.0048 after structural features are added to surface controls.
- Fixed-gamma and entropy-adaptive sharpening cannot change exact top-k membership because both are monotone transformations; their matched-budget rows therefore reproduce B0.
- Natural bridge-minus-base answer-logprob means are positive at all six HotpotQA/QASPER budget cells, but every paired 95% bootstrap interval includes zero.
- The natural validation-fitted S5 selector is negative in five of six cells; all four natural held-out removal-utility regressions have negative `R^2`.

## Decision gate

The controlled bridge result passes the synthetic utility/frontier gate. The natural candidate-level implementation closes the instrumentation and materialization gap and yields a consistent favorable bridge direction, but not a statistically resolved or learned-selector result. Paper 2 online consolidation remains blocked pending a larger fixed-protocol, multi-seed natural replication.
