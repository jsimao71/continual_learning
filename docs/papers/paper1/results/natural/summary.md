# Paper 1 natural candidate-level gate

## Scope

- Frozen Qwen/Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca` on identity-disjoint official HotpotQA and QASPER validation identities.
- 8 validation and 8 test identities per dataset; 12 fixed 32-token candidates and budgets 64/128/192 tokens.
- Every evaluation performs native causal K/V prefill; bytes and latency are measured from the materialized tensors and forward path.

## Main results

- HOTPOTQA bridge-minus-base answer-logprob delta (95% paired bootstrap CI): 64: +1.086 [-0.688, +2.943]; 128: +0.413 [-0.497, +1.710]; 192: +0.640 [-0.210, +1.846].
- QASPER bridge-minus-base answer-logprob delta (95% paired bootstrap CI): 64: +0.542 [-0.021, +1.363]; 128: +0.011 [-0.003, +0.035]; 192: +0.503 [-0.042, +1.550].

## Decision

Bridge preservation has a positive mean delta at all six dataset/budget cells, but every paired interval includes zero. The validation-fitted combined selector is mostly negative, and held-out candidate-removal models retain negative R2. This diagnostic run therefore narrows the uncertainty but does not pass the reproducibility gate for online consolidation; Paper 2 remains stopped.

## Functional-equivalence revisit

- Paper 0.5 membership is represented only by a frozen dynamical proxy because candidate-conditioned output distributions were not retained; it is not relabeled as functional equivalence.
- Paper 0.6 semantic/type features are unavailable because all Paper 0.6 runs failed their completion-competence gate.
- All feature-family models have negative held-out R2 on both task families. The best rank-only change is QASPER's predictive proxy (Spearman 0.517), but its R2 is -1.370.
- This is the second natural causal null. Persistent learning remains stopped.
