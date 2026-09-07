# Paper 1 natural candidate-level gate

## Scope

- Frozen Qwen/Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca` on identity-disjoint official HotpotQA and QASPER validation identities.
- 24 validation and 24 test identities per dataset; 12 fixed 32-token candidates and budgets 64/128/192 tokens.
- Every evaluation performs native causal K/V prefill; bytes and latency are measured from the materialized tensors and forward path.

## Main results

- HOTPOTQA bridge-minus-base answer-logprob delta (95% paired bootstrap CI): 64: +0.263 [-0.201, +0.851]; 128: +0.054 [-0.543, +0.667]; 192: +0.458 [+0.093, +0.942].
- QASPER bridge-minus-base answer-logprob delta (95% paired bootstrap CI): 64: +0.416 [-0.353, +1.295]; 128: +0.168 [-0.319, +0.784]; 192: +0.281 [-0.020, +0.806].

## Decision

Bridge preservation has a positive mean delta at all six dataset/budget cells, but every paired interval includes zero. The validation-fitted combined selector is mostly negative, and held-out candidate-removal models retain negative R2. This diagnostic run therefore narrows the uncertainty but does not pass the reproducibility gate for online consolidation; Paper 2 remains stopped.
