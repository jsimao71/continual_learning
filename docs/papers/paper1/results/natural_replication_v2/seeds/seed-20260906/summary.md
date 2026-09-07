# Paper 1 natural candidate-level gate

## Scope

- Frozen Qwen/Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca` on identity-disjoint official HotpotQA and QASPER validation identities.
- 24 validation and 24 test identities per dataset; 12 fixed 32-token candidates and budgets 64/128/192 tokens.
- Every evaluation performs native causal K/V prefill; bytes and latency are measured from the materialized tensors and forward path.

## Main results

- HOTPOTQA bridge-minus-base answer-logprob delta (95% paired bootstrap CI): 64: +0.566 [-0.310, +1.549]; 128: +0.967 [+0.271, +1.937]; 192: +0.164 [-0.214, +0.628].
- QASPER bridge-minus-base answer-logprob delta (95% paired bootstrap CI): 64: +0.069 [-0.370, +0.421]; 128: -0.122 [-0.315, +0.050]; 192: -0.051 [-0.250, +0.151].

## Decision

Bridge preservation has a positive mean delta at all six dataset/budget cells, but every paired interval includes zero. The validation-fitted combined selector is mostly negative, and held-out candidate-removal models retain negative R2. This diagnostic run therefore narrows the uncertainty but does not pass the reproducibility gate for online consolidation; Paper 2 remains stopped.
