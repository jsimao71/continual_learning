# Paper 1 natural replication v2 audit

- Frozen protocol: Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`, 12 candidates, 10 selectors, and 64/128/192 native K/V tokens.
- Completed sampling seeds: 20260906, 20260917, and 20260929.
- Each seed passes the independent artifact validator with 1,440 selector rows, 288 candidate-removal rows, 1,152 candidate records, and 1,440 selection traces.
- Pooled selector rows: 4,320.
- Unique test identities after averaging cross-seed repeats: 71 HotpotQA and 63 QASPER. The detailed overlap ledger is `tables/identity_overlap_audit.csv`.
- Every one of the six bridge-minus-base means is positive, but all unique-identity bootstrap intervals cross zero.
- Structural-minus-surface held-out R2 is negative for all three seeds on both datasets.
- Complete-only analysis validates the frozen configuration hash `d89d2cec0e79b7ece5fa43c2006a5b4bb2de3f93e5218a1cbe7b111022c18e36`.
- Authoritative decision: frontier branch fails on both datasets, causal-prediction branch fails on both datasets, and `persistent_learning_gate=0`.

Accordingly, no persistent-learning, adapter, override, rollback, or Paper 2 consolidation experiment is authorized by this result.
