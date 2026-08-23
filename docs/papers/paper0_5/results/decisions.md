# Decisions

- Pretrained causal replication is now complete: equivalent transfer exceeds syntax/semantic mismatch in trained Pythia and Qwen, but not reliably at random initialization.
- E7 remains stopped: common-direction addition is not beneficial and Paper 1's natural held-out utility gate remains negative.
- Controlled local training is used so training frequency is known exactly.
- Zero ablation supports causal pilot measurements; all logit-lens values remain explicitly diagnostic.
- Variance refinement is conditional rather than universal; Qwen does not show monotone depthwise JS reduction.
