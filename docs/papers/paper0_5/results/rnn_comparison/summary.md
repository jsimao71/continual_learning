# Matched recurrent comparison

All four architectures saw identical generated examples, optimizer family, training steps, vocabulary, target entropy, and token budget. Exact parameter counts are {'transformer': 40896, 'rnn': 16640, 'gru': 33280, 'lstm': 41600}.

Linear accuracy slopes (descriptive over the tested range) are:

| model | length | nuisance | predictive_order | span |
|---|---|---|---|---|
| gru | 0.00000 | -0.00000 | -0.00000 | -0.00000 |
| lstm | -0.00334 | -0.00000 | -0.02987 | -0.00000 |
| rnn | -0.01116 | -0.00000 | -0.13189 | -0.00712 |
| transformer | 0.00000 | -0.00049 | -0.11834 | -0.00000 |

The fixed-order length and span hypotheses are supported only when the recurrent slopes are more negative than the Transformer slope and the relevant models are behaviorally competent. Predictive-order and nuisance curves are reported separately. RNN time is sequential transport; Transformer depth is parallel refinement, so their internal indices are plotted but never equated. This benchmark excludes state-space and recurrent-attention models.
