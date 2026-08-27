# Manually constructed micro-Transformer witnesses

## Scope

Milestone 1 provides deterministic fixed-weight constructions for three M0 witnesses executed with the Paper 0.1 C0 fixed-pass protocol. No optimization or fitting occurs. These results establish only that the declared architectures can implement the tasks under the declared encodings. They do not establish architectural minimality, weight uniqueness, SGD acquisition, robustness outside the legal domain, or a cognitively canonical mechanism.

The implementation uses row-vector weights, causal finite-softmax attention, identity residuals, ReLU FF blocks, zero positional embeddings, and no LayerNorm or dropout. Each topology records all matrices, including explicit zero matrices for absent components. Canonical traces include embeddings, Q, K, raw QK transpose products, scaled/masked scores, probabilities, V, head output, post-SA residual, FF preactivation and activation, post-FF residual, and final logits.

## Milestone-1 result

| Task | Controller | Data | Primary topology | L | H | d | Legal domain | Result | Controls |
|---|---|---|---|---:|---:|---:|---:|---|---|
| successor | M0 witness / C0 protocol | weights | FF-only | 1 | 0 | 4 | 3 inputs | 3/3, margin 1 | SA-only 0/3; SA+FF 3/3 |
| associative pair lookup | M0 witness / C0 protocol | context | SA-only | 1 | 1 | 9 | 6 bijections x 3 queries | 18/18, minimum margin 0.99999955 | FF-only 6/18 ties; SA+FF 18/18 |
| grandparent | M0 witness / C0 protocol | context | SA-only | 2 | 1 | 20 | both depth-2 nodes in fixed tree | 2/2, minimum margin 0.99999955 | FF-only 0/2; SA+FF 2/2; one-layer SA 0/2 |

All passing results use no recurrence, external iteration, information service, or tool. The contextual pair and edge records are atomic tokens with explicitly separated key/value features. This choice makes the circuit readable but is not claimed to be the smallest serialization.

## Milestone-2 result: bounded autoregressive reuse

Milestone 2 adds two M1 witnesses executed with a C1 external autoregressive model loop. The driver appends the model's argmax output and calls the identical one-layer, one-head circuit (`d_model=15`, `d_head=5`) again; it is not a tool and performs no parent or unary transition itself. Atomic map tokens separate keys and values for readability.

| Task | Controller | Data | Primary topology | Tested domain | Result | Controls |
|---|---|---|---|---|---|---|
| root traversal | M1 witness / C1 protocol | contextual parent map | SA-only | all 6 three-node orderings x 4 starts; root depths 0--3 | 24/24 argmax-correct trajectories; minimum signed target margin 0.99999910 | FF-only 6/24; SA+FF 24/24 |
| unary chain traversal | M1 witness / C1 protocol | contextual unary map | SA-only | all 24 four-symbol orderings x 4 starts; chain lengths 1--4 | 96/96 argmax-correct trajectories; minimum signed target margin 0.99999910 | FF-only 24/96; SA+FF 96/96 |

For either task, a contextual `MAP_X_Y` token carries `X` in the key subspace and `Y` in the value subspace. A `STATE_X` token carries `X` only in the query subspace. One forward call retrieves `Y`; the autoregressive harness appends `STATE_Y` and invokes the identical block again:

```
x_(t+1) = transition(x_t; fixed contextual map).
```

The desired scaled score is 16 at every step. Previously generated state tokens have zero keys and add finite softmax leakage, so the desired probability decreases slightly with trajectory length; at the longest declared step it is 0.9999992122543974. The minimum observed signed target margin remains 0.99999910. “Exact” means exhaustive argmax correctness with positive margin, not one-hot softmax. Every step's scores, probabilities, residuals, FF intermediates, logits, winner margin, and signed target margin are recorded. `STOP` is explicit. FF-only's 25% trajectory accuracy consists of deterministic tie-break `STOP` cases with zero margin.

These results demonstrate reuse through root depth three and unary-chain length four. Reversed and fixed-shuffle map serializations also pass every case. The legal domain requires one atomic, unique, complete record per key and excludes cycles, missing records, duplicate/conflicting keys, larger vocabularies, non-atomic encodings, and lengths beyond the positional budget. The results do not prove arbitrary-depth correctness, minimality, universal FF dispensability, or acquisition. Unary successor recurrence was not added because it would merely reapply the already-exact local FF successor.

## Witness 1: local successor

For one-hot inputs A--D, the legal task is A->B, B->C, C->D. Let `M` contain the desired output one-hot in each source row. The primary block has zero attention, `W1=I`, and `W2=M-I`. ReLU is the identity on the legal one-hot inputs, so the residual update is

```
x + ReLU(x I)(M-I) = x + x(M-I) = xM.
```

The FF-only and SA+FF variants therefore have exact unit margin. SA-only is an identity map and fails every legal successor. This is a local transformation calibration; it does not show that FF is necessary for every possible successor encoding.

## Witness 2: contextual associative lookup

Each contextual pair token stores its left-hand symbol in a three-dimensional key subspace and its right-hand symbol in a separate value/output subspace. The final query token stores its symbol in a query-only subspace. One head maps query identity to Q, pair LHS to K, and pair RHS to V. With alpha 16, three pair tokens, and one final query, the desired scaled score is 16 and the other scores are zero:

```
p(desired) = exp(16) / (exp(16) + 3) = 0.999999662395.
```

The attention is finite and leaky, not hard attention. Nevertheless the desired output logit exceeds every competitor for all six A/B/C bijections and all three queries. FF-only cannot move contextual values to the query site; its 6/18 accuracy is deterministic tie-breaking, with zero logit margin, rather than successful lookup.

## Witness 3: bounded two-hop grandparent

The fixed context describes `R <- A <- B <- C` using edge tokens that contain a child key and parent value. Query tokens cover the entire declared depth-2 domain: B targets R and C targets A. Layer 1 retrieves the queried node's parent into a first-hop state subspace. Layer 2 queries from that state and retrieves the next parent into a distinct output subspace. Edge key/value subspaces remain invariant through the identity residual.

Both finite-softmax layers assign more than 0.99999966 probability to their intended edge, and both legal queries have output margin above 0.99999954. FF-only cannot inspect edge tokens. The one-layer SA control transports the parent but has no grandparent output support, so it fails both cases. Thus the construction witnesses bounded two-hop composition under this encoding. It is not evidence of unbounded closure or a lower bound that every grandparent circuit needs two layers.

## Artifact map and reproducibility

Artifacts live under `results/manual_witnesses/{successor,pair_lookup,grandparent,root_recurrence,unary_chain_recurrence}/{topology}`. Each topology contains embedding, positional, Q/K/V/O, FF, bias, and unembedding CSVs; exhaustive legal-domain results; a canonical example; and traced activations. M1 topology directories additionally contain every exhaustive generation step and canonical per-step trace trees. Each M1 task root also contains tracked reversed and fixed-shuffle serialization controls. Each task root contains `construction.md`. Cross-task files are:

- `manual_witness_summary.csv`: accuracy, margins, and expectation gates;
- `manual_architectures.csv`: L/H/d, controller, data location, residual, and normalization choices;
- `manual_component_necessity.csv`: encoding-relative SA/FF control outcomes;
- `manual_sa_vs_ff.png`: component comparison;
- `manual_two_hop_attention.png`: layerwise parent/grandparent selection.
- `manual_autoregressive_summary.csv`, `manual_autoregressive_architectures.csv`, and `manual_autoregressive_component_necessity.csv`: M1-only gates and architecture metadata;
- `manual_autoregressive_depth.png`: argmax-correct trajectory accuracy by generated transition count, with distinct line and marker styles.

Regenerate with:

```bash
PYTHONPATH=src python -m cl.manual_transformers.run_milestone1
PYTHONPATH=src python -m cl.manual_transformers.run_milestone2
PYTHONPATH=src pytest -q tests/test_manual_transformers_milestone1.py
PYTHONPATH=src pytest -q tests/test_manual_transformers_milestone2.py
```

Later milestones may add structured counters, local tools, and interpreters. Those are not results of Milestones 1--2.
