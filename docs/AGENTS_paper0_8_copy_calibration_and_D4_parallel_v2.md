# AGENTS.md — Paper 0.8 Parallel Iteration
## Copy/Induction Calibration + D4 ICL Continuation

### Mission
Run two coordinated workstreams in parallel.

A — Minimal contextual copy/induction calibration:
find the smallest dataset and smallest Transformer that learns genuine contextual copying, then verify that tracing reconstructs the known induction-style circuit at token, head, sublayer, and output-logit level.

B — D4 genuine ICL continuation:
continue the already-running leakage-audited D4 phase sweep and replication. Do not redesign D4 until the current three-seed/model-boundary evidence is complete.

Workstream A is calibration, not the main ICL claim.

# PART I — COPY TASK

## Behavioral criterion
Canonical pattern:
X,Y,...,X -> Y

Require rank(Y|X) >> 1 but rank(Y|D_context,X)=1.

Primary metric:
G_copy(Y)=margin(Y|context,X)-margin(Y|X)

## C0 — Minimal copy
Use:
X Y X -> Y

Examples:
A B A -> B
A C A -> C
B D B -> D
C A C -> A

Across episodes, make P(Y|X) approximately balanced so no fixed X->Y map can solve the task.

## C1 — Copy with distractors
Examples:
A B C D A -> B
E F A C B D A -> C

Sweep:
num_distractor_pairs=[0,1,2,4,8]
query distance=[1,2,4,8,16]
pair order and query position.

## C2 — Multiple candidate keys
Examples:
A B
C D
E F
C -> D

and:
A F
C B
E D
A -> F

This forces key matching plus value retrieval.

## C3-copy — Role-preserving copy
Use many isomorphic symbol families:
A B ... A -> B
M N ... M -> N
u v ... u -> v

Hold literal symbols out where possible.

# PART II — MODEL GRID

## Architecture families
Run both:
- SA-only: attention + residual, no FF
- SA+FF: standard attention + residual + FF + residual

## Layers
L=[1,2,3,4]

Important:
1 vs 2 layers
2 vs 3
2 vs 4

Do not assume deeper is better.

## Residual / embedding width
d_model=[4,8,16,32,64,128]

Record:
embedding dimension
residual dimension
Q/K/V dimension
head dimension
FF hidden dimension

## Heads
H=[1,2,4,8] where head dimension is sensible.

Suggested:
d=8 -> H=[1,2]
d=16 -> H=[1,2,4]
d=32 -> H=[1,2,4]
d=64 -> H=[1,2,4,8]

## FF width
For SA+FF:
d_ff/d_model=[1,2,4]

# PART III — DATASET SIZE / VOCAB GRID

## Vocabulary
V=[2,3,4,6,8,16,32]

Treat V=2 as potentially degenerate.
Primary clean start: V=4.

## Number of distinct mappings
M=[1,2,3,4,8,16,32,fresh-random]

Hold total training examples fixed when comparing M.

## Number of training episodes
E=[4,8,16,32,64,128,256,512,1024]

Once threshold is bracketed, binary/local search.

## Repetitions per mapping
E=M*R
Sweep M and R independently.

## Prompt size
pairs_per_prompt=[1,2,3,4,8]

# PART IV — LEAKAGE / DEGENERACY AUDIT

Verify:
- global P(Y|X) is balanced enough that weights cannot solve query
- target is not position-unique
- query position does not leak target
- pair order does not leak target
- V=2 complement shortcuts are separately labeled
- shuffled pairings destroy association
- context-free target rank remains nontrivial

# PART V — TOKEN-LEVEL HORIZONTAL SA TRACING

## Capture tensors
For every layer/head/token capture:
Q_lh, K_lh, V_lh

Compute:
S_lh = Q_lh K_lh^T / sqrt(d_h)
A_lh = softmax(S_lh)

Save full token-by-token matrices.

## QK matrix structure
Measure:
- diagonal mass
- previous-token band mass
- repeated-key matching mass
- query-to-associated-value mass
- query-to-key mass
- positional-distance profile
- row entropy
- effective rank
- singular values
- Frobenius norm
- row cosine similarity
- role-aligned averages

For canonical induction:
Layer 1: look for previous-token/pair-binding structure.
Layer 2: look for repeated-key/prefix matching.

## Role-aligned attention statistics
Roles:
demo_key
demo_value
distractor_key
distractor_value
query_key

Aggregate E[A(destination_role,source_role)] across episodes.

# PART VI — VALUE TRANSPORT / TARGET SUPPORT

For destination j and source i:
c_lh(i->j)=A_lh[j,i] V_lh[i]

After output projection:
ctilde_lh(i->j)=W_O c_lh(i->j)

Project to target unembedding:
F_lh(i->j;Y)=w_Y^T ctilde_lh(i->j)

Call this target-support flow, not probability mass.

Build source-token -> head/layer -> destination-token -> target-logit graphs.

Compare correct vs shuffled/no-context:
Delta Q, Delta K, Delta V, Delta S, Delta A, Delta F.

For q^T k decompose:
Delta(q^T k)
=(Delta q)^T k0 + q0^T(Delta k) + (Delta q)^T(Delta k)

# PART VII — CAUSAL SA TESTS

## Token key masking
Mask selected source positions as keys.
Measure:
accuracy
target rank
margin
JS

## Value ablation
Keep routing but set selected V_i=0.

## Q/K/V patching
Patch correct-run tensors from control:
Q_correct <- Q_control
K_correct <- K_control
V_correct <- V_control

at selected head/layer/token positions.

## Head ablation
Zero output, mean replace, role-matched replace, wrong-role replace.

Measure target rank, margin damage, top1 flip, JS.

# PART VIII — FF / RESIDUAL TRACING

At every token/layer capture:
r_preSA
Delta r_SA
r_postSA
Delta r_FF
r_postFF

Project each onto:
target unembedding direction
top competitor directions

Compute:
Delta z_Y^(SA,l)=w_Y^T Delta r_SA
Delta z_Y^(FF,l)=w_Y^T Delta r_FF

Also compute margin increments.

## FF neuron-level
For FF:
a=sigma(W1 r)
FF=W2 a

Neuron u target contribution:
C_u(Y)=a_u * w_Y^T W2[:,u]

Rank neurons by:
positive target support
negative competitor support
correct-vs-control delta

Causally zero/patch high-contribution neurons or groups.

# PART IX — OUTPUT HEAD

At every residual boundary:
z_l=W_U r_l

Track:
target rank
target probability
target margin
top-k candidates
competitor margins

Track target trajectory:
pre-context -> L1-SA -> L1-FF -> ... -> final

Determine whether rank-1 arises from positive target support, competitor suppression, or both.

# PART X — MATRIX / SUBSPACE CHARACTERIZATION

Across role-aligned episodes compute:
Q cosine by role
K cosine by role
V cosine by role
within-role vs between-role dispersion
singular spectra
effective dimension
head subspace overlap

Characterize W_Q W_K^T and W_O W_V where useful, but validate with activations and causality.

# PART XI — KNOWN-CIRCUIT CALIBRATION

Instrumentation should recover, if present:

Layer 1:
previous-token/pair-binding transport.

Layer 2:
prefix/repeated-key matching.

Layer 2 OV:
copy associated value into query/output direction.

If the tracing stack cannot recover a plausible known induction circuit on a competent tiny copy model, debug instrumentation before using it on D4.

# PART XII — COPY ACQUISITION BOUNDARY

Find C- and C+ models immediately below/above copy acquisition.

Compare:
QK motifs
attention entropy
target-support flow
OV copy structure
SA/FF contribution
head utility
residual target projection

# PART XIII — AGGRESSIVE MINIMALITY ORDER

Once a stable positive copy regime exists:
1. reduce vocabulary
2. reduce mappings
3. reduce episodes
4. reduce prompt pairs
5. reduce d_model
6. reduce heads
7. reduce layers
8. remove FF
9. remove positional regularity
10. remove separators

At every step require replication.
Use binary/local search after bracketing.

# PART XIV — WORKSTREAM B: D4 ICL

Continue current D0-D4 sweep unchanged until:
- D0-D4 comparison completes
- seed-11 positive cell is replicated or rejected
- minimum competent architecture is bracketed

Current provisional D4 reference:
L=2
d_model=32
H=2
seed=11

Provisional:
correct demos: 100% accuracy, rank 1
no demos: 12.5% accuracy, mean target rank 18
negative controls <=25%
zero critical direct mapping leakage

Treat as provisional.

Pending:
seeds 23 and 37
architecture boundary
context-free rank control
shuffled/wrong/irrelevant controls
D-/D+ pair
layerwise promotion
SA/FF/head tracing
QKV token tracing
causal replacement
minimality deletion

# PART XV — D3 DEFINITION

D3 is the bridge between:
D1/D2: local relational structure exists, but context need not determine the answer
and
D4: every episode defines a fresh mapping, so the answer is necessarily context-dependent.

D3 should introduce the smallest possible pressure to align two relational systems without full fresh-mapping meta-training.

## D3-A — partial cross-domain alignment
Train local orders:
A->B->C->D
1->2->3->4

Add only partial paired examples:
A 1
B 2

Withhold:
C 3
D 4

Test:
A 1
B 2
C ?

Question:
Is a little cross-domain alignment enough to induce extension by relational position?

## D3-B — multiple partial alignments
Use several families:
A B C
1 2 3

D E F
4 5 6

Expose only partial paired positions in each family:
A 1
B 2

D 4
E 5

Withhold final completions.

Question:
Does repeated role-alignment across families induce a reusable relation?

## D3-C — low-entropy mapping variation
Use only M=2,3,4 mappings rather than a fresh D4 mapping every episode.

Example:
M1: A->1 B->2 C->3
M2: A->2 B->3 C->1
M3: A->3 B->1 C->2

Training alternates among this small set.

This may be the cleanest D3 if the key question becomes:
how much mapping inconsistency is required before the model starts using context?

## D3 interpretation
D3 is an intermediate-pressure family, not one fixed dataset.

If D3 succeeds:
D4 contains unnecessary complexity; shrink D3.

If D3 fails but D4 succeeds:
fully dynamic/context-dependent mappings may be the critical training pressure.

If D3 partially succeeds:
use it as the acquisition-boundary regime for mechanistic comparison.

# PART XVI — SHARED INSTRUMENTATION

Implement one reusable tracer for Copy and D4:
full Q/K/V capture
QK matrices
attention probabilities
token/head target-support flow
residual projections
FF neuron contributions
output-head rank/margin
Q/K/V patching
token key/value masking
head ablation
role-aligned averaging

Avoid separate interpretability codepaths.

# REQUIRED OUTPUTS

Copy directory:
docs/papers/paper0_8/results/copy/

Tables:
copy_generator_validation.csv
copy_phase_grid.csv
copy_minimality.csv
copy_qk_structure.csv
copy_attention_role_mass.csv
copy_target_support_flow.csv
copy_sa_ff_contributions.csv
copy_ff_neuron_contributions.csv
copy_head_utility.csv
copy_qkv_patching.csv
copy_acquisition_boundary.csv

Figures:
copy_phase_boundary.png
copy_vocab_vs_model_size.png
copy_qk_heatmaps.png
copy_attention_role_matrix.png
copy_target_support_graph.png
copy_target_rank_vs_depth.png
copy_sa_ff_support_vs_depth.png
copy_head_utility.png
copy_qkv_patch_damage.png
copy_before_after_acquisition.png

# QUESTIONS TO ANSWER

Copy:
1. Smallest non-degenerate vocabulary?
2. Minimum mappings?
3. Minimum episodes?
4. Is 1 layer sufficient behaviorally?
5. Is 2-layer SA sufficient?
6. Does FF lower acquisition threshold?
7. Does FF materially participate after copy is learned?
8. Minimum d_model?
9. Minimum heads?
10. Does canonical previous-token + induction-head structure appear?
11. Can tracing recover it automatically?
12. Which Q/K/V changes are causally necessary?
13. Where does target support propagate horizontally?
14. Positive promotion vs competitor suppression?
15. What changes at C-/C+ threshold?

D4:
16. Does seed-11 replicate?
17. Smallest competent D4 architecture?
18. Does D4 pass all context-dependence controls?
19. Where does target support arise?
20. Does D4 reuse copy/induction circuitry or require a qualitatively different mechanism?

# CLAIM DISCIPLINE

Call Workstream A:
contextual copy / induction

Do not call it general ICL.

Reserve ICL for D4 after held-out episode-specific mapping and negative controls pass.

Do not call attention weight mass flow.
Use:
attention probability
value transport
signed target-support contribution

Do not infer mechanism from QK motifs without causal tests.

# COMPLETION GATE

Workstream A:
- stable copy regime across seeds
- minimal dataset/model boundary bracketed
- SA-only vs SA+FF compared
- L=1..4 tested
- vocabulary and embedding width swept
- full Q/K/V/QK captured
- horizontal target-support flow computed
- FF/residual/output-head contributions measured
- Q/K/V patching and ablations validate circuit
- instrumentation recovers plausible known induction mechanism if present

Workstream B:
- D4 three-seed comparison completes
- architecture boundary bracketed
- leakage controls stay clean
- only then full D4 tracing begins

# Final target
Use contextual copy as a calibrated microscopic circuit:
match key -> transport associated value -> promote copied target.

Then determine whether D4 genuine ICL:
- reuses the same transport circuit plus additional transformation,
- introduces an FF-mediated relation circuit,
- or implements a distinct contextual algorithm.
