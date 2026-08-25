# AGENTS.md — Paper 0.6 Next Iteration
## Predicate Semantics, Deep-Tree Phase Diagrams, and Causal Mechanism

### Mission
Extend Paper 0.6 to answer:

> Does a Transformer learn reusable meanings/algorithms for predicates such as parent, grandparent, ancestor_k, isAncestor, and root, or does it merely learn finite-depth mappings or shallow compositions that happen to solve the training-depth regime?

Combine:
1. train-shallow/test-deep predicate generalization phase diagrams;
2. competence-gated causal replacement, SA/FF ablation, head utility, intermediate-node decoding, and path masking.

Operational criterion for a learned predicate:
The model applies the same relation rule to unseen identities, unseen tree topologies, unseen depths, and unseen path lengths, while causal computations remain organized by relational role rather than token identity.

# PART I — FORMAL PREDICATES

Let P(v) be the unique parent of non-root node v.

parent(x)=P(x)

grandparent(x)=P^2(x)

ancestor_k(x)=P^k(x)

isAncestor(x,y)=1 iff there exists k>=1 such that P^k(x)=y

root(x)=P^{d(x)}(x), where d(x) is distance to root.

# PART II — EVIDENCE LEVELS

A. Training-depth competence only:
solves D<=D_train, fails deeper. Interpret as finite-depth interpolation, not general predicate learning.

B. Fixed-hop generalization:
learns P^k for fixed k and applies it inside much deeper unseen trees.

C. Unseen-hop extrapolation:
train k=[1,2,3], test k=[4,5,6,8,12] using compositional encoding of k.

D. Depth-general root/transitive ancestor:
train shallow, test root/isAncestor on D=[4,5,6,8,12,16,24,32].

# PART III — GENERATOR

Independently vary:
- required path distance d=[1,2,3,4,6,8,12,16,24,32]
- total tree depth D=[2,3,4,6,8,12,16,24,32]
- branching factor b=[1,2,4,8]
- distractor nodes N=[0,4,16,64,256] where feasible
- 4+ surface templates
- aligned/randomized positions
- arbitrary symbolic labels primary; natural labels secondary

Do not conflate d, D, b, and N.

# PART IV — TRAIN-SHALLOW / TEST-DEEP

Primary split:
train D<=3 or 4, required d<=3.
Test D=[2,3,4,5,6,8,12,16,24,32] with disjoint node identities and new topologies.

Predicate-specific:
- parent: d=1 regardless of D
- grandparent: d=2 regardless of D
- ancestor_k: train k 1-3, test unseen k 4+
- isAncestor: balanced positives/negatives including sibling, cousin, descendant, different branch
- root: required d equals node depth

# PART V — MODEL PHASE DIAGRAM

Transformer grid:
L=[2,4,6,8,12,16,24]
d_model=[32,64,128,256] or designed subset
>=3 seeds for central claims
1x/2x/4x training budgets for difficult cells

Primary competence surface:
A(L,d,D,b,N)

Required plots:
- accuracy_vs_required_path_d.png
- accuracy_vs_total_depth_D.png
- accuracy_vs_branching_b.png
- accuracy_vs_distractors_N.png
- phase_L_vs_d_root.png
- phase_L_vs_k_ancestor.png
- phase_L_vs_D_parent.png
- phase_L_vs_b_root.png

# PART VI — SCALING OF REQUIRED MODEL DEPTH

Define L_min(d) as minimum model depth achieving accuracy >=0.80. Also report 0.90.

Compare:
- constant: L_min(d)=c
- logarithmic: a+b log2 d
- linear: a+b d
- sublinear power: a+b d^alpha, 0<alpha<1
- piecewise / threshold failure

Use AIC/BIC, held-out fit, bootstrap CI, seed stability.

Interpretation:
- constant: fixed-depth global strategy
- logarithmic: pointer-jumping/doubling candidate
- linear: iterative relation composition
- threshold failure: finite-depth specialization

Do not infer mechanism from scaling alone.

# PART VII — INTERMEDIATE PREDICTION TRAJECTORIES

For chain x->a1->a2->...->ad=root, decode every path node at each residual boundary.

Record:
- prob(x)
- prob(a1)...prob(root)
- top path node per layer
- target margin

Mechanism signatures:
- sequential hop: layer 1 a1, layer 2 a2, ...
- pointer doubling: preferred hop grows roughly 1,2,4,8,...
- direct global: root becomes dominant early independent of d
- distributed: no intermediate node trajectory, yet final generalization succeeds

# PART VIII — ATTENTION PATH ANALYSIS

At each layer/head aggregate attention mass to:
- parent
- grandparent
- ancestor distance k
- root
- off-path nodes
- sibling branches

Define A_lh(k) as attention mass to nodes at relation distance k.

Test whether preferred k grows with depth, including possible k~2^l.

Attention is descriptive until causal tests agree.

# PART IX — CAUSAL PATH MASKING

Selectively block:
- parent
- intermediate ancestor
- root
- off-path distractors
- sibling branches

Measure:
- accuracy delta
- margin delta
- stable-depth delta
- JS divergence

Key test:
If masking an intermediate ancestor destroys root prediction, chain structure is causally used.
If intermediate nodes can be masked while root remains accurate, a more direct/global strategy is plausible.

Also test relation-distance masks where possible.

# PART X — CAUSAL REPLACEMENT

For target paths, replace layer updates with donors from:
1. same node, different template
2. sibling, same parent
3. different node at same depth
4. same hop role on another branch/tree
5. unrelated predicate/tree
6. random matched-norm donor

Measure:
- target-margin damage
- JS damage
- top1 flip
- stable-depth delay

Important cross-depth test:
Use donors from different absolute tree depths but the same relational stage. If "two hops composed" transfers across trees, that supports role abstraction.

# PART XI — SA/FF CAUSAL ANALYSIS

For competent S1 and S3:
- zero SA update
- zero FF update
- mean replacement
- role-matched donor replacement

Measure accuracy, target margin, stable depth.

Cross-paper hypothesis to test, not assume:
SA may be more context-sensitive transport; FF may show more reusable relation transformation.

Compare SA/FF update similarity and causal substitutability across:
- same predicate/new tree
- same hop stage/different tree
- different predicate
- different required depth

# PART XII — HEAD UTILITY

Per head:
- zero ablation
- mean replacement
- role-matched replacement
- cross-role replacement

Measure utility for:
parent, grandparent, ancestor_k, isAncestor, root.

As path depth grows, determine whether:
- more heads are recruited
- recruitment occurs later
- the same heads are reused iteratively
- different predicates use distinct circuits

Report top-k overlap, Jaccard, utility Spearman, new-head count, last recruitment layer.

# PART XIII — S3 CAUSAL COMPLETION

For:
f1=a1
f2=f1 xor a3
f3=f2 xor a6

Test whether order-2 computation is reused inside order-3.

At every layer, replace order-3 updates with:
- matched order-2 donor
- unrelated same-target donor
- cross-target donor

Measure:
- JS damage
- margin damage
- top1 flip
- stable-depth change

Also run SA/FF ablation and head-utility overlap across orders 1-3.

Determine whether extra order-3 depth comes from:
- more SA integration
- more FF transformation
- new heads
- same computation reused across more layers

# PART XIV — S2 REMAINS BLOCKED

Do not interpret S2 representations mechanistically while held-out competence fails.

Permitted separately:
- curriculum pilot
- explicit intermediate parity supervision
- depth/width/training sweep

# PART XV — OPERATIONAL PREDICATE-LEARNING SCORECARD

For each predicate report:
- identity generalization
- topology generalization
- depth generalization
- hop generalization
- branching generalization
- distractor robustness
- causal role invariance
- mechanism consistency

Only predicates passing most of these should be described as depth-general learned relations.

# PART XVI — DECISIVE OUTCOME CLASSES

Outcome A — Finite mapping:
excellent shallow performance, sharp failure past training depth.
Conclusion: finite-depth interpolation.

Outcome B — Linear composition:
L_min(d) proportional to d plus path-node progression.
Conclusion: learned predicate implemented by iterative relation composition.

Outcome C — Logarithmic/pointer-jumping:
L_min(d) proportional to log d plus doubling-like intermediate/causal evidence.
Conclusion: parallel relation-composition algorithm.

Outcome D — Constant-depth global predicate:
L_min approximately constant over substantial d extrapolation plus causal evidence for direct relational endpoint selection.
Conclusion: strong depth-general predicate implementation.

Outcome E — Mixed:
different regimes by depth/branching/distractors. Report separately.

# PART XVII — STATISTICS

Inferential units:
- tree generator seed
- model seed
- predicate family

Bootstrap over trees/seeds.
Do not treat nodes/examples as independent replicates.

Avoid scaling claims with fewer than 4 competent extrapolation points beyond training range.

# PART XVIII — REQUIRED FIGURES

Generate:
- s1_accuracy_vs_tree_depth.png
- s1_accuracy_vs_required_path.png
- s1_accuracy_vs_branching.png
- s1_accuracy_vs_distractors.png
- s1_phase_model_depth_vs_path_depth.png
- s1_Lmin_scaling_fits.png
- s1_root_intermediate_node_trajectory.png
- s1_attention_preferred_hop_vs_layer.png
- s1_path_masking_effects.png
- s1_cross_depth_replacement_damage.png
- s1_sa_ff_causal_effects.png
- s1_head_utility_by_predicate.png
- s1_head_recruitment_vs_path_depth.png
- s3_nested_replacement_by_layer.png
- s3_sa_ff_by_predictive_order.png
- s3_head_recruitment_by_order.png

# PART XIX — REQUIRED TABLES

Generate:
- s1_predicate_competence.csv
- s1_depth_extrapolation.csv
- s1_Lmin_scaling_fits.csv
- s1_intermediate_node_decoding.csv
- s1_path_masking.csv
- s1_causal_replacement.csv
- s1_sa_ff_ablation.csv
- s1_head_utility.csv
- s1_predicate_scorecard.csv
- s3_causal_replacement.csv
- s3_sa_ff_ablation.csv
- s3_head_utility.csv

# PART XX — REQUIRED SUMMARIES

Create:
- docs/papers/paper0_6/results/v4/s1_deep_predicates_summary.md
- docs/papers/paper0_6/results/v4/s1_causal_summary.md
- docs/papers/paper0_6/results/v4/s3_causal_summary.md
- docs/papers/paper0_6/results/v4/predicate_mechanism_classification.md

# PART XXI — QUESTIONS THAT MUST BE ANSWERED

S1:
1. Does parent generalize to deeper unseen trees?
2. Does grandparent generalize independently of total tree depth?
3. Does unseen k-ancestor extrapolate beyond trained k?
4. Does isAncestor generalize to longer positive paths and hard negatives?
5. Does root generalize beyond training depth?
6. What is L_min(d)?
7. Is constant/log/linear/sublinear/threshold the best description?
8. Does branching independently change required model depth?
9. Does distractor count increase stable decision depth?
10. Do intermediate predictions trace actual ancestors?
11. Is there evidence of pointer jumping?
12. Does masking an intermediate path node break root prediction?
13. Does relational-role matched causal replacement transfer across trees?
14. Are SA and FF roles different?
15. Are the same heads reused at greater path depth?
16. Which predicates pass the learned-relation scorecard?

S3:
17. Is order-2 computation causally reused in order-3?
18. Does extra order-3 depth come from SA, FF, heads, or repeated reuse?
19. Does causal structure support the claim that predictive semantic order consumes iterative transformation depth?

# PART XXII — PAPER INTEGRATION

If successful, organize Paper 0.6 around:
1. competence-first S1/S3 and S2 failure
2. shallow-train/deep-test predicate extrapolation
3. mechanism class: finite vs linear vs logarithmic vs constant-depth vs mixed
4. causal relation reuse in S1 and compositional reuse in S3
5. SA/FF/head mechanism

Keep legacy six-tree and pretrained work in appendices.

# PART XXIII — CLAIM DISCIPLINE

Do not write "the Transformer learned the concept of ancestor" unless depth/hop/topology generalization and causal role invariance support it.

Prefer:
"The model learned a depth-general implementation of the ancestor relation over the tested range."

Do not call logarithmic scaling unless fit plus causal/intermediate evidence support pointer jumping.

Do not infer one-hop-per-layer from final accuracy alone.

# COMPLETION GATE

Do not declare complete until:
1. train-shallow/test-deep S1 extrapolation runs
2. parent/grandparent/ancestor_k/isAncestor/root separately evaluated
3. model-depth × required-path phase diagram exists
4. branching and distractor controls exist
5. >=3 seeds for central claims
6. competing L_min fits compared
7. intermediate path-node trajectories measured
8. path masking performed
9. S1 causal replacement complete
10. S1 SA/FF ablation complete
11. S1 head-utility complete
12. S3 causal replacement complete
13. S3 SA/FF/head analysis complete
14. S2 remains competence-gated
15. all claims backed by tracked raw artifacts
16. tests pass
17. PDFs build without warnings

# Final Scientific Target

Distinguish:
finite-depth lookup
vs linear relational composition
vs logarithmic pointer jumping
vs constant-depth global predicate execution.

Strong positive outcome:
A Transformer trained only on shallow trees applies root/ancestor relations to substantially deeper unseen trees, with causal and intermediate dynamics revealing a reusable relation-composition algorithm whose required network depth follows a measurable law in path depth rather than a memorized training-depth boundary.

Strong negative outcome:
If competence collapses immediately beyond trained path lengths, shallow S1 success reflects finite-depth functional interpolation rather than a depth-general learned predicate.
