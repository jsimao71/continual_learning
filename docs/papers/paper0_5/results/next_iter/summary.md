# Paper 0.5 next-iteration results

## Milestone 1 — expanded pretrained families

- The frozen matrix contains 32 externally defined families: four syntax templates crossed with eight semantic domains, with four identities per family.
- Identities 0--1 are reserved for fitting and identities 2--3 for evaluation; identity rows are not counted as independent families.
- At the final selected depth, pinned Qwen3-0.6B achieves 0.688 top-1 accuracy and mean target log-probability -0.873 nats over the evaluation matrix.
- Qwen is strongest on the explicit mapping template (0.938 accuracy, -0.200 nats) and weaker on arrow, alternation, and question templates (0.562--0.625 accuracy).
- Pythia-70M-deduped at step 143,000 has weak behavioral competence on this broadened matrix (0.016 accuracy, -6.188 nats). Its causal-transfer results therefore require competence-stratified reporting rather than pooling with Qwen.

Interpretation: family breadth exposes a material architecture/capability boundary. The broader causal claim must be evaluated per family and conditioned on behavioral competence; the 32-family inventory alone does not establish transfer.

## Milestone 2 — syntax × semantics causal transfer

- Across 32 family units, equivalent replacement is better than fully nonequivalent replacement by 2.552 nats for final Pythia (family bootstrap 95% CI [1.991, 3.070], 90.6% positive families) and 0.763 nats for Qwen (CI [0.623, 0.899], 100% positive).
- Changing syntax while retaining the semantic domain is strongly harmful: the equivalent advantage is 2.186 nats for Pythia (CI [1.656, 2.681]) and 0.624 for Qwen (CI [0.491, 0.764]).
- Changing semantics while retaining syntax has a smaller but positive effect: 0.184 nats for Pythia (CI [0.075, 0.293]) and 0.118 for Qwen (CI [0.040, 0.230]); sign consistency is 68.8% for both models.

Interpretation: both axes contribute, but template/syntax compatibility explains substantially more transfer than semantic-domain compatibility in this matrix. The full equivalence advantage is broad and sign-consistent for Qwen, while Pythia's strong causal contrast must remain qualified by its weak task competence.
