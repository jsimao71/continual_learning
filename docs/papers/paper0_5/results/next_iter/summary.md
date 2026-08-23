# Paper 0.5 next-iteration results

## Milestone 1 — expanded pretrained families

- The frozen matrix contains 32 externally defined families: four syntax templates crossed with eight semantic domains, with four identities per family.
- Identities 0--1 are reserved for fitting and identities 2--3 for evaluation; identity rows are not counted as independent families.
- At the final selected depth, pinned Qwen3-0.6B achieves 0.688 top-1 accuracy and mean target log-probability -0.873 nats over the evaluation matrix.
- Qwen is strongest on the explicit mapping template (0.938 accuracy, -0.200 nats) and weaker on arrow, alternation, and question templates (0.562--0.625 accuracy).
- Pythia-70M-deduped at step 143,000 has weak behavioral competence on this broadened matrix (0.016 accuracy, -6.188 nats). Its causal-transfer results therefore require competence-stratified reporting rather than pooling with Qwen.

Interpretation: family breadth exposes a material architecture/capability boundary. The broader causal claim must be evaluated per family and conditioned on behavioral competence; the 32-family inventory alone does not establish transfer.
