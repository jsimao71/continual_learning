# Paper 0.6 controlled experiment summary

## What was run

- E1--E8 over two model settings x 2 seeds.
- Balanced noun, action, and relation hierarchies; natural labels, arbitrary synthetic labels, and permuted controls.
- Geometry aggregate rows: 108; component aggregate rows: 72.

## Main pilot observations

- Mean post-block normalized hierarchy separation: 0.1100.
- Mean post-block hierarchy RSA: 0.1941.
- Tree-neighbor recovery versus permuted control: 0.4774 versus 0.2535.
- Mean cross-template cosine: 0.9597.
- Mean semantic motif specificity: -0.0121.
- Final training loss range: 0.4574--0.5295.
- Held-out completion accuracy is 1.4%--15.3% across runs; all four fail the preregistered 80% competence gate.
- Same-parent sibling versus cross-category output JS divergence is 0.660 versus 0.746 bits.
- Layer-0 FFN sibling versus cross-category replacement changes output distributions by 0.241 versus 0.284 JS bits; the contrast is diagnostic only.
- Parent-query entropy rises from 0.629 bits pre-SA to 1.107 post-block; root-query entropy rises from 0.596 to 1.139.

## Interpretation and failures

This is a controlled negative result. Because every run fails the held-out completion gate, geometry, entropy, and replacement measurements cannot be interpreted as semantic invariants. Above-permutation geometry in an incompetent model is itself a warning against representation-first claims. No Paper 1 learning feature is enabled by this run.

## Next falsifiable question

Do hierarchy geometry and causal component effects survive residualization against natural-corpus n-gram statistics in a pretrained checkpoint series and transfer across unseen paraphrases?
