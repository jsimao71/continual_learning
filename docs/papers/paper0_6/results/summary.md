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

## Interpretation and failures

This is a controlled local-model abstraction pilot, not evidence that pretrained Transformers encode a literal taxonomy. Templates, token length, training frequency, and continuation statistics are balanced by construction. Geometry is compared with permuted hierarchy and causal component interventions, but generic layer utility and off-manifold interventions remain limitations. Flat or non-monotonic profiles are retained.

## Next falsifiable question

Do hierarchy geometry and causal component effects survive residualization against natural-corpus n-gram statistics in a pretrained checkpoint series and transfer across unseen paraphrases?
