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
- Held-out accuracy is 1.4%--15.3%; all four runs fail the 80% competence gate.
- Diagnostic post-block JS-to-category-centroid dispersion falls 0.376 -> 0.328 -> 0.213 bits while entropy rises 0.797 -> 1.224 -> 1.547 bits.
- Diagnostic Fisher-style between/within residual ratio rises 1.668 -> 1.724 -> 1.957, but total within-category variance also rises.

## Interpretation and failures

This is a controlled negative result. Since every run fails held-out completion competence, generator categories remain evaluation metadata: geometry and variance curves cannot be interpreted as learned semantic invariants. The coexistence of attractive separation with failed behavior is the central warning.

## Next falsifiable question

Can a revised training regime cross the frozen competence gate before repeating the same identity-disjoint variance and causal analyses?
