# Group 2 — Self-attention transport

The smoke grid covers eight span/window/depth regimes with an exactly verified local causal mask. Four predictive families are trained jointly with a local-pattern control.

Graph-unreachable long-range cells remain near four-way chance: accuracy is 0.188 for span 2/window 1/depth 1, 0.234 for span 4/window 1/depth 2, and 0.234 for span 8/window 2/depth 2. Their final separation ratios are only 0.09–0.15.

Reachable cells succeed when transport is learned: span 2/window 1/depth 2 reaches 0.797 accuracy; span 4/window 1/depth 4, span 8/window 4/depth 2, and span 8/full/depth 2 reach 1.0. Their very large `R` values reflect near-zero within-family denominators and should be read jointly with accuracy rather than as stable effect sizes.

Reachability is not sufficient. Span 8/window 2/depth 4 is graph-reachable but reaches only 0.078 accuracy and `R=0.060`. The simple `depth × window ≥ span` rule therefore specifies a causal cone, not an optimization or capacity guarantee.

The mixed local task remains substantially more robust to window restriction (0.75–0.953 final accuracy across most cells), although the one-layer cell reaches 0.828 rather than ceiling. Nested override is not yet independently identified by this smoke grid; its displayed trajectory is diagnostic and must not be cited as a completed override result.
