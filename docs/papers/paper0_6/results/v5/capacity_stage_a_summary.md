# Paper 0.6 v5 Stage-A capacity diagnostic

## Observed 1x results

The official designed subset contains seven architectures, three independently trained seeds, five predicates, and easy/medium/hard diagnostic tiers. It comprises 21 checkpoints, 315 seed-level aggregate cells, and 45,360 raw held-out rows. The competence criterion is a worst-seed accuracy of at least 0.80.

Depth 8 is the only tested allocation that creates any seed-stable competent cells. For `isAncestor`, its three-seed mean/worst-seed accuracies are 0.935/0.889 on easy, 0.928/0.882 on medium, and 0.869/0.759 on hard. It therefore passes easy and medium but not hard. Relative to the L4/W64/H4 baseline, depth 8 raises `isAncestor` mean accuracy by 0.213, 0.211, and 0.193 across the three tiers and raises worst-seed accuracy by 0.389, 0.382, and 0.264.

This improvement is allocation-specific rather than a monotone parameter-count effect in the measured subset. Doubling width to 128 changes `isAncestor` mean accuracy relative to baseline by only +0.049, -0.002, and +0.022; its worst seed remains near chance. At fixed L4/W64 and identical parameter count, two heads collapse `isAncestor` to approximately 0.50, while eight heads improve its mean but retain a hard-tier worst seed of 0.611. The L12/W48 deep/narrow and L4/W96 shallow/wide controls both fail all gates. These controls are not exactly parameter matched (246,048 versus 336,192 parameters), so they support an allocation contrast but cannot by themselves identify a pure depth effect.

The result is predicate-specific. No architecture brings `parent`, `grandparent`, `ancestor_k`, or `root` close to competence. Their best easy-tier three-seed means are 0.330, 0.267, 0.312, and 0.287, respectively, and performance generally declines further on medium and hard tiers. Consequently the observed depth-8 result is finite stabilization of binary contextual membership, not evidence that capacity has produced a general predicate-composition procedure. It does not reopen an `L_min(d)` fit.

## Planned training-budget control (not yet run)

The next run should be limited to `baseline` and `depth8` at 2x training over seeds 11, 23, and 37. This six-checkpoint comparison tests whether the depth-8 advantage persists under a matched optimization control and whether its hard-tier worst seed crosses 0.80. It also distinguishes a generic training rescue of the baseline from a depth-specific competence transition.

Do not run a broad 4x grid. Run 4x for these architectures only if the 2x result either approaches/crosses a gate or materially changes the easy-to-hard diagnostic gap. Stop an arm if all non-`isAncestor` predicates remain far below the gate and `isAncestor` shows no meaningful improvement. If depth 8 becomes hard-tier seed-stable, expand only its independently controlled D/d/b/N frontiers before any new mechanism analysis.

Recommended command:

```bash
PYTHONPATH=src python -u -m cl.experiments.paper06_capacity_v5 \
  --device mps --budget 2 --architectures baseline,depth8 \
  --output docs/papers/paper0_6/results/v5/stage_a_2x_selected
```

## Manuscript integration boundary

The v4 conclusions remain intact for the original 15-cell phase. A v5 results section may add that Stage A found a depth-sensitive `isAncestor` transition on easy and medium diagnostics, while the hard tier and all other predicates remain below the central gate. Until the selected 2x controls run, it must not label the failure capacity-limited or optimization-limited, and it must describe training-budget recovery as planned rather than observed.
