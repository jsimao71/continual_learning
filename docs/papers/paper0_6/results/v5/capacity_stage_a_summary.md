# Paper 0.6 v5 Stage-A capacity diagnostic

## Observed 1x results

The official designed subset contains seven architectures, three independently trained seeds, five predicates, and easy/medium/hard diagnostic tiers. It comprises 21 checkpoints, 315 seed-level aggregate cells, and 45,360 raw held-out rows. The competence criterion is a worst-seed accuracy of at least 0.80.

Depth 8 is the only tested allocation that creates any seed-stable competent cells. For `isAncestor`, its three-seed mean/worst-seed accuracies are 0.935/0.889 on easy, 0.928/0.882 on medium, and 0.869/0.759 on hard. It therefore passes easy and medium but not hard. Relative to the L4/W64/H4 baseline, depth 8 raises `isAncestor` mean accuracy by 0.213, 0.211, and 0.193 across the three tiers and raises worst-seed accuracy by 0.389, 0.382, and 0.264.

This improvement is allocation-specific rather than a monotone parameter-count effect in the measured subset. Doubling width to 128 changes `isAncestor` mean accuracy relative to baseline by only +0.049, -0.002, and +0.022; its worst seed remains near chance. At fixed L4/W64 and identical parameter count, two heads collapse `isAncestor` to approximately 0.50, while eight heads improve its mean but retain a hard-tier worst seed of 0.611. The L12/W48 deep/narrow and L4/W96 shallow/wide controls both fail all gates. These controls are not exactly parameter matched (246,048 versus 336,192 parameters), so they support an allocation contrast but cannot by themselves identify a pure depth effect.

The result is predicate-specific. No architecture brings `parent`, `grandparent`, `ancestor_k`, or `root` close to competence. Their best easy-tier three-seed means are 0.330, 0.267, 0.312, and 0.287, respectively, and performance generally declines further on medium and hard tiers. Consequently the observed depth-8 result is finite stabilization of binary contextual membership, not evidence that capacity has produced a general predicate-composition procedure. It does not reopen an `L_min(d)` fit.

## Complete 2x training-budget control

All six selected 2x checkpoints are observed. The baseline's `isAncestor` mean/worst-seed accuracies are 0.944/0.833, 0.935/0.840, and 0.900/0.725 on easy, medium, and hard. Training alone therefore rescues the baseline on easy and medium, while the hard tier remains below the seed-stable gate. Its easy-minus-hard mean gap is 0.045, essentially unchanged from 0.046 at 1x. This is direct evidence that optimization budget moves finite-regime competence and weakens a depth-only account.

Depth 8 at 2x is complete over seeds 11, 23, and 37. `isAncestor` reaches mean/worst-seed accuracies of 1.000/1.000, 0.998/0.993, and 0.997/0.995; its mean easy-minus-hard gap is 0.003. It therefore passes the 0.80 worst-seed gate on every tested tier. Together with the baseline rescue, this supports a capacity--optimization interaction in the selected finite diagnostic, not a depth-only scaling law.

No broad 4x grid is justified. The next informative experiment is an independently controlled D/d/b/N frontier before mechanism analysis or an `L_min(d)` fit.

Completion command:

```bash
PYTHONPATH=src python -u -m cl.experiments.paper06_capacity_v5 \
  --device mps --budget 2 --architectures depth8 --seeds 37 \
  --output docs/papers/paper0_6/results/v5/stage_a_2x_selected
```

## Manuscript integration boundary

The v4 conclusions remain intact for the original 15-cell phase. The v5 results add that 1x Stage A found a depth-sensitive `isAncestor` transition on easy and medium diagnostics, the complete baseline 2x rescue proves that training budget also moves this boundary, and depth 8 at 2x passes the selected hard tier across three seeds. The result is seed-stable for these diagnostic cells, but it is not a complete capacity frontier and must not be called a depth-only transition or general predicate closure.
