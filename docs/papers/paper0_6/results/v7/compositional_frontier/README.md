# Paper 0.6 cue-free compositional frontier (v7)

This is the official CPU run of the preregistered cue-free prerequisite. It uses
two disjoint, equal-length chains and matches candidate exposure and endpoint roles
across labels. The generator audit records balanced label/order cells, disjoint split
symbols and paths, exact reachability labels, and a 0.500 endpoint-membership baseline.

All nine planned models completed: L2/W64/H4, L4/W64/H4, and L8/W64/H4 with seeds
11, 23, and 37. Training used required paths d<=3; evaluation used
d in {1,2,3,4,6,8,12,16}. The 55,296 raw rows cover both held-out splits, four
templates, two position modes, three topology seeds, and 16 examples per stratum.

Mean test accuracies across all seed--depth cells were 0.496 (L2), 0.502 (L4), and
0.498 (L8). No architecture passed the strict 0.8 three-seed, all-strata criterion
at any depth, including d=1. All measured frontiers are therefore zero and L_min(d)
is ineligible. Under the preregistered stopping rule, no deeper scaling follows.

This is evidence of acquisition failure for the tested architecture and optimization
regime, not an impossibility result. It shows that the positive v6 path-length result
does not survive removal of the direct endpoint-membership cue.

The manifest records CPU execution, configuration and generator hashes. Aggregate
model runtime was 1,599 seconds (26.7 minutes); checkpoints remain locally resumable
but are excluded from version control.
