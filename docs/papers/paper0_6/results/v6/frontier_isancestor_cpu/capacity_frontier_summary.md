# Paper 0.6 v6 measured frontier

This directory is the clean, complete six-model CPU run of the preregistered
`isAncestor` stress sweep. It contains the baseline (`L4/W64/H4`) and depth-8
(`L8/W64/H4`) architectures for seeds 11, 23, and 37, trained for 1,000 steps.
The acquisition and frontier gate is worst-seed accuracy at least 0.80.

| Architecture | D max | d max | b max | N max | Training-regime accuracy | Extrapolation accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 32 | 16 | 8 | 256 | 0.9948 | 0.9888 |
| Depth 8 | 32 | 16 | 8 | 64 | 0.9983 | 0.9800 |

The baseline and depth-8 models pass every tested total-depth, required-position,
and branching value. These values are right-censored by the tested ranges and do
not establish an unbounded frontier. At 256 distractors, the baseline worst-seed
accuracy is 0.9583 and the depth-8 worst-seed accuracy is 0.7917, just below the
gate.

The required-position result is not evidence of iterative ancestor composition.
For positive examples, this encoding exposes the candidate among the path/edge
records; negative candidates are distractors. A direct membership comparison can
therefore succeed without traversing parent links. The v6 result measures robust
finite contextual lookup and shows no advantage from additional model depth.

An earlier MPS attempt in the sibling `frontier_isancestor` directory repeatedly
stalled inside optimization after a checkpoint. It is incomplete and is excluded
from all reported statistics. The present directory was generated independently
on CPU and its manifest records all six completed cells.
