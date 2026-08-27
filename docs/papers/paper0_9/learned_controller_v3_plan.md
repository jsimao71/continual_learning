# Paper 0.9 learned-controller v3 staged plan

This is a plan, not learned-model evidence. It follows the leakage-free shared-64-symbol split (`pair_split_seed=8502`, 20% held-out pairs) used by Paper 0.85 and the completed Paper 0.9 v1 run. The v1 observation is narrow: M3 and M4 both pass exact final answers through the training boundary `K=3`, neither passes `K=4`, `6`, or `8`, and no long-depth M4 advantage is observed. The old 1,620-cell factorial is quarantined because it predates the corrected split and would confound too many resources.

## Invariants and scoring

Every stage uses seeds 11, 23, and 37, held-out implication pairs, 64 evaluation examples per depth and seed, and test depths `1,2,3,4,6,8`. Results are reported by seed and depth. A frontier is the largest contiguous passing prefix, never the maximum isolated passing depth and never an average across train-range and extrapolation depths.

M3 makes exactly one T1 call. Its corrected execution metrics are: one-call coverage (fraction of the required proof completed by that single valid call), selected-edge validity, post-tool answer accuracy, invalid-call rate, and final accuracy. The misleading v1 quantity `valid_call / depth` must not be called transition accuracy. M4 is scored at every attempted transition and separately on exact trajectory, final answer, termination precision/recall/accuracy, invalid calls, nontermination, calls, and model forwards.

All checkpoints store model/optimizer state, every RNG state, the master-stream cursor, and config/dataset hashes. Resume must reproduce an uninterrupted CPU reference byte-for-byte at the metric-row level. Since v1 did not save its data RNG/cursor, Stage A must replay from step zero under the v3 stateless step-indexed stream unless exact reconstruction is demonstrated; merely loading v1 weights and repeating early batches is not an exact 2x continuation.

## Stages

**A — budget rescue.** Re-run the current L2/W64/H4 architecture with exact nested snapshots at 2,000 and 4,000 updates. The 4x trajectory continues the 2x trajectory; it is not a fresh run. This is six training trajectories and 12 snapshot evaluation cells.

**B — depth and allocation.** Compare L4/W64/H4 (capacity-increased), L4/W48/H4 (85,728 parameters), and L1/W88/H4 (81,048 parameters). The latter two are within 10% of the 80,128-parameter baseline, giving a deep/narrow versus shallow/wide allocation contrast. Maximum: 18 cells.

**C — width/head controls.** If B does not give a decisive staged answer, test L2 widths 32 and 96 at four heads, plus L2/W64 with one and eight heads. Head controls have exactly the baseline parameter count; width controls intentionally change capacity. Maximum: 24 cells.

**D — example diversity at fixed `Kmax=3`.** Fix the winning architecture, 1,000 updates, batch 96, and a 96,000-draw schedule. Generate a canonical deduplicated 96k master pool; 24k and 48k pools are exact row-prefixes verified by SHA-256. Smaller pools repeat deterministically to fill the same draw count. The row identities, repetition schedule, and batch order are serialized, making accessible diversity the intended changed variable. Maximum: 18 cells.

**E — training-depth boundary.** Fix architecture, diversity, updates, and draws, then compare `Kmax=3` with `Kmax=4`. The first 72,000 ordered examples are identical K1–3 examples in both conditions. The remaining 24,000 are a preregistered matched control: balanced K1–3 continuation for the K3 condition versus K4 examples for the K4 condition. Evaluate both beyond their own boundary at K4, K6, and K8 and label K4 interpolation for the K4-trained condition, not extrapolation. Maximum: 12 cells.

## Gates and compute

A condition passes only if all three seeds reach 95% final accuracy plus its machine-specific execution gate. Advance a resource branch only if it increases the three-seed contiguous frontier or improves every seed by at least ten points at the first failed depth without damaging train-range competence. Stop after two successive resource levels fail that rescue rule, or after all seeds pass K8.

The ungated ceiling is 78 training trajectories, 84 evaluation cells, and 90,000 optimizer updates—about 15 completed-v1 run equivalents. A provisional serial estimate is 8–14 MPS-hours, widened because deep/wide cells are slower than the baseline; it is not a launch promise. Because C is conditional and D/E use only the selected architecture, expected staged compute should be lower. After A, report observed updates per second and replace this range with architecture-specific wall-clock estimates before launching B.
