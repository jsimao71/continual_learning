# Predictive-order phase diagram

The compute-feasible grid trained 14 models over depths [2, 4, 8, 12, 16, 24] and widths [32, 64], with 2x/4x training at selected difficult cells. All generators passed the no-shortcut information audit.

Best held-out accuracy by predictive order was {'1': 1.0, '2': 1.0, '3': 0.9722222222222222, '4': 0.5069444444444444, '6': 0.3871527777777778, '8': 0.390625}. Orders reaching the primary 0.80 threshold were [1, 2, 3]; unresolved orders were [4, 6, 8]. Minimum-depth and minimum-width tables leave a cell blank when competence was never reached rather than treating failure as a large depth estimate.

The interaction fits are descriptive competence-surface comparisons, not universal scaling laws. Internal first/stable decision depths include only architecture/order strata whose aggregate accuracy clears 0.80.
