"""Constant-prediction sanity check of the seed x subject balanced-accuracy bootstrap.

Runs the unmodified bootstrap() from aggregate_foundation_multiseed_predictions.py on
synthetic prediction vectors built on the real TUAB evaluation structure (253 subjects,
36,945 windows):
  A. all three seeds predict the same class for every window   -> BA must be exactly 0.5
  B. seeds predict different constant classes (0, 1, 0)         -> pooled BA can deviate
  C. per-seed BA averaged instead of pooled (reference only)

Usage: python ba_bootstrap_sanity.py <aggregator dir> <single-seed predictions csv> [n_boot]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1])
import aggregate_foundation_multiseed_predictions as agg  # noqa: E402

pred = pd.read_csv(sys.argv[2])
n_boot = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
base = pred[pred.test_input == "raw_eeg"][["task", "subject", "label", "y_true"]].reset_index(drop=True)


def frame(seed_classes):
    rows = []
    for seed, cls in seed_classes.items():
        for ti in agg.TEST_INPUTS:
            f = base.copy()
            f["test_input"] = ti
            f["seed"] = seed
            f["model"] = "SYNTH"
            # raw input keeps the true labels so the baseline is perfect; edited inputs are constant
            f["y_pred"] = f["y_true"] if ti == "raw_eeg" else cls
            rows.append(f)
    return pd.concat(rows, ignore_index=True)


def run(name, seed_classes):
    _, _, labels, values = agg.build_arrays(frame(seed_classes))
    point, boot = agg.bootstrap(values, labels, "balanced_accuracy", n_boot, 42, 500)
    lo, hi = agg.ci_bounds(boot["sham"], 0.95)
    print(f"{name}: sham BA point {point['sham']:.6f}  95% CI [{lo:.6f}, {hi:.6f}]  "
          f"min {boot['sham'].min():.6f} max {boot['sham'].max():.6f}")


print(f"windows {len(base)}, subjects {base.subject.nunique()}, bootstrap resamples {n_boot}")
run("A  all seeds constant class 0", {1: 0, 2: 0, 3: 0})
run("A' all seeds constant class 1", {1: 1, 2: 1, 3: 1})
run("B  seeds constant 0 / 1 / 0  ", {1: 0, 2: 1, 3: 0})
