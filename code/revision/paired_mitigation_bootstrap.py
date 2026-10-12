#!/usr/bin/env python
"""Paired bootstrap of augmented vs unaugmented training.

Both arms are stacked into one seed x unit cube (conditions prefixed "base|"
and "aug|"); seeds are paired by seed value and every bootstrap replicate
resamples the same seeds and units for both arms (schemes and resampling as in
aggregate_revision_bootstrap.py). Reports, per test condition C:
  performance per arm, drop per arm (raw - C), and
  reliance reduction  = drop_base(C) - drop_aug(C)
  raw cost            = BA_base(raw) - BA_aug(raw)
with percentile CIs and two-sided bootstrap p-values.
"""
from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

import aggregate_revision_bootstrap as agg


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", action="append", required=True, help="seed=path (unaugmented)")
    p.add_argument("--aug", action="append", required=True, help="seed=path (augmented)")
    p.add_argument("--label", required=True)
    p.add_argument("--scheme", required=True, choices=["subject_mean", "pooled_fm", "pooled_ptbxl"])
    p.add_argument("--conditions", required=True)
    p.add_argument("--baseline", default="raw_eeg")
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--output-csv", type=Path, required=True)
    return p.parse_args()


def load(items, arm):
    frames = []
    for it in items:
        seed, path = it.split("=", 1)
        f = pd.read_csv(path)
        f["seed"] = int(seed)
        f["test_input"] = arm + "|" + f["test_input"].astype(str)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def main():
    a = parse_args()
    conds = a.conditions.split(",")
    df = pd.concat([load(a.base, "base"), load(a.aug, "aug")], ignore_index=True)
    allc = [f"{arm}|{c}" for arm in ("base", "aug") for c in conds]
    df = df[df.test_input.isin(allc)]
    cube, seeds, units, labels = agg.build_cube(df, allc, "subject")
    rng = np.random.default_rng(42 + zlib.crc32(a.label.encode()) % 1_000_000)
    point, boot = agg.bootstrap(cube, labels if a.scheme == "pooled_fm" else None, a.scheme,
                                a.n_bootstrap, 250, rng)
    ix = {c: i for i, c in enumerate(allc)}
    rows = []

    def add(est, cond, pt, dist):
        dist = dist[np.isfinite(dist)]
        rows.append({"model": a.label, "estimate": est, "condition": cond, "point": float(pt),
                     "ci_lower": float(np.quantile(dist, 0.025)), "ci_upper": float(np.quantile(dist, 0.975)),
                     "p_two_sided_zero": float(min(1.0, 2 * min(np.mean(dist <= 0), np.mean(dist >= 0)))),
                     "n_seeds": len(seeds), "n_units": len(units), "scheme": a.scheme})

    def val(arr, arm, c):
        return arr[..., ix[f"{arm}|{c}"]]

    for arm in ("base", "aug"):
        for c in conds:
            add(f"performance_{arm}", c, val(point, arm, c), val(boot, arm, c))
            if c != a.baseline:
                add(f"drop_{arm}", c, val(point, arm, a.baseline) - val(point, arm, c),
                    val(boot, arm, a.baseline) - val(boot, arm, c))
    add("raw_cost", a.baseline, val(point, "base", a.baseline) - val(point, "aug", a.baseline),
        val(boot, "base", a.baseline) - val(boot, "aug", a.baseline))
    for c in conds:
        if c == a.baseline:
            continue
        db = lambda arr: val(arr, "base", a.baseline) - val(arr, "base", c)  # noqa: E731
        da = lambda arr: val(arr, "aug", a.baseline) - val(arr, "aug", c)  # noqa: E731
        add("reliance_reduction", c, db(point) - da(point), db(boot) - da(boot))
    out = pd.DataFrame(rows)
    a.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output_csv, index=False)
    show = out.copy()
    for c in ["point", "ci_lower", "ci_upper"]:
        show[c] = show[c].map(lambda v: f"{v:.3f}")
    print(show[["model", "estimate", "condition", "point", "ci_lower", "ci_upper", "p_two_sided_zero",
                "n_seeds", "n_units"]].to_string(index=False))


if __name__ == "__main__":
    main()
