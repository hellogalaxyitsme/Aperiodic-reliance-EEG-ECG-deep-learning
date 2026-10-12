#!/usr/bin/env python
"""Hierarchical seed x unit bootstrap for revision experiments with arbitrary conditions.

Generalises the three aggregation schemes of the original analyses so that new
conditions (alternative envelope estimators) are summarised in exactly the same way:

  --scheme subject_mean   mean of per-subject balanced accuracy over seed x subject
                          (aggregate_multiseed_subject_bootstrap.py; TUAB/Sleep-EDF
                          conventional neural models)
  --scheme pooled_fm      pooled confusion; per sampled seed, subjects resampled
                          independently and stratified by label
                          (aggregate_foundation_multiseed_predictions.py)
  --scheme pooled_ptbxl   pooled confusion; one unit resample per replicate shared
                          across sampled seeds (aggregate_ptbxl_prediction_bootstrap.py)
  --scheme pooled_seed_subject
                          pooled confusion; per sampled seed, units resampled
                          independently without stratification (datasets whose
                          subjects contain both classes: Sleep-EDF, PhysioNet MI)

For each condition the output has: performance, drop vs. baseline (one-sided
bootstrap p for drop > 0) and, for each --compare A,B pair, the paired
difference drop_A - drop_B (two-sided p).
"""
from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", action="append", required=True, help="model:seed=path/to/predictions.csv")
    p.add_argument("--scheme", required=True,
                   choices=["subject_mean", "pooled_fm", "pooled_ptbxl", "pooled_seed_subject"])
    p.add_argument("--unit-column", default="subject")
    p.add_argument("--baseline", default="raw_eeg")
    p.add_argument("--conditions", default=None, help="Comma list; default = all present")
    p.add_argument("--compare", action="append", default=[], help="condA,condB -> drop_A - drop_B")
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--batch", type=int, default=250)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include-units", type=Path, default=None, help="Text file, one unit id per line")
    p.add_argument("--output-csv", type=Path, required=True)
    return p.parse_args()


def ba_from_counts(c):
    """c[..., 4] = tn, fp, fn, tp. Balanced accuracy over classes present."""
    tn, fp, fn, tp = (c[..., i].astype(float) for i in range(4))
    n0, n1 = tn + fp, fn + tp
    r0 = np.where(n0 > 0, tn / np.maximum(n0, 1), np.nan)
    r1 = np.where(n1 > 0, tp / np.maximum(n1, 1), np.nan)
    return np.nanmean(np.stack([r0, r1]), axis=0)


def build_cube(df, conditions, unit_col):
    seeds = np.array(sorted(df["seed"].unique()))
    units = np.array(sorted(df[unit_col].astype(str).unique()))
    s_ix = {s: i for i, s in enumerate(seeds)}
    u_ix = {u: i for i, u in enumerate(units)}
    c_ix = {c: i for i, c in enumerate(conditions)}
    cube = np.zeros((len(seeds), len(units), len(conditions), 4), dtype=np.int64)
    present = np.zeros((len(seeds), len(units), len(conditions)), dtype=bool)
    si = df["seed"].map(s_ix).to_numpy()
    ui = df[unit_col].astype(str).map(u_ix).to_numpy()
    ci = df["test_input"].map(c_ix).to_numpy()
    k = 2 * df["y_true"].to_numpy(dtype=int) + df["y_pred"].to_numpy(dtype=int)
    np.add.at(cube, (si, ui, ci, k), 1)
    present[si, ui, ci] = True
    complete = present.all(axis=(0, 2))
    labels = None
    if "label" in df.columns:
        lab = df.groupby(df[unit_col].astype(str))["label"].first()
        labels = lab.reindex(units).astype(str).to_numpy()
    elif "y_true" in df.columns:
        lab = df.groupby(df[unit_col].astype(str))["y_true"].first()
        labels = lab.reindex(units).astype(str).to_numpy()
    return cube[:, complete], seeds, units[complete], (labels[complete] if labels is not None else None)


def bootstrap(cube, labels, scheme, n_boot, batch, rng):
    S, U, C, _ = cube.shape
    out = np.empty((n_boot, C))
    if scheme == "subject_mean":
        vals = ba_from_counts(cube)  # (S, U, C)
        point = np.nanmean(vals, axis=(0, 1))
    else:
        point = ba_from_counts(cube.sum(axis=(0, 1)))
    groups = [np.flatnonzero(labels == g) for g in np.unique(labels)] if labels is not None else None
    for start in range(0, n_boot, batch):
        b = min(batch, n_boot - start)
        sidx = rng.integers(0, S, size=(b, S))
        if scheme == "subject_mean":
            uidx = rng.integers(0, U, size=(b, U))
            samp = vals[sidx[:, :, None], uidx[:, None, :], :]
            out[start:start + b] = np.nanmean(samp, axis=(1, 2))
        elif scheme == "pooled_ptbxl":
            uidx = rng.integers(0, U, size=(b, U))
            samp = cube[sidx[:, :, None], uidx[:, None, :]].sum(axis=(1, 2))
            out[start:start + b] = ba_from_counts(samp)
        elif scheme == "pooled_seed_subject":
            uidx = rng.integers(0, U, size=(b, S, U))
            samp = cube[sidx[:, :, None], uidx].sum(axis=(1, 2))
            out[start:start + b] = ba_from_counts(samp)
        else:  # pooled_fm: per-seed, label-stratified unit resampling
            parts = [g[rng.integers(0, len(g), size=(b, S, len(g)))] for g in groups]
            uidx = np.concatenate(parts, axis=2)
            samp = cube[sidx[:, :, None], uidx].sum(axis=(1, 2))
            out[start:start + b] = ba_from_counts(samp)
    return point, out


def main():
    args = parse_args()
    frames = []
    for item in args.input:
        spec, path = item.split("=", 1)
        model, seed = spec.rsplit(":", 1)
        f = pd.read_csv(path)
        f["model"] = model
        f["seed"] = int(seed)
        frames.append(f)
    data = pd.concat(frames, ignore_index=True)
    conditions = args.conditions.split(",") if args.conditions else list(dict.fromkeys(data["test_input"]))
    if args.baseline not in conditions:
        raise ValueError("baseline missing")
    data = data[data["test_input"].isin(conditions)]
    if args.include_units is not None:
        keep = {u.strip() for u in args.include_units.read_text().splitlines() if u.strip()}
        data = data[data[args.unit_column].astype(str).isin(keep)]
    alpha = 1 - args.ci
    rows = []
    for model, df in data.groupby("model", sort=False):
        cube, seeds, units, labels = build_cube(df, conditions, args.unit_column)
        rng = np.random.default_rng(args.seed + zlib.crc32(model.encode()) % 1_000_000)
        point, boot = bootstrap(cube, labels if args.scheme == "pooled_fm" else None, args.scheme,
                                args.n_bootstrap, args.batch, rng)
        b0 = conditions.index(args.baseline)

        def add(estimate, cond, pt, dist, p_one=None, p_two=None):
            dist = dist[np.isfinite(dist)]
            rows.append({"model": model, "estimate": estimate, "condition": cond, "point": float(pt),
                         "ci_lower": float(np.quantile(dist, alpha / 2)),
                         "ci_upper": float(np.quantile(dist, 1 - alpha / 2)),
                         "p_one_sided_positive": p_one, "p_two_sided_zero": p_two,
                         "n_seeds": len(seeds), "n_units": len(units), "n_bootstrap": args.n_bootstrap,
                         "scheme": args.scheme})

        for i, c in enumerate(conditions):
            add("performance", c, point[i], boot[:, i])
            if i != b0:
                d = boot[:, b0] - boot[:, i]
                add("drop", c, point[b0] - point[i], d,
                    p_one=float(np.mean(d <= 0)),
                    p_two=float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))))
        for pair in args.compare:
            a, bb = pair.split(",")
            ia, ib = conditions.index(a), conditions.index(bb)
            d = (boot[:, b0] - boot[:, ia]) - (boot[:, b0] - boot[:, ib])
            add("drop_difference", f"{a}-minus-{bb}", (point[b0] - point[ia]) - (point[b0] - point[ib]), d,
                p_two=float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))))
    out = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    show = out.copy()
    for c in ["point", "ci_lower", "ci_upper"]:
        show[c] = show[c].map(lambda v: f"{v:.3f}")
    print(show[["model", "estimate", "condition", "point", "ci_lower", "ci_upper",
                "p_one_sided_positive", "p_two_sided_zero", "n_seeds", "n_units"]].to_string(index=False))


if __name__ == "__main__":
    main()
