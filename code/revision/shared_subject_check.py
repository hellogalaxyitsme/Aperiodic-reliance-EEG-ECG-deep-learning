#!/usr/bin/env python
"""Sensitivity check: hierarchical bootstrap with one subject draw shared across sampled seeds.

The reported TUAB pooled results (foundation-model evaluator, scheme pooled_fm) resample subjects
independently within each sampled seed (stratified by label); the reported pooled Sleep-EDF/MI values
(Table S9, scheme pooled_seed_subject) do the same without stratification. Here every replicate draws
seeds with replacement and ONE subject sample (TUAB: stratified by label), applied to all sampled seeds,
so subject identity is preserved across seeds and conditions.

Outputs (in <out>):
  shared_vs_reported_<block>.csv  per block: point, CI and p under both schemes (same RNG stream sizes)
  formal_primary_shared.csv, formal_sham_shared.csv  complete families with the TUAB rows replaced by
                                  shared-scheme p values and BH recomputed
  summary.txt
Usage: python shared_subject_check.py <revision root R> <results dir M> <formal dir> <out dir>
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

import aggregate_revision_bootstrap as agg

R, M, FORMAL, OUT = (Path(a) for a in sys.argv[1:5])
OUT.mkdir(parents=True, exist_ok=True)
T = M / "tuab_full_v3_0_1"
NB, BATCH = 10000, 250
COND = ["raw_eeg", "phase_sham", "phase_aperiodic", "phase_flattened"]
log = []


def say(s):
    print(s, flush=True)
    log.append(s)


def load(files, conds):
    fr = []
    for seed, p in files.items():
        f = pd.read_csv(p)
        f["seed"] = int(seed)
        fr.append(f)
    d = pd.concat(fr, ignore_index=True)
    return d[d.test_input.isin(conds)]


def boot_shared(cube, labels, rng, stratified):
    S, U, C, _ = cube.shape
    out = np.empty((NB, C))
    groups = [np.flatnonzero(labels == g) for g in np.unique(labels)] if stratified else None
    for st in range(0, NB, BATCH):
        b = min(BATCH, NB - st)
        sidx = rng.integers(0, S, size=(b, S))
        if stratified:
            uidx = np.concatenate([g[rng.integers(0, len(g), size=(b, len(g)))] for g in groups], axis=1)
        else:
            uidx = rng.integers(0, U, size=(b, U))
        out[st:st + b] = agg.ba_from_counts(cube[sidx[:, :, None], uidx[:, None, :]].sum(axis=(1, 2)))
    return out


def summarise(point, dist):
    d = dist[np.isfinite(dist)]
    return (float(point), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)),
            float(np.mean(d <= 0)), float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))))


def run_block(name, df, conds, baseline, stratified, independent_scheme, compares=()):
    cube, seeds, units, labels = agg.build_cube(df, conds, "subject")
    rng_seed = 42 + zlib.crc32(name.encode()) % 1_000_000
    point = agg.ba_from_counts(cube.sum(axis=(0, 1)))
    shared = boot_shared(cube, labels, np.random.default_rng(rng_seed), stratified)
    _, indep = agg.bootstrap(cube, labels if independent_scheme == "pooled_fm" else None, independent_scheme,
                             NB, BATCH, np.random.default_rng(rng_seed))
    b0 = conds.index(baseline)
    rows = []
    for i, c in enumerate(conds):
        items = [("performance", c, point[i], lambda bt, i=i: bt[:, i])]
        if i != b0:
            items.append(("drop", c, point[b0] - point[i], lambda bt, i=i: bt[:, b0] - bt[:, i]))
        for est, cond, pt, f in items:
            r = {"block": name, "estimate": est, "condition": cond, "n_seeds": len(seeds), "n_units": len(units)}
            for lab, bt in (("independent", indep), ("shared", shared)):
                p, lo, hi, p1, p2 = summarise(pt, f(bt))
                r.update({"point": p, f"{lab}_lo": lo, f"{lab}_hi": hi, f"{lab}_p_one": p1, f"{lab}_p_two": p2})
            rows.append(r)
    for a, bb in compares:
        ia, ib = conds.index(a), conds.index(bb)
        f = lambda bt: (bt[:, b0] - bt[:, ia]) - (bt[:, b0] - bt[:, ib])  # noqa: E731
        pt = (point[b0] - point[ia]) - (point[b0] - point[ib])
        r = {"block": name, "estimate": "drop_difference", "condition": f"{a}-minus-{bb}",
             "n_seeds": len(seeds), "n_units": len(units)}
        for lab, bt in (("independent", indep), ("shared", shared)):
            p, lo, hi, p1, p2 = summarise(pt, f(bt))
            r.update({"point": p, f"{lab}_lo": lo, f"{lab}_hi": hi, f"{lab}_p_one": p1, f"{lab}_p_two": p2})
        rows.append(r)
    return rows, (cube, labels, rng_seed)


def paired_block(name, base_files, aug_files, conds, stratified):
    fr = []
    for arm, files in (("base", base_files), ("aug", aug_files)):
        d = load(files, conds)
        d["test_input"] = arm + "|" + d.test_input
        fr.append(d)
    df = pd.concat(fr, ignore_index=True)
    allc = [f"{arm}|{c}" for arm in ("base", "aug") for c in conds]
    cube, seeds, units, labels = agg.build_cube(df, allc, "subject")
    rng_seed = 42 + zlib.crc32(name.encode()) % 1_000_000
    point = agg.ba_from_counts(cube.sum(axis=(0, 1)))
    shared = boot_shared(cube, labels, np.random.default_rng(rng_seed), stratified)
    _, indep = agg.bootstrap(cube, labels, "pooled_fm", NB, BATCH, np.random.default_rng(rng_seed))
    ix = {c: i for i, c in enumerate(allc)}
    v = lambda arr, arm, c: arr[..., ix[f"{arm}|{c}"]]  # noqa: E731
    rows = []

    def add(est, cond, fn):
        r = {"block": name, "estimate": est, "condition": cond, "n_seeds": len(seeds), "n_units": len(units)}
        for lab, bt in (("independent", indep), ("shared", shared)):
            p, lo, hi, p1, p2 = summarise(fn(point), fn(bt))
            r.update({"point": p, f"{lab}_lo": lo, f"{lab}_hi": hi, f"{lab}_p_one": p1, f"{lab}_p_two": p2})
        rows.append(r)

    for arm in ("base", "aug"):
        add(f"performance_{arm}", "raw_eeg", lambda a, arm=arm: v(a, arm, "raw_eeg"))
        for c in conds[1:]:
            add(f"drop_{arm}", c, lambda a, arm=arm, c=c: v(a, arm, "raw_eeg") - v(a, arm, c))
    add("raw_cost", "raw_eeg", lambda a: v(a, "base", "raw_eeg") - v(a, "aug", "raw_eeg"))
    for c in conds[1:]:
        add("reliance_reduction", c, lambda a, c=c: (v(a, "base", "raw_eeg") - v(a, "base", c))
            - (v(a, "aug", "raw_eeg") - v(a, "aug", c)))
    return rows


all_rows = []
fam_p = {}  # (domain, model) -> (p_primary_shared, p_sham_shared)
NAMES = {"eegnet": "EEGNet", "shallow_fbcsp": "ShallowFBCSPNet", "deep4": "Deep4Net"}

# 1. TUAB conventional, full and matched
for cohort, dom in (("full", "TUAB full"), ("matched", "TUAB full age/sex-matched")):
    for m, nm in NAMES.items():
        files = {s: (T / f"multiseed_neural/{m}_seed{s}/tuab_full_{m}_intervention_predictions.csv" if cohort == "full"
                     else T / f"age_matched/multiseed_neural/{m}_seed{s}/tuab_full_age_matched_{m}_intervention_predictions.csv")
                 for s in (42, 43, 44)}
        rows, _ = run_block(f"tuab_{cohort}_{m}", load(files, COND), COND, "raw_eeg", True, "pooled_fm")
        all_rows += rows
        fam_p[(dom, nm)] = rows

# 2. TUAB foundation models
first = {"biot": "biot_interventions_prest_full", "labram": "labram_interventions_base_full",
         "eegpt": "eegpt_interventions_braindecode_full", "cbramod": "cbramod_interventions_braindecode_full",
         "reve": "reve_base_interventions_full", "eegmamba": "eegmamba_interventions_official_full"}
FMN = {"biot": "BIOT", "labram": "LaBraM", "eegpt": "EEGPT", "cbramod": "CBraMod", "reve": "REVE",
       "eegmamba": "EEGMamba"}
for k, nm in FMN.items():
    files = {0: T / first[k] / f"tuab_{k}_intervention_predictions.csv"}
    files.update({s: T / f"foundation_multiseed/{k}_seed{s}/tuab_{k}_intervention_predictions.csv" for s in (20260526, 20260527)})
    rows, _ = run_block(f"tuab_fm_{k}", load(files, COND), COND, "raw_eeg", True, "pooled_fm")
    all_rows += rows
    fam_p[("TUAB full foundation models", nm)] = rows
files = {s: R / f"results/bendr_fixed/bendr_seed{s}/tuab_bendr_intervention_predictions.csv" for s in (20260525, 20260526, 20260527)}
rows, _ = run_block("tuab_fm_bendr", load(files, COND), COND, "raw_eeg", True, "pooled_fm")
all_rows += rows
fam_p[("TUAB full foundation models", "BENDR")] = rows

# 3. 250-Hz subset
keep = {u.strip() for u in (R / "results/aggregates/tuab_eval_subjects_250hz.txt").read_text().splitlines() if u.strip()}
for m in NAMES:
    files = {s: T / f"multiseed_neural/{m}_seed{s}/tuab_full_{m}_intervention_predictions.csv" for s in (42, 43, 44)}
    d = load(files, COND)
    d = d[d.subject.astype(str).isin(keep)]
    rows, _ = run_block(f"tuab_250hz_{m}", d, COND, "raw_eeg", True, "pooled_fm")
    all_rows += rows

# 4. Estimator sensitivity (Table S5)
C5 = COND + ["lin_aperiodic", "lin_flattened"]
files = {s: R / f"results/tuab_fit_sensitivity/eegnet_seed{s}/predictions.csv" for s in (42, 43, 44)}
rows, _ = run_block("s5_tuab_eegnet", load(files, C5), C5, "raw_eeg", True, "pooled_fm",
                    compares=[("phase_flattened", "lin_flattened"), ("phase_aperiodic", "lin_aperiodic")])
all_rows += rows

# 5. Mitigation, TUAB (Table S8)
G = R / "results/mitigation"
all_rows += paired_block("s8_tuab_eegnet",
                         {s: R / f"results/tuab_fit_sensitivity/eegnet_seed{s}/predictions.csv" for s in (42, 43, 44)},
                         {s: G / f"tuab_eegnet_flataug_p0.5_seed{s}/predictions.csv" for s in (42, 43, 44)}, C5, True)

# 6. Pooled Sleep-EDF / PhysioNet MI (Table S9)
O = R / "results/ba_correction_binary"
for f in sorted(O.glob("pred_*_seed42.csv")):
    tag = f.name[len("pred_"):-len("_seed42.csv")]
    if tag.startswith("sleep_MLP"):
        cs, bl = ["full_log_psd", "aperiodic_spectrum", "flattened_log_psd"], "full_log_psd"
    else:
        cs, bl = COND, "raw_eeg"
    files = {s: O / f"pred_{tag}_seed{s}.csv" for s in (42, 43, 44)}
    rows, _ = run_block(f"s9_{tag}", load(files, cs), cs, bl, False, "pooled_seed_subject")
    all_rows += rows

res = pd.DataFrame(all_rows)
res.to_csv(OUT / "shared_vs_independent_all.csv", index=False)

# CI shifts
drops = res[res.estimate.str.startswith(("drop", "raw_cost", "reliance"))].copy()
drops["max_ci_shift"] = np.maximum((drops.shared_lo - drops.independent_lo).abs(), (drops.shared_hi - drops.independent_hi).abs())
say(f"rows {len(res)}; drop-type rows {len(drops)}; max CI-limit shift {drops.max_ci_shift.max():.4f}; "
    f"median {drops.max_ci_shift.median():.4f}")
perf = res[res.estimate.str.startswith("performance")].copy()
perf["max_ci_shift"] = np.maximum((perf.shared_lo - perf.independent_lo).abs(), (perf.shared_hi - perf.independent_hi).abs())
say(f"performance rows: max CI-limit shift {perf.max_ci_shift.max():.4f}; median {perf.max_ci_shift.median():.4f}")
# CI excludes zero under both schemes?
for _, r in drops.iterrows():
    ind_ex = r.independent_lo > 0 or r.independent_hi < 0
    sh_ex = r.shared_lo > 0 or r.shared_hi < 0
    if ind_ex != sh_ex:
        say(f"CI-EXCLUDES-ZERO CHANGES: {r.block} {r.estimate} {r.condition}: indep [{r.independent_lo:.3f}, "
            f"{r.independent_hi:.3f}] shared [{r.shared_lo:.3f}, {r.shared_hi:.3f}]")


# Formal families with shared-scheme p values for all TUAB rows
def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1.0)
    return out


for fam, cond, pcol in (("primary", "phase_flattened", "shared_p_one"), ("sham", "phase_sham", "shared_p_two")):
    d = pd.read_csv(FORMAL / f"formal_hypothesis_tests_{fam}_ba_corrected.csv")
    old_sig = d.significant_fdr.astype(bool).copy()
    old_pfdr = d.p_fdr.copy()
    n = 0
    for i, row in d.iterrows():
        key = (row.domain, row.model)
        if key in fam_p:
            r = [x for x in fam_p[key] if x["estimate"] == "drop" and x["condition"] == cond][0]
            if abs(r["point"] - row.effect) > 5e-4:
                raise ValueError(f"point mismatch {key}: {r['point']} vs {row.effect}")
            d.loc[i, ["ci_lower", "ci_upper", "p_raw"]] = [r["shared_lo"], r["shared_hi"], r[pcol]]
            n += 1
    d["p_fdr"] = bh(d.p_raw.values)
    d["significant_fdr"] = d.p_fdr < 0.05
    d.to_csv(OUT / f"formal_{fam}_shared.csv", index=False)
    changed = (d.significant_fdr != old_sig)
    say(f"{fam}: TUAB rows replaced {n}; significant {int(d.significant_fdr.sum())}/{len(d)} "
        f"(reported {int(old_sig.sum())}/{len(d)}); decisions changed: {int(changed.sum())}; "
        f"max |p_fdr change| {np.max(np.abs(d.p_fdr - old_pfdr)):.4f}")
    for i in np.flatnonzero(changed):
        say(f"  CHANGED: {d.domain[i]} {d.task[i]} {d.model[i]} p_fdr {old_pfdr[i]:.4f} -> {d.p_fdr[i]:.4f}")
    for i in range(len(d)):
        if abs(d.p_fdr[i] - old_pfdr[i]) >= 5e-4:
            say(f"  p_fdr {d.domain[i]} {d.task[i]} {d.model[i]}: {old_pfdr[i]:.4f} -> {d.p_fdr[i]:.4f}")

(OUT / "summary.txt").write_text("\n".join(log) + "\n")
