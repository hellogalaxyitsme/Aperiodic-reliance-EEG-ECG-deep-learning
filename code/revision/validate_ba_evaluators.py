#!/usr/bin/env python
"""Independent validation of the balanced-accuracy evaluators used for the revision.

1. Point estimates: pooled BA recomputed with sklearn.metrics.balanced_accuracy_score on the
   concatenated predictions must equal both evaluators' point estimates.
2. Bootstrap: a deliberately simple loop implementation of the seed x subject (label-stratified)
   bootstrap, with its own random generator, must give 95% CIs that agree with the evaluators'
   within Monte-Carlo error.
3. Synthetic tests on every scheme of aggregate_revision_bootstrap.py and on the original
   foundation-model evaluator: constant predictions -> BA exactly 0.5 in every resample;
   perfect predictions -> exactly 1.0; seeds constant on opposite classes -> point 0.5.

Usage: python validate_ba_evaluators.py <results dir> <ba_correction dir> <code/scripts dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

M, A, S = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(S))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregate_foundation_multiseed_predictions as fmagg  # noqa: E402
import aggregate_revision_bootstrap as revagg  # noqa: E402

T = M / "tuab_full_v3_0_1"
FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""), flush=True)
    if not ok:
        FAIL.append(name)


def load(cohort, m):
    fr = []
    for s in (42, 43, 44):
        f = (T / f"multiseed_neural/{m}_seed{s}/tuab_full_{m}_intervention_predictions.csv" if cohort == "full" else
             T / f"age_matched/multiseed_neural/{m}_seed{s}/tuab_full_age_matched_{m}_intervention_predictions.csv")
        d = pd.read_csv(f, usecols=["test_input", "subject", "label", "y_true", "y_pred"])
        d["seed"] = s
        fr.append(d)
    return pd.concat(fr, ignore_index=True)


# ---- 1. point estimates vs sklearn ----
data = {}
for cohort in ("full", "matched"):
    fm = pd.read_csv(A / f"tuab_{cohort}_neural_pooled_fm_evaluator.csv")
    fm = fm[fm.metric == "balanced_accuracy"]
    for m in ("eegnet", "shallow_fbcsp", "deep4"):
        d = load(cohort, m)
        data[(cohort, m)] = d
        ind = pd.read_csv(A / f"tuab_{cohort}_{m}_pooled_independent.csv")
        for cond, est in [("raw_eeg", "baseline"), ("phase_sham", "sham"), ("phase_aperiodic", "aperiodic"),
                          ("phase_flattened", "flattened")]:
            e = d[d.test_input == cond]
            sk = balanced_accuracy_score(e.y_true, e.y_pred)
            p_fm = fm[(fm.model == m) & (fm.estimate == est)].point.iloc[0]
            p_in = ind[(ind.estimate == "performance") & (ind.condition == cond)].point.iloc[0]
            check(f"point {cohort} {m} {cond}", abs(sk - p_fm) < 1e-12 and abs(sk - p_in) < 1e-12,
                  f"sklearn {sk:.6f} fm-evaluator {p_fm:.6f} independent {p_in:.6f}")

r = pd.read_csv(A / "ptbxl_psd_ridge_pooled.csv")
d = pd.read_csv(A / "ptbxl_psd_ridge_full_trained_predictions.csv")
for cond in ("full_log_psd", "aperiodic_spectrum", "flattened_log_psd"):
    e = d[d.test_input == cond]
    sk = balanced_accuracy_score(e.y_true, e.y_pred)
    pt = r[(r.estimate == "performance") & (r.condition == cond)].point.iloc[0]
    check(f"point PTB-XL ridge {cond}", abs(sk - pt) < 1e-12, f"sklearn {sk:.6f} evaluator {pt:.6f}")


# ---- 2. simple loop bootstrap (own RNG) vs evaluators, drop of flattening ----
def simple_bootstrap(d, n_boot=2000, seed=12345):
    rng = np.random.default_rng(seed)
    seeds = sorted(d.seed.unique())
    by = {}
    for (s, subj), g in d.groupby(["seed", "subject"]):
        by[(s, subj)] = {c: (gg.y_true.to_numpy(), gg.y_pred.to_numpy()) for c, gg in g.groupby("test_input")}
    lab = d.groupby("subject").label.first()
    groups = [lab.index[lab == v].to_numpy() for v in sorted(lab.unique())]
    out = []
    for _ in range(n_boot):
        ys = {"raw_eeg": ([], []), "phase_flattened": ([], [])}
        for s in rng.choice(seeds, size=len(seeds), replace=True):
            for g in groups:
                for subj in rng.choice(g, size=len(g), replace=True):
                    for c in ys:
                        yt, yp = by[(s, subj)][c]
                        ys[c][0].append(yt)
                        ys[c][1].append(yp)
        ba = {c: balanced_accuracy_score(np.concatenate(v[0]), np.concatenate(v[1])) for c, v in ys.items()}
        out.append(ba["raw_eeg"] - ba["phase_flattened"])
    return np.quantile(out, [0.025, 0.975])


for cohort, m in [("full", "eegnet"), ("matched", "deep4")]:
    lo, hi = simple_bootstrap(data[(cohort, m)])
    fm = pd.read_csv(A / f"tuab_{cohort}_neural_pooled_fm_evaluator.csv")
    row = fm[(fm.model == m) & (fm.metric == "balanced_accuracy") & (fm.estimate == "drop_flattened")].iloc[0]
    ok = abs(lo - row.ci_lower) < 0.006 and abs(hi - row.ci_upper) < 0.006
    check(f"bootstrap CI {cohort} {m} drop_flattened", ok,
          f"simple loop [{lo:.3f}, {hi:.3f}] vs evaluator [{row.ci_lower:.3f}, {row.ci_upper:.3f}]")

# ---- 3. synthetic tests on every scheme ----
base = data[("full", "eegnet")]
base = base[(base.seed == 42) & (base.test_input == "raw_eeg")][["subject", "label", "y_true"]].reset_index(drop=True)


def synth(pred_by_seed):
    fr = []
    for s, rule in pred_by_seed.items():
        for c in ("raw_eeg", "phase_flattened"):
            f = base.copy()
            f["seed"], f["test_input"] = s, c
            f["y_pred"] = f.y_true if rule == "perfect" else rule
            fr.append(f)
    return pd.concat(fr, ignore_index=True)


for scheme in ("subject_mean", "pooled_fm", "pooled_ptbxl"):
    for label, rule, expect in [("constant 0", {1: 0, 2: 0, 3: 0}, 0.5), ("constant 1", {1: 1, 2: 1, 3: 1}, 0.5),
                                ("perfect", {1: "perfect", 2: "perfect", 3: "perfect"}, 1.0)]:
        cube, _, _, labels = revagg.build_cube(synth(rule), ["raw_eeg", "phase_flattened"], "subject")
        point, boot = revagg.bootstrap(cube, labels if scheme == "pooled_fm" else None, scheme, 500, 250,
                                       np.random.default_rng(0))
        if scheme == "subject_mean" and label.startswith("constant"):
            # subject-averaged: a constant classifier scores 1 on one class's subjects and 0 on the other's,
            # so its value is the subject share of that class, not 0.5 (the quantity replaced by pooled BA)
            share = (base.groupby("subject").y_true.first() == int(label[-1])).mean()
            check(f"synthetic {scheme} {label} equals class share (not BA)", abs(point[0] - share) < 1e-12,
                  f"value {point[0]:.4f} = share of subjects in class {label[-1]} {share:.4f}")
            continue
        ok = np.allclose(boot[:, 0], expect) and abs(point[0] - expect) < 1e-12
        check(f"synthetic {scheme} {label}", ok, f"point {point[0]:.4f}, resamples {boot[:, 0].min():.4f}-{boot[:, 0].max():.4f}")
    cube, _, _, labels = revagg.build_cube(synth({1: 0, 2: 1, 3: 0}), ["raw_eeg", "phase_flattened"], "subject")
    point, boot = revagg.bootstrap(cube, labels if scheme == "pooled_fm" else None, scheme, 500, 250,
                                   np.random.default_rng(0))
    if scheme != "subject_mean":
        check(f"synthetic {scheme} opposite-constant seeds point", abs(point[0] - 0.5) < 1e-12, f"point {point[0]:.4f}")

print("\nVALIDATION", "FAILED: " + ", ".join(FAIL) if FAIL else "ALL PASSED")
