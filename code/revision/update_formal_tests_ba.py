#!/usr/bin/env python
"""Replace the six TUAB conventional-model rows (EEGNet, ShallowFBCSPNet, Deep4Net; full and
age/sex-matched) of both formal test families with pooled balanced-accuracy results from the
foundation-model evaluator, then recompute Benjamini-Hochberg FDR (q = 0.05) over each complete
family (31 primary flattening tests; 28 sham tests). Input families already contain the
corrected BENDR rows (update_formal_tests_bendr.py).

Usage: python update_formal_tests_ba.py <aggregates dir> <ba_correction dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AGG, BC = Path(sys.argv[1]), Path(sys.argv[2])
MODEL = {"EEGNet": "eegnet", "ShallowFBCSPNet": "shallow_fbcsp", "Deep4Net": "deep4"}
COHORT = {"TUAB full": "full", "TUAB full age/sex-matched": "matched"}
src = {c: pd.read_csv(BC / f"tuab_{c}_neural_pooled_fm_evaluator.csv") for c in ("full", "matched")}


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def disp(p):
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


changes = []
for fam, est, pcol in [("primary", "drop_flattened", "p_one_sided_positive"), ("sham", "drop_sham", "p_two_sided_zero")]:
    d = pd.read_csv(AGG / f"formal_hypothesis_tests_{fam}_revised.csv")
    old = d.copy()
    n_replaced = 0
    for i, row in d.iterrows():
        if row.domain in COHORT and row.model in MODEL:
            s = src[COHORT[row.domain]]
            r = s[(s.model == MODEL[row.model]) & (s.metric == "balanced_accuracy") & (s.estimate == est)]
            if len(r) != 1:
                raise ValueError((fam, row.domain, row.model))
            r = r.iloc[0]
            d.loc[i, ["effect", "ci_lower", "ci_upper", "p_raw"]] = [r.point, r.ci_lower, r.ci_upper, r[pcol]]
            d.loc[i, "source_file"] = f"revision: pooled balanced accuracy, tuab_{COHORT[row.domain]}_neural_pooled_fm_evaluator.csv"
            d.loc[i, "note"] = "pooled balanced accuracy (original value was subject-averaged accuracy)"
            n_replaced += 1
    if n_replaced != 6:
        raise ValueError(f"{fam}: replaced {n_replaced} rows, expected 6")
    d["p_fdr"] = bh(d.p_raw.values)
    d["significant_fdr"] = d.p_fdr < 0.05
    d["effect_95ci"] = [f"{e:.3f} [{lo:.3f}, {hi:.3f}]" for e, lo, hi in zip(d.effect, d.ci_lower, d.ci_upper)]
    d["p_raw_display"] = d.p_raw.map(disp)
    d["p_fdr_display"] = d.p_fdr.map(disp)
    d["significance"] = np.where(d.significant_fdr, "*", "")
    d.to_csv(AGG / f"formal_hypothesis_tests_{fam}_ba_corrected.csv", index=False)
    for i in range(len(d)):
        if (abs(d.effect[i] - old.effect[i]) > 5e-4 or d.significant_fdr[i] != old.significant_fdr[i]
                or abs(d.p_fdr[i] - old.p_fdr[i]) > 5e-4):
            changes.append({"family": fam, "domain": d.domain[i], "task": d.task[i], "model": d.model[i],
                            "old": old.effect_95ci[i], "new": d.effect_95ci[i],
                            "old_p_fdr": old.p_fdr_display[i], "new_p_fdr": d.p_fdr_display[i],
                            "old_sig": bool(old.significant_fdr[i]), "new_sig": bool(d.significant_fdr[i])})
    print(f"{fam}: {int(d.significant_fdr.sum())}/{len(d)} significant after BH "
          f"(before: {int(old.significant_fdr.sum())}/{len(old)}); max |sham effect| "
          f"{d.effect.abs().max():.4f}" if fam == "sham" else
          f"{fam}: {int(d.significant_fdr.sum())}/{len(d)} significant after BH (before: {int(old.significant_fdr.sum())}/{len(old)})")
ch = pd.DataFrame(changes)
ch.to_csv(AGG / "formal_tests_ba_correction_changes.csv", index=False)
pd.set_option("display.width", 250)
print(ch.to_string(index=False))
