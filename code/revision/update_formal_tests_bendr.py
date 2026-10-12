#!/usr/bin/env python
"""Replace the BENDR rows of the formal test families with the corrected-pipeline
result and recompute Benjamini-Hochberg FDR (q = 0.05) within each family
(31 primary flattening tests; 28 sham tests), exactly as in the submission."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TABLES = Path(sys.argv[1])        # original results/tables
BENDR = Path(sys.argv[2])         # aggregate_revision_bootstrap output for corrected BENDR
OUT = Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
agg = pd.read_csv(BENDR)


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


summary = {}
for fam, cond, pcol, note in [
    ("primary", "phase_flattened", "p_one_sided_positive",
     "Corrected pipeline (single uV->V input scaling); sham verified neutral."),
    ("sham", "phase_sham", "p_two_sided_zero", "Corrected pipeline (single uV->V input scaling)."),
]:
    d = pd.read_csv(TABLES / f"formal_hypothesis_tests_{fam}.csv")
    r = agg[(agg.estimate == "drop") & (agg.condition == cond)].iloc[0]
    m = d.model == "BENDR"
    if m.sum() != 1:
        raise ValueError("expected one BENDR row")
    d.loc[m, ["effect", "ci_lower", "ci_upper", "p_raw"]] = [r.point, r.ci_lower, r.ci_upper, r[pcol]]
    d.loc[m, "note"] = note
    d.loc[m, "source_file"] = "revision: results/bendr_fixed (3 seeds, pooled_fm bootstrap)"
    d["p_fdr"] = bh(d.p_raw.values)
    d["significant_fdr"] = d.p_fdr < 0.05
    d["effect_95ci"] = [f"{e:.3f} [{lo:.3f}, {hi:.3f}]" for e, lo, hi in zip(d.effect, d.ci_lower, d.ci_upper)]
    d["p_raw_display"] = d.p_raw.map(disp)
    d["p_fdr_display"] = d.p_fdr.map(disp)
    d["significance"] = np.where(d.significant_fdr, "*", "")
    d.to_csv(OUT / f"formal_hypothesis_tests_{fam}_revised.csv", index=False)
    summary[fam] = (int(d.significant_fdr.sum()), len(d))
    print(fam, "significant:", summary[fam])
    print(d.loc[m, ["model", "effect_95ci", "p_raw_display", "p_fdr_display", "significance"]].to_string(index=False))
    if fam == "sham":
        print("significant sham tests:", d.loc[d.significant_fdr, "model"].tolist())
