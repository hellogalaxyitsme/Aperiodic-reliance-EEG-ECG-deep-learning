#!/usr/bin/env python
"""Raw time-domain overlays, original vs. flattened (and sham), one
representative window per dataset (the window whose original-flattened correlation
is closest to the dataset median), plus a zoom on the largest transient.
Also writes the distortion summary as a LaTeX table.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

IN = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
FLAT, SHAM = "#008300", "#2a78d6"

ex, meta = {}, {}
for part in ["part_noTUAB", "part_TUAB"]:
    z = np.load(IN / part / "overlay_examples.npz", allow_pickle=True)
    for k in z.files:
        ds, field = k.split("__")
        ex.setdefault(ds, {})[field] = z[k]
    meta.update(json.loads((IN / part / "overlay_examples_meta.json").read_text()))
summary = pd.concat([pd.read_csv(IN / p / "waveform_distortion_summary.csv") for p in ["part_noTUAB", "part_TUAB"]])

#           dataset        title                     excerpt_s  zoom_s
rows = [("sleep_edf", "Sleep-EDF (N2, Fpz-Cz)", 6.0, 1.0),
        ("tuab", "TUAB (FP1)", 6.0, 1.0),
        ("physionet_mi", "PhysioNet MI (FC5)", None, 1.0),
        ("ptbxl", "PTB-XL ECG (lead II)", None, 2.0)]
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.linewidth": 0.7})
fig, axes = plt.subplots(4, 2, figsize=(7.16, 6.6), gridspec_kw={"width_ratios": [2.0, 1]})
letters = iter("abcdefgh")


def window(n, center, length_s, sf):
    if length_s is None:
        return 0, n
    half = int(length_s / 2 * sf)
    lo = min(max(0, center - half), max(0, n - 2 * half))
    return lo, min(n, lo + 2 * half)


for i, (ds, title, excerpt_s, zoom_s) in enumerate(rows):
    e = ex[ds]
    sf = float(e["sfreq"])
    o, fl = e["original"], e["phase_flattened"]
    t = np.arange(len(o)) / sf
    tstar = int(np.abs(o - o.mean()).argmax())
    for j, ax in enumerate(axes[i]):
        lo, hi = window(len(o), tstar, excerpt_s if j == 0 else zoom_s, sf)
        sl = slice(lo, hi)
        ax.plot(t[sl], o[sl], color=INK, lw=0.9, label="Original (sham reconstruction is identical)")
        ax.plot(t[sl], fl[sl], color=FLAT, lw=0.9, ls="--", label="Flattened")
        if j == 0:
            zlo, zhi = window(len(o), tstar, zoom_s, sf)
            ax.axvspan(t[zlo], t[zhi - 1], color=GRID, alpha=0.7, lw=0, zorder=0)
            r = float(e["r_flattened"])
            span = "full window" if excerpt_s is None else f"{excerpt_s:.0f}-s excerpt"
            ax.set_title(f"{title}, {span};  r(original, flattened) = {r:.2f}", loc="left", fontsize=7.5,
                         color=INK)
            ax.set_ylabel("Amplitude (µV)")
        else:
            ax.set_title("zoom on largest transient", loc="left", fontsize=7, color=MUTED)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.12 if j == 0 else -0.24, 1.06, next(letters), transform=ax.transAxes, fontsize=10,
                fontweight="bold", color=INK)
        if i == 3:
            ax.set_xlabel("Time (s)")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(h_pad=1.3, rect=(0, 0.03, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_r3_3_time_domain_overlays.{ext}", dpi=300, bbox_inches="tight")

# --- LaTeX summary table (flattened and aperiodic-shaped; sham is identity) ---
names = {"sleep_edf": "Sleep-EDF", "tuab": "TUAB", "physionet_mi": "PhysioNet MI", "ptbxl": "PTB-XL ECG"}
metr = [("pearson_r", "Waveform $r$"), ("p2p_ratio", "Peak-to-peak ratio"),
        ("transient_amp_ratio", "Transient amplitude ratio"), ("transient_shift_ms", "Transient shift (ms)")]
ecg = [("r_peak_preserved_frac", "R-peaks preserved ($\\pm$20 ms)"), ("r_amp_ratio", "R-wave amplitude ratio"),
       ("median_beat_r", "Median-beat $r$")]
lines = [r"\begin{table}[t]", r"\centering",
         r"\caption{Time-domain effect of the interventions (2{,}000 random windows per dataset; median [IQR] over "
         r"windows $\times$ channels). The sham reconstruction is the identity on every metric (all medians 1.000; shift 0 ms).}",
         r"\label{tab:waveform}", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
         r"\begin{tabular}{llcc}", r"\toprule",
         r"\textbf{Dataset} & \textbf{Metric} & \textbf{Flattened} & \textbf{Aperiodic-shaped} \\", r"\midrule"]
fmt = lambda r: f"{r['median']:.2f} [{r['q25']:.2f}, {r['q75']:.2f}]"
csv_rows = []
for ds, nice in names.items():
    first = True
    for m, label in metr + (ecg if ds == "ptbxl" else []):
        cells = []
        for c in ["phase_flattened", "phase_aperiodic"]:
            r = summary[(summary.dataset == ds) & (summary.condition == c) & (summary.metric == m)].iloc[0]
            cells.append(fmt(r))
            csv_rows.append({"dataset": nice, "metric": m, "condition": c, "median": r["median"], "q25": r.q25,
                             "q75": r.q75, "p5": r.p5, "p95": r.p95})
        lines.append(f"{nice if first else ''} & {label} & {cells[0]} & {cells[1]} \\\\")
        first = False
    lines.append(r"\midrule" if ds != "ptbxl" else r"\bottomrule")
lines += [r"\end{tabular}", r"\end{table}"]
(OUT / "table_r3_3_waveform_distortion.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
pd.DataFrame(csv_rows).to_csv(OUT / "table_r3_3_waveform_distortion.csv", index=False)
sham = summary[summary.condition == "phase_sham"]
print("sham max deviation from identity:", float(sham.mean_abs_dev_from_identity.max()))
print("transient shift p5/p95 (flattened):")
print(summary[(summary.metric == "transient_shift_ms") & (summary.condition == "phase_flattened")][["dataset", "p5", "p95"]])
