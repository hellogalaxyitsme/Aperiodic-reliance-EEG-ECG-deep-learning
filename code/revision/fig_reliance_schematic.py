#!/usr/bin/env python
"""Conceptual schematic - how flattening affects an aperiodic-reliant
versus an oscillation-reliant classifier.

Spectra are the noise-free class means of the manuscript's simulation generator
(run_aperiodic_simulation_validation.generate_psd: offset 0.5, exponent 1.0,
aperiodic_only: offset +/-0.18, exponent +/-0.22; oscillatory_only: +/-0.45 x
[alpha(10 Hz) - 0.65 sigma(13.5 Hz)]). Balanced accuracies are the reported
simulation results (train on full spectrum; test on full / flattened).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
FULL, FLAT = "#2a78d6", "#008300"   # paper convention: full = blue, flattened = green

sim = pd.read_csv(ROOT / "results/tables/aperiodic_simulation_validation/simulation_validation_metrics.csv")


def ba(scenario, test):
    r = sim[(sim.scenario == scenario) & (sim.train_input == "full_log_psd") & (sim.test_input == test)]
    return float(r.balanced_accuracy_mean.iloc[0])


f = np.linspace(1.0, 45.0, 177)
lf = np.log10(f)
g = lambda c, w: np.exp(-0.5 * ((f - c) / w) ** 2)
template = g(10.0, 1.2) - 0.65 * g(13.5, 1.5)


def spectra(scenario):
    out = {}
    for name, s in [("Class 1", -1), ("Class 2", +1)]:
        if scenario == "aperiodic_only":
            ap = (0.5 + 0.18 * s) - (1.0 + 0.22 * s) * lf
            osc = 0 * f
        else:
            ap = 0.5 - 1.0 * lf
            osc = 0.45 * s * template
        out[name] = (ap + osc, osc)  # full log-PSD, flattened residual
    return out


plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8})
fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.9), gridspec_kw={"width_ratios": [1.15, 1.15, 0.8]})
rows = [("aperiodic_only", "Aperiodic-reliant task\n(classes differ in broadband slope)"),
        ("oscillatory_only", "Oscillation-reliant task\n(classes differ in a band-limited peak)")]
letters = iter("abcdef")
for r, (scen, title) in enumerate(rows):
    sp = spectra(scen)
    ax_full, ax_flat, ax_bar = axes[r]
    for name, ls, col in [("Class 1", "-", INK), ("Class 2", "--", MUTED)]:
        full, flat = sp[name]
        ax_full.plot(f, full, ls=ls, color=col, lw=1.6)
        ax_flat.plot(f, flat, ls=ls, color=col, lw=1.6)
        # direct labels (identity is never colour-alone), placed in clear space
        if scen == "aperiodic_only":
            xi = 44 if name == "Class 1" else 22
            yi = full[np.argmin(abs(f - xi))]
            ax_full.text(xi, yi + (0.07 if name == "Class 1" else -0.08), name, color=col, fontsize=7,
                         ha="right" if name == "Class 1" else "left", va="bottom" if name == "Class 1" else "top")
        else:
            ax_full.text(30, 0.25 if name == "Class 1" else 0.05, ("— " if ls == "-" else "-- ") + name,
                         color=col, fontsize=7, ha="left")
    ax_full.set_title("Full spectrum", fontsize=8, color=INK, loc="left")
    ax_flat.set_title("After flattening", fontsize=8, color=INK, loc="left")
    ax_flat.axhline(0, color=GRID, lw=0.8, zorder=0)
    if scen == "aperiodic_only":
        ax_flat.text(23, 0.12, "class curves coincide:\nno evidence left", ha="center", fontsize=7, color=INK)
        ax_flat.set_ylim(-0.6, 0.6)
    else:
        ax_flat.text(30, 0.3, "peak difference\nsurvives", ha="center", fontsize=7, color=INK)
        ax_flat.set_ylim(-0.6, 0.6)
    for ax in (ax_full, ax_flat):
        ax.set_xlim(1, 45)
        ax.set_xlabel("Frequency (Hz)")
        ax.spines[["top", "right"]].set_visible(False)
    ax_full.set_ylabel("log$_{10}$ power")
    ax_full.text(-0.30, 1.30 if r == 0 else 1.30, title, transform=ax_full.transAxes, fontsize=8,
                 fontweight="bold", color=INK, va="bottom")
    vals = [ba(scen, "full_log_psd"), ba(scen, "flattened_log_psd")]
    bars = ax_bar.bar([0, 1], vals, width=0.62, color=[FULL, FLAT], edgecolor="white", linewidth=1.5)
    bars[1].set_hatch("////")
    for x, v in zip([0, 1], vals):
        ax_bar.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    ax_bar.axhline(0.5, color=MUTED, ls=":", lw=0.9)
    ax_bar.text(1.55, 0.5, "chance", fontsize=6.5, color=MUTED, ha="left", va="center", clip_on=False)
    ax_bar.set_xticks([0, 1], ["Full", "Flattened"])
    ax_bar.set_xlim(-0.5, 1.5)
    ax_bar.set_ylim(0.4, 1.08)
    ax_bar.set_ylabel("Balanced accuracy")
    drop = vals[0] - vals[1]
    ax_bar.set_title(f"Test BA ($\\Delta_{{flat}}$ = {drop:.2f})", fontsize=8, color=INK, loc="left")
    ax_bar.spines[["top", "right"]].set_visible(False)
    for ax in (ax_full, ax_flat, ax_bar):
        ax.text(-0.22 if ax is not ax_bar else -0.42, 1.02, next(letters), transform=ax.transAxes,
                fontsize=10, fontweight="bold", color=INK)
fig.tight_layout(h_pad=2.6, w_pad=1.2)
fig.subplots_adjust(top=0.86, hspace=0.95)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_r2_reliance_schematic.{ext}", dpi=300, bbox_inches="tight")
print("saved", OUT)
