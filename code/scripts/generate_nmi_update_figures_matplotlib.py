#!/usr/bin/env python3
"""Generate new NMI extended-data figures with matplotlib."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "Journal Images" / "Extended Data Figures"

BLUE = "#2166ac"
GREY = "#878787"
RED = "#b2182b"
GREEN = "#4daf4a"
ORANGE = "#ff7f00"


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.08, label, transform=ax.transAxes, fontweight="bold", va="bottom")


def draw_error_bars(ax: plt.Axes, x: np.ndarray, y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> None:
    ax.errorbar(
        x,
        y,
        yerr=np.vstack([y - lo, hi - y]),
        fmt="none",
        ecolor="black",
        elinewidth=0.7,
        capsize=2,
        capthick=0.7,
        zorder=5,
    )


def finish_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.6, length=3)


def make_ecg_figure() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels = ["PSD\nridge", "ResNet1D-\nWang", "Inception1D", "XResNet\n1D101"]
    unmatched = np.array([0.100, 0.347, 0.358, 0.322])
    unmatched_lo = np.array([0.079, 0.331, 0.343, 0.291])
    unmatched_hi = np.array([0.122, 0.363, 0.373, 0.354])

    neural = ["ResNet1D-\nWang", "Inception1D", "XResNet\n1D101"]
    unmatched_neural = np.array([0.347, 0.358, 0.322])
    unmatched_neural_lo = np.array([0.331, 0.343, 0.291])
    unmatched_neural_hi = np.array([0.363, 0.373, 0.354])
    matched_neural = np.array([0.312, 0.311, 0.299])
    matched_neural_lo = np.array([0.293, 0.291, 0.279])
    matched_neural_hi = np.array([0.331, 0.330, 0.319])

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 2.35),
        gridspec_kw={"width_ratios": [1.22, 1.22, 0.95]},
    )

    ax = axes[0]
    x = np.arange(len(labels))
    ax.bar(x, unmatched, color=[GREY, BLUE, BLUE, BLUE], width=0.68)
    draw_error_bars(ax, x, unmatched, unmatched_lo, unmatched_hi)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Flattening drop (BA)")
    ax.set_ylim(0, 0.40)
    ax.set_title("Unmatched PTB-XL")
    panel_label(ax, "a")
    finish_axes(ax)

    ax = axes[1]
    x = np.arange(len(neural))
    width = 0.34
    ax.bar(x - width / 2, unmatched_neural, width=width, color=BLUE, label="Unmatched")
    ax.bar(x + width / 2, matched_neural, width=width, color=GREEN, label="Age/sex matched")
    draw_error_bars(ax, x - width / 2, unmatched_neural, unmatched_neural_lo, unmatched_neural_hi)
    draw_error_bars(ax, x + width / 2, matched_neural, matched_neural_lo, matched_neural_hi)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(neural)
    ax.set_ylim(0, 0.40)
    ax.set_title("Neural drops after matching")
    ax.legend(frameon=False, loc="upper right", fontsize=7)
    panel_label(ax, "b")
    finish_axes(ax)

    ax = axes[2]
    q10, q25, q50, q75, q90 = 0.138, 0.198, 0.273, 0.356, 0.442
    ax.plot([q10, q90], [0, 0], color=GREY, linewidth=5, solid_capstyle="butt")
    ax.plot([q25, q75], [0, 0], color=BLUE, linewidth=10, solid_capstyle="butt")
    ax.scatter([q50], [0], color=RED, zorder=5, s=18)
    ax.set_xlim(0, 0.65)
    ax.set_ylim(-0.45, 0.45)
    ax.set_yticks([])
    ax.set_xlabel("SpecParam $R^2$")
    ax.set_title("ECG fit quality")
    ax.text(q50, 0.19, "median 0.273", ha="center", va="bottom", fontsize=7)
    ax.text(q10, -0.19, "10th", ha="center", va="top", fontsize=7)
    ax.text(q90, -0.19, "90th", ha="center", va="top", fontsize=7)
    panel_label(ax, "c")
    finish_axes(ax)

    fig.tight_layout(w_pad=1.25)
    fig.savefig(FIG_DIR / "ed_fig9_ptbxl_ecg_spectral_audit.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "ed_fig9_ptbxl_ecg_spectral_audit.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def make_tuab_temporal_figure() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels = [
        "Within\nearly",
        "Within\nlate",
        "Train early\neval late",
        "Train late\neval early",
    ]
    values = np.array([0.118, 0.204, 0.146, 0.164])
    lo = np.array([0.068, 0.153, 0.107, 0.105])
    hi = np.array([0.165, 0.253, 0.184, 0.221])
    colors = [BLUE, RED, ORANGE, GREEN]

    fig, ax = plt.subplots(figsize=(4.0, 2.3))
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, width=0.62)
    draw_error_bars(ax, x, values, lo, hi)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Flattening drop (BA)")
    ax.set_ylim(0, 0.28)
    ax.set_title("TUAB temporal acquisition-proxy audit")
    finish_axes(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ed_fig10_tuab_temporal_audit.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "ed_fig10_tuab_temporal_audit.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    make_ecg_figure()
    make_tuab_temporal_figure()


if __name__ == "__main__":
    main()
