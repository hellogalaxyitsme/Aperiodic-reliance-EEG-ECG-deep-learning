#!/usr/bin/env python3
"""Generate main NMI manuscript figures from local result artifacts.

The quantitative panels read the existing CSV result tables. Figure 1 uses a
deterministic representative signal because the heavy raw epoch caches are not
kept in this local working folder.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Journal Images"

COLORS = {
    "raw": "#2166ac",
    "sham": "#878787",
    "aperiodic": "#b2182b",
    "flattened": "#4daf4a",
    "foundation": "#ff7f00",
    "black": "#1a1a1a",
}

PATHS = {
    "sleep_full": ROOT / "reports/tables/full_sleep_edf_multiseed_subject_bootstrap.csv",
    "sleep_reviewer": ROOT / "reports/tables/sleep_edf_reviewer_resistance_bootstrap.csv",
    "tuab_neural": ROOT / "reports/tables/tuab_full_multiseed_neural_subject_bootstrap.csv",
    "tuab_matched_neural": ROOT / "reports/tables/tuab_full_age_matched_multiseed_neural_subject_bootstrap.csv",
    "mi_neural": ROOT / "reports/tables/physionet_mi_multiseed_neural_subject_bootstrap.csv",
    "tuab_foundation": ROOT / "reports/tables/tuab_full_foundation_multiseed_subject_bootstrap.csv",
    "tuab_psd": ROOT / "results/tuab_full_v3_0_1/psd_interventions_specparam/tuab_psd_intervention_subject_bootstrap.csv",
    "tuab_matched_psd": ROOT / "results/tuab_full_v3_0_1/age_matched/psd_interventions_specparam_tuab_full_age_sex_matched_caliper5/tuab_psd_intervention_subject_bootstrap.csv",
    "tuab_age": ROOT / "results/tuab_full_v3_0_1/age_matched/tuab_full_age_sex_matched_caliper5_header_metadata_subjects.csv",
    "tuab_matched_age": ROOT / "results/tuab_full_v3_0_1/age_matched/tuab_full_age_sex_matched_caliper5_subjects.csv",
    "simulation": ROOT / "reports/tables/aperiodic_simulation_validation/simulation_validation_metrics.csv",
    "irasa": ROOT / "reports/tables/irasa_specparam_agreement_stage_balanced_5k_volts/irasa_specparam_agreement.csv",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fval(value: str | float | int | None, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except ValueError:
        return default


def row_basic(
    rows: list[dict[str, str]],
    model: str,
    task: str,
    estimate: str,
    metric: str = "balanced_accuracy",
) -> dict[str, str]:
    for row in rows:
        if (
            row.get("model") == model
            and row.get("task") == task
            and row.get("metric") == metric
            and row.get("estimate") == estimate
        ):
            return row
    raise KeyError((model, task, estimate, metric))


def row_matrix(
    rows: list[dict[str, str]],
    train_input: str,
    test_input: str,
    estimate: str = "performance",
    metric: str = "balanced_accuracy",
    model: str | None = None,
) -> dict[str, str]:
    for row in rows:
        if (
            row.get("train_input") == train_input
            and row.get("test_input") == test_input
            and row.get("metric") == metric
            and row.get("estimate") == estimate
            and (model is None or row.get("model") == model)
        ):
            return row
    raise KeyError((model, train_input, test_input, estimate, metric))


def point_ci(row: dict[str, str]) -> tuple[float, float, float]:
    return fval(row["point"]), fval(row["ci_lower"]), fval(row["ci_upper"])


def fmt_ci(row: dict[str, str]) -> str:
    p, lo, hi = point_ci(row)
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


def clean_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, width=0.7)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def savefig(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_box(ax: plt.Axes, xy: tuple[float, float], w: float, h: float, text: str, fc: str, ec: str = "#333333") -> None:
    ax.add_patch(Rectangle(xy, w, h, facecolor=fc, edgecolor=ec, linewidth=0.9))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=7)


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.8,
            color="#333333",
        )
    )


def generate_representative_interventions() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    sfreq = 128
    duration = 30
    t = np.arange(0, duration, 1 / sfreq)
    freqs = np.fft.rfftfreq(t.size, 1 / sfreq)
    amp = 1 / np.maximum(freqs, 0.5) ** 0.9
    phases = rng.uniform(0, 2 * np.pi, size=freqs.size)
    spectrum = amp * np.exp(1j * phases)
    noise = np.fft.irfft(spectrum, n=t.size)
    raw = (
        0.55 * noise / np.std(noise)
        + 0.55 * np.sin(2 * np.pi * 10 * t + 0.2)
        + 0.35 * np.sin(2 * np.pi * 2.2 * t)
        + 0.18 * rng.normal(size=t.size)
    )
    raw = raw / np.std(raw)

    fft = np.fft.rfft(raw)
    mag = np.abs(fft)
    phase = np.angle(fft)
    valid = (freqs >= 1) & (freqs <= 45)
    logf = np.log10(np.maximum(freqs[valid], 1e-6))
    logmag = np.log10(np.maximum(mag[valid], 1e-12))
    slope, intercept = np.polyfit(logf, logmag, 1)
    ap_mag = mag.copy()
    ap_mag[valid] = 10 ** (intercept + slope * logf)
    flat_mag = mag.copy()
    flat_mag[valid] = mag[valid] / np.maximum(ap_mag[valid], 1e-12) * np.median(ap_mag[valid])

    sham = np.fft.irfft(mag * np.exp(1j * phase), n=t.size)
    aperiodic = np.fft.irfft(ap_mag * np.exp(1j * phase), n=t.size)
    flattened = np.fft.irfft(flat_mag * np.exp(1j * phase), n=t.size)
    for arr in (sham, aperiodic, flattened):
        arr *= np.std(raw) / np.std(arr)

    return {
        "t": t,
        "freqs": freqs,
        "raw": raw,
        "mag": mag,
        "ap_mag": ap_mag,
        "flat_mag": flat_mag,
        "sham": sham,
        "aperiodic": aperiodic,
        "flattened": flattened,
    }


def fig1_pipeline() -> None:
    data = generate_representative_interventions()
    fig = plt.figure(figsize=(11.5, 3.7))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1.1, 1.35, 0.9, 0.65], wspace=0.35)

    ax_a = fig.add_subplot(gs[0, 0])
    freqs = data["freqs"]
    psd = (data["mag"] ** 2) / len(data["raw"])
    ap_psd = (data["ap_mag"] ** 2) / len(data["raw"])
    valid = (freqs >= 0.8) & (freqs <= 45)
    ax_a.loglog(freqs[valid], psd[valid], color=COLORS["raw"], linewidth=1.2)
    ax_a.loglog(freqs[valid], ap_psd[valid], color=COLORS["aperiodic"], linewidth=1.1)
    ax_a.axvspan(1, 45, color="#e8e8e8", alpha=0.4, zorder=0)
    ax_a.annotate("aperiodic fit", xy=(7, ap_psd[np.argmin(abs(freqs - 7))]), xytext=(13, np.nanmax(psd[valid]) / 7), arrowprops=dict(arrowstyle="->", lw=0.6), fontsize=7)
    ax_a.annotate("peaks", xy=(10, psd[np.argmin(abs(freqs - 10))]), xytext=(18, np.nanmax(psd[valid]) / 2.5), arrowprops=dict(arrowstyle="->", lw=0.6), fontsize=7)
    ax_a.set_xlabel("Frequency (Hz)")
    ax_a.set_ylabel("Power")
    clean_ax(ax_a)
    panel_label(ax_a, "a")

    ax_b = fig.add_subplot(gs[0, 1])
    t = data["t"]
    mask = (t >= 0.1) & (t <= 5.1)
    traces = [
        ("Raw", data["raw"], COLORS["raw"]),
        ("Sham", data["sham"], COLORS["sham"]),
        ("Aperiodic", data["aperiodic"], COLORS["aperiodic"]),
        ("Flattened", data["flattened"], COLORS["flattened"]),
    ]
    offsets = np.arange(len(traces))[::-1] * 3.0
    for offset, (label, trace, color) in zip(offsets, traces):
        ax_b.plot(t[mask], trace[mask] + offset, color=color, linewidth=0.8)
        ax_b.text(-0.08, offset, label, ha="right", va="center", fontsize=7, color=color)
    ax_b.set_yticks([])
    ax_b.set_xlabel("Time (s)")
    ax_b.set_xlim(0.1, 5.1)
    clean_ax(ax_b)
    panel_label(ax_b, "b")

    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.axis("off")
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(0, 1)
    add_box(ax_c, (0.05, 0.72), 0.9, 0.16, "Train on\noriginal", "#dbe9f6")
    add_box(ax_c, (0.05, 0.43), 0.9, 0.20, "Evaluate\nraw / sham /\naperiodic / flat", "#f2f2f2")
    add_box(ax_c, (0.05, 0.14), 0.9, 0.16, "Compare\nbalanced accuracy", "#e5f5e0")
    add_arrow(ax_c, (0.5, 0.72), (0.5, 0.63))
    add_arrow(ax_c, (0.5, 0.43), (0.5, 0.30))
    panel_label(ax_c, "c")

    ax_d = fig.add_subplot(gs[0, 3])
    ax_d.axis("off")
    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(0, 1)
    labels = ["Sham\ncontrol", "Simulation\nvalidation", "IRASA\nagreement", "Train-on-\nrepresentation"]
    y0s = [0.76, 0.52, 0.28, 0.04]
    for y, label in zip(y0s, labels):
        add_box(ax_d, (0.06, y), 0.88, 0.16, label, "#ffffff")
    panel_label(ax_d, "d")
    savefig(fig, "fig1_spectral_audit_framework")


def domain_rows_for_fig2() -> list[dict[str, object]]:
    sleep_rev = read_csv(PATHS["sleep_reviewer"])
    sleep_full = read_csv(PATHS["sleep_full"])
    tuab = read_csv(PATHS["tuab_neural"])
    mi = read_csv(PATHS["mi_neural"])
    foundation = read_csv(PATHS["tuab_foundation"])

    rows: list[dict[str, object]] = []

    def add(domain: str, model: str, row: dict[str, str], foundation: bool = False) -> None:
        p, lo, hi = point_ci(row)
        rows.append({"domain": domain, "model": model, "drop": p, "lo": lo, "hi": hi, "foundation": foundation})

    add("Sleep-EDF\n(Wake vs Sleep)", "EEGNet", row_basic(sleep_rev, "braindecode_eegnet", "wake_vs_sleep", "drop_flattened"))
    add("Sleep-EDF\n(Wake vs Sleep)", "Shallow", row_basic(sleep_rev, "braindecode_shallow_fbcsp", "wake_vs_sleep", "drop_flattened"))
    add("Sleep-EDF\n(Wake vs Sleep)", "Deep4Net", row_basic(sleep_rev, "braindecode_deep4", "wake_vs_sleep", "drop_flattened"))
    add("Sleep-EDF\n(Wake vs Sleep)", "CNN", row_basic(sleep_rev, "raw_cnn_sham", "wake_vs_sleep", "drop_flattened"))
    add("Sleep-EDF\n(Wake vs Sleep)", "MLP", row_basic(sleep_full, "deep_mlp", "wake_vs_sleep", "drop_flattened"))

    for model, label in [("eegnet", "EEGNet"), ("shallow_fbcsp", "Shallow"), ("deep4", "Deep4Net")]:
        add("TUAB\n(Normal vs Abnormal)", label, row_basic(tuab, model, "tuab_normal_vs_abnormal", "drop_flattened"))
    for model, label in [
        ("BIOT", "BIOT"),
        ("LaBraM", "LaBraM"),
        ("EEGPT", "EEGPT"),
        ("CBraMod", "CBraMod"),
        ("REVE-base", "REVE"),
        ("EEGMamba", "EEGMamba"),
        ("BENDR", "BENDR"),
    ]:
        add("TUAB\n(Normal vs Abnormal)", label, row_basic(foundation, model, "tuab_normal_vs_abnormal", "drop_flattened"), True)

    for model, label in [("eegnet", "EEGNet"), ("shallow_fbcsp", "Shallow"), ("deep4", "Deep4Net")]:
        add("PhysioNet MI\n(Left vs Right)", label, row_basic(mi, model, "imagined_left_vs_right_fist", "drop_flattened"))
    return rows


def fig2_cross_domain() -> None:
    rows = domain_rows_for_fig2()
    fig, ax = plt.subplots(figsize=(11.0, 4.1))
    x = np.arange(len(rows))
    groups = [r["domain"] for r in rows]
    spans = []
    start = 0
    for i in range(1, len(groups) + 1):
        if i == len(groups) or groups[i] != groups[start]:
            spans.append((start, i - 1, groups[start]))
            start = i

    shade = {
        "Sleep-EDF\n(Wake vs Sleep)": "#fde0dd",
        "TUAB\n(Normal vs Abnormal)": "#fee8c8",
        "PhysioNet MI\n(Left vs Right)": "#e5f5e0",
    }
    for start, end, domain in spans:
        ax.axvspan(start - 0.5, end + 0.5, color=shade[domain], alpha=0.45, zorder=0)
        ax.text((start + end) / 2, 0.535, domain, ha="center", va="bottom", fontsize=7)
        if end < len(rows) - 1:
            ax.axvline(end + 0.5, color="#cccccc", linewidth=0.8)

    for i, r in enumerate(rows):
        p, lo, hi = float(r["drop"]), float(r["lo"]), float(r["hi"])
        marker = "*" if r["foundation"] else "o"
        color = COLORS["foundation"] if r["foundation"] else COLORS["black"]
        ms = 9 if r["foundation"] else 5
        ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], fmt=marker, markersize=ms, color=color, ecolor=color, elinewidth=0.9, capsize=2.5, zorder=3)

    ax.axhline(0, color="#666666", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Flattening drop (balanced accuracy)")
    ax.set_ylim(-0.05, 0.56)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r["model"]) for r in rows], rotation=35, ha="right")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["black"], markeredgecolor=COLORS["black"], label="Standard architecture"),
            Line2D([0], [0], marker="*", color="none", markerfacecolor=COLORS["foundation"], markeredgecolor=COLORS["foundation"], markersize=9, label="Foundation model"),
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
        ncol=2,
    )
    clean_ax(ax)
    savefig(fig, "fig2_cross_domain_flattening_drop")


def fig3_sleep_tasks() -> None:
    rows = read_csv(PATHS["sleep_reviewer"])
    tasks = [
        ("Wake vs Sleep", "wake_vs_sleep", 0.5),
        ("Five-stage", "five_stage", 0.2),
        ("N2 vs N3", "n2_vs_n3", 0.5),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), sharey=True)
    conditions = [("Raw", "baseline", "raw"), ("Sham", "sham", "sham"), ("Aperiodic", "aperiodic", "aperiodic"), ("Flattened", "flattened", "flattened")]
    for ax, (title, task, chance) in zip(axes, tasks):
        vals = [point_ci(row_basic(rows, "braindecode_eegnet", task, est))[0] for _, est, _ in conditions]
        cis = [point_ci(row_basic(rows, "braindecode_eegnet", task, est)) for _, est, _ in conditions]
        x = np.arange(len(conditions))
        ax.bar(x, vals, color=[COLORS[key] for _, _, key in conditions], edgecolor="#222222", linewidth=0.5, width=0.68)
        for i, (p, lo, hi) in enumerate(cis):
            ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], color="#222222", elinewidth=0.7, capsize=2, fmt="none")
        drop = row_basic(rows, "braindecode_eegnet", task, "drop_flattened")
        ax.axhline(chance, color="#777777", linestyle="--", linewidth=0.7)
        ax.text(3.55, chance + 0.015, f"chance={chance:.1f}", fontsize=6, color="#666666", ha="right")
        ax.set_xticks(x)
        ax.set_xticklabels([c[0] for c in conditions], rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.text(0.5, -0.30, f"Delta(flat) = {fmt_ci(drop)}", transform=ax.transAxes, ha="center", va="top", fontsize=7)
        ax.text(0.5, 1.04, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=8, fontweight="bold")
        clean_ax(ax)
    axes[0].set_ylabel("Balanced accuracy")
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    panel_label(axes[2], "c")
    savefig(fig, "fig3_sleep_task_dissociation")


def ages_by_label(path: Path, age_col: str) -> dict[str, list[float]]:
    out = {"normal": [], "abnormal": []}
    for row in read_csv(path):
        label = row.get("label", "").lower()
        age = fval(row.get(age_col))
        if label in out and math.isfinite(age) and 0 < age < 110:
            out[label].append(age)
    return out


def fig4_tuab_audit() -> None:
    tuab_psd = read_csv(PATHS["tuab_psd"])
    matched_psd = read_csv(PATHS["tuab_matched_psd"])
    tuab_neural = read_csv(PATHS["tuab_neural"])
    matched_neural = read_csv(PATHS["tuab_matched_neural"])
    foundation = read_csv(PATHS["tuab_foundation"])

    fig = plt.figure(figsize=(10.5, 7.2))
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)

    ax_a = fig.add_subplot(gs[0, 0])
    labels = ["Full", "Aperiodic", "Flattened"]
    test_inputs = ["full_log_psd", "aperiodic_spectrum", "flattened_log_psd"]
    x = np.arange(2)
    width = 0.23
    for j, (label, test_input, color_key) in enumerate(zip(labels, test_inputs, ["raw", "aperiodic", "flattened"])):
        vals, los, his = [], [], []
        for rows in [tuab_psd, matched_psd]:
            p, lo, hi = point_ci(row_matrix(rows, "full_log_psd", test_input))
            vals.append(p)
            los.append(lo)
            his.append(hi)
        xpos = x + (j - 1) * width
        ax_a.bar(xpos, vals, width=width, color=COLORS[color_key], edgecolor="#222222", linewidth=0.5, label=label)
        ax_a.errorbar(xpos, vals, yerr=[np.array(vals) - np.array(los), np.array(his) - np.array(vals)], fmt="none", color="#222222", elinewidth=0.7, capsize=2)
    ax_a.axhline(0.5, color="#777777", linestyle="--", linewidth=0.7)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(["Unmatched", "Age/sex\nmatched"])
    ax_a.set_ylabel("PSD ridge balanced accuracy")
    ax_a.set_ylim(0.45, 0.82)
    ax_a.text(0.02, 0.03, "y-axis starts at 0.45", transform=ax_a.transAxes, fontsize=6, color="#555555")
    ax_a.legend(frameon=False, loc="upper right")
    clean_ax(ax_a)
    panel_label(ax_a, "a")

    ax_b = fig.add_subplot(gs[0, 1])
    unmatched = ages_by_label(PATHS["tuab_age"], "age_years_first_available")
    matched = ages_by_label(PATHS["tuab_matched_age"], "age_years")
    box_data = [unmatched["normal"], unmatched["abnormal"], matched["normal"], matched["abnormal"]]
    bp = ax_b.boxplot(box_data, positions=[0, 1, 3, 4], widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color="#222222", linewidth=1.0))
    for patch, color in zip(bp["boxes"], ["#9ecae1", "#f4a3a3", "#9ecae1", "#f4a3a3"]):
        patch.set_facecolor(color)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(0.7)
    ax_b.set_xticks([0.5, 3.5])
    ax_b.set_xticklabels(["Unmatched", "Age/sex matched"])
    ax_b.set_ylabel("Age (years)")
    ax_b.text(
        0.98,
        0.96,
        "full-corpus age imbalance",
        transform=ax_b.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.5),
    )
    ax_b.legend(
        handles=[
            Rectangle((0, 0), 1, 1, facecolor="#9ecae1", edgecolor="#333333", label="Normal"),
            Rectangle((0, 0), 1, 1, facecolor="#f4a3a3", edgecolor="#333333", label="Abnormal"),
        ],
        frameon=False,
        loc="lower right",
    )
    clean_ax(ax_b)
    panel_label(ax_b, "b")

    ax_c = fig.add_subplot(gs[1, 0])
    models = [("eegnet", "EEGNet"), ("shallow_fbcsp", "Shallow"), ("deep4", "Deep4Net")]
    for i, (model, label) in enumerate(models):
        u = point_ci(row_basic(tuab_neural, model, "tuab_normal_vs_abnormal", "drop_flattened"))
        m = point_ci(row_basic(matched_neural, model, "tuab_normal_vs_abnormal", "drop_flattened"))
        xs = [i - 0.14, i + 0.14]
        ax_c.plot(xs, [u[0], m[0]], color="#999999", linewidth=0.8)
        ax_c.errorbar(xs[0], u[0], yerr=[[u[0] - u[1]], [u[2] - u[0]]], fmt="o", color=COLORS["black"], capsize=2, markersize=4)
        ax_c.errorbar(xs[1], m[0], yerr=[[m[0] - m[1]], [m[2] - m[0]]], fmt="o", color=COLORS["flattened"], capsize=2, markersize=4)
        ax_c.text(i, -0.02, label, transform=ax_c.get_xaxis_transform(), ha="center", va="top", fontsize=7)
    ax_c.axhline(0, color="#777777", linestyle="--", linewidth=0.7)
    ax_c.set_xticks([])
    ax_c.set_ylabel("Flattening drop")
    ax_c.set_ylim(0.0, 0.35)
    ax_c.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["black"], markeredgecolor=COLORS["black"], label="Unmatched"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["flattened"], markeredgecolor=COLORS["flattened"], label="Age/sex matched"),
        ],
        frameon=False,
        loc="upper left",
    )
    clean_ax(ax_c)
    panel_label(ax_c, "c")

    ax_d = fig.add_subplot(gs[1, 1])
    fm_rows = [
        ("BIOT", "BIOT"),
        ("LaBraM", "LaBraM"),
        ("EEGPT", "EEGPT"),
        ("CBraMod", "CBraMod"),
        ("REVE", "REVE-base"),
        ("EEGMamba", "EEGMamba"),
        ("BENDR*", "BENDR"),
    ]
    drop_rows = [row_basic(foundation, model, "tuab_normal_vs_abnormal", "drop_flattened") for _, model in fm_rows]
    flat_rows = [row_basic(foundation, model, "tuab_normal_vs_abnormal", "flattened") for _, model in fm_rows]
    vals = [point_ci(r)[0] for r in drop_rows]
    los = [point_ci(r)[1] for r in drop_rows]
    his = [point_ci(r)[2] for r in drop_rows]
    xx = np.arange(len(fm_rows))
    bar_colors = [COLORS["foundation"]] * (len(fm_rows) - 1) + [COLORS["sham"]]
    ax_d.bar(xx, vals, color=bar_colors, edgecolor="#222222", linewidth=0.5, width=0.58)
    ax_d.errorbar(xx, vals, yerr=[np.array(vals) - np.array(los), np.array(his) - np.array(vals)], fmt="none", color="#222222", elinewidth=0.7, capsize=2)
    for i, perf_row in enumerate(flat_rows):
        ax_d.text(i, vals[i] + 0.025, f"{point_ci(perf_row)[0]:.3f}", ha="center", fontsize=6, rotation=90 if len(fm_rows) > 5 else 0)
    ax_d.axhline(0, color="#777777", linestyle="--", linewidth=0.7)
    ax_d.set_xticks(xx)
    ax_d.set_xticklabels([r[0] for r in fm_rows], rotation=35, ha="right")
    ax_d.set_ylabel("Flattening drop")
    ax_d.set_ylim(0, 0.34)
    ax_d.text(0.02, 0.96, "numbers above bars: flattened BA", transform=ax_d.transAxes, va="top", fontsize=6)
    clean_ax(ax_d)
    panel_label(ax_d, "d")

    savefig(fig, "fig4_tuab_clinical_benchmark_audit")


def collect_sham_pairs() -> list[tuple[float, float, str]]:
    pairs = []

    def add_basic(path_key: str, models: list[str], task: str, label: str) -> None:
        rows = read_csv(PATHS[path_key])
        for model in models:
            try:
                raw = point_ci(row_basic(rows, model, task, "baseline"))[0]
                sham = point_ci(row_basic(rows, model, task, "sham"))[0]
                pairs.append((raw, sham, label))
            except KeyError:
                continue

    add_basic("sleep_reviewer", ["raw_cnn_sham", "braindecode_eegnet", "braindecode_shallow_fbcsp", "braindecode_deep4"], "wake_vs_sleep", "Sleep")
    add_basic("sleep_reviewer", ["raw_cnn_sham", "braindecode_eegnet", "braindecode_shallow_fbcsp", "braindecode_deep4"], "five_stage", "Sleep")
    add_basic("sleep_reviewer", ["raw_cnn_sham", "braindecode_eegnet", "braindecode_shallow_fbcsp", "braindecode_deep4"], "n2_vs_n3", "Sleep")
    add_basic("tuab_neural", ["raw_cnn", "eegnet", "shallow_fbcsp", "deep4"], "tuab_normal_vs_abnormal", "TUAB")
    add_basic("tuab_matched_neural", ["eegnet", "shallow_fbcsp", "deep4"], "tuab_normal_vs_abnormal", "TUAB matched")
    add_basic("mi_neural", ["eegnet", "shallow_fbcsp", "deep4"], "imagined_left_vs_right_fist", "MI")

    foundation = read_csv(PATHS["tuab_foundation"])
    for model in ["BIOT", "LaBraM", "EEGPT", "CBraMod", "REVE-base", "EEGMamba", "BENDR"]:
        raw = point_ci(row_basic(foundation, model, "tuab_normal_vs_abnormal", "baseline"))[0]
        sham = point_ci(row_basic(foundation, model, "tuab_normal_vs_abnormal", "sham"))[0]
        pairs.append((raw, sham, "Foundation"))
    return pairs


def fig5_validation_controls() -> None:
    fig = plt.figure(figsize=(10.5, 7.2))
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)

    ax_a = fig.add_subplot(gs[0, 0])
    pairs = collect_sham_pairs()
    raw = np.array([p[0] for p in pairs])
    sham = np.array([p[1] for p in pairs])
    ax_a.scatter(raw, sham, s=24, color=COLORS["sham"], edgecolor="#222222", linewidth=0.4)
    mn = min(raw.min(), sham.min()) - 0.02
    mx = max(raw.max(), sham.max()) + 0.02
    ax_a.plot([mn, mx], [mn, mx], color="#555555", linestyle="--", linewidth=0.8)
    corr = np.corrcoef(raw, sham)[0, 1] if len(raw) > 1 else math.nan
    ax_a.text(0.04, 0.94, f"Pearson r = {corr:.5f}", transform=ax_a.transAxes, va="top", fontsize=8)
    ax_a.set_xlabel("Raw balanced accuracy")
    ax_a.set_ylabel("Sham balanced accuracy")
    ax_a.set_xlim(mn, mx)
    ax_a.set_ylim(mn, mx)
    clean_ax(ax_a)
    panel_label(ax_a, "a")

    ax_b = fig.add_subplot(gs[0, 1])
    sim_rows = [r for r in read_csv(PATHS["simulation"]) if r["train_input"] == "full_log_psd"]
    scenarios = ["aperiodic_only", "oscillatory_only", "mixed", "train_confound_test_unconfounded"]
    tests = ["full_log_psd", "aperiodic_spectrum", "flattened_log_psd"]
    mat = np.empty((len(scenarios), len(tests)))
    for i, scenario in enumerate(scenarios):
        for j, test in enumerate(tests):
            row = next(r for r in sim_rows if r["scenario"] == scenario and r["test_input"] == test)
            mat[i, j] = fval(row["balanced_accuracy_mean"])
    im = ax_b.imshow(mat, vmin=0.45, vmax=1.0, cmap="viridis", aspect="auto")
    ax_b.set_xticks(np.arange(len(tests)))
    ax_b.set_xticklabels(["Full", "Aperiodic", "Flattened"], rotation=25, ha="right")
    ax_b.set_yticks(np.arange(len(scenarios)))
    ax_b.set_yticklabels(["Aperiodic", "Oscillatory", "Mixed", "Confound"])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax_b.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if mat[i, j] < 0.72 else "black")
    cbar = fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.03)
    cbar.set_label("Balanced accuracy", fontsize=7)
    panel_label(ax_b, "b")

    ax_c = fig.add_subplot(gs[1, 0])
    irasa_row = next(r for r in read_csv(PATHS["irasa"]) if r["comparison"] == "aperiodic_shape_corr")
    mean, std, median, p05, p95 = [fval(irasa_row[k]) for k in ["mean", "std", "median", "p05", "p95"]]
    ax_c.hlines(0, p05, p95, color=COLORS["aperiodic"], linewidth=3)
    ax_c.plot(mean, 0, "o", color=COLORS["black"], label=f"mean={mean:.3f}")
    ax_c.plot(median, 0, "s", color=COLORS["aperiodic"], label=f"median={median:.3f}")
    ax_c.vlines([p05, p95], -0.08, 0.08, color=COLORS["aperiodic"], linewidth=1.0)
    ax_c.text((p05 + p95) / 2, 0.16, f"5th-95th percentile: {p05:.3f}-{p95:.3f}", ha="center", fontsize=7)
    ax_c.set_xlim(0.65, 1.01)
    ax_c.set_ylim(-0.35, 0.35)
    ax_c.set_yticks([])
    ax_c.set_xlabel("SpecParam-IRASA aperiodic shape correlation")
    ax_c.legend(frameon=False, loc="lower left")
    clean_ax(ax_c)
    panel_label(ax_c, "c")

    ax_d = fig.add_subplot(gs[1, 1])
    sleep = read_csv(PATHS["sleep_reviewer"])
    task_defs = [("Wake vs Sleep", "wake_vs_sleep"), ("Five-stage", "five_stage")]
    estimates = [("Full", "baseline", "raw"), ("Aperiodic", "aperiodic", "aperiodic"), ("Flattened", "flattened", "flattened")]
    x = np.arange(len(task_defs))
    width = 0.23
    for j, (label, est, color_key) in enumerate(estimates):
        vals, los, his = [], [], []
        for _, task in task_defs:
            p, lo, hi = point_ci(row_basic(sleep, "psd_train_control", task, est))
            vals.append(p)
            los.append(lo)
            his.append(hi)
        xpos = x + (j - 1) * width
        ax_d.bar(xpos, vals, width=width, color=COLORS[color_key], edgecolor="#222222", linewidth=0.5, label=label)
        ax_d.errorbar(xpos, vals, yerr=[np.array(vals) - np.array(los), np.array(his) - np.array(vals)], fmt="none", color="#222222", elinewidth=0.7, capsize=2)
    ax_d.axhline(0.5, color="#777777", linestyle="--", linewidth=0.7)
    ax_d.axhline(0.2, color="#aaaaaa", linestyle=":", linewidth=0.7)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([t[0] for t in task_defs])
    ax_d.set_ylabel("Balanced accuracy")
    ax_d.set_ylim(0.15, 1.0)
    ax_d.legend(frameon=False, loc="upper right")
    clean_ax(ax_d)
    panel_label(ax_d, "d")

    savefig(fig, "fig5_validation_and_controls")


def table1_summary() -> None:
    sleep_rev = read_csv(PATHS["sleep_reviewer"])
    sleep_full = read_csv(PATHS["sleep_full"])
    tuab = read_csv(PATHS["tuab_neural"])
    mi = read_csv(PATHS["mi_neural"])
    foundation = read_csv(PATHS["tuab_foundation"])

    rows: list[dict[str, str]] = []

    def add_basic(domain: str, task_label: str, model_label: str, source_rows: list[dict[str, str]], model: str, task: str) -> None:
        def maybe(est: str) -> str:
            try:
                return fmt_ci(row_basic(source_rows, model, task, est))
            except KeyError:
                return "NA"

        rows.append(
            {
                "Domain": domain,
                "Task": task_label,
                "Model": model_label,
                "Raw BA [95% CI]": maybe("baseline"),
                "Sham BA [95% CI]": maybe("sham"),
                "Aperiodic BA [95% CI]": maybe("aperiodic"),
                "Flattened BA [95% CI]": maybe("flattened"),
                "Flattening Drop [95% CI]": maybe("drop_flattened"),
            }
        )

    for model, label in [
        ("braindecode_eegnet", "EEGNet"),
        ("braindecode_shallow_fbcsp", "ShallowFBCSPNet"),
        ("braindecode_deep4", "Deep4Net"),
        ("raw_cnn_sham", "CNN"),
    ]:
        add_basic("Sleep-EDF", "Wake vs Sleep", label, sleep_rev, model, "wake_vs_sleep")
    add_basic("Sleep-EDF", "Wake vs Sleep", "MLP", sleep_full, "deep_mlp", "wake_vs_sleep")

    for model, label in [
        ("eegnet", "EEGNet"),
        ("shallow_fbcsp", "ShallowFBCSPNet"),
        ("deep4", "Deep4Net"),
    ]:
        add_basic("TUAB", "Normal vs Abnormal", label, tuab, model, "tuab_normal_vs_abnormal")

    def add_foundation(model: str, label: str | None = None) -> None:
        mapping = {
            "Raw BA [95% CI]": "baseline",
            "Sham BA [95% CI]": "sham",
            "Aperiodic BA [95% CI]": "aperiodic",
            "Flattened BA [95% CI]": "flattened",
            "Flattening Drop [95% CI]": "drop_flattened",
        }
        out = {"Domain": "TUAB", "Task": "Normal vs Abnormal", "Model": label or model}
        for column, estimate in mapping.items():
            out[column] = fmt_ci(row_basic(foundation, model, "tuab_normal_vs_abnormal", estimate))
        rows.append(out)

    for model, label in [
        ("BIOT", None),
        ("LaBraM", None),
        ("EEGPT", None),
        ("CBraMod", None),
        ("REVE-base", "REVE"),
        ("EEGMamba", None),
        ("BENDR", "BENDR*"),
    ]:
        add_foundation(model, label)

    for model, label in [("eegnet", "EEGNet"), ("shallow_fbcsp", "ShallowFBCSPNet"), ("deep4", "Deep4Net")]:
        add_basic("PhysioNet MI", "Left vs Right imagery", label, mi, model, "imagined_left_vs_right_fist")

    OUT.mkdir(exist_ok=True)
    csv_path = OUT / "table1_cross_domain_summary.csv"
    md_path = OUT / "table1_cross_domain_summary.md"
    tex_path = OUT / "table1_cross_domain_summary.tex"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    widths = {field: max(len(field), *(len(row[field]) for row in rows)) for field in fieldnames}
    drop_field = "Flattening Drop [95% CI]"
    with md_path.open("w") as f:
        f.write("| " + " | ".join(field.ljust(widths[field]) for field in fieldnames) + " |\n")
        f.write("| " + " | ".join("-" * widths[field] for field in fieldnames) + " |\n")
        prev_domain = None
        for row in rows:
            if prev_domain is not None and row["Domain"] != prev_domain:
                f.write("| " + " | ".join("-" * widths[field] for field in fieldnames) + " |\n")
            display = dict(row)
            display[drop_field] = f"**{row[drop_field]}**"
            f.write("| " + " | ".join(display[field].ljust(widths[field]) for field in fieldnames) + " |\n")
            prev_domain = row["Domain"]
    with tex_path.open("w") as f:
        f.write("\\begin{tabular}{lllccccc}\n\\toprule\n")
        f.write("Domain & Task & Model & Raw BA & Sham BA & Aperiodic BA & Flattened BA & Flattening Drop \\\\\n\\midrule\n")
        prev_domain = None
        for row in rows:
            if prev_domain is not None and row["Domain"] != prev_domain:
                f.write("\\midrule\n")
            vals = [row[field].replace("%", "\\%") for field in fieldnames]
            vals[-1] = f"\\textbf{{{vals[-1]}}}"
            f.write(" & ".join(vals) + " \\\\\n")
            prev_domain = row["Domain"]
        f.write("\\bottomrule\n\\end{tabular}\n")


def readme() -> None:
    text = """# Journal Images

Generated from `code/scripts/generate_nmi_journal_images.py`.

Files:
- `fig1_spectral_audit_framework.*`: deterministic representative schematic for the decomposition/intervention pipeline. The local folder does not contain the heavy raw Sleep-EDF epoch caches referenced in the planning document.
- `fig2_cross_domain_flattening_drop.*`: cross-domain flattening-drop hero figure from saved bootstrap CSVs.
- `fig3_sleep_task_dissociation.*`: Sleep-EDF EEGNet task-specific intervention results from saved bootstrap CSVs.
- `fig4_tuab_clinical_benchmark_audit.*`: TUAB PSD, age-matched, neural, and foundation-model audit panels from saved CSVs.
- `fig5_validation_and_controls.*`: sham control, simulation validation, IRASA agreement, and train-on-representation controls from saved CSVs.
- `table1_cross_domain_summary.*`: compact display table for the headline cross-domain numbers.

No dataset files are modified by this script.
"""
    (OUT / "README.md").write_text(text)


def main() -> None:
    setup_style()
    OUT.mkdir(exist_ok=True)
    fig1_pipeline()
    fig2_cross_domain()
    fig3_sleep_tasks()
    fig4_tuab_audit()
    fig5_validation_controls()
    table1_summary()
    readme()
    print(f"Generated figures and table in: {OUT}")


if __name__ == "__main__":
    main()
