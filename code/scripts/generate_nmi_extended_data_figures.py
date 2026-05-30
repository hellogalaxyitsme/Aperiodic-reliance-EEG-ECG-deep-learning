#!/usr/bin/env python3
"""Generate Extended Data figures for the NMI manuscript draft."""

from __future__ import annotations

import csv
import json
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
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Journal Images" / "Extended Data Figures"

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
    "mi_neural": ROOT / "reports/tables/physionet_mi_multiseed_neural_subject_bootstrap.csv",
    "mi_psd": ROOT / "results/physionet_mi/psd_interventions/psd_intervention_subject_bootstrap.csv",
    "tuab_specparam_summary": ROOT / "results/tuab_full_v3_0_1/preprocess_20s_100hz/specparam/specparam_fixed_20s.summary.json",
    "raw_diag": ROOT / "reports/tables/raw_intervention_diagnostics/raw_intervention_distribution_diagnostics.csv",
    "tuab_psd": ROOT / "results/tuab_full_v3_0_1/psd_interventions_specparam/tuab_psd_intervention_subject_bootstrap.csv",
    "tuab_neural": ROOT / "reports/tables/tuab_full_multiseed_neural_subject_bootstrap.csv",
    "tuab_matched_neural": ROOT / "reports/tables/tuab_full_age_matched_multiseed_neural_subject_bootstrap.csv",
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


def fval(value, default=math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except ValueError:
        return default


def row_basic(rows, model, task, estimate, metric="balanced_accuracy"):
    for row in rows:
        if (
            row.get("model") == model
            and row.get("task") == task
            and row.get("estimate") == estimate
            and row.get("metric") == metric
        ):
            return row
    raise KeyError((model, task, estimate, metric))


def row_matrix(rows, train_input, test_input, estimate="performance", metric="balanced_accuracy"):
    for row in rows:
        if (
            row.get("train_input") == train_input
            and row.get("test_input") == test_input
            and row.get("estimate") == estimate
            and row.get("metric") == metric
        ):
            return row
    raise KeyError((train_input, test_input, estimate, metric))


def point_ci(row):
    return fval(row["point"]), fval(row["ci_lower"]), fval(row["ci_upper"])


def fmt_ci(row) -> str:
    p, lo, hi = point_ci(row)
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


def clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, width=0.7)


def panel_label(ax, label):
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")


def savefig(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_representative_signal():
    rng = np.random.default_rng(17)
    sfreq = 100
    t = np.arange(0, 5, 1 / sfreq)
    raw = (
        0.6 * np.sin(2 * np.pi * 10 * t + 0.3)
        + 0.35 * np.sin(2 * np.pi * 2.2 * t)
        + 0.15 * rng.normal(size=t.size)
    )
    fft = np.fft.rfft(raw)
    sham = np.fft.irfft(fft, n=t.size).real
    return t, raw, sham


def ed1_sleep_full_results() -> None:
    sleep_rev = read_csv(PATHS["sleep_reviewer"])
    sleep_full = read_csv(PATHS["sleep_full"])
    tasks = [("Wake vs Sleep", "wake_vs_sleep", 0.5), ("N2 vs N3", "n2_vs_n3", 0.5), ("Five-stage", "five_stage", 0.2)]
    models = [
        ("CNN", "raw_cnn_sham", sleep_rev),
        ("EEGNet", "braindecode_eegnet", sleep_rev),
        ("Shallow", "braindecode_shallow_fbcsp", sleep_rev),
        ("Deep4Net", "braindecode_deep4", sleep_rev),
        ("MLP", "deep_mlp", sleep_full),
    ]
    conds = [("Raw", "baseline", "raw"), ("Sham", "sham", "sham"), ("Aperiodic", "aperiodic", "aperiodic"), ("Flattened", "flattened", "flattened")]
    fig, axes = plt.subplots(len(tasks), len(models), figsize=(11.8, 6.4), sharey="row")
    for r, (task_label, task, chance) in enumerate(tasks):
        for c, (model_label, model, rows) in enumerate(models):
            ax = axes[r, c]
            x = np.arange(len(conds))
            vals = []
            cis = []
            for _, est, _ in conds:
                try:
                    row = row_basic(rows, model, task, est)
                    vals.append(point_ci(row)[0])
                    cis.append(point_ci(row))
                except KeyError:
                    vals.append(np.nan)
                    cis.append((np.nan, np.nan, np.nan))
            ax.bar(x, np.nan_to_num(vals), color=[COLORS[k] for _, _, k in conds], edgecolor="#222222", linewidth=0.4, width=0.68)
            for i, (p, lo, hi) in enumerate(cis):
                if math.isfinite(p):
                    ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], fmt="none", color="#222222", elinewidth=0.6, capsize=1.8)
                else:
                    ax.text(i, 0.05, "NA", ha="center", va="bottom", fontsize=6, rotation=90)
            ax.axhline(chance, color="#777777", linestyle="--", linewidth=0.6)
            ax.set_ylim(0, 1.02)
            if r == 0:
                ax.text(0.5, 1.04, model_label, transform=ax.transAxes, ha="center", va="bottom", fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{task_label}\nBalanced accuracy")
            if r == len(tasks) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels([a for a, _, _ in conds], rotation=45, ha="right")
            else:
                ax.set_xticks([])
            clean_ax(ax)
    panel_label(axes[0, 0], "a")
    savefig(fig, "ed_fig1_sleep_edf_all_tasks_architectures")


def ed2_sleep_n2n3_drop() -> None:
    rows = read_csv(PATHS["sleep_reviewer"])
    full = read_csv(PATHS["sleep_full"])
    raw_models = [
        ("CNN", "raw_cnn_sham", rows),
        ("EEGNet", "braindecode_eegnet", rows),
        ("Shallow", "braindecode_shallow_fbcsp", rows),
        ("Deep4Net", "braindecode_deep4", rows),
    ]
    context_models = [("PSD train-control", "psd_train_control", rows), ("MLP", "deep_mlp", full)]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3), gridspec_kw={"width_ratios": [1.5, 0.85]}, sharey=True)
    for ax, models, title in zip(axes, [raw_models, context_models], ["Raw EEG architectures", "PSD/MLP context"]):
        x = np.arange(len(models))
        for i, (_, model, source) in enumerate(models):
            p, lo, hi = point_ci(row_basic(source, model, "n2_vs_n3", "drop_flattened"))
            color = COLORS["flattened"] if p <= 0 else COLORS["black"]
            ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], fmt="o", color=color, ecolor=color, capsize=2.5, markersize=5)
        ax.axhline(0, color="#777777", linestyle="--", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([m[0] for m in models], rotation=30, ha="right")
        ax.text(0.5, 1.04, title, transform=ax.transAxes, ha="center", va="bottom", fontweight="bold")
        clean_ax(ax)
    axes[0].set_ylabel("N2 vs N3 flattening drop")
    axes[0].set_ylim(-0.06, 0.45)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    savefig(fig, "ed_fig2_sleep_n2_vs_n3_flattening_drops")


def ed3_physionet_mi_full() -> None:
    neural = read_csv(PATHS["mi_neural"])
    psd = read_csv(PATHS["mi_psd"])
    fig = plt.figure(figsize=(10.8, 4.2))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.35, 1.0], wspace=0.35)
    ax_a = fig.add_subplot(gs[0, 0])
    models = [("EEGNet", "eegnet"), ("Shallow", "shallow_fbcsp"), ("Deep4Net", "deep4")]
    conds = [("Raw", "baseline", "raw"), ("Sham", "sham", "sham"), ("Aperiodic", "aperiodic", "aperiodic"), ("Flattened", "flattened", "flattened")]
    x = np.arange(len(models))
    width = 0.18
    for j, (label, est, color_key) in enumerate(conds):
        vals, los, his = [], [], []
        for _, model in models:
            p, lo, hi = point_ci(row_basic(neural, model, "imagined_left_vs_right_fist", est))
            vals.append(p)
            los.append(lo)
            his.append(hi)
        xpos = x + (j - 1.5) * width
        ax_a.bar(xpos, vals, width=width, color=COLORS[color_key], edgecolor="#222222", linewidth=0.4, label=label)
        ax_a.errorbar(xpos, vals, yerr=[np.array(vals) - np.array(los), np.array(his) - np.array(vals)], fmt="none", color="#222222", elinewidth=0.6, capsize=1.8)
    ax_a.axhline(0.5, color="#777777", linestyle="--", linewidth=0.7)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([m[0] for m in models])
    ax_a.set_ylabel("Balanced accuracy")
    ax_a.set_ylim(0.48, 0.78)
    ax_a.legend(frameon=False, ncol=2, loc="upper right")
    clean_ax(ax_a)
    panel_label(ax_a, "a")

    ax_b = fig.add_subplot(gs[0, 1])
    rep = ["full_log_psd", "aperiodic_spectrum", "flattened_log_psd"]
    labels = ["Full", "Aperiodic", "Flattened"]
    mat = np.empty((3, 3))
    for i, train in enumerate(rep):
        for j, test in enumerate(rep):
            mat[i, j] = point_ci(row_matrix(psd, train, test))[0]
    im = ax_b.imshow(mat, vmin=0.49, vmax=0.58, cmap="viridis", aspect="auto")
    ax_b.set_xticks(np.arange(3))
    ax_b.set_xticklabels(labels, rotation=30, ha="right")
    ax_b.set_yticks(np.arange(3))
    ax_b.set_yticklabels(labels)
    ax_b.set_xlabel("Test input")
    ax_b.set_ylabel("Train input")
    for i in range(3):
        for j in range(3):
            ax_b.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="white" if mat[i, j] < 0.535 else "black", fontsize=7)
    cbar = fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.03)
    cbar.set_label("Balanced accuracy", fontsize=7)
    panel_label(ax_b, "b")
    savefig(fig, "ed_fig3_physionet_mi_full_results")


def ed4_tuab_specparam_qc() -> None:
    summary = json.loads(PATHS["tuab_specparam_summary"].read_text())
    fig = plt.figure(figsize=(10.0, 6.7))
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

    ax_a = fig.add_subplot(gs[0, 0])
    labels = ["p10", "mean", "median"]
    vals = [summary["p10_r_squared"], summary["mean_r_squared"], summary["median_r_squared"]]
    ax_a.bar(np.arange(3), vals, color=COLORS["raw"], edgecolor="#222222", linewidth=0.5)
    ax_a.set_xticks(np.arange(3))
    ax_a.set_xticklabels(labels)
    ax_a.set_ylabel("SpecParam R2")
    ax_a.set_ylim(0.88, 1.0)
    ax_a.text(0.02, 0.04, f"n={summary['shape'][0]:,} epochs x {summary['shape'][1]} channels", transform=ax_a.transAxes, fontsize=7)
    clean_ax(ax_a)
    panel_label(ax_a, "a")

    ax_b = fig.add_subplot(gs[0, 1])
    vals = [summary["mean_exponent"], summary["median_exponent"]]
    ax_b.bar(np.arange(2), vals, color=COLORS["aperiodic"], edgecolor="#222222", linewidth=0.5)
    ax_b.set_xticks(np.arange(2))
    ax_b.set_xticklabels(["Mean", "Median"])
    ax_b.set_ylabel("Aperiodic exponent")
    ax_b.set_ylim(1.35, 1.65)
    clean_ax(ax_b)
    panel_label(ax_b, "b")

    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.bar([0], [summary["mean_n_peaks"]], color=COLORS["foundation"], edgecolor="#222222", linewidth=0.5)
    ax_c.axhline(summary["settings"]["max_n_peaks"], color="#777777", linestyle="--", linewidth=0.8)
    ax_c.set_xticks([0])
    ax_c.set_xticklabels(["Mean peaks"])
    ax_c.set_ylabel("Peaks per spectrum")
    ax_c.set_ylim(0, summary["settings"]["max_n_peaks"] + 0.5)
    ax_c.text(0.03, 0.93, f"max peak cap={summary['settings']['max_n_peaks']}", transform=ax_c.transAxes, va="top", fontsize=7)
    clean_ax(ax_c)
    panel_label(ax_c, "c")

    ax_d = fig.add_subplot(gs[1, 1])
    vals = [summary["mean_error_mae"], summary["median_error_mae"]]
    ax_d.bar(np.arange(2), vals, color=COLORS["sham"], edgecolor="#222222", linewidth=0.5)
    ax_d.set_xticks(np.arange(2))
    ax_d.set_xticklabels(["Mean", "Median"])
    ax_d.set_ylabel("SpecParam MAE")
    ax_d.set_ylim(0, 0.11)
    ax_d.text(0.03, 0.93, f"ok fraction={summary['ok_fraction']:.3f}", transform=ax_d.transAxes, va="top", fontsize=7)
    clean_ax(ax_d)
    panel_label(ax_d, "d")
    savefig(fig, "ed_fig4_tuab_specparam_fit_quality")


def aggregate_raw_diag():
    rows = read_csv(PATHS["raw_diag"])
    out = {}
    for cond in ["raw_eeg", "phase_sham", "phase_aperiodic", "phase_flattened"]:
        for stat in ["rms", "kurtosis", "corr_vs_raw", "rmse_vs_raw"]:
            vals = [r for r in rows if r["test_input"] == cond and r["statistic"] == stat]
            if vals:
                out[(cond, stat)] = {
                    k: float(np.nanmean([fval(v[k]) for v in vals]))
                    for k in ["mean", "median", "p05", "p95"]
                }
    return out


def ed5_raw_intervention_diagnostics() -> None:
    stats = aggregate_raw_diag()
    fig = plt.figure(figsize=(10.2, 6.4))
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.34)

    ax_a = fig.add_subplot(gs[0, 0])
    t, raw, sham = generate_representative_signal()
    ax_a.plot(t, raw, color=COLORS["raw"], linewidth=1.0, label="Raw")
    ax_a.plot(t, sham, color=COLORS["sham"], linewidth=0.8, linestyle="--", label="Sham")
    ax_a.set_xlabel("Time (s)")
    ax_a.set_ylabel("Amplitude (a.u.)")
    ax_a.legend(frameon=False)
    clean_ax(ax_a)
    panel_label(ax_a, "a")

    conditions = [("Raw", "raw_eeg", "raw"), ("Sham", "phase_sham", "sham"), ("Aperiodic", "phase_aperiodic", "aperiodic"), ("Flattened", "phase_flattened", "flattened")]
    for ax, stat, ylabel, label in [
        (fig.add_subplot(gs[0, 1]), "rms", "RMS (uV)", "b"),
        (fig.add_subplot(gs[1, 0]), "kurtosis", "Excess kurtosis", "c"),
    ]:
        x = np.arange(len(conditions))
        med = [stats[(cond, stat)]["median"] for _, cond, _ in conditions]
        lo = [stats[(cond, stat)]["p05"] for _, cond, _ in conditions]
        hi = [stats[(cond, stat)]["p95"] for _, cond, _ in conditions]
        ax.bar(x, med, color=[COLORS[k] for _, _, k in conditions], edgecolor="#222222", linewidth=0.5)
        ax.errorbar(x, med, yerr=[np.array(med) - np.array(lo), np.array(hi) - np.array(med)], fmt="none", color="#222222", elinewidth=0.7, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels([c[0] for c in conditions], rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        clean_ax(ax)
        panel_label(ax, label)

    ax_d = fig.add_subplot(gs[1, 1])
    conds = [("Sham", "phase_sham", "sham"), ("Aperiodic", "phase_aperiodic", "aperiodic"), ("Flattened", "phase_flattened", "flattened")]
    x = np.arange(len(conds))
    corr = [stats[(cond, "corr_vs_raw")]["median"] for _, cond, _ in conds]
    rmse = [stats[(cond, "rmse_vs_raw")]["median"] for _, cond, _ in conds]
    ax_d.plot(x, corr, "o-", color=COLORS["black"], label="corr vs raw")
    ax_d.set_ylabel("Correlation")
    ax_d.set_ylim(0, 1.05)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([c[0] for c in conds], rotation=30, ha="right")
    ax2 = ax_d.twinx()
    ax2.plot(x, rmse, "s--", color=COLORS["foundation"], label="RMSE vs raw")
    ax2.set_ylabel("RMSE (uV)")
    handles = ax_d.get_lines() + ax2.get_lines()
    ax_d.legend(handles, [h.get_label() for h in handles], frameon=False, loc="center left")
    clean_ax(ax_d)
    ax2.spines["top"].set_visible(False)
    panel_label(ax_d, "d")
    savefig(fig, "ed_fig5_raw_intervention_diagnostics")


def ed6_tuab_psd_matrix() -> None:
    rows = read_csv(PATHS["tuab_psd"])
    rep = ["full_log_psd", "aperiodic_spectrum", "flattened_log_psd"]
    labels = ["Full", "Aperiodic", "Flattened"]
    mat = np.empty((3, 3))
    ci_txt = [["" for _ in rep] for _ in rep]
    for i, train in enumerate(rep):
        for j, test in enumerate(rep):
            row = row_matrix(rows, train, test)
            mat[i, j] = point_ci(row)[0]
            ci_txt[i][j] = fmt_ci(row)
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    im = ax.imshow(mat, vmin=0.45, vmax=0.75, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Test input")
    ax.set_ylabel("Train input")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, ci_txt[i][j].replace(" ", "\n", 1), ha="center", va="center", fontsize=7, color="white" if mat[i, j] < 0.60 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Balanced accuracy", fontsize=7)
    savefig(fig, "ed_fig6_tuab_psd_train_test_matrix")


def ed7_specparam_sensitivity() -> None:
    # Logged in experiments.md because the remote sensitivity-grid artifact is not
    # present in the local result folder.
    data = {
        "Wake vs Sleep": {"Aperiodic": (0.894, 0.896), "Residual": (0.916, 0.917)},
        "N2 vs N3": {"Aperiodic": (0.851, 0.858), "Residual": (0.768, 0.772)},
        "Five-stage": {"Aperiodic": (0.527, 0.530), "Residual": (0.636, 0.637)},
    }
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    tasks = list(data.keys())
    x = np.arange(len(tasks))
    offsets = {"Aperiodic": -0.08, "Residual": 0.08}
    colors = {"Aperiodic": COLORS["aperiodic"], "Residual": COLORS["flattened"]}
    for label in ["Aperiodic", "Residual"]:
        mids, lows, highs = [], [], []
        for task in tasks:
            lo, hi = data[task][label]
            mids.append((lo + hi) / 2)
            lows.append(lo)
            highs.append(hi)
        xpos = x + offsets[label]
        ax.errorbar(xpos, mids, yerr=[np.array(mids) - np.array(lows), np.array(highs) - np.array(mids)], fmt="o", color=colors[label], capsize=3, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("Balanced accuracy range across settings")
    ax.set_ylim(0.48, 0.94)
    ax.legend(frameon=False)
    clean_ax(ax)
    savefig(fig, "ed_fig7_sleep_specparam_sensitivity")


def ed8_tuab_age_matched_neural_table() -> None:
    unmatched = read_csv(PATHS["tuab_neural"])
    matched = read_csv(PATHS["tuab_matched_neural"])
    models = [("EEGNet", "eegnet"), ("ShallowFBCSPNet", "shallow_fbcsp"), ("Deep4Net", "deep4")]
    rows_out = []
    for model_label, model in models:
        for cohort, source in [("Unmatched", unmatched), ("Age/sex matched", matched)]:
            rows_out.append(
                [
                    cohort,
                    model_label,
                    fmt_ci(row_basic(source, model, "tuab_normal_vs_abnormal", "baseline")),
                    fmt_ci(row_basic(source, model, "tuab_normal_vs_abnormal", "sham")),
                    fmt_ci(row_basic(source, model, "tuab_normal_vs_abnormal", "aperiodic")),
                    fmt_ci(row_basic(source, model, "tuab_normal_vs_abnormal", "flattened")),
                    fmt_ci(row_basic(source, model, "tuab_normal_vs_abnormal", "drop_flattened")),
                ]
            )
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "ed_fig8_tuab_age_matched_neural_table.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Cohort", "Model", "Raw", "Sham", "Aperiodic", "Flattened", "Drop"])
        writer.writerows(rows_out)

    fig, ax = plt.subplots(figsize=(11.3, 3.1))
    ax.axis("off")
    table = ax.table(
        cellText=rows_out,
        colLabels=["Cohort", "Model", "Raw BA", "Sham BA", "Aperiodic BA", "Flattened BA", "Drop"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#bbbbbb")
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(weight="bold")
        elif c == 6:
            cell.set_text_props(weight="bold")
        elif r in [1, 2, 5, 6]:
            cell.set_facecolor("#fbfbfb")
    savefig(fig, "ed_fig8_tuab_age_matched_neural_results")


def readme() -> None:
    text = """# Extended Data Figures

Generated by `code/scripts/generate_nmi_extended_data_figures.py`.

- `ed_fig1_sleep_edf_all_tasks_architectures.*`: full Sleep-EDF task x architecture intervention results.
- `ed_fig2_sleep_n2_vs_n3_flattening_drops.*`: N2-vs-N3 flattening-drop pattern, with PSD/MLP context.
- `ed_fig3_physionet_mi_full_results.*`: PhysioNet MI neural intervention results and PSD train-test matrix.
- `ed_fig4_tuab_specparam_fit_quality.*`: full-TUAB SpecParam R2, exponent, peak-count, and error summaries from the local full-corpus summary.
- `ed_fig5_raw_intervention_diagnostics.*`: sham overlay schematic plus full Sleep-EDF raw intervention diagnostic summaries.
- `ed_fig6_tuab_psd_train_test_matrix.*`: full-TUAB PSD ridge train-test intervention matrix.
- `ed_fig7_sleep_specparam_sensitivity.*`: Sleep-EDF sensitivity ranges recorded in `experiments.md`; the remote grid CSVs are not local.
- `ed_fig8_tuab_age_matched_neural_results.*`: full-TUAB unmatched vs age/sex-matched neural result table.

No dataset files are modified by this script.
"""
    (OUT / "README.md").write_text(text)


def main() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    ed1_sleep_full_results()
    ed2_sleep_n2n3_drop()
    ed3_physionet_mi_full()
    ed4_tuab_specparam_qc()
    ed5_raw_intervention_diagnostics()
    ed6_tuab_psd_matrix()
    ed7_specparam_sensitivity()
    ed8_tuab_age_matched_neural_table()
    readme()
    print(f"Generated Extended Data figures in: {OUT}")


if __name__ == "__main__":
    main()
