#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path


TASK_LABELS = {
    "wake_vs_sleep": "Wake vs Sleep",
    "n2_vs_n3": "N2 vs N3",
    "five_stage": "Five-stage",
}

MODEL_LABELS = {
    "linear_ridge": "Linear ridge PSD",
    "deep_mlp": "Deep MLP PSD",
    "raw_cnn": "Raw EEG CNN",
    "braindecode_eegnet": "Braindecode EEGNet",
}

CANONICAL_LABELS = {
    "baseline": "Original",
    "aperiodic": "Aperiodic",
    "flattened": "Flattened",
}

CHANCE = {
    "wake_vs_sleep": 0.5,
    "n2_vs_n3": 0.5,
    "five_stage": 0.2,
}

MODEL_ORDER = ["linear_ridge", "deep_mlp", "raw_cnn", "braindecode_eegnet"]
TASK_ORDER = ["wake_vs_sleep", "n2_vs_n3", "five_stage"]
CANONICAL_ORDER = ["baseline", "aperiodic", "flattened"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create combined paper-style tables and figures for intervention CIs."
    )
    parser.add_argument(
        "--bootstrap-csv",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/reports/tables/intervention_bootstrap_ci.csv"),
    )
    parser.add_argument(
        "--figure-png",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/reports/figures/combined_intervention_performance.png"),
    )
    parser.add_argument(
        "--figure-pdf",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/reports/figures/combined_intervention_performance.pdf"),
    )
    parser.add_argument(
        "--table-csv",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/reports/tables/combined_intervention_summary.csv"),
    )
    parser.add_argument(
        "--table-md",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/reports/tables/combined_intervention_summary.md"),
    )
    return parser.parse_args()


def direct_estimate_kind(row) -> str | None:
    if row["estimate"] == row["baseline_input"]:
        return "baseline"
    if row["estimate"] == row["aperiodic_input"]:
        return "aperiodic"
    if row["estimate"] == row["flattened_input"]:
        return "flattened"
    return None


def ci_text(point: float, low: float, high: float) -> str:
    return f"{point:.3f} [{low:.3f}, {high:.3f}]"


def write_markdown_table(path: Path, rows, columns) -> None:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in rows.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def make_summary_table(df):
    import pandas as pd

    ba = df[df["metric"] == "balanced_accuracy"].copy()
    ba["kind"] = ba.apply(direct_estimate_kind, axis=1)

    rows = []
    for model in MODEL_ORDER:
        for task in TASK_ORDER:
            sub = ba[(ba["model"] == model) & (ba["task"] == task)]
            if sub.empty:
                continue
            out = {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "task": task,
                "task_label": TASK_LABELS[task],
            }
            for kind in CANONICAL_ORDER:
                row = sub[sub["kind"] == kind]
                if row.empty:
                    continue
                row = row.iloc[0]
                out[f"{kind}_point"] = float(row["point"])
                out[f"{kind}_ci_lower"] = float(row["ci_lower"])
                out[f"{kind}_ci_upper"] = float(row["ci_upper"])
                out[f"{kind}_ci"] = ci_text(
                    float(row["point"]),
                    float(row["ci_lower"]),
                    float(row["ci_upper"]),
                )
            for estimate in ["retention_aperiodic", "retention_flattened", "drop_aperiodic", "drop_flattened"]:
                row = sub[sub["estimate"] == estimate]
                if row.empty:
                    continue
                row = row.iloc[0]
                out[f"{estimate}_point"] = float(row["point"])
                out[f"{estimate}_ci_lower"] = float(row["ci_lower"])
                out[f"{estimate}_ci_upper"] = float(row["ci_upper"])
                out[f"{estimate}_ci"] = ci_text(
                    float(row["point"]),
                    float(row["ci_lower"]),
                    float(row["ci_upper"]),
                )
            rows.append(out)
    return pd.DataFrame(rows)


def main() -> int:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    args = parse_args()
    df = pd.read_csv(args.bootstrap_csv)
    summary = make_summary_table(df)

    numeric_cols = [
        "model",
        "model_label",
        "task",
        "task_label",
        "baseline_point",
        "baseline_ci_lower",
        "baseline_ci_upper",
        "aperiodic_point",
        "aperiodic_ci_lower",
        "aperiodic_ci_upper",
        "flattened_point",
        "flattened_ci_lower",
        "flattened_ci_upper",
        "retention_aperiodic_point",
        "retention_aperiodic_ci_lower",
        "retention_aperiodic_ci_upper",
        "drop_flattened_point",
        "drop_flattened_ci_lower",
        "drop_flattened_ci_upper",
    ]
    args.table_csv.parent.mkdir(parents=True, exist_ok=True)
    summary[numeric_cols].to_csv(args.table_csv, index=False)

    md_cols = [
        "model_label",
        "task_label",
        "baseline_ci",
        "aperiodic_ci",
        "flattened_ci",
        "retention_aperiodic_ci",
        "drop_flattened_ci",
    ]
    md = summary[md_cols].copy()
    md = md.rename(
        columns={
            "model_label": "Model",
            "task_label": "Task",
            "baseline_ci": "Original BA",
            "aperiodic_ci": "Aperiodic BA",
            "flattened_ci": "Flattened BA",
            "retention_aperiodic_ci": "Aperiodic retention",
            "drop_flattened_ci": "Flattening drop",
        }
    )
    write_markdown_table(args.table_md, md, list(md.columns))

    colors = {
        "baseline": "#2f5d8c",
        "aperiodic": "#3f8f6b",
        "flattened": "#b65f5a",
    }
    fig, axes = plt.subplots(4, 3, figsize=(11.4, 10.8), sharey=False)
    for row_idx, model in enumerate(MODEL_ORDER):
        for col_idx, task in enumerate(TASK_ORDER):
            ax = axes[row_idx, col_idx]
            row = summary[(summary["model"] == model) & (summary["task"] == task)].iloc[0]
            values = np.array([row[f"{kind}_point"] for kind in CANONICAL_ORDER], dtype=float)
            lower = np.array([row[f"{kind}_ci_lower"] for kind in CANONICAL_ORDER], dtype=float)
            upper = np.array([row[f"{kind}_ci_upper"] for kind in CANONICAL_ORDER], dtype=float)
            yerr = np.vstack([values - lower, upper - values])
            x = np.arange(len(CANONICAL_ORDER))
            ax.bar(
                x,
                values,
                yerr=yerr,
                color=[colors[kind] for kind in CANONICAL_ORDER],
                edgecolor="#222222",
                linewidth=0.7,
                capsize=3,
            )
            ax.axhline(CHANCE[task], color="#555555", linestyle="--", linewidth=1.0)
            ax.set_ylim(0.0, 1.0)
            ax.grid(axis="y", alpha=0.25, linewidth=0.7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if row_idx == 0:
                ax.set_title(TASK_LABELS[task], fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(f"{MODEL_LABELS[model]}\nBalanced accuracy")
            else:
                ax.set_ylabel("")
            ax.set_xticks(x)
            if row_idx == len(MODEL_ORDER) - 1:
                ax.set_xticklabels([CANONICAL_LABELS[kind] for kind in CANONICAL_ORDER], rotation=25, ha="right")
            else:
                ax.set_xticklabels([])

    fig.tight_layout()
    args.figure_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure_png, dpi=300, bbox_inches="tight")
    fig.savefig(args.figure_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote figure: {args.figure_png}")
    print(f"Wrote figure: {args.figure_pdf}")
    print(f"Wrote table: {args.table_csv}")
    print(f"Wrote table: {args.table_md}")
    print(md.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
