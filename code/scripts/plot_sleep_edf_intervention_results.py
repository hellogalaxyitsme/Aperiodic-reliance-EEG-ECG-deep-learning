#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path


TASK_LABELS = {
    "wake_vs_sleep": "Wake vs Sleep",
    "n2_vs_n3": "N2 vs N3",
    "five_stage": "Five-stage",
}

INPUT_LABELS = {
    "full_log_psd": "Full",
    "aperiodic_spectrum": "Aperiodic-only",
    "flattened_log_psd": "Flattened",
}

CHANCE = {
    "wake_vs_sleep": 0.5,
    "n2_vs_n3": 0.5,
    "five_stage": 0.2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create paper-style tables and figures for Sleep-EDF interventions."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path(
            "/mnt/data/aperiodic_confounds/results/sleep_edf_subset/interventions_specparam_flattening/intervention_summary_metrics.csv"
        ),
    )
    parser.add_argument(
        "--figure-png",
        type=Path,
        default=Path(
            "/mnt/data/aperiodic_confounds/reports/figures/sleep_edf_intervention_performance.png"
        ),
    )
    parser.add_argument(
        "--figure-pdf",
        type=Path,
        default=Path(
            "/mnt/data/aperiodic_confounds/reports/figures/sleep_edf_intervention_performance.pdf"
        ),
    )
    parser.add_argument(
        "--table-csv",
        type=Path,
        default=Path(
            "/mnt/data/aperiodic_confounds/reports/tables/sleep_edf_intervention_summary.csv"
        ),
    )
    parser.add_argument(
        "--table-md",
        type=Path,
        default=Path(
            "/mnt/data/aperiodic_confounds/reports/tables/sleep_edf_intervention_summary.md"
        ),
    )
    return parser.parse_args()


def write_markdown_table(path: Path, rows, columns) -> None:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in rows.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    args = parse_args()
    df = pd.read_csv(args.summary_csv)
    direct = df[df["test_input"].isin(INPUT_LABELS)].copy()
    direct["task_label"] = direct["task"].map(TASK_LABELS)
    direct["input_label"] = direct["test_input"].map(INPUT_LABELS)
    direct["chance"] = direct["task"].map(CHANCE)

    drop = df[df["test_input"].astype(str).str.startswith("drop::")].copy()
    drop["test_input_clean"] = drop["test_input"].str.replace("drop::", "", regex=False)
    retention = df[df["test_input"].astype(str).str.startswith("retention::")].copy()
    retention["test_input_clean"] = retention["test_input"].str.replace(
        "retention::", "", regex=False
    )

    table = direct.pivot(
        index="task",
        columns="test_input",
        values="balanced_accuracy_mean",
    ).reset_index()
    ret_table = retention.pivot(
        index="task",
        columns="test_input_clean",
        values="balanced_accuracy_mean",
    ).reset_index()
    ret_table = ret_table.rename(
        columns={
            "aperiodic_spectrum": "aperiodic_retention",
            "flattened_log_psd": "flattened_retention",
        }
    )
    drop_table = drop.pivot(
        index="task",
        columns="test_input_clean",
        values="balanced_accuracy_mean",
    ).reset_index()
    drop_table = drop_table.rename(
        columns={
            "aperiodic_spectrum": "drop_to_aperiodic",
            "flattened_log_psd": "drop_flat",
        }
    )

    summary = table.merge(ret_table, on="task").merge(drop_table, on="task")
    summary.insert(1, "task_label", summary["task"].map(TASK_LABELS))
    columns = [
        "task",
        "task_label",
        "full_log_psd",
        "aperiodic_spectrum",
        "flattened_log_psd",
        "aperiodic_retention",
        "flattened_retention",
        "drop_flat",
    ]
    summary = summary[columns]
    args.table_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.table_csv, index=False)

    rounded = summary.copy()
    for col in rounded.columns:
        if col not in {"task", "task_label"}:
            rounded[col] = rounded[col].map(lambda val: f"{val:.3f}")
    write_markdown_table(args.table_md, rounded, list(rounded.columns))

    task_order = ["wake_vs_sleep", "n2_vs_n3", "five_stage"]
    input_order = ["full_log_psd", "aperiodic_spectrum", "flattened_log_psd"]
    colors = {
        "full_log_psd": "#2f5d8c",
        "aperiodic_spectrum": "#3f8f6b",
        "flattened_log_psd": "#b65f5a",
    }

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=False)
    for ax, task in zip(axes, task_order):
        sub = direct[direct["task"] == task].set_index("test_input").loc[input_order]
        x = np.arange(len(input_order))
        values = sub["balanced_accuracy_mean"].to_numpy()
        errors = sub["balanced_accuracy_std"].fillna(0.0).to_numpy()
        ax.bar(
            x,
            values,
            yerr=errors,
            color=[colors[item] for item in input_order],
            edgecolor="#222222",
            linewidth=0.7,
            capsize=3,
        )
        ax.axhline(CHANCE[task], color="#555555", linestyle="--", linewidth=1.0)
        ax.set_title(TASK_LABELS[task], fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([INPUT_LABELS[item] for item in input_order], rotation=25, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Balanced accuracy")
    fig.tight_layout()
    args.figure_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure_png, dpi=300, bbox_inches="tight")
    fig.savefig(args.figure_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote figure: {args.figure_png}")
    print(f"Wrote figure: {args.figure_pdf}")
    print(f"Wrote table: {args.table_csv}")
    print(f"Wrote table: {args.table_md}")
    print(rounded.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

