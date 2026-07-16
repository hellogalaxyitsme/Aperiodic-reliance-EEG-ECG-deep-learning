#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a compact table for intervention/flattening results."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path(
            "results/sleep_edf_subset/interventions_specparam_flattening/intervention_summary_metrics.csv"
        ),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "results/sleep_edf_subset/interventions_specparam_flattening/intervention_summary_table.md"
        ),
    )
    return parser.parse_args()


def main() -> int:
    import pandas as pd

    args = parse_args()
    df = pd.read_csv(args.summary_csv)

    direct = df[
        df["test_input"].isin(["full_log_psd", "aperiodic_spectrum", "flattened_log_psd"])
    ].copy()
    pivot = direct.pivot(
        index="task",
        columns="test_input",
        values="balanced_accuracy_mean",
    ).reset_index()

    retention = df[df["test_input"].astype(str).str.startswith("retention::")].copy()
    retention["test_input"] = retention["test_input"].str.replace("retention::", "", regex=False)
    ret_pivot = retention.pivot(
        index="task",
        columns="test_input",
        values="balanced_accuracy_mean",
    ).reset_index()
    ret_pivot = ret_pivot.rename(
        columns={
            "aperiodic_spectrum": "aperiodic_retention",
            "flattened_log_psd": "flattened_retention",
        }
    )

    drops = df[df["test_input"].astype(str).str.startswith("drop::")].copy()
    drops["test_input"] = drops["test_input"].str.replace("drop::", "", regex=False)
    drop_pivot = drops.pivot(
        index="task",
        columns="test_input",
        values="balanced_accuracy_mean",
    ).reset_index()
    drop_pivot = drop_pivot.rename(
        columns={
            "aperiodic_spectrum": "drop_aperiodic_only",
            "flattened_log_psd": "drop_flattened",
        }
    )

    table = pivot.merge(ret_pivot, on="task").merge(drop_pivot, on="task")
    column_order = [
        "task",
        "full_log_psd",
        "aperiodic_spectrum",
        "flattened_log_psd",
        "aperiodic_retention",
        "flattened_retention",
        "drop_flattened",
    ]
    table = table[column_order]

    rounded = table.copy()
    for col in rounded.columns:
        if col != "task":
            rounded[col] = rounded[col].map(lambda val: f"{val:.3f}")

    headers = list(rounded.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in rounded.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n")
    print(rounded.to_string(index=False))
    print(f"Wrote: {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
