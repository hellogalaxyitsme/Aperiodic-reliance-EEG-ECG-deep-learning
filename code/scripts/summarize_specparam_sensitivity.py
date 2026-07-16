#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize specparam sensitivity outputs.")
    parser.add_argument(
        "--sensitivity-root",
        type=Path,
        default=Path("results/sleep_edf_subset/specparam_sensitivity"),
    )
    parser.add_argument(
        "--main-baseline-dir",
        type=Path,
        default=Path("results/sleep_edf_subset/baselines_specparam"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/sleep_edf_subset/specparam_sensitivity/sensitivity_summary.csv"),
    )
    return parser.parse_args()


def load_setting_rows(setting_name: str, baseline_dir: Path, diagnostic_path: Path):
    import pandas as pd

    summary_path = baseline_dir / "summary_metrics.csv"
    if not summary_path.exists():
        return []
    metrics = pd.read_csv(summary_path)
    direct = metrics[~metrics["feature_set"].astype(str).str.startswith("retention::")].copy()

    diagnostics = {}
    if diagnostic_path.exists():
        diagnostics = json.loads(diagnostic_path.read_text())

    rows = []
    for _, row in direct.iterrows():
        rows.append(
            {
                "setting": setting_name,
                "task": row["task"],
                "feature_set": row["feature_set"],
                "balanced_accuracy_mean": row["balanced_accuracy_mean"],
                "balanced_accuracy_std": row["balanced_accuracy_std"],
                "macro_f1_mean": row["macro_f1_mean"],
                "mean_r_squared": diagnostics.get("mean_r_squared"),
                "median_r_squared": diagnostics.get("median_r_squared"),
                "low_r2_fraction": diagnostics.get("low_r2_fraction"),
                "mean_n_peaks": diagnostics.get("mean_n_peaks"),
                "peak_cap_fraction": diagnostics.get("peak_cap_fraction"),
            }
        )

    retention = metrics[metrics["feature_set"].astype(str).str.startswith("retention::")].copy()
    for _, row in retention.iterrows():
        rows.append(
            {
                "setting": setting_name,
                "task": row["task"],
                "feature_set": row["feature_set"],
                "balanced_accuracy_mean": row["balanced_accuracy_mean"],
                "balanced_accuracy_std": row.get("balanced_accuracy_std"),
                "macro_f1_mean": row["macro_f1_mean"],
                "mean_r_squared": diagnostics.get("mean_r_squared"),
                "median_r_squared": diagnostics.get("median_r_squared"),
                "low_r2_fraction": diagnostics.get("low_r2_fraction"),
                "mean_n_peaks": diagnostics.get("mean_n_peaks"),
                "peak_cap_fraction": diagnostics.get("peak_cap_fraction"),
            }
        )
    return rows


def main() -> int:
    import pandas as pd

    args = parse_args()
    rows = []

    main_diag = Path("results/sleep_edf_subset/specparam/diagnostics/specparam_diagnostics_summary.json")
    rows.extend(load_setting_rows("main_p6_h010", args.main_baseline_dir, main_diag))

    for setting_dir in sorted(args.sensitivity_root.iterdir() if args.sensitivity_root.exists() else []):
        if not setting_dir.is_dir():
            continue
        rows.extend(
            load_setting_rows(
                setting_dir.name,
                setting_dir / "baselines",
                setting_dir / "diagnostics" / "specparam_diagnostics_summary.json",
            )
        )

    if not rows:
        raise SystemExit(f"No sensitivity rows found under {args.sensitivity_root}")

    out = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Wrote: {args.output_csv}")

    direct = out[~out["feature_set"].astype(str).str.startswith("retention::")]
    print(
        direct[
            [
                "setting",
                "task",
                "feature_set",
                "balanced_accuracy_mean",
                "mean_r_squared",
                "peak_cap_fraction",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

