#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize specparam fit diagnostics.")
    parser.add_argument(
        "--decomp-npz",
        type=Path,
        default=Path("results/sleep_edf_subset/specparam/specparam_fixed.npz"),
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=Path("results/sleep_edf_subset/psd_index.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/sleep_edf_subset/specparam/diagnostics"),
    )
    parser.add_argument("--r2-threshold", type=float, default=0.90)
    parser.add_argument("--exponent-min", type=float, default=0.0)
    parser.add_argument("--exponent-max", type=float, default=6.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import numpy as np
    import pandas as pd

    decomp = np.load(args.decomp_npz)
    index = pd.read_csv(args.index_csv)
    channels = [str(ch) for ch in decomp["channels"].tolist()]
    settings = json.loads(str(decomp["settings_json"])) if "settings_json" in decomp else {}
    max_n_peaks = int(settings.get("max_n_peaks", np.nanmax(decomp["n_peaks"])))

    if len(index) != decomp["r_squared"].shape[0]:
        raise ValueError(
            f"Index rows {len(index)} != decomposition epochs {decomp['r_squared'].shape[0]}"
        )

    rows = []
    for channel_idx, channel in enumerate(channels):
        frame = index[["psd_index", "recording", "subject", "night", "stage"]].copy()
        frame["channel"] = channel
        frame["r_squared"] = decomp["r_squared"][:, channel_idx]
        frame["error_mae"] = decomp["error_mae"][:, channel_idx]
        frame["offset"] = decomp["offset"][:, channel_idx]
        frame["exponent"] = decomp["exponent"][:, channel_idx]
        frame["n_peaks"] = decomp["n_peaks"][:, channel_idx]
        frame["fit_ok"] = decomp["ok"][:, channel_idx]
        rows.append(frame)
    long_df = pd.concat(rows, ignore_index=True)

    long_df["low_r2"] = long_df["r_squared"] < args.r2_threshold
    long_df["peak_cap"] = long_df["n_peaks"] >= max_n_peaks
    long_df["bad_exponent"] = (
        (long_df["exponent"] < args.exponent_min)
        | (long_df["exponent"] > args.exponent_max)
    )

    group_cols = ["stage", "channel"]
    by_stage_channel = (
        long_df.groupby(group_cols)
        .agg(
            n=("r_squared", "size"),
            r2_mean=("r_squared", "mean"),
            r2_median=("r_squared", "median"),
            error_mae_mean=("error_mae", "mean"),
            exponent_mean=("exponent", "mean"),
            exponent_std=("exponent", "std"),
            offset_mean=("offset", "mean"),
            n_peaks_mean=("n_peaks", "mean"),
            low_r2_fraction=("low_r2", "mean"),
            peak_cap_fraction=("peak_cap", "mean"),
            bad_exponent_fraction=("bad_exponent", "mean"),
            fit_ok_fraction=("fit_ok", "mean"),
        )
        .reset_index()
    )

    by_recording = (
        long_df.groupby(["recording", "channel"])
        .agg(
            n=("r_squared", "size"),
            r2_mean=("r_squared", "mean"),
            r2_median=("r_squared", "median"),
            exponent_mean=("exponent", "mean"),
            n_peaks_mean=("n_peaks", "mean"),
            low_r2_fraction=("low_r2", "mean"),
            peak_cap_fraction=("peak_cap", "mean"),
        )
        .reset_index()
    )

    overall = {
        "decomp_npz": str(args.decomp_npz),
        "index_csv": str(args.index_csv),
        "settings": settings,
        "n_epochs": int(len(index)),
        "n_channels": int(len(channels)),
        "r2_threshold": float(args.r2_threshold),
        "max_n_peaks": max_n_peaks,
        "fit_ok_fraction": float(long_df["fit_ok"].mean()),
        "mean_r_squared": float(long_df["r_squared"].mean()),
        "median_r_squared": float(long_df["r_squared"].median()),
        "low_r2_fraction": float(long_df["low_r2"].mean()),
        "mean_error_mae": float(long_df["error_mae"].mean()),
        "mean_n_peaks": float(long_df["n_peaks"].mean()),
        "peak_cap_fraction": float(long_df["peak_cap"].mean()),
        "bad_exponent_fraction": float(long_df["bad_exponent"].mean()),
        "exponent_mean": float(long_df["exponent"].mean()),
        "exponent_std": float(long_df["exponent"].std()),
        "offset_mean": float(long_df["offset"].mean()),
        "offset_std": float(long_df["offset"].std()),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(args.output_dir / "specparam_epoch_channel_diagnostics.csv", index=False)
    by_stage_channel.to_csv(args.output_dir / "specparam_by_stage_channel.csv", index=False)
    by_recording.to_csv(args.output_dir / "specparam_by_recording.csv", index=False)
    (args.output_dir / "specparam_diagnostics_summary.json").write_text(
        json.dumps(overall, indent=2)
    )

    print(json.dumps(overall, indent=2))
    print(f"Wrote diagnostics to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

