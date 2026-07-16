#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run stratified specparam fit-quality QC for TUAB PSDs."
    )
    parser.add_argument(
        "--psd-npz",
        type=Path,
        default=Path("results/tuab_subset_200/psd_20s_multitaper.npz"),
    )
    parser.add_argument(
        "--psd-index-csv",
        type=Path,
        default=Path("results/tuab_subset_200/psd_20s_multitaper_index.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/tuab_subset_200/specparam_qc_20s_metrics.csv"),
    )
    parser.add_argument(
        "--output-summary-json",
        type=Path,
        default=Path("results/tuab_subset_200/specparam_qc_20s_summary.json"),
    )
    parser.add_argument("--freq-min", type=float, default=1.0)
    parser.add_argument("--freq-max", type=float, default=45.0)
    parser.add_argument("--max-n-peaks", type=int, default=6)
    parser.add_argument("--min-peak-height", type=float, default=0.1)
    parser.add_argument("--peak-threshold", type=float, default=2.0)
    parser.add_argument("--peak-width-min", type=float, default=0.5)
    parser.add_argument("--peak-width-max", type=float, default=8.0)
    parser.add_argument("--epochs-per-split-label", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--n-jobs", type=int, default=16)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fit_one(freqs, spectrum, settings):
    import numpy as np
    from specparam import SpectralModel

    model = SpectralModel(
        aperiodic_mode="fixed",
        peak_width_limits=[settings["peak_width_min"], settings["peak_width_max"]],
        max_n_peaks=settings["max_n_peaks"],
        min_peak_height=settings["min_peak_height"],
        peak_threshold=settings["peak_threshold"],
        verbose=False,
    )
    try:
        model.fit(freqs, spectrum, [settings["freq_min"], settings["freq_max"]])
        aperiodic_params = model.results.params.aperiodic.asdict()["aperiodic_fit"]
        metrics = model.results.metrics.results
        periodic = model.results.params.periodic.asdict()["peak_fit"]
        n_peaks = periodic.shape[0] if periodic.ndim == 2 else 0
        return {
            "ok": True,
            "offset": float(aperiodic_params[0]),
            "exponent": float(aperiodic_params[1]),
            "r_squared": float(metrics["gof_rsquared"]),
            "error_mae": float(metrics["error_mae"]),
            "n_peaks": int(n_peaks),
            "error": "",
        }
    except Exception as exc:
        return {
            "ok": False,
            "offset": np.nan,
            "exponent": np.nan,
            "r_squared": np.nan,
            "error_mae": np.nan,
            "n_peaks": np.nan,
            "error": repr(exc),
        }


def summarize(values: list[float]) -> dict[str, float | int]:
    import numpy as np

    arr = np.array(values, dtype="float64")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "p10": float("nan"), "p90": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def main() -> int:
    args = parse_args()

    import numpy as np
    from joblib import Parallel, delayed

    rows = read_rows(args.psd_index_csv)
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[f"{row['official_split']}/{row['label']}"].append(idx)

    rng = np.random.default_rng(args.seed)
    selected_epoch_indices: list[int] = []
    selected_counts: dict[str, int] = {}
    for group, indices in sorted(groups.items()):
        n = min(len(indices), args.epochs_per_split_label)
        chosen = sorted(rng.choice(indices, size=n, replace=False).tolist())
        selected_epoch_indices.extend(chosen)
        selected_counts[group] = n

    selected_epoch_indices = sorted(selected_epoch_indices)
    selected_index_lookup = {epoch_idx: pos for pos, epoch_idx in enumerate(selected_epoch_indices)}

    bundle = np.load(args.psd_npz)
    freqs = bundle["freqs"]
    channels = [str(ch) for ch in bundle["channels"]]
    psd = bundle["psd"][selected_epoch_indices]

    freq_mask = (freqs >= args.freq_min) & (freqs <= args.freq_max)
    freqs_selected = freqs[freq_mask]
    freqs_fit = np.linspace(
        float(freqs_selected[0]),
        float(freqs_selected[-1]),
        int(freqs_selected.shape[0]),
        dtype="float64",
    )
    psd_fit = psd[:, :, freq_mask]

    settings = {
        "freq_min": float(args.freq_min),
        "freq_max": float(args.freq_max),
        "max_n_peaks": int(args.max_n_peaks),
        "min_peak_height": float(args.min_peak_height),
        "peak_threshold": float(args.peak_threshold),
        "peak_width_min": float(args.peak_width_min),
        "peak_width_max": float(args.peak_width_max),
    }

    tasks = [
        (epoch_idx, channel_idx, psd_fit[selected_index_lookup[epoch_idx], channel_idx])
        for epoch_idx in selected_epoch_indices
        for channel_idx in range(psd_fit.shape[1])
    ]
    print(
        f"Fitting TUAB specparam QC: epochs={len(selected_epoch_indices)}, "
        f"channels={len(channels)}, spectra={len(tasks)}, freqs={len(freqs_fit)}, "
        f"n_jobs={args.n_jobs}",
        flush=True,
    )
    fit_results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(fit_one)(freqs_fit, spectrum, settings)
        for _, _, spectrum in tasks
    )

    out_rows: list[dict[str, object]] = []
    for (epoch_idx, channel_idx, _), fit in zip(tasks, fit_results):
        row = rows[epoch_idx]
        out_rows.append(
            {
                "psd_index": epoch_idx,
                "subject": row["subject"],
                "recording": row["recording"],
                "official_split": row["official_split"],
                "label": row["label"],
                "channel": channels[channel_idx],
                "ok": int(bool(fit["ok"])),
                "offset": fit["offset"],
                "exponent": fit["exponent"],
                "r_squared": fit["r_squared"],
                "error_mae": fit["error_mae"],
                "n_peaks": fit["n_peaks"],
                "error": fit["error"],
            }
        )

    ok_rows = [row for row in out_rows if int(row["ok"]) == 1]
    by_group: dict[str, dict[str, object]] = {}
    for group in sorted(groups):
        group_rows = [
            row
            for row in out_rows
            if f"{row['official_split']}/{row['label']}" == group
        ]
        by_group[group] = {
            "n_spectra": len(group_rows),
            "ok_fraction": (
                sum(int(row["ok"]) for row in group_rows) / len(group_rows)
                if group_rows
                else 0.0
            ),
            "r_squared": summarize([float(row["r_squared"]) for row in group_rows]),
            "error_mae": summarize([float(row["error_mae"]) for row in group_rows]),
            "exponent": summarize([float(row["exponent"]) for row in group_rows]),
            "n_peaks": summarize([float(row["n_peaks"]) for row in group_rows]),
        }

    summary = {
        "psd_npz": str(args.psd_npz),
        "psd_index_csv": str(args.psd_index_csv),
        "output_csv": str(args.output_csv),
        "n_total_epochs": len(rows),
        "n_selected_epochs": len(selected_epoch_indices),
        "n_channels": len(channels),
        "n_spectra": len(out_rows),
        "selected_epoch_counts_by_split_label": selected_counts,
        "settings": settings,
        "seed": int(args.seed),
        "ok_fraction": float(len(ok_rows) / len(out_rows)) if out_rows else 0.0,
        "r_squared": summarize([float(row["r_squared"]) for row in out_rows]),
        "error_mae": summarize([float(row["error_mae"]) for row in out_rows]),
        "exponent": summarize([float(row["exponent"]) for row in out_rows]),
        "n_peaks": summarize([float(row["n_peaks"]) for row in out_rows]),
        "by_split_label": by_group,
        "first_errors": [row["error"] for row in out_rows if row["error"]][:10],
    }

    write_csv(args.output_csv, out_rows)
    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if summary["ok_fraction"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
