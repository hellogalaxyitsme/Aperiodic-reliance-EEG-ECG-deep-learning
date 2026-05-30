#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TEST_INPUTS = ["raw_eeg", "phase_sham", "phase_aperiodic", "phase_flattened"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quantify time-domain and spectral side effects of raw EEG "
            "phase-preserving aperiodic interventions."
        )
    )
    parser.add_argument(
        "--raw-npz",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/results/sleep_edf_full/raw_epochs_fpz_pz_100hz.npz"),
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/results/sleep_edf_full/raw_epochs_index.csv"),
    )
    parser.add_argument(
        "--decomp-npz",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/results/sleep_edf_full/specparam/specparam_fixed.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/reports/tables/raw_intervention_diagnostics"),
    )
    parser.add_argument("--band-min", type=float, default=1.0)
    parser.add_argument("--band-max", type=float, default=45.0)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--no-match-rms", action="store_true")
    return parser.parse_args()


def interpolate_aperiodic(ap_log_psd, decomp_freqs, target_freqs):
    import numpy as np

    out = np.empty(ap_log_psd.shape[:2] + (len(target_freqs),), dtype="float32")
    for epoch_idx in range(ap_log_psd.shape[0]):
        for channel_idx in range(ap_log_psd.shape[1]):
            out[epoch_idx, channel_idx] = np.interp(
                target_freqs,
                decomp_freqs,
                ap_log_psd[epoch_idx, channel_idx],
            ).astype("float32")
    return out


def make_phase_preserving_inputs(
    x_test,
    ap_log_psd,
    decomp_freqs,
    sfreq: float,
    band_min: float,
    band_max: float,
    match_rms: bool,
):
    import numpy as np

    n_times = x_test.shape[-1]
    fft_freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    band_mask = (fft_freqs >= band_min) & (fft_freqs <= band_max)
    band_freqs = fft_freqs[band_mask]
    fft = np.fft.rfft(x_test, axis=-1)
    band_fft = fft[:, :, band_mask]
    band_amp = np.maximum(np.abs(band_fft), 1e-12)
    band_phase = band_fft / band_amp

    ap_interp = interpolate_aperiodic(ap_log_psd, decomp_freqs, band_freqs)
    ap_centered = ap_interp - ap_interp.mean(axis=-1, keepdims=True)
    ap_amp_shape = np.power(10.0, 0.5 * ap_centered).astype("float32")

    sham_fft = fft.copy()
    sham_fft[:, :, band_mask] = band_phase * band_amp

    aperiodic_fft = fft.copy()
    geom_amp = np.exp(np.mean(np.log(band_amp), axis=-1, keepdims=True))
    aperiodic_fft[:, :, band_mask] = band_phase * geom_amp * ap_amp_shape

    flattened_fft = fft.copy()
    flattened_fft[:, :, band_mask] = band_fft / np.maximum(ap_amp_shape, 1e-6)

    edited = {
        "raw_eeg": x_test,
        "phase_sham": np.fft.irfft(sham_fft, n=n_times, axis=-1),
        "phase_aperiodic": np.fft.irfft(aperiodic_fft, n=n_times, axis=-1),
        "phase_flattened": np.fft.irfft(flattened_fft, n=n_times, axis=-1),
    }
    original_std = x_test.std(axis=-1, keepdims=True)
    for name in ["phase_sham", "phase_aperiodic", "phase_flattened"]:
        array = edited[name].real.astype("float32", copy=False)
        array -= array.mean(axis=-1, keepdims=True)
        if match_rms:
            edited_std = array.std(axis=-1, keepdims=True)
            array *= original_std / np.maximum(edited_std, 1e-6)
        edited[name] = array.astype("float32", copy=False)
    return edited


def summarize_array(name, x, reference=None):
    import numpy as np
    from scipy.stats import kurtosis, skew

    flat = x.reshape(x.shape[0], x.shape[1], -1)
    rms = np.sqrt(np.mean(flat**2, axis=-1))
    std = flat.std(axis=-1)
    ptp = np.ptp(flat, axis=-1)
    line_length = np.mean(np.abs(np.diff(flat, axis=-1)), axis=-1)
    zcr = np.mean(np.diff(np.signbit(flat), axis=-1), axis=-1)
    rows = []
    stats = {
        "mean": flat.mean(axis=-1),
        "std": std,
        "rms": rms,
        "peak_to_peak": ptp,
        "skew": skew(flat, axis=-1, bias=False),
        "kurtosis": kurtosis(flat, axis=-1, fisher=True, bias=False),
        "line_length": line_length,
        "zero_crossing_rate": zcr,
    }
    if reference is not None:
        diff = (flat - reference.reshape(reference.shape[0], reference.shape[1], -1))
        stats["rmse_vs_raw"] = np.sqrt(np.mean(diff**2, axis=-1))
        stats["corr_vs_raw"] = np.array(
            [
                [
                    np.corrcoef(flat[i, ch], reference[i, ch])[0, 1]
                    for ch in range(flat.shape[1])
                ]
                for i in range(flat.shape[0])
            ]
        )

    for stat_name, values in stats.items():
        values = np.asarray(values, dtype=float).reshape(-1)
        rows.append(
            {
                "test_input": name,
                "statistic": stat_name,
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values, ddof=1)),
                "median": float(np.nanmedian(values)),
                "p05": float(np.nanquantile(values, 0.05)),
                "p95": float(np.nanquantile(values, 0.95)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    keep = [
        row
        for row in rows
        if row["statistic"] in ["std", "rms", "peak_to_peak", "kurtosis", "rmse_vs_raw", "corr_vs_raw"]
    ]
    lines = [
        "| test_input | statistic | mean | median | p05 | p95 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in keep:
        lines.append(
            "| {test_input} | {statistic} | {mean:.4f} | {median:.4f} | {p05:.4f} | {p95:.4f} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    import numpy as np
    import pandas as pd

    args = parse_args()
    raw = np.load(args.raw_npz)
    decomp = np.load(args.decomp_npz)
    index = pd.read_csv(args.index_csv)
    n_available = len(index)
    if args.max_epochs is None or args.max_epochs >= n_available:
        selected = np.arange(n_available)
    else:
        rng = np.random.default_rng(args.seed)
        selected = np.sort(rng.choice(n_available, size=args.max_epochs, replace=False))

    x = raw["x"][selected].astype("float32", copy=False)
    ap = decomp["aperiodic_log_psd"][selected].astype("float32", copy=False)
    freqs = decomp["freqs"].astype("float32", copy=False)
    match_rms = not args.no_match_rms

    rows = []
    for start in range(0, len(selected), args.batch_size):
        stop = min(start + args.batch_size, len(selected))
        edited = make_phase_preserving_inputs(
            x[start:stop],
            ap[start:stop],
            freqs,
            sfreq=float(raw["sfreq"]),
            band_min=args.band_min,
            band_max=args.band_max,
            match_rms=match_rms,
        )
        reference = edited["raw_eeg"]
        for name in TEST_INPUTS:
            rows.extend(summarize_array(name, edited[name], reference=reference if name != "raw_eeg" else None))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "raw_intervention_distribution_diagnostics.csv"
    md_path = args.output_dir / "raw_intervention_distribution_diagnostics.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    (args.output_dir / "raw_intervention_distribution_diagnostics_metadata.json").write_text(
        json.dumps(
            {
                "raw_npz": str(args.raw_npz),
                "index_csv": str(args.index_csv),
                "decomp_npz": str(args.decomp_npz),
                "n_available": int(n_available),
                "n_analyzed": int(len(selected)),
                "seed": int(args.seed),
                "band_min": float(args.band_min),
                "band_max": float(args.band_max),
                "match_rms": bool(match_rms),
            },
            indent=2,
        )
    )
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
