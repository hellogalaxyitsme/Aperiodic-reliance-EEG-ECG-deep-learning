#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit fixed-mode specparam models to PhysioNet MI multitaper PSDs."
    )
    parser.add_argument(
        "--psd-npz",
        type=Path,
        default=Path("results/physionet_mi/imagined_fists_psd_multitaper.npz"),
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=Path("results/physionet_mi/specparam/imagined_fists_specparam_fixed.npz"),
    )
    parser.add_argument("--freq-min", type=float, default=2.0)
    parser.add_argument("--freq-max", type=float, default=45.0)
    parser.add_argument("--max-n-peaks", type=int, default=4)
    parser.add_argument("--min-peak-height", type=float, default=0.1)
    parser.add_argument("--peak-threshold", type=float, default=2.0)
    parser.add_argument("--peak-width-min", type=float, default=1.0)
    parser.add_argument("--peak-width-max", type=float, default=10.0)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--max-trials", type=int, default=None)
    return parser.parse_args()


def fit_one_spectrum(freqs, spectrum, settings):
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
    n_freqs = len(freqs)
    log_psd = np.log10(np.maximum(spectrum, settings["eps"]))
    try:
        model.fit(freqs, spectrum, [settings["freq_min"], settings["freq_max"]])
        aperiodic_params = model.results.params.aperiodic.asdict()["aperiodic_fit"]
        metrics = model.results.metrics.results
        aperiodic_log = model.results.model.get_component("aperiodic")
        residual_log = log_psd - aperiodic_log
        periodic = model.results.params.periodic.asdict()["peak_fit"]
        n_peaks = periodic.shape[0] if periodic.ndim == 2 else 0
        return {
            "ok": True,
            "log_psd": log_psd.astype("float32"),
            "aperiodic_log_psd": aperiodic_log.astype("float32"),
            "residual_log_psd": residual_log.astype("float32"),
            "offset": float(aperiodic_params[0]),
            "exponent": float(aperiodic_params[1]),
            "r_squared": float(metrics["gof_rsquared"]),
            "error_mae": float(metrics["error_mae"]),
            "n_peaks": int(n_peaks),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": repr(exc),
            "log_psd": log_psd.astype("float32"),
            "aperiodic_log_psd": np.full(n_freqs, np.nan, dtype="float32"),
            "residual_log_psd": np.full(n_freqs, np.nan, dtype="float32"),
            "offset": np.nan,
            "exponent": np.nan,
            "r_squared": np.nan,
            "error_mae": np.nan,
            "n_peaks": np.nan,
        }


def main() -> int:
    args = parse_args()

    import numpy as np
    from joblib import Parallel, delayed

    bundle = np.load(args.psd_npz)
    psd = bundle["psd"]
    freqs = bundle["freqs"]
    channels = bundle["channels"]
    if len(channels) != psd.shape[1] and "edf_channels" in bundle:
        edf_channels = bundle["edf_channels"]
        if len(edf_channels) == psd.shape[1]:
            channels = edf_channels

    if args.max_trials is not None:
        psd = psd[: args.max_trials]

    freq_mask = (freqs >= args.freq_min) & (freqs <= args.freq_max)
    freqs_fit = freqs[freq_mask]
    psd_fit = psd[:, :, freq_mask]
    settings = {
        "freq_min": float(args.freq_min),
        "freq_max": float(args.freq_max),
        "max_n_peaks": int(args.max_n_peaks),
        "min_peak_height": float(args.min_peak_height),
        "peak_threshold": float(args.peak_threshold),
        "peak_width_min": float(args.peak_width_min),
        "peak_width_max": float(args.peak_width_max),
        "eps": 1e-30,
        "short_window_note": "PhysioNet MI uses cue-locked short trials; fit quality must be reported.",
    }

    n_trials, n_channels, n_freqs = psd_fit.shape
    spectra = [
        psd_fit[trial_idx, channel_idx]
        for trial_idx in range(n_trials)
        for channel_idx in range(n_channels)
    ]
    print(
        f"Fitting PhysioNet MI specparam: trials={n_trials}, channels={n_channels}, "
        f"spectra={len(spectra)}, freqs={n_freqs}, n_jobs={args.n_jobs}",
        flush=True,
    )
    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(fit_one_spectrum)(freqs_fit, spectrum, settings)
        for spectrum in spectra
    )

    log_psd = np.stack([res["log_psd"] for res in results], axis=0).reshape(
        n_trials, n_channels, n_freqs
    )
    aperiodic_log_psd = np.stack(
        [res["aperiodic_log_psd"] for res in results], axis=0
    ).reshape(n_trials, n_channels, n_freqs)
    residual_log_psd = np.stack(
        [res["residual_log_psd"] for res in results], axis=0
    ).reshape(n_trials, n_channels, n_freqs)
    offset = np.array([res["offset"] for res in results], dtype="float32").reshape(
        n_trials, n_channels
    )
    exponent = np.array([res["exponent"] for res in results], dtype="float32").reshape(
        n_trials, n_channels
    )
    r_squared = np.array([res["r_squared"] for res in results], dtype="float32").reshape(
        n_trials, n_channels
    )
    error_mae = np.array([res["error_mae"] for res in results], dtype="float32").reshape(
        n_trials, n_channels
    )
    n_peaks = np.array([res["n_peaks"] for res in results], dtype="float32").reshape(
        n_trials, n_channels
    )
    ok = np.array([res["ok"] for res in results], dtype=bool).reshape(n_trials, n_channels)
    errors = [res.get("error", "") for res in results if not res["ok"]]

    summary = {
        "settings": settings,
        "psd_npz": str(args.psd_npz),
        "output_npz": str(args.output_npz),
        "shape": [int(n_trials), int(n_channels), int(n_freqs)],
        "channels": channels.tolist(),
        "ok_fraction": float(ok.mean()),
        "mean_r_squared": float(np.nanmean(r_squared)),
        "median_r_squared": float(np.nanmedian(r_squared)),
        "p10_r_squared": float(np.nanpercentile(r_squared, 10)),
        "mean_error_mae": float(np.nanmean(error_mae)),
        "mean_n_peaks": float(np.nanmean(n_peaks)),
        "n_errors": int(len(errors)),
        "first_errors": errors[:10],
    }

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        log_psd=log_psd,
        aperiodic_log_psd=aperiodic_log_psd,
        residual_log_psd=residual_log_psd,
        offset=offset,
        exponent=exponent,
        r_squared=r_squared,
        error_mae=error_mae,
        n_peaks=n_peaks,
        ok=ok,
        freqs=freqs_fit.astype("float32"),
        channels=channels,
        settings_json=np.array(json.dumps(settings)),
    )
    summary_path = args.output_npz.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Wrote: {args.output_npz}")
    print(f"Wrote: {summary_path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
