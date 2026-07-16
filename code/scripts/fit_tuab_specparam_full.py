#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit full TUAB fixed-mode specparam decomposition in chunks."
    )
    parser.add_argument(
        "--psd-npz",
        type=Path,
        default=Path("results/tuab_subset_200/psd_20s_multitaper.npz"),
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=Path("results/tuab_subset_200/specparam/specparam_fixed_20s.npz"),
    )
    parser.add_argument("--freq-min", type=float, default=1.0)
    parser.add_argument("--freq-max", type=float, default=45.0)
    parser.add_argument("--max-n-peaks", type=int, default=6)
    parser.add_argument("--min-peak-height", type=float, default=0.1)
    parser.add_argument("--peak-threshold", type=float, default=2.0)
    parser.add_argument("--peak-width-min", type=float, default=0.5)
    parser.add_argument("--peak-width-max", type=float, default=8.0)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--chunk-spectra", type=int, default=8192)
    parser.add_argument("--max-epochs", type=int, default=None)
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
    log_psd = np.log10(np.maximum(spectrum, settings["eps"])).astype("float32")
    try:
        model.fit(freqs, spectrum, [settings["freq_min"], settings["freq_max"]])
        aperiodic_params = model.results.params.aperiodic.asdict()["aperiodic_fit"]
        metrics = model.results.metrics.results
        aperiodic_log = model.results.model.get_component("aperiodic").astype("float32")
        residual_log = (log_psd - aperiodic_log).astype("float32")
        periodic = model.results.params.periodic.asdict()["peak_fit"]
        n_peaks = periodic.shape[0] if periodic.ndim == 2 else 0
        return (
            True,
            log_psd,
            aperiodic_log,
            residual_log,
            float(aperiodic_params[0]),
            float(aperiodic_params[1]),
            float(metrics["gof_rsquared"]),
            float(metrics["error_mae"]),
            int(n_peaks),
            "",
        )
    except Exception as exc:
        nan_curve = np.full(n_freqs, np.nan, dtype="float32")
        return (
            False,
            log_psd,
            nan_curve,
            nan_curve,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            repr(exc),
        )


def main() -> int:
    args = parse_args()

    import numpy as np
    from joblib import Parallel, delayed

    bundle = np.load(args.psd_npz)
    psd = bundle["psd"]
    freqs = bundle["freqs"]
    channels = bundle["channels"]
    if args.max_epochs is not None:
        psd = psd[: args.max_epochs]

    freq_mask = (freqs >= args.freq_min) & (freqs <= args.freq_max)
    freqs_selected = freqs[freq_mask]
    freqs_fit = np.linspace(
        float(freqs_selected[0]),
        float(freqs_selected[-1]),
        int(freqs_selected.shape[0]),
        dtype="float64",
    )
    psd_fit = psd[:, :, freq_mask]
    n_epochs, n_channels, n_freqs = psd_fit.shape
    n_spectra = n_epochs * n_channels

    settings = {
        "freq_min": float(args.freq_min),
        "freq_max": float(args.freq_max),
        "max_n_peaks": int(args.max_n_peaks),
        "min_peak_height": float(args.min_peak_height),
        "peak_threshold": float(args.peak_threshold),
        "peak_width_min": float(args.peak_width_min),
        "peak_width_max": float(args.peak_width_max),
        "eps": 1e-30,
    }

    log_psd_flat = np.empty((n_spectra, n_freqs), dtype="float32")
    aperiodic_flat = np.empty((n_spectra, n_freqs), dtype="float32")
    residual_flat = np.empty((n_spectra, n_freqs), dtype="float32")
    offset_flat = np.empty(n_spectra, dtype="float32")
    exponent_flat = np.empty(n_spectra, dtype="float32")
    r_squared_flat = np.empty(n_spectra, dtype="float32")
    error_mae_flat = np.empty(n_spectra, dtype="float32")
    n_peaks_flat = np.empty(n_spectra, dtype="float32")
    ok_flat = np.empty(n_spectra, dtype=bool)
    errors: list[str] = []

    flat_psd = psd_fit.reshape(n_spectra, n_freqs)
    print(
        f"Fitting TUAB specparam fixed models: epochs={n_epochs}, "
        f"channels={n_channels}, spectra={n_spectra}, freqs={n_freqs}, "
        f"n_jobs={args.n_jobs}, chunk_spectra={args.chunk_spectra}",
        flush=True,
    )

    for start in range(0, n_spectra, args.chunk_spectra):
        stop = min(start + args.chunk_spectra, n_spectra)
        chunk = flat_psd[start:stop]
        results = Parallel(n_jobs=args.n_jobs, verbose=0)(
            delayed(fit_one_spectrum)(freqs_fit, spectrum, settings)
            for spectrum in chunk
        )
        for local_idx, result in enumerate(results):
            (
                ok,
                log_psd,
                aperiodic_log,
                residual_log,
                offset,
                exponent,
                r_squared,
                error_mae,
                n_peaks,
                error,
            ) = result
            idx = start + local_idx
            ok_flat[idx] = ok
            log_psd_flat[idx] = log_psd
            aperiodic_flat[idx] = aperiodic_log
            residual_flat[idx] = residual_log
            offset_flat[idx] = offset
            exponent_flat[idx] = exponent
            r_squared_flat[idx] = r_squared
            error_mae_flat[idx] = error_mae
            n_peaks_flat[idx] = n_peaks
            if error and len(errors) < 20:
                errors.append(error)
        print(
            f"Finished spectra {start}:{stop} ok_fraction_so_far={ok_flat[:stop].mean():.6f}",
            flush=True,
        )

    log_psd = log_psd_flat.reshape(n_epochs, n_channels, n_freqs)
    aperiodic_log_psd = aperiodic_flat.reshape(n_epochs, n_channels, n_freqs)
    residual_log_psd = residual_flat.reshape(n_epochs, n_channels, n_freqs)
    offset = offset_flat.reshape(n_epochs, n_channels)
    exponent = exponent_flat.reshape(n_epochs, n_channels)
    r_squared = r_squared_flat.reshape(n_epochs, n_channels)
    error_mae = error_mae_flat.reshape(n_epochs, n_channels)
    n_peaks = n_peaks_flat.reshape(n_epochs, n_channels)
    ok = ok_flat.reshape(n_epochs, n_channels)

    summary = {
        "settings": settings,
        "psd_npz": str(args.psd_npz),
        "output_npz": str(args.output_npz),
        "shape": [int(n_epochs), int(n_channels), int(n_freqs)],
        "channels": channels.tolist(),
        "ok_fraction": float(ok.mean()),
        "mean_r_squared": float(np.nanmean(r_squared)),
        "median_r_squared": float(np.nanmedian(r_squared)),
        "p10_r_squared": float(np.nanpercentile(r_squared, 10)),
        "mean_error_mae": float(np.nanmean(error_mae)),
        "median_error_mae": float(np.nanmedian(error_mae)),
        "mean_exponent": float(np.nanmean(exponent)),
        "median_exponent": float(np.nanmedian(exponent)),
        "mean_n_peaks": float(np.nanmean(n_peaks)),
        "n_errors": int((~ok).sum()),
        "first_errors": errors,
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
    return 0 if bool(ok.all()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
