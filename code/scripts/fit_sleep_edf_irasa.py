#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from fractions import Fraction
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit an IRASA aperiodic spectrum estimate to Sleep-EDF raw epochs."
    )
    parser.add_argument(
        "--raw-npz",
        type=Path,
        default=Path("results/sleep_edf_full/raw_epochs_fpz_pz_100hz.npz"),
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=Path("results/sleep_edf_full/raw_epochs_index.csv"),
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=Path("results/sleep_edf_full/irasa/irasa_aperiodic.npz"),
    )
    parser.add_argument(
        "--output-index-csv",
        type=Path,
        default=None,
        help="CSV mapping IRASA rows back to raw/specparam epoch rows.",
    )
    parser.add_argument("--freq-min", type=float, default=1.0)
    parser.add_argument("--freq-max", type=float, default=45.0)
    parser.add_argument("--freq-step", type=float, default=0.25)
    parser.add_argument("--nperseg-seconds", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--input-unit",
        choices=["auto", "as_stored", "volts"],
        default="auto",
        help=(
            "Unit handling for raw cache. 'auto' divides by raw['scale'] when "
            "available so IRASA matches EDF-volts PSD artifacts."
        ),
    )
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument(
        "--sample-per-stage",
        type=int,
        default=None,
        help="Randomly sample up to this many epochs per sleep stage.",
    )
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument(
        "--hset",
        default="1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9",
    )
    return parser.parse_args()


def read_index(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_index(path: Path, rows: list[dict[str, str]], selected_raw_indices) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["irasa_index"] + list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for irasa_idx, raw_idx in enumerate(selected_raw_indices):
            writer.writerow({"irasa_index": irasa_idx, **rows[int(raw_idx)]})


def select_indices(rows: list[dict[str, str]], sample_per_stage: int | None, max_epochs: int | None, seed: int):
    import numpy as np

    n_rows = len(rows)
    if sample_per_stage is None:
        n = n_rows if max_epochs is None else min(max_epochs, n_rows)
        return np.arange(n, dtype=int)

    rng = np.random.default_rng(seed)
    by_stage: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        by_stage.setdefault(str(row.get("stage", "")), []).append(idx)

    selected: list[int] = []
    for stage in sorted(by_stage):
        indices = np.array(by_stage[stage], dtype=int)
        n = min(sample_per_stage, len(indices))
        selected.extend(rng.choice(indices, size=n, replace=False).tolist())

    selected = sorted(selected)
    if max_epochs is not None and len(selected) > max_epochs:
        selected = sorted(rng.choice(np.array(selected, dtype=int), size=max_epochs, replace=False).tolist())
    return np.array(selected, dtype=int)


def welch_interp(x, sfreq: float, nperseg: int, target_freqs):
    import numpy as np
    from scipy.signal import welch

    freqs, psd = welch(
        x,
        fs=sfreq,
        nperseg=min(nperseg, x.shape[-1]),
        noverlap=min(nperseg // 2, max(0, x.shape[-1] // 2 - 1)),
        axis=-1,
        detrend="constant",
    )
    return np.interp(target_freqs, freqs, psd).astype("float32")


def resample_by_factor(x, factor: float):
    from scipy.signal import resample_poly

    frac = Fraction(str(factor)).limit_denominator(100)
    return resample_poly(x, frac.numerator, frac.denominator, axis=-1)


def irasa_batch(x, sfreq: float, target_freqs, hset, nperseg: int):
    import numpy as np

    n_epochs, n_channels, _ = x.shape
    original = np.empty((n_epochs, n_channels, len(target_freqs)), dtype="float32")
    fractal = np.empty_like(original)
    for epoch_idx in range(n_epochs):
        for ch_idx in range(n_channels):
            signal = x[epoch_idx, ch_idx]
            original[epoch_idx, ch_idx] = welch_interp(signal, sfreq, nperseg, target_freqs)
            h_psds = []
            for h in hset:
                up = resample_by_factor(signal, h)
                down = resample_by_factor(signal, 1.0 / h)
                psd_up = welch_interp(up, sfreq * h, nperseg, target_freqs)
                psd_down = welch_interp(down, sfreq / h, nperseg, target_freqs)
                h_psds.append(np.sqrt(np.maximum(psd_up, 1e-30) * np.maximum(psd_down, 1e-30)))
            fractal[epoch_idx, ch_idx] = np.median(np.stack(h_psds, axis=0), axis=0)
    return original, fractal


def main() -> int:
    import numpy as np

    args = parse_args()
    raw = np.load(args.raw_npz)
    rows = read_index(args.index_csv)
    selected_indices = select_indices(
        rows,
        sample_per_stage=args.sample_per_stage,
        max_epochs=args.max_epochs,
        seed=args.seed,
    )
    x_all = raw["x"]
    x = x_all[selected_indices].astype("float32", copy=False)
    unit_note = "as_stored"
    if args.input_unit in {"auto", "volts"} and "scale" in raw:
        scale = float(raw["scale"])
        if scale != 0:
            x = (x / scale).astype("float32", copy=False)
            unit_note = f"converted_to_volts_by_dividing_scale_{scale:g}"
    sfreq = float(raw["sfreq"])
    target_freqs = np.arange(
        args.freq_min,
        args.freq_max + args.freq_step / 2.0,
        args.freq_step,
        dtype="float32",
    )
    hset = [float(item) for item in args.hset.split(",") if item.strip()]
    nperseg = max(8, int(round(args.nperseg_seconds * sfreq)))

    psd = np.empty((len(x), x.shape[1], len(target_freqs)), dtype="float32")
    ap = np.empty_like(psd)
    for start in range(0, len(x), args.batch_size):
        stop = min(start + args.batch_size, len(x))
        batch_psd, batch_ap = irasa_batch(
            x[start:stop],
            sfreq=sfreq,
            target_freqs=target_freqs,
            hset=hset,
            nperseg=nperseg,
        )
        psd[start:stop] = batch_psd
        ap[start:stop] = batch_ap
        print(f"IRASA epochs {start}:{stop} done", flush=True)

    log_psd = np.log10(np.maximum(psd, 1e-30)).astype("float32")
    aperiodic_log_psd = np.log10(np.maximum(ap, 1e-30)).astype("float32")
    residual_log_psd = (log_psd - aperiodic_log_psd).astype("float32")
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_index_csv = args.output_index_csv
    if output_index_csv is None:
        output_index_csv = args.output_npz.with_suffix(".index.csv")
    np.savez_compressed(
        args.output_npz,
        freqs=target_freqs,
        log_psd=log_psd,
        aperiodic_log_psd=aperiodic_log_psd,
        residual_log_psd=residual_log_psd,
        sfreq=np.array(sfreq, dtype="float32"),
        channels=raw["channels"],
        hset=np.array(hset, dtype="float32"),
        raw_indices=selected_indices.astype("int64"),
    )
    write_index(output_index_csv, rows, selected_indices)
    args.output_npz.with_suffix(".json").write_text(
        json.dumps(
            {
                "raw_npz": str(args.raw_npz),
                "index_csv": str(args.index_csv),
                "output_npz": str(args.output_npz),
                "output_index_csv": str(output_index_csv),
                "shape": list(log_psd.shape),
                "n_raw_epochs": int(len(rows)),
                "n_selected_epochs": int(len(selected_indices)),
                "sample_per_stage": args.sample_per_stage,
                "freq_min": float(args.freq_min),
                "freq_max": float(args.freq_max),
                "freq_step": float(args.freq_step),
                "hset": hset,
                "nperseg_seconds": float(args.nperseg_seconds),
                "max_epochs": args.max_epochs,
                "seed": int(args.seed),
                "input_unit": args.input_unit,
                "unit_note": unit_note,
            },
            indent=2,
        )
    )
    print(f"Wrote: {args.output_npz}")
    print(f"Wrote: {output_index_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
