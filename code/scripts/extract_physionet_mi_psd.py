#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract multitaper PSDs from cached PhysioNet MI raw trials."
    )
    parser.add_argument(
        "--raw-npz",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/results/physionet_mi/imagined_fists_raw_160hz.npz"),
    )
    parser.add_argument(
        "--raw-index-csv",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/results/physionet_mi/imagined_fists_raw_index.csv"),
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/results/physionet_mi/imagined_fists_psd_multitaper.npz"),
    )
    parser.add_argument(
        "--output-index-csv",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/results/physionet_mi/imagined_fists_psd_index.csv"),
    )
    parser.add_argument("--freq-min", type=float, default=2.0)
    parser.add_argument("--freq-max", type=float, default=45.0)
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=4.0,
        help="Multitaper bandwidth in Hz. Wider than Sleep-EDF because MI trials are short.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-trials", type=int, default=None)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_index_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["psd_index"] + [name for name in rows[0].keys() if name != "raw_index"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows):
            out = {name: row[name] for name in fieldnames if name not in {"psd_index"}}
            out["psd_index"] = idx
            writer.writerow(out)


def main() -> int:
    args = parse_args()

    import mne
    import numpy as np

    bundle = np.load(args.raw_npz)
    x = bundle["x"]
    sfreq = float(bundle["sfreq"])
    channels = bundle["channels"]
    edf_channels = bundle["edf_channels"]
    if len(channels) != x.shape[1] and len(edf_channels) == x.shape[1]:
        channels = edf_channels
    rows = read_rows(args.raw_index_csv)
    if args.max_trials is not None:
        x = x[: args.max_trials]
        rows = rows[: args.max_trials]
    if len(rows) != x.shape[0]:
        raise ValueError(f"Index rows {len(rows)} != raw trials {x.shape[0]}")

    psd_chunks: list[np.ndarray] = []
    freqs_out = None
    for start in range(0, x.shape[0], args.batch_size):
        stop = min(start + args.batch_size, x.shape[0])
        psd, freqs = mne.time_frequency.psd_array_multitaper(
            x[start:stop],
            sfreq=sfreq,
            fmin=args.freq_min,
            fmax=args.freq_max,
            bandwidth=args.bandwidth,
            adaptive=False,
            low_bias=True,
            normalization="full",
            verbose="ERROR",
        )
        if freqs_out is None:
            freqs_out = freqs.astype("float32")
        psd_chunks.append(psd.astype("float32", copy=False))
        print(f"PSD batch {start}:{stop} shape {psd.shape}", flush=True)

    psd_all = np.concatenate(psd_chunks, axis=0)
    assert freqs_out is not None

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        psd=psd_all,
        freqs=freqs_out,
        channels=channels,
        edf_channels=edf_channels,
        sfreq=np.float32(sfreq),
        method=np.array("multitaper"),
        bandwidth=np.float32(args.bandwidth),
    )
    write_index_csv(args.output_index_csv, rows)

    print(f"Wrote PSD: {args.output_npz}")
    print(f"Wrote index: {args.output_index_csv}")
    print(f"PSD array shape: {psd_all.shape}")
    print(f"Frequency bins: {len(freqs_out)} from {freqs_out[0]:.2f} to {freqs_out[-1]:.2f} Hz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
