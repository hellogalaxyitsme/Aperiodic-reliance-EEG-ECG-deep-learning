#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_SRC))

from aperiodic_eeg.sleep_edf import resolve_channel_name  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Welch PSDs for Sleep-EDF epochs.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/sleep-edf/sleep-cassette"),
        help="Kept for consistency; paths are read from the epoch manifest.",
    )
    parser.add_argument(
        "--epochs-csv",
        type=Path,
        default=Path("results/sleep_edf_subset/epochs.csv"),
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=Path("results/sleep_edf_subset/psd_welch.npz"),
    )
    parser.add_argument(
        "--output-index-csv",
        type=Path,
        default=Path("results/sleep_edf_subset/psd_index.csv"),
    )
    parser.add_argument("--channels", nargs="+", default=["Fpz-Cz", "Pz-Oz"])
    parser.add_argument("--freq-min", type=float, default=1.0)
    parser.add_argument("--freq-max", type=float, default=45.0)
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--overlap-fraction", type=float, default=0.5)
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Optional smoke-test limit after reading the epoch manifest.",
    )
    return parser.parse_args()


def read_epoch_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_index_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "psd_index",
        "epoch_index",
        "recording",
        "subject",
        "night",
        "recording_epoch_index",
        "onset_sec",
        "duration_sec",
        "stage",
        "stage_original",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for psd_idx, row in enumerate(rows):
            out = {name: row[name] for name in fieldnames if name != "psd_index"}
            out["psd_index"] = psd_idx
            writer.writerow(out)


def main() -> int:
    args = parse_args()

    import mne
    import numpy as np
    from scipy import signal

    rows = read_epoch_rows(args.epochs_csv)
    if args.max_epochs is not None:
        rows = rows[: args.max_epochs]
    if not rows:
        print(f"ERROR: no epoch rows found in {args.epochs_csv}", file=sys.stderr)
        return 2

    psd_chunks: list[np.ndarray] = []
    freqs_out: np.ndarray | None = None
    resolved_channels: list[str] | None = None
    used_rows: list[dict[str, str]] = []

    rows_by_recording: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        rows_by_recording.setdefault(row["recording"], []).append(row)

    for recording, recording_rows in rows_by_recording.items():
        psg_path = Path(recording_rows[0]["psg_path"])
        raw = mne.io.read_raw_edf(psg_path, preload=False, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])

        if resolved_channels is None:
            resolved_channels = [
                resolve_channel_name(channel, raw.ch_names) for channel in args.channels
            ]
        picks = [raw.ch_names.index(channel) for channel in resolved_channels]
        full_data = raw.get_data(picks=picks)

        nperseg = int(round(args.window_seconds * sfreq))
        noverlap = int(round(nperseg * args.overlap_fraction))
        expected_samples = int(round(float(recording_rows[0]["duration_sec"]) * sfreq))
        epoch_data: list[np.ndarray] = []
        recording_used_rows: list[dict[str, str]] = []

        for row in recording_rows:
            onset_sec = float(row["onset_sec"])
            duration_sec = float(row["duration_sec"])
            start = int(round(onset_sec * sfreq))
            stop = int(round((onset_sec + duration_sec) * sfreq))
            data = full_data[:, start:stop]
            if data.shape[-1] != expected_samples:
                print(
                    f"Skipping {recording} epoch {row['recording_epoch_index']}: "
                    f"expected {expected_samples} samples, found {data.shape[-1]}",
                    file=sys.stderr,
                )
                continue
            epoch_data.append(data)
            recording_used_rows.append(row)

        if not epoch_data:
            print(f"WARNING: no usable epochs for {recording}", file=sys.stderr)
            continue

        epoch_array = np.stack(epoch_data, axis=0)
        freqs, psd = signal.welch(
            epoch_array,
            fs=sfreq,
            nperseg=nperseg,
            noverlap=noverlap,
            axis=-1,
            scaling="density",
        )
        freq_mask = (freqs >= args.freq_min) & (freqs <= args.freq_max)
        recording_psd = psd[:, :, freq_mask].astype("float32")

        if freqs_out is None:
            freqs_out = freqs[freq_mask].astype("float32")

        psd_chunks.append(recording_psd)
        used_rows.extend(recording_used_rows)
        print(f"{recording}: PSD shape {recording_psd.shape}", flush=True)

    psds = np.concatenate(psd_chunks, axis=0)
    assert freqs_out is not None
    assert resolved_channels is not None

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        psd=psds,
        freqs=freqs_out,
        channels=np.array(args.channels),
        edf_channels=np.array(resolved_channels),
    )
    write_index_csv(args.output_index_csv, used_rows)

    print(f"Wrote PSD: {args.output_npz}")
    print(f"Wrote index: {args.output_index_csv}")
    print(f"PSD array shape: {psds.shape}")
    print(f"Frequency bins: {len(freqs_out)} from {freqs_out[0]:.2f} to {freqs_out[-1]:.2f} Hz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
