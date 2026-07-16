#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_SRC))

from aperiodic_eeg.physionet_mi import resolve_channel_names  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract PhysioNet MI cue-locked raw EEG trials into a NumPy cache."
    )
    parser.add_argument(
        "--trials-csv",
        type=Path,
        default=Path("results/physionet_mi/imagined_fists_trials.csv"),
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=Path("results/physionet_mi/imagined_fists_raw_160hz.npz"),
    )
    parser.add_argument(
        "--output-index-csv",
        type=Path,
        default=Path("results/physionet_mi/imagined_fists_raw_index.csv"),
    )
    parser.add_argument("--channels", nargs="+", default=["all"])
    parser.add_argument("--target-sfreq", type=float, default=160.0)
    parser.add_argument("--scale", type=float, default=1e6, help="Scale EDF volts to microvolts.")
    parser.add_argument("--bandpass-low", type=float, default=1.0)
    parser.add_argument("--bandpass-high", type=float, default=45.0)
    parser.add_argument("--filter-order", type=int, default=4)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--compressed", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_index_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["raw_index"] + list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows):
            writer.writerow({"raw_index": idx, **row})


def maybe_filter(data, sfreq: float, low: float | None, high: float | None, order: int):
    if low is None and high is None:
        return data
    from scipy import signal

    nyquist = sfreq / 2.0
    if high is not None and high >= nyquist:
        high = None
    if low is not None and low <= 0:
        low = None
    if low is None and high is None:
        return data
    if low is None:
        sos = signal.butter(order, high / nyquist, btype="lowpass", output="sos")
    elif high is None:
        sos = signal.butter(order, low / nyquist, btype="highpass", output="sos")
    else:
        sos = signal.butter(order, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, data, axis=-1).astype("float32", copy=False)


def maybe_resample(epoch, sfreq: float, target_sfreq: float, target_samples: int):
    if abs(sfreq - target_sfreq) < 1e-6 and epoch.shape[-1] == target_samples:
        return epoch
    from scipy import signal

    return signal.resample(epoch, target_samples, axis=-1).astype("float32", copy=False)


def main() -> int:
    args = parse_args()

    import mne
    import numpy as np

    rows = read_rows(args.trials_csv)
    if args.max_trials is not None:
        rows = rows[: args.max_trials]
    if not rows:
        print(f"ERROR: no trial rows found in {args.trials_csv}", file=sys.stderr)
        return 2

    rows_by_recording: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        rows_by_recording.setdefault(row["recording"], []).append(row)

    chunks: list[np.ndarray] = []
    used_rows: list[dict[str, str]] = []
    resolved_channels: list[str] | None = None
    samples_per_trial: int | None = None

    for recording, recording_rows in rows_by_recording.items():
        edf_path = Path(recording_rows[0]["edf_path"])
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        eeg_channel_names = [raw.ch_names[idx] for idx in mne.pick_types(raw.info, eeg=True, stim=False)]
        if resolved_channels is None:
            resolved_channels = resolve_channel_names(args.channels, eeg_channel_names)
        picks = [raw.ch_names.index(channel) for channel in resolved_channels]

        full_data = raw.get_data(picks=picks).astype("float32", copy=False)
        full_data = maybe_filter(
            full_data,
            sfreq=sfreq,
            low=args.bandpass_low,
            high=args.bandpass_high,
            order=args.filter_order,
        )
        full_data *= np.float32(args.scale)

        duration_sec = float(recording_rows[0]["duration_sec"])
        expected_samples = int(round(duration_sec * sfreq))
        target_samples = int(round(duration_sec * args.target_sfreq))
        if samples_per_trial is None:
            samples_per_trial = target_samples
        elif samples_per_trial != target_samples:
            raise ValueError(
                f"Inconsistent trial sample count: {samples_per_trial} vs {target_samples}"
            )

        trial_data: list[np.ndarray] = []
        recording_used_rows: list[dict[str, str]] = []
        for row in recording_rows:
            onset_sec = float(row["onset_sec"])
            start = int(round(onset_sec * sfreq))
            stop = start + expected_samples
            trial = full_data[:, start:stop]
            if trial.shape[-1] != expected_samples:
                print(
                    f"Skipping {recording} trial {row['recording_trial_index']}: "
                    f"expected {expected_samples} samples, found {trial.shape[-1]}",
                    file=sys.stderr,
                )
                continue
            trial = maybe_resample(trial, sfreq, args.target_sfreq, target_samples)
            trial = trial - trial.mean(axis=-1, keepdims=True)
            trial_data.append(trial.astype("float32", copy=False))
            recording_used_rows.append(row)

        if not trial_data:
            print(f"WARNING: no usable trials for {recording}", file=sys.stderr)
            continue

        recording_array = np.stack(trial_data, axis=0)
        chunks.append(recording_array)
        used_rows.extend(recording_used_rows)
        print(f"{recording}: raw trial shape {recording_array.shape}", flush=True)

    if not chunks:
        print("ERROR: no raw trials extracted", file=sys.stderr)
        return 2

    x = np.concatenate(chunks, axis=0).astype("float32", copy=False)
    assert resolved_channels is not None
    assert samples_per_trial is not None

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    save_fn = np.savez_compressed if args.compressed else np.savez
    save_fn(
        args.output_npz,
        x=x,
        sfreq=np.float32(args.target_sfreq),
        channels=np.array(resolved_channels),
        requested_channels=np.array(args.channels),
        edf_channels=np.array(resolved_channels),
        scale=np.float32(args.scale),
        bandpass_low=np.float32(args.bandpass_low),
        bandpass_high=np.float32(args.bandpass_high),
    )
    write_index_csv(args.output_index_csv, used_rows)

    print(f"Wrote raw trials: {args.output_npz}")
    print(f"Wrote index: {args.output_index_csv}")
    print(f"Raw trial array shape: {x.shape}")
    print(f"Target sampling rate: {args.target_sfreq:g} Hz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
