#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_SRC))

from aperiodic_eeg.sleep_edf import (  # noqa: E402
    discover_recording_pairs,
    normalize_stage,
    trim_wake_epochs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 30-second Sleep-EDF epoch manifest from hypnograms."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/sleep-edf/sleep-cassette"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/sleep_edf_subset/epochs.csv"),
    )
    parser.add_argument("--epoch-seconds", type=float, default=30.0)
    parser.add_argument(
        "--wake-trim-minutes",
        type=float,
        default=30.0,
        help="Keep wake epochs within this many minutes of sleep. Use -1 to disable.",
    )
    parser.add_argument(
        "--max-recordings",
        type=int,
        default=None,
        help="Optional smoke-test limit on number of PSG/Hypnogram pairs.",
    )
    return parser.parse_args()


def load_recording_epochs(pair, raw_duration_sec: float, epoch_seconds: float) -> list[dict[str, object]]:
    import mne

    annotations = mne.read_annotations(pair.hypnogram_path)
    rows: list[dict[str, object]] = []

    for onset, duration, description in zip(
        annotations.onset, annotations.duration, annotations.description
    ):
        stage = normalize_stage(str(description))
        if stage is None:
            continue

        n_epochs = int(duration // epoch_seconds)
        for within_annotation_idx in range(n_epochs):
            epoch_onset = float(onset) + within_annotation_idx * epoch_seconds
            epoch_end = epoch_onset + epoch_seconds
            if epoch_onset < 0 or epoch_end > raw_duration_sec:
                continue

            rows.append(
                {
                    "recording": pair.key,
                    "subject": pair.subject_code,
                    "night": pair.night_code,
                    "psg_path": str(pair.psg_path),
                    "hypnogram_path": str(pair.hypnogram_path),
                    "onset_sec": f"{epoch_onset:.3f}",
                    "duration_sec": f"{epoch_seconds:.3f}",
                    "stage": stage,
                    "stage_original": str(description),
                }
            )

    return rows


def main() -> int:
    args = parse_args()
    if args.wake_trim_minutes < 0:
        wake_trim_minutes = None
    else:
        wake_trim_minutes = args.wake_trim_minutes

    import mne

    pairs = discover_recording_pairs(args.data_root)
    if args.max_recordings is not None:
        pairs = pairs[: args.max_recordings]
    if not pairs:
        print(f"ERROR: no recording pairs found in {args.data_root}", file=sys.stderr)
        return 2

    all_rows: list[dict[str, object]] = []
    for pair in pairs:
        raw = mne.io.read_raw_edf(pair.psg_path, preload=False, verbose="ERROR")
        raw_duration_sec = raw.n_times / raw.info["sfreq"]
        rows = load_recording_epochs(pair, raw_duration_sec, args.epoch_seconds)
        rows = trim_wake_epochs(rows, wake_trim_minutes, args.epoch_seconds)

        for local_idx, row in enumerate(rows):
            row["recording_epoch_index"] = local_idx
        all_rows.extend(rows)

        counts = Counter(row["stage"] for row in rows)
        print(
            f"{pair.key}: {len(rows)} epochs after trimming "
            f"({dict(sorted(counts.items()))})"
        )

    for global_idx, row in enumerate(all_rows):
        row["epoch_index"] = global_idx

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch_index",
        "recording",
        "subject",
        "night",
        "recording_epoch_index",
        "onset_sec",
        "duration_sec",
        "stage",
        "stage_original",
        "psg_path",
        "hypnogram_path",
    ]
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    total_counts = Counter(row["stage"] for row in all_rows)
    print(f"Wrote: {args.output_csv}")
    print(f"Total epochs: {len(all_rows)}")
    print(f"Stage counts: {dict(sorted(total_counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

