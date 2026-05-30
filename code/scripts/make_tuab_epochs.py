#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_SRC))

from aperiodic_eeg.tuab import (  # noqa: E402
    TUAB_STANDARD_CHANNELS,
    binary_label,
    local_edf_path,
    resolve_tuab_channels,
    tuab_recording_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a fixed-window TUAB epoch manifest from selected EDF files."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/data/tuab/v3.0.1_random_stratified_200"),
    )
    parser.add_argument(
        "--selected-files-csv",
        type=Path,
        default=Path(
            "/mnt/data/aperiodic_confounds/results/tuab_subset_200/"
            "tuab_v3_0_1_random_stratified_200_files.csv"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/results/tuab_subset_200/epochs_20s.csv"),
    )
    parser.add_argument(
        "--output-summary-json",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/results/tuab_subset_200/epochs_20s_summary.json"),
    )
    parser.add_argument("--channels", nargs="+", default=TUAB_STANDARD_CHANNELS)
    parser.add_argument("--epoch-seconds", type=float, default=20.0)
    parser.add_argument("--stride-seconds", type=float, default=20.0)
    parser.add_argument(
        "--drop-initial-seconds",
        type=float,
        default=0.0,
        help="Optional lead-in to skip. Default keeps the official recording from t=0.",
    )
    parser.add_argument(
        "--max-epochs-per-file",
        type=int,
        default=None,
        help="Optional bounded smoke-test cap per EDF file.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
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


def main() -> int:
    args = parse_args()

    import mne

    selected = read_csv(args.selected_files_csv)
    epoch_rows: list[dict[str, object]] = []
    skipped_files: list[dict[str, object]] = []
    counts = Counter()

    for file_idx, row in enumerate(selected):
        edf_path = local_edf_path(args.data_root, row["remote_rel_path"])
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
        try:
            resolved_channels = resolve_tuab_channels(args.channels, raw.ch_names)
        except ValueError as exc:
            skipped_files.append(
                {
                    "remote_rel_path": row["remote_rel_path"],
                    "reason": repr(exc),
                }
            )
            continue

        sfreq = float(raw.info["sfreq"])
        duration_sec = float(raw.n_times / sfreq)
        recording = tuab_recording_id(row)
        start = float(args.drop_initial_seconds)
        max_start = duration_sec - float(args.epoch_seconds)
        if max_start < start:
            skipped_files.append(
                {
                    "remote_rel_path": row["remote_rel_path"],
                    "reason": f"duration {duration_sec:.3f}s shorter than requested window",
                }
            )
            continue

        n_windows = int(math.floor((max_start - start) / args.stride_seconds)) + 1
        if args.max_epochs_per_file is not None:
            n_windows = min(n_windows, args.max_epochs_per_file)

        for recording_epoch_index in range(n_windows):
            onset_sec = start + recording_epoch_index * float(args.stride_seconds)
            epoch_rows.append(
                {
                    "epoch_index": len(epoch_rows),
                    "recording": recording,
                    "file_index": file_idx,
                    "recording_epoch_index": recording_epoch_index,
                    "subject": row["subject_id"],
                    "session": row["session"],
                    "token": row["token"],
                    "official_split": row["official_split"],
                    "label": row["label"],
                    "target": binary_label(row["label"]),
                    "montage": row["montage"],
                    "onset_sec": f"{onset_sec:.6f}",
                    "duration_sec": f"{float(args.epoch_seconds):.6f}",
                    "sfreq_original": f"{sfreq:.6f}",
                    "n_times_original": int(raw.n_times),
                    "duration_recording_sec": f"{duration_sec:.6f}",
                    "remote_rel_path": row["remote_rel_path"],
                    "edf_path": str(edf_path),
                    "channels": "|".join(args.channels),
                    "edf_channels": "|".join(resolved_channels),
                }
            )
            counts.update([f"{row['official_split']}/{row['label']}"])

        if (file_idx + 1) % 25 == 0:
            print(f"Created epochs for {file_idx + 1}/{len(selected)} EDF files", flush=True)

    summary = {
        "data_root": str(args.data_root),
        "selected_files_csv": str(args.selected_files_csv),
        "output_csv": str(args.output_csv),
        "n_input_files": len(selected),
        "n_used_files": len(selected) - len(skipped_files),
        "n_skipped_files": len(skipped_files),
        "first_skipped_files": skipped_files[:10],
        "n_epochs": len(epoch_rows),
        "epoch_seconds": float(args.epoch_seconds),
        "stride_seconds": float(args.stride_seconds),
        "drop_initial_seconds": float(args.drop_initial_seconds),
        "max_epochs_per_file": args.max_epochs_per_file,
        "requested_channels": list(args.channels),
        "epoch_counts_by_split_label": dict(sorted(counts.items())),
    }

    write_csv(args.output_csv, epoch_rows)
    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
