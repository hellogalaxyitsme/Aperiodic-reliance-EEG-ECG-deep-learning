#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_SRC))

from aperiodic_eeg.physionet_mi import (  # noqa: E402
    condition_for_event,
    discover_recordings,
    normalize_event_code,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cue-locked trial manifest for PhysioNet EEG Motor Movement/Imagery."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/physionet-eegmmidb"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/physionet_mi/imagined_fists_trials.csv"),
    )
    parser.add_argument(
        "--task",
        choices=[
            "imagined_fists",
            "executed_fists",
            "imagined_fists_feet",
            "executed_fists_feet",
            "all_imagery",
            "all_execution",
            "all_tasks",
        ],
        default="imagined_fists",
    )
    parser.add_argument("--tmin", type=float, default=0.5)
    parser.add_argument("--tmax", type=float, default=4.0)
    parser.add_argument(
        "--include-rest",
        action="store_true",
        help="Include T0 rest periods as trials. By default only T1/T2 task cues are kept.",
    )
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--max-recordings", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import mne

    if args.tmax <= args.tmin:
        raise ValueError("--tmax must be greater than --tmin")

    recordings = discover_recordings(
        args.data_root,
        task=args.task,
        max_subjects=args.max_subjects,
        max_recordings=args.max_recordings,
    )
    if not recordings:
        print(f"ERROR: no matching recordings found under {args.data_root}", file=sys.stderr)
        return 2

    rows: list[dict[str, object]] = []
    for recording in recordings:
        raw = mne.io.read_raw_edf(recording.path, preload=False, verbose="ERROR")
        duration = raw.n_times / float(raw.info["sfreq"])
        local_count = 0

        for onset, annotation_duration, description in zip(
            raw.annotations.onset,
            raw.annotations.duration,
            raw.annotations.description,
        ):
            event_code = normalize_event_code(str(description))
            if event_code is None:
                continue
            if event_code == "T0" and not args.include_rest:
                continue

            condition = condition_for_event(recording.run, event_code)
            if condition is None:
                continue

            trial_onset = float(onset) + args.tmin
            trial_end = float(onset) + args.tmax
            if trial_onset < 0 or trial_end > duration:
                continue

            rows.append(
                {
                    "recording": recording.recording,
                    "subject": recording.subject,
                    "run": recording.run,
                    "task_family": recording.task_family,
                    "condition": condition,
                    "event_code": event_code,
                    "event_onset_sec": f"{float(onset):.3f}",
                    "event_duration_sec": f"{float(annotation_duration):.3f}",
                    "onset_sec": f"{trial_onset:.3f}",
                    "duration_sec": f"{args.tmax - args.tmin:.3f}",
                    "tmin": f"{args.tmin:.3f}",
                    "tmax": f"{args.tmax:.3f}",
                    "edf_path": str(recording.path),
                    "recording_trial_index": local_count,
                }
            )
            local_count += 1

        counts = Counter(row["condition"] for row in rows if row["recording"] == recording.recording)
        print(f"{recording.recording}: {local_count} trials {dict(sorted(counts.items()))}")

    for idx, row in enumerate(rows):
        row["trial_index"] = idx

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial_index",
        "recording",
        "subject",
        "run",
        "task_family",
        "condition",
        "event_code",
        "event_onset_sec",
        "event_duration_sec",
        "onset_sec",
        "duration_sec",
        "tmin",
        "tmax",
        "recording_trial_index",
        "edf_path",
    ]
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote: {args.output_csv}")
    print(f"Total trials: {len(rows)}")
    print(f"Subjects: {len(set(row['subject'] for row in rows))}")
    print(f"Condition counts: {dict(sorted(Counter(row['condition'] for row in rows).items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
