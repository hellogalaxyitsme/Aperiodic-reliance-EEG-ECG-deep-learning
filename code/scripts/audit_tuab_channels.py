#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_SRC))

from aperiodic_eeg.tuab import (  # noqa: E402
    TUAB_STANDARD_CHANNELS,
    local_edf_path,
    normalize_channel_name,
    resolve_tuab_channels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit TUAB channel availability for the selected EDF subset."
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
        "--output-file-csv",
        type=Path,
        default=Path(
            "/mnt/data/aperiodic_confounds/results/tuab_subset_200/"
            "tuab_channel_audit_files.csv"
        ),
    )
    parser.add_argument(
        "--output-summary-json",
        type=Path,
        default=Path(
            "/mnt/data/aperiodic_confounds/results/tuab_subset_200/"
            "tuab_channel_audit_summary.json"
        ),
    )
    parser.add_argument("--channels", nargs="+", default=TUAB_STANDARD_CHANNELS)
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
    file_rows: list[dict[str, object]] = []
    normalized_counter: Counter[str] = Counter()
    n_ok = 0
    missing_counter: Counter[str] = Counter()
    n_by_split_label: Counter[str] = Counter()
    n_ok_by_split_label: Counter[str] = Counter()
    sfreq_counter: Counter[str] = Counter()

    for idx, row in enumerate(selected):
        path = local_edf_path(args.data_root, row["remote_rel_path"])
        raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        normalized = [normalize_channel_name(name) for name in raw.ch_names]
        normalized_counter.update(normalized)
        sfreq_counter.update([f"{float(raw.info['sfreq']):g}"])

        split_label = f"{row['official_split']}/{row['label']}"
        n_by_split_label.update([split_label])
        try:
            resolved = resolve_tuab_channels(args.channels, raw.ch_names)
            ok = True
            missing = []
            n_ok += 1
            n_ok_by_split_label.update([split_label])
        except ValueError:
            resolved_map = {
                normalize_channel_name(name): name
                for name in raw.ch_names
            }
            missing = [
                channel
                for channel in args.channels
                if normalize_channel_name(channel) not in resolved_map
            ]
            missing_counter.update(missing)
            resolved = []
            ok = False

        file_rows.append(
            {
                "remote_rel_path": row["remote_rel_path"],
                "subject_id": row["subject_id"],
                "official_split": row["official_split"],
                "label": row["label"],
                "n_channels": len(raw.ch_names),
                "sfreq": float(raw.info["sfreq"]),
                "duration_sec": float(raw.n_times / raw.info["sfreq"]),
                "has_standard_channels": int(ok),
                "missing_standard_channels": ",".join(missing),
                "resolved_standard_channels": "|".join(resolved),
                "all_channels": "|".join(raw.ch_names),
            }
        )
        if (idx + 1) % 25 == 0:
            print(f"Audited {idx + 1}/{len(selected)} EDF headers", flush=True)

    channel_presence = {
        channel: int(normalized_counter[normalize_channel_name(channel)])
        for channel in args.channels
    }
    summary = {
        "data_root": str(args.data_root),
        "selected_files_csv": str(args.selected_files_csv),
        "n_files": len(selected),
        "requested_channels": list(args.channels),
        "n_files_with_all_requested_channels": int(n_ok),
        "fraction_files_with_all_requested_channels": float(n_ok / len(selected)) if selected else 0.0,
        "file_counts_by_split_label": dict(sorted(n_by_split_label.items())),
        "ok_file_counts_by_split_label": dict(sorted(n_ok_by_split_label.items())),
        "sfreq_counts": dict(sorted(sfreq_counter.items())),
        "requested_channel_presence_counts": channel_presence,
        "missing_requested_channel_counts": dict(sorted(missing_counter.items())),
        "top_normalized_channel_counts": dict(normalized_counter.most_common(80)),
    }

    write_csv(args.output_file_csv, file_rows)
    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if n_ok == len(selected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
