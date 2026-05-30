#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build full TUAB v3.0.1 rsync files-from manifest from parsed EDF listing."
    )
    parser.add_argument(
        "--all-edf-csv",
        type=Path,
        default=Path("results/tuab_subset_200/tuab_v3_0_1_all_edf_files.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tuab_full_v3_0_1"),
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
    rows = read_csv(args.all_edf_csv)
    if not rows:
        raise ValueError(f"No rows in {args.all_edf_csv}")

    rows = sorted(
        rows,
        key=lambda row: (
            row["official_split"],
            row["label"],
            row["subject_id"],
            int(row["session"]),
            int(row["token"]),
            row["remote_rel_path"],
        ),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_csv = args.output_dir / "tuab_v3_0_1_full_edf_files.csv"
    write_csv(full_csv, rows)

    files_from = args.output_dir / "tuab_v3_0_1_full_files_from.txt"
    files_from.write_text(
        "\n".join(["AAREADME.txt"] + [f"edf/{row['remote_rel_path']}" for row in rows])
        + "\n"
    )

    subject_rows = []
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["official_split"], row["label"], row["subject_id"]), []).append(row)
    for (split, label, subject), subject_files in sorted(grouped.items()):
        subject_rows.append(
            {
                "subject_id": subject,
                "official_split": split,
                "label": label,
                "n_files": len(subject_files),
                "total_size_bytes": sum(int(row["size_bytes"]) for row in subject_files),
            }
        )
    subjects_csv = args.output_dir / "tuab_v3_0_1_full_subjects.csv"
    write_csv(subjects_csv, subject_rows)

    file_counter = Counter((row["official_split"], row["label"]) for row in rows)
    subject_counter = Counter((row["official_split"], row["label"]) for row in subject_rows)
    summary = {
        "source_all_edf_csv": str(args.all_edf_csv),
        "full_edf_csv": str(full_csv),
        "files_from": str(files_from),
        "subjects_csv": str(subjects_csv),
        "n_edf_files": len(rows),
        "n_subject_label_rows": len(subject_rows),
        "n_unique_subject_ids": len({row["subject_id"] for row in rows}),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in rows),
        "total_size_gib": sum(int(row["size_bytes"]) for row in rows) / 1024**3,
        "files_by_split_label": {"/".join(key): value for key, value in sorted(file_counter.items())},
        "subjects_by_split_label": {
            "/".join(key): value for key, value in sorted(subject_counter.items())
        },
        "remote_base": "nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/tuh_eeg_abnormal/v3.0.1/",
        "download_note": (
            "Use code/scripts/download_tuab_full_resumable.sh from H200 with SSH agent "
            "forwarding. The script uses rsync without --delete and is safe to rerun."
        ),
    }
    summary_json = args.output_dir / "tuab_v3_0_1_full_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2) + "\n")

    summary_md = args.output_dir / "tuab_v3_0_1_full_summary.md"
    lines = [
        "# TUAB v3.0.1 Full Download Manifest",
        "",
        f"- EDF files: {summary['n_edf_files']}",
        f"- Unique subject IDs: {summary['n_unique_subject_ids']}",
        f"- Subject/split/label rows: {summary['n_subject_label_rows']}",
        f"- Total EDF size: {summary['total_size_gib']:.2f} GiB",
        f"- Files-from: `{files_from}`",
        "",
        "## Files By Split/Label",
        "",
        "| split/label | files | subjects |",
        "| --- | ---: | ---: |",
    ]
    for key in sorted(file_counter):
        key_str = "/".join(key)
        lines.append(f"| {key_str} | {file_counter[key]} | {subject_counter[key]} |")
    summary_md.write_text("\n".join(lines) + "\n")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
