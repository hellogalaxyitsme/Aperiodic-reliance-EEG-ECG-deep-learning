#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract non-signal TUAB EDF header metadata for a selected subset."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/tuab/v3.0.1_random_stratified_200"),
    )
    parser.add_argument(
        "--selected-files-csv",
        type=Path,
        default=Path(
            "results/tuab_subset_200/"
            "tuab_v3_0_1_random_stratified_200_files.csv"
        ),
    )
    parser.add_argument(
        "--output-file-csv",
        type=Path,
        default=Path(
            "results/tuab_subset_200/"
            "tuab_v3_0_1_random_stratified_200_header_metadata_files.csv"
        ),
    )
    parser.add_argument(
        "--output-subject-csv",
        type=Path,
        default=Path(
            "results/tuab_subset_200/"
            "tuab_v3_0_1_random_stratified_200_header_metadata_subjects.csv"
        ),
    )
    parser.add_argument(
        "--output-summary-json",
        type=Path,
        default=Path(
            "results/tuab_subset_200/"
            "tuab_v3_0_1_random_stratified_200_header_metadata_summary.json"
        ),
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


def json_default(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def sex_label(value) -> str:
    if value in (1, "1"):
        return "male"
    if value in (2, "2"):
        return "female"
    return "unknown"


def maybe_age(subject_info, meas_date) -> float | None:
    birthday = subject_info.get("birthday") if isinstance(subject_info, dict) else None
    if birthday is None or meas_date is None:
        return None
    if isinstance(meas_date, datetime):
        meas_day = meas_date.date()
    elif isinstance(meas_date, date):
        meas_day = meas_date
    else:
        return None
    if isinstance(birthday, datetime):
        birth_day = birthday.date()
    elif isinstance(birthday, date):
        birth_day = birthday
    else:
        return None
    return (meas_day - birth_day).days / 365.25


def read_fixed_edf_header(path: Path) -> dict[str, object]:
    """Read TUAB demographic strings from the fixed 256-byte EDF header.

    MNE parses the anonymized TUAB birth date as year 0000 and therefore does
    not expose age. TUAB v3.0.1 stores it in the patient field as, for example:
    ``aaaaabdo F 01-JAN-0000 aaaaabdo Age:75``.
    """
    with path.open("rb") as f:
        header = f.read(256)
    patient = header[8:88].decode("latin1", errors="replace").strip()
    recording = header[88:168].decode("latin1", errors="replace").strip()

    age_match = re.search(r"\bAge\s*:\s*(\d{1,3})\b", patient, flags=re.IGNORECASE)
    age = int(age_match.group(1)) if age_match else None
    if age is not None and age > 120:
        age = None

    sex = None
    parts = patient.split()
    if len(parts) >= 2 and parts[1].upper() in {"M", "F"}:
        sex = parts[1].upper()

    return {
        "patient_header": patient,
        "recording_header": recording,
        "age_years_header": age,
        "sex_header": sex,
    }


def main() -> int:
    args = parse_args()

    import mne

    selected = read_csv(args.selected_files_csv)
    file_rows: list[dict[str, object]] = []

    for idx, row in enumerate(selected):
        rel_path = "edf/" + row["remote_rel_path"]
        path = args.data_root / rel_path
        if not path.exists():
            raise FileNotFoundError(path)
        raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        subject_info = raw.info.get("subject_info") or {}
        meas_date = raw.info.get("meas_date")
        parsed_header = read_fixed_edf_header(path)
        header_age = parsed_header["age_years_header"]
        age = header_age if header_age is not None else maybe_age(subject_info, meas_date)
        sex = subject_info.get("sex") if isinstance(subject_info, dict) else None
        header_sex = parsed_header["sex_header"]
        sex_text = sex_label(sex)
        if sex_text == "unknown" and header_sex in {"M", "F"}:
            sex_text = "male" if header_sex == "M" else "female"

        file_rows.append(
            {
                "remote_rel_path": row["remote_rel_path"],
                "local_rel_path": rel_path,
                "subject_id": row["subject_id"],
                "official_split": row["official_split"],
                "label": row["label"],
                "session": row["session"],
                "token": row["token"],
                "size_bytes": row["size_bytes"],
                "n_channels": len(raw.ch_names),
                "sfreq": float(raw.info["sfreq"]),
                "n_times": int(raw.n_times),
                "duration_sec": float(raw.n_times / raw.info["sfreq"]),
                "meas_date": meas_date.isoformat() if hasattr(meas_date, "isoformat") else "",
                "sex_code": "" if sex is None else sex,
                "sex": sex_text,
                "sex_header": "" if header_sex is None else header_sex,
                "age_years": "" if age is None else f"{age:.3f}",
                "age_source": "edf_patient_header" if header_age is not None else "",
                "patient_header": parsed_header["patient_header"],
                "recording_header": parsed_header["recording_header"],
                "subject_info_json": json.dumps(subject_info, default=json_default, sort_keys=True),
            }
        )
        if (idx + 1) % 25 == 0:
            print(f"Read {idx + 1}/{len(selected)} EDF headers", flush=True)

    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in file_rows:
        grouped[(str(row["official_split"]), str(row["label"]), str(row["subject_id"]))].append(
            row
        )

    subject_rows: list[dict[str, object]] = []
    for (split, label, subject), rows in sorted(grouped.items()):
        sex_values = sorted({str(row["sex"]) for row in rows if row["sex"] != "unknown"})
        age_values = [
            float(row["age_years"])
            for row in rows
            if str(row["age_years"]).strip() not in {"", "nan"}
        ]
        subject_rows.append(
            {
                "subject_id": subject,
                "official_split": split,
                "label": label,
                "n_files": len(rows),
                "total_duration_sec": sum(float(row["duration_sec"]) for row in rows),
                "mean_duration_sec": sum(float(row["duration_sec"]) for row in rows) / len(rows),
                "min_n_channels": min(int(row["n_channels"]) for row in rows),
                "max_n_channels": max(int(row["n_channels"]) for row in rows),
                "sfreq_values": ",".join(sorted({f"{float(row['sfreq']):g}" for row in rows})),
                "sex": sex_values[0] if len(sex_values) == 1 else "unknown_or_mixed",
                "age_years_first_available": age_values[0] if age_values else "",
            }
        )

    def count_by(rows, *keys):
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            key = "/".join(str(row[k]) for k in keys)
            counts[key] += 1
        return dict(sorted(counts.items()))

    summary = {
        "data_root": str(args.data_root),
        "selected_files_csv": str(args.selected_files_csv),
        "n_files": len(file_rows),
        "n_subject_label_rows": len(subject_rows),
        "n_unique_subjects": len({row["subject_id"] for row in subject_rows}),
        "file_counts_by_split_label": count_by(file_rows, "official_split", "label"),
        "subject_counts_by_split_label": count_by(subject_rows, "official_split", "label"),
        "subject_counts_by_split_label_sex": count_by(
            subject_rows, "official_split", "label", "sex"
        ),
        "age_available_subject_rows": sum(
            1 for row in subject_rows if str(row["age_years_first_available"]).strip()
        ),
        "age_note": (
            "Ages are parsed from the EDF patient header Age: field. Values above "
            "120 are treated as missing sentinels."
        ),
    }

    write_csv(args.output_file_csv, file_rows)
    write_csv(args.output_subject_csv, subject_rows)
    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
