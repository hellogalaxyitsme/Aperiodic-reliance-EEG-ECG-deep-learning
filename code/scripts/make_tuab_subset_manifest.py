#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path


DEFAULT_COUNTS = {
    ("train", "normal"): 60,
    ("train", "abnormal"): 60,
    ("eval", "normal"): 40,
    ("eval", "abnormal"): 40,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a reproducible 200-subject TUAB subset manifest from an "
            "rsync --list-only listing while preserving official train/eval splits."
        )
    )
    parser.add_argument(
        "--listing",
        type=Path,
        default=Path("results/tuab_subset_200/tuab_v3_0_1_edf_rsync_listing.txt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tuab_subset_200"),
    )
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--train-normal", type=int, default=DEFAULT_COUNTS[("train", "normal")])
    parser.add_argument(
        "--train-abnormal", type=int, default=DEFAULT_COUNTS[("train", "abnormal")]
    )
    parser.add_argument("--eval-normal", type=int, default=DEFAULT_COUNTS[("eval", "normal")])
    parser.add_argument("--eval-abnormal", type=int, default=DEFAULT_COUNTS[("eval", "abnormal")])
    parser.add_argument(
        "--allow-train-label-conflicts",
        action="store_true",
        help=(
            "Allow train subjects that appear under both normal and abnormal. "
            "Default excludes them from the subject sampling pool."
        ),
    )
    return parser.parse_args()


def parse_rsync_listing(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pattern = re.compile(
        r"^(?P<mode>[-dl][rwx-]{9})\s+"
        r"(?P<size>[0-9,]+)\s+"
        r"(?P<date>\d{4}/\d{2}/\d{2})\s+"
        r"(?P<time>\d{2}:\d{2}:\d{2})\s+"
        r"(?P<path>.+)$"
    )
    file_pattern = re.compile(
        r"^(?P<split>train|eval)/(?P<label>normal|abnormal)/01_tcp_ar/"
        r"(?P<filename>(?P<subject>[^_/]+)_s(?P<session>\d+)_t(?P<token>\d+)\.edf)$"
    )
    for line in path.read_text().splitlines():
        match = pattern.match(line.strip())
        if match is None:
            continue
        rel_path = match.group("path")
        file_match = file_pattern.match(rel_path)
        if file_match is None:
            continue
        rows.append(
            {
                "remote_rel_path": rel_path,
                "file_name": file_match.group("filename"),
                "subject_id": file_match.group("subject"),
                "session": int(file_match.group("session")),
                "token": int(file_match.group("token")),
                "official_split": file_match.group("split"),
                "label": file_match.group("label"),
                "montage": "01_tcp_ar",
                "size_bytes": int(match.group("size").replace(",", "")),
                "mtime_date": match.group("date"),
                "mtime_time": match.group("time"),
            }
        )
    if not rows:
        raise ValueError(f"No TUAB EDF rows parsed from {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_subject_pool(file_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    labels_by_split_subject: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in file_rows:
        key = (str(row["official_split"]), str(row["label"]), str(row["subject_id"]))
        grouped[key].append(row)
        labels_by_split_subject[(str(row["official_split"]), str(row["subject_id"]))].add(
            str(row["label"])
        )

    pool = []
    for (split, label, subject), rows in sorted(grouped.items()):
        pool.append(
            {
                "subject_id": subject,
                "official_split": split,
                "label": label,
                "n_files": len(rows),
                "total_size_bytes": sum(int(row["size_bytes"]) for row in rows),
                "sessions": ",".join(
                    f"s{int(row['session']):03d}_t{int(row['token']):03d}" for row in rows
                ),
                "has_label_conflict_within_split": len(labels_by_split_subject[(split, subject)])
                > 1,
            }
        )
    return pool


def select_subjects(
    pool: list[dict[str, object]],
    counts: dict[tuple[str, str], int],
    seed: int,
    allow_train_label_conflicts: bool,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    selected = []
    for split_label, n_subjects in counts.items():
        split, label = split_label
        candidates = [
            row
            for row in pool
            if row["official_split"] == split and row["label"] == label
        ]
        if split == "train" and not allow_train_label_conflicts:
            candidates = [
                row for row in candidates if not bool(row["has_label_conflict_within_split"])
            ]
        if len(candidates) < n_subjects:
            raise ValueError(
                f"Requested {n_subjects} {split}/{label} subjects, "
                f"but only {len(candidates)} candidates are available."
            )
        candidates = sorted(candidates, key=lambda row: str(row["subject_id"]))
        picked = rng.sample(candidates, n_subjects)
        for row in sorted(picked, key=lambda item: str(item["subject_id"])):
            selected.append(
                {
                    **row,
                    "subset_name": "random_stratified_200",
                    "selection_seed": seed,
                    "target_n_for_split_label": n_subjects,
                }
            )
    return selected


def selected_file_rows(
    file_rows: list[dict[str, object]],
    selected_subjects: list[dict[str, object]],
) -> list[dict[str, object]]:
    selected_keys = {
        (str(row["official_split"]), str(row["label"]), str(row["subject_id"]))
        for row in selected_subjects
    }
    out = [
        {
            **row,
            "subset_name": "random_stratified_200",
            "source_base": "nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/tuh_eeg_abnormal/v3.0.1/",
        }
        for row in file_rows
        if (str(row["official_split"]), str(row["label"]), str(row["subject_id"]))
        in selected_keys
    ]
    return sorted(
        out,
        key=lambda row: (
            str(row["official_split"]),
            str(row["label"]),
            str(row["subject_id"]),
            int(row["session"]),
            int(row["token"]),
        ),
    )


def main() -> int:
    args = parse_args()
    counts = {
        ("train", "normal"): args.train_normal,
        ("train", "abnormal"): args.train_abnormal,
        ("eval", "normal"): args.eval_normal,
        ("eval", "abnormal"): args.eval_abnormal,
    }

    file_rows = parse_rsync_listing(args.listing)
    pool = make_subject_pool(file_rows)
    selected_subject_rows = select_subjects(
        pool,
        counts=counts,
        seed=args.seed,
        allow_train_label_conflicts=args.allow_train_label_conflicts,
    )
    selected_files = selected_file_rows(file_rows, selected_subject_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "tuab_v3_0_1_all_edf_files.csv", file_rows)
    write_csv(args.output_dir / "tuab_v3_0_1_subject_pool.csv", pool)
    write_csv(
        args.output_dir / "tuab_v3_0_1_random_stratified_200_subjects.csv",
        selected_subject_rows,
    )
    write_csv(
        args.output_dir / "tuab_v3_0_1_random_stratified_200_files.csv",
        selected_files,
    )

    files_from = args.output_dir / "tuab_v3_0_1_random_stratified_200_files_from.txt"
    files_from.write_text(
        "\n".join(
            ["AAREADME.txt"]
            + [f"edf/{row['remote_rel_path']}" for row in selected_files]
        )
        + "\n"
    )

    summary = {
        "listing": str(args.listing),
        "subset_name": "random_stratified_200",
        "selection_seed": int(args.seed),
        "allow_train_label_conflicts": bool(args.allow_train_label_conflicts),
        "target_counts": {f"{k[0]}/{k[1]}": v for k, v in counts.items()},
        "n_all_files": len(file_rows),
        "n_all_subject_label_rows": len(pool),
        "n_selected_subject_label_rows": len(selected_subject_rows),
        "n_selected_unique_subjects": len({row["subject_id"] for row in selected_subject_rows}),
        "n_selected_files": len(selected_files),
        "selected_total_size_bytes": sum(int(row["size_bytes"]) for row in selected_files),
        "selected_total_size_gib": sum(int(row["size_bytes"]) for row in selected_files)
        / (1024**3),
        "outputs": {
            "all_files": "tuab_v3_0_1_all_edf_files.csv",
            "subject_pool": "tuab_v3_0_1_subject_pool.csv",
            "selected_subjects": "tuab_v3_0_1_random_stratified_200_subjects.csv",
            "selected_files": "tuab_v3_0_1_random_stratified_200_files.csv",
            "files_from": "tuab_v3_0_1_random_stratified_200_files_from.txt",
        },
        "note": (
            "Age/sex matching is not possible from the rsync listing alone. "
            "Use downloaded EDF headers to audit demographics and build an "
            "age-matched follow-up subset if needed."
        ),
    }
    (args.output_dir / "tuab_v3_0_1_random_stratified_200_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
