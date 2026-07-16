#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a TUAB abnormal-normal age-matched subject subset."
    )
    parser.add_argument(
        "--metadata-subjects-csv",
        type=Path,
        default=Path(
            "results/tuab_subset_200/"
            "tuab_v3_0_1_random_stratified_200_header_metadata_subjects.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tuab_subset_200/age_matched"),
    )
    parser.add_argument("--caliper-years", type=float, default=5.0)
    parser.add_argument("--same-sex", action="store_true", default=True)
    parser.add_argument("--allow-cross-sex", action="store_false", dest="same_sex")
    parser.add_argument("--max-valid-age", type=float, default=120.0)
    parser.add_argument("--prefix", default="tuab_age_sex_matched_caliper5")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(
    path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0].keys()))
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def parse_age(row: dict[str, str], max_valid_age: float) -> float | None:
    raw = str(row.get("age_years_first_available", "")).strip()
    if not raw:
        return None
    age = float(raw)
    if age < 0 or age > max_valid_age:
        return None
    return age


def linear_sum_assignment(cost):
    try:
        from scipy.optimize import linear_sum_assignment as scipy_lsa

        return scipy_lsa(cost)
    except Exception:
        return None


def optimal_pairs(abnormal, normal, caliper_years: float):
    import numpy as np

    if not abnormal or not normal:
        return []
    cost = np.zeros((len(abnormal), len(normal)), dtype=float)
    high_cost = 1e9
    for i, ab_row in enumerate(abnormal):
        for j, norm_row in enumerate(normal):
            diff = abs(float(ab_row["age"]) - float(norm_row["age"]))
            cost[i, j] = diff if diff <= caliper_years else high_cost

    assigned = linear_sum_assignment(cost)
    if assigned is not None:
        row_idx, col_idx = assigned
        pairs = []
        for i, j in zip(row_idx, col_idx):
            if cost[i, j] <= caliper_years:
                pairs.append((abnormal[int(i)], normal[int(j)], float(cost[i, j])))
        return pairs

    used_normals = set()
    pairs = []
    for ab_row in sorted(abnormal, key=lambda row: (float(row["age"]), row["subject_id"])):
        candidates = []
        for norm_row in normal:
            if norm_row["subject_id"] in used_normals:
                continue
            diff = abs(float(ab_row["age"]) - float(norm_row["age"]))
            if diff <= caliper_years:
                candidates.append((diff, norm_row))
        if candidates:
            diff, norm_row = min(candidates, key=lambda item: (item[0], item[1]["subject_id"]))
            used_normals.add(norm_row["subject_id"])
            pairs.append((ab_row, norm_row, float(diff)))
    return pairs


def make_markdown(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# TUAB Age-Matched Subject Subset",
        "",
        f"Caliper: +/-{summary['caliper_years']} years",
        f"Same-sex matching: {summary['same_sex']}",
        "",
        "| Split | Pairs | Mean absolute age difference | Max absolute age difference |",
        "| --- | ---: | ---: | ---: |",
    ]
    for split, item in summary["by_split"].items():
        lines.append(
            "| {split} | {n_pairs} | {mean_abs_age_diff:.2f} | {max_abs_age_diff:.2f} |".format(
                split=split,
                n_pairs=item["n_pairs"],
                mean_abs_age_diff=item["mean_abs_age_diff"],
                max_abs_age_diff=item["max_abs_age_diff"],
            )
        )
    lines.extend(
        [
            "",
            f"Total pairs: {summary['n_pairs_total']}",
            f"Total subjects: {summary['n_subject_rows']}",
            "",
            "This subset preserves the official TUAB train/eval boundary and matches",
            "abnormal subjects to normal subjects within each split.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()

    rows = []
    skipped = []
    for row in read_csv(args.metadata_subjects_csv):
        age = parse_age(row, args.max_valid_age)
        if age is None:
            skipped.append(
                {
                    "subject_id": row["subject_id"],
                    "official_split": row["official_split"],
                    "label": row["label"],
                    "sex": row.get("sex", ""),
                    "age_years_first_available": row.get("age_years_first_available", ""),
                    "reason": "missing_or_invalid_age",
                }
            )
            continue
        rows.append({**row, "age": age})

    grouped: dict[tuple[str, str], dict[str, list[dict[str, object]]]] = defaultdict(
        lambda: {"abnormal": [], "normal": []}
    )
    for row in rows:
        split = str(row["official_split"])
        sex_key = str(row.get("sex", "unknown")) if args.same_sex else "any"
        grouped[(split, sex_key)][str(row["label"])].append(row)

    pair_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    by_split_diffs: dict[str, list[float]] = defaultdict(list)
    pair_counter_by_split: dict[str, int] = defaultdict(int)

    for (split, sex_key), group in sorted(grouped.items()):
        pairs = optimal_pairs(group["abnormal"], group["normal"], args.caliper_years)
        for ab_row, norm_row, diff in pairs:
            pair_counter_by_split[split] += 1
            pair_id = f"{split}_pair_{pair_counter_by_split[split]:03d}"
            by_split_diffs[split].append(diff)
            pair_rows.append(
                {
                    "pair_id": pair_id,
                    "official_split": split,
                    "sex_match_key": sex_key,
                    "abnormal_subject_id": ab_row["subject_id"],
                    "normal_subject_id": norm_row["subject_id"],
                    "abnormal_age": f"{float(ab_row['age']):.3f}",
                    "normal_age": f"{float(norm_row['age']):.3f}",
                    "abnormal_sex": ab_row.get("sex", ""),
                    "normal_sex": norm_row.get("sex", ""),
                    "abs_age_diff": f"{diff:.3f}",
                }
            )
            for label, row, matched in [
                ("abnormal", ab_row, norm_row),
                ("normal", norm_row, ab_row),
            ]:
                subject_rows.append(
                    {
                        "pair_id": pair_id,
                        "official_split": split,
                        "label": label,
                        "subject_id": row["subject_id"],
                        "sex": row.get("sex", ""),
                        "age_years": f"{float(row['age']):.3f}",
                        "matched_subject_id": matched["subject_id"],
                        "matched_age_years": f"{float(matched['age']):.3f}",
                        "abs_age_diff": f"{diff:.3f}",
                    }
                )

    by_split = {}
    for split, diffs in sorted(by_split_diffs.items()):
        by_split[split] = {
            "n_pairs": len(diffs),
            "mean_abs_age_diff": float(sum(diffs) / len(diffs)) if diffs else 0.0,
            "max_abs_age_diff": float(max(diffs)) if diffs else 0.0,
        }

    summary = {
        "metadata_subjects_csv": str(args.metadata_subjects_csv),
        "caliper_years": float(args.caliper_years),
        "same_sex": bool(args.same_sex),
        "max_valid_age": float(args.max_valid_age),
        "n_input_subject_rows": int(len(rows) + len(skipped)),
        "n_valid_age_subject_rows": int(len(rows)),
        "n_skipped_subject_rows": int(len(skipped)),
        "skipped_subjects": skipped,
        "n_pairs_total": int(len(pair_rows)),
        "n_subject_rows": int(len(subject_rows)),
        "by_split": by_split,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / f"{args.prefix}_pairs.csv", pair_rows)
    write_csv(args.output_dir / f"{args.prefix}_subjects.csv", subject_rows)
    write_csv(
        args.output_dir / f"{args.prefix}_skipped_subjects.csv",
        skipped,
        fieldnames=[
            "subject_id",
            "official_split",
            "label",
            "sex",
            "age_years_first_available",
            "reason",
        ],
    )
    (args.output_dir / f"{args.prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    make_markdown(args.output_dir / f"{args.prefix}_summary.md", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
