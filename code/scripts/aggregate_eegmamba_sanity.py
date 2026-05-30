#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate EEGMamba sanity-check runs.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/results/tuab_subset_200/eegmamba_sanity_checks"),
    )
    args = parser.parse_args()

    summary_rows: list[dict[str, object]] = []
    for run_dir in sorted(args.root.glob("*_seed*")):
        boot_path = run_dir / "tuab_eegmamba_intervention_subject_bootstrap.csv"
        meta_path = run_dir / "tuab_eegmamba_intervention_metadata.json"
        if not boot_path.exists() or not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        run_name = run_dir.name.rsplit("_seed", 1)[0]
        seed = run_dir.name.rsplit("_seed", 1)[1]
        for row in read_rows(boot_path):
            if row.get("metric") != "balanced_accuracy":
                continue
            estimate = row.get("estimate")
            if estimate != "performance" and estimate != "drop::phase_flattened":
                continue
            test_input = row.get("test_input")
            if estimate == "performance" and test_input not in {"raw_eeg", "phase_flattened"}:
                continue
            summary_rows.append(
                {
                    "run": run_name,
                    "seed": seed,
                    "input_normalization": meta.get("input_normalization"),
                    "input_divisor": meta.get("input_divisor"),
                    "freeze_backbone": meta.get("freeze_backbone"),
                    "selection_metric": meta.get("selection_metric"),
                    "best_epoch": meta.get("best_epoch"),
                    "best_val_balanced_accuracy": meta.get("best_val_balanced_accuracy"),
                    "test_input": test_input,
                    "estimate": estimate,
                    "point": float(row["point"]),
                    "ci_lower": float(row["ci_lower"]),
                    "ci_upper": float(row["ci_upper"]),
                }
            )

    if not summary_rows:
        raise SystemExit(f"No completed sanity-check runs found under {args.root}")

    out_csv = args.root / "eegmamba_sanity_summary.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "# EEGMamba TUAB Sanity Checks",
        "",
        "| run | seed | raw BA | flattened BA | flattening drop | best val BA | best epoch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_run_seed: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    for row in summary_rows:
        by_run_seed.setdefault((str(row["run"]), str(row["seed"])), {})[
            f"{row['estimate']}::{row['test_input']}"
        ] = row

    for (run, seed), rows in sorted(by_run_seed.items()):
        raw = rows.get("performance::raw_eeg")
        flat = rows.get("performance::phase_flattened")
        drop = rows.get("drop::phase_flattened::phase_flattened")
        if raw is None or flat is None or drop is None:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    run,
                    seed,
                    f"{raw['point']:.3f}",
                    f"{flat['point']:.3f}",
                    f"{drop['point']:.3f}",
                    f"{float(raw['best_val_balanced_accuracy']):.3f}",
                    str(raw["best_epoch"]),
                ]
            )
            + " |"
        )

    out_md = args.root / "eegmamba_sanity_summary.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(out_csv)
    print(out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
