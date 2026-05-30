#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SpecParam and IRASA aperiodic decompositions."
    )
    parser.add_argument(
        "--specparam-npz",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/results/sleep_edf_full/specparam/specparam_fixed.npz"),
    )
    parser.add_argument(
        "--irasa-npz",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/results/sleep_edf_full/irasa/irasa_aperiodic.npz"),
    )
    parser.add_argument(
        "--irasa-index-csv",
        type=Path,
        default=None,
        help="Optional IRASA index CSV containing raw_index/specparam row mapping.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/reports/tables/irasa_specparam_agreement"),
    )
    parser.add_argument("--max-epochs", type=int, default=None)
    return parser.parse_args()


def read_raw_indices(path: Path) -> list[int]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return [int(row["raw_index"]) for row in rows]


def summarize(values):
    import numpy as np

    values = np.asarray(values, dtype=float).reshape(-1)
    return {
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values, ddof=1)),
        "median": float(np.nanmedian(values)),
        "p05": float(np.nanquantile(values, 0.05)),
        "p95": float(np.nanquantile(values, 0.95)),
    }


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows) -> None:
    lines = [
        "| comparison | mean | median | p05 | p95 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {comparison} | {mean:.4f} | {median:.4f} | {p05:.4f} | {p95:.4f} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    import numpy as np

    args = parse_args()
    spec = np.load(args.specparam_npz)
    irasa = np.load(args.irasa_npz)
    n = len(irasa["aperiodic_log_psd"])
    if args.max_epochs is not None:
        n = min(n, args.max_epochs)
    if args.irasa_index_csv is not None:
        raw_indices = np.array(read_raw_indices(args.irasa_index_csv)[:n], dtype=int)
    elif "raw_indices" in irasa:
        raw_indices = irasa["raw_indices"][:n].astype(int)
    else:
        raw_indices = np.arange(n, dtype=int)

    target_freqs = irasa["freqs"].astype("float32")
    spec_freqs = spec["freqs"].astype("float32")
    spec_ap = spec["aperiodic_log_psd"][raw_indices]
    irasa_ap = irasa["aperiodic_log_psd"][:n]
    spec_interp = np.empty_like(irasa_ap)
    for epoch_idx in range(n):
        for ch_idx in range(irasa_ap.shape[1]):
            spec_interp[epoch_idx, ch_idx] = np.interp(
                target_freqs,
                spec_freqs,
                spec_ap[epoch_idx, ch_idx],
            )

    diff = spec_interp - irasa_ap
    centered_spec = spec_interp - spec_interp.mean(axis=-1, keepdims=True)
    centered_irasa = irasa_ap - irasa_ap.mean(axis=-1, keepdims=True)
    centered_diff = centered_spec - centered_irasa
    corr = np.sum(centered_spec * centered_irasa, axis=-1) / np.maximum(
        np.sqrt(np.sum(centered_spec**2, axis=-1) * np.sum(centered_irasa**2, axis=-1)),
        1e-12,
    )
    rows = []
    for name, values in {
        "aperiodic_mae_log10_power": np.abs(diff),
        "aperiodic_bias_specparam_minus_irasa": diff,
        "aperiodic_centered_mae_log10_power": np.abs(centered_diff),
        "aperiodic_shape_corr": corr,
    }.items():
        rows.append({"comparison": name, **summarize(values)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "irasa_specparam_agreement.csv"
    md_path = args.output_dir / "irasa_specparam_agreement.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    (args.output_dir / "irasa_specparam_agreement_metadata.json").write_text(
        json.dumps(
            {
                "specparam_npz": str(args.specparam_npz),
                "irasa_npz": str(args.irasa_npz),
                "irasa_index_csv": None if args.irasa_index_csv is None else str(args.irasa_index_csv),
                "n_epochs": int(n),
                "n_channels": int(irasa_ap.shape[1]),
                "n_freqs": int(irasa_ap.shape[2]),
                "raw_index_min": int(raw_indices.min()) if len(raw_indices) else None,
                "raw_index_max": int(raw_indices.max()) if len(raw_indices) else None,
            },
            indent=2,
        )
    )
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
