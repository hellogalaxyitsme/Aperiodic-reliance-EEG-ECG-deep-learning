#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose ridge logits into aperiodic and flattened input contributions."
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=Path("results/sleep_edf_subset/psd_index.csv"),
    )
    parser.add_argument(
        "--decomp-npz",
        type=Path,
        default=Path("results/sleep_edf_subset/specparam/specparam_fixed.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/sleep_edf_subset/linear_logit_contributions"),
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    return parser.parse_args()


def make_task_labels(stages):
    import numpy as np

    stages = np.asarray(stages)
    return {
        "wake_vs_sleep": np.where(stages == "W", "W", "Sleep"),
        "n2_vs_n3": np.where(np.isin(stages, ["N2", "N3"]), stages, None),
        "five_stage": stages,
    }


def encode_labels(labels):
    import numpy as np
    from sklearn.preprocessing import LabelEncoder

    labels = np.asarray(labels)
    mask = labels != None  # noqa: E711
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels[mask])
    return mask, y, encoder.classes_.tolist()


def make_group_folds(groups, n_splits: int):
    import numpy as np
    from sklearn.model_selection import GroupKFold

    unique_groups = np.unique(groups)
    n_splits = min(n_splits, len(unique_groups))
    splitter = GroupKFold(n_splits=n_splits)
    placeholder_y = np.zeros(len(groups), dtype=int)
    return list(splitter.split(placeholder_y, groups=groups))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_ratio(numerator, denominator):
    import numpy as np

    denominator = np.asarray(denominator)
    return numerator / np.maximum(denominator, 1e-12)


def summarize(rows):
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(rows)
    summaries = []
    metrics = [
        "mean_abs_full_margin",
        "mean_abs_aperiodic_margin",
        "mean_abs_flattened_margin",
        "abs_aperiodic_margin_fraction",
        "abs_flattened_margin_fraction",
        "mean_signed_aperiodic_fraction",
        "mean_signed_flattened_fraction",
    ]

    for key, group in df.groupby(["task", "class_label"], sort=False):
        task, class_label = key
        row = {"task": task, "class_label": class_label, "n_folds": int(len(group))}
        for metric in metrics:
            vals = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.nanmean(vals))
            row[f"{metric}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        summaries.append(row)

    for task, group in df.groupby("task", sort=False):
        row = {"task": task, "class_label": "__overall__", "n_folds": int(len(group))}
        for metric in metrics:
            vals = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.nanmean(vals))
            row[f"{metric}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        summaries.append(row)
    return summaries


def main() -> int:
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import RidgeClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    args = parse_args()
    index = pd.read_csv(args.index_csv)
    decomp = np.load(args.decomp_npz)

    full = decomp["log_psd"].reshape(decomp["log_psd"].shape[0], -1)
    aperiodic = decomp["aperiodic_log_psd"].reshape(decomp["aperiodic_log_psd"].shape[0], -1)
    flattened = (
        decomp["log_psd"] - decomp["aperiodic_log_psd"]
    ).reshape(decomp["log_psd"].shape[0], -1)

    task_labels = make_task_labels(index["stage"].to_numpy())
    groups_all = index["subject"].to_numpy()

    fold_rows = []
    for task_name, labels in task_labels.items():
        mask, y, classes = encode_labels(labels)
        task_groups = groups_all[mask]
        folds = make_group_folds(task_groups, args.n_splits)

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            clf = make_pipeline(
                StandardScaler(),
                RidgeClassifier(
                    alpha=args.ridge_alpha,
                    class_weight="balanced",
                    solver="lsqr",
                ),
            )
            clf.fit(full[mask][train_idx], y[train_idx])

            scaler = clf.named_steps["standardscaler"]
            ridge = clf.named_steps["ridgeclassifier"]
            coef = ridge.coef_
            intercept = ridge.intercept_
            if coef.ndim == 1:
                coef = coef[None, :]
            if intercept.ndim == 0:
                intercept = intercept[None]

            full_test = scaler.transform(full[mask][test_idx])
            ap_test = (aperiodic[mask][test_idx] - scaler.mean_) / scaler.scale_
            flat_test = flattened[mask][test_idx] / scaler.scale_

            full_logits = full_test @ coef.T + intercept
            ap_logits = ap_test @ coef.T + intercept
            flat_logits_no_intercept = flat_test @ coef.T

            if len(classes) == 2 and coef.shape[0] == 1:
                class_specs = [
                    (classes[1], full_logits[:, 0], ap_logits[:, 0], flat_logits_no_intercept[:, 0]),
                    (classes[0], -full_logits[:, 0], -ap_logits[:, 0], -flat_logits_no_intercept[:, 0]),
                ]
            else:
                class_specs = [
                    (
                        classes[class_idx],
                        full_logits[:, class_idx],
                        ap_logits[:, class_idx],
                        flat_logits_no_intercept[:, class_idx],
                    )
                    for class_idx in range(len(classes))
                ]

            for class_label, full_margin, ap_margin, flat_margin in class_specs:
                abs_full = np.abs(full_margin)
                abs_ap = np.abs(ap_margin)
                abs_flat = np.abs(flat_margin)
                abs_total_parts = abs_ap + abs_flat

                fold_rows.append(
                    {
                        "task": task_name,
                        "fold": fold_idx,
                        "class_label": class_label,
                        "n_test": int(len(test_idx)),
                        "mean_abs_full_margin": float(abs_full.mean()),
                        "mean_abs_aperiodic_margin": float(abs_ap.mean()),
                        "mean_abs_flattened_margin": float(abs_flat.mean()),
                        "abs_aperiodic_margin_fraction": float(
                            safe_ratio(abs_ap.mean(), abs_total_parts.mean())
                        ),
                        "abs_flattened_margin_fraction": float(
                            safe_ratio(abs_flat.mean(), abs_total_parts.mean())
                        ),
                        "mean_signed_aperiodic_fraction": float(
                            np.nanmean(safe_ratio(ap_margin, np.abs(ap_margin) + np.abs(flat_margin)))
                        ),
                        "mean_signed_flattened_fraction": float(
                            np.nanmean(safe_ratio(flat_margin, np.abs(ap_margin) + np.abs(flat_margin)))
                        ),
                        "train_subjects": ",".join(sorted(np.unique(task_groups[train_idx]))),
                        "test_subjects": ",".join(sorted(np.unique(task_groups[test_idx]))),
                    }
                )

    summary_rows = summarize(fold_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "linear_logit_contributions_by_fold.csv", fold_rows)
    write_csv(args.output_dir / "linear_logit_contributions_summary.csv", summary_rows)

    print(f"Wrote outputs to: {args.output_dir}")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

