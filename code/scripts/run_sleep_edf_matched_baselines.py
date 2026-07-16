#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_SRC))

from aperiodic_eeg.spectral import flatten_param_features, flatten_spectral_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offset/exponent matched Sleep-EDF aperiodic-control baselines."
    )
    parser.add_argument(
        "--psd-npz",
        type=Path,
        default=Path("results/sleep_edf_subset/psd_welch_fpz_pz.npz"),
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
        "--reference-summary-csv",
        type=Path,
        default=Path("results/sleep_edf_subset/baselines_specparam/summary_metrics.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/sleep_edf_subset/matched_specparam"),
    )
    parser.add_argument("--n-bins", type=int, default=4)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--min-epochs", type=int, default=200)
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
    if n_splits < 2:
        raise ValueError("Need at least two subject groups for group-level evaluation.")
    splitter = GroupKFold(n_splits=n_splits)
    placeholder_y = np.zeros(len(groups), dtype=int)
    return list(splitter.split(placeholder_y, groups=groups))


def evaluate_feature_set(features, y, groups, n_splits: int, ridge_alpha: float):
    import numpy as np
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(make_group_folds(groups, n_splits)):
        clf = make_pipeline(
            StandardScaler(),
            RidgeClassifier(alpha=ridge_alpha, class_weight="balanced", solver="lsqr"),
        )
        clf.fit(features[train_idx], y[train_idx])
        pred = clf.predict(features[test_idx])
        rows.append(
            {
                "fold": fold_idx,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "train_subjects": ",".join(sorted(np.unique(groups[train_idx]))),
                "test_subjects": ",".join(sorted(np.unique(groups[test_idx]))),
                "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], pred)),
                "macro_f1": float(f1_score(y[test_idx], pred, average="macro")),
                "accuracy": float(accuracy_score(y[test_idx], pred)),
            }
        )
    return rows


def summarize_fold_rows(rows):
    import numpy as np

    out = {}
    for metric in ["balanced_accuracy", "macro_f1", "accuracy"]:
        vals = np.array([row[metric] for row in rows], dtype=float)
        out[f"{metric}_mean"] = float(vals.mean())
        out[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return out


def quantile_bin(values, n_bins: int):
    import numpy as np

    values = np.asarray(values, dtype=float)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, quantiles)
    edges = np.unique(edges)
    if len(edges) <= 2:
        return np.zeros(len(values), dtype=int)
    return np.digitize(values, edges[1:-1], right=True)


def matched_indices(labels, covariates, n_bins: int, seed: int):
    import numpy as np

    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    covariates = np.asarray(covariates)
    bin_columns = [quantile_bin(covariates[:, idx], n_bins) for idx in range(covariates.shape[1])]
    strata = np.array(["_".join(str(col[row]) for col in bin_columns) for row in range(len(labels))])

    selected = []
    class_labels = np.unique(labels)
    stratum_rows = []
    for stratum in sorted(np.unique(strata)):
        stratum_idx = np.flatnonzero(strata == stratum)
        by_class = {
            label: stratum_idx[labels[stratum_idx] == label]
            for label in class_labels
        }
        counts = {str(label): int(len(indices)) for label, indices in by_class.items()}
        min_count = min(counts.values()) if counts else 0
        if min_count <= 0:
            stratum_rows.append(
                {
                    "match_bin": stratum,
                    "kept_per_class": 0,
                    "kept_total": 0,
                    "counts_json": json.dumps(counts),
                }
            )
            continue
        for label, indices in by_class.items():
            selected.extend(rng.choice(indices, size=min_count, replace=False).tolist())
        stratum_rows.append(
            {
                "match_bin": stratum,
                "kept_per_class": int(min_count),
                "kept_total": int(min_count * len(class_labels)),
                "counts_json": json.dumps(counts),
            }
        )

    selected = np.array(sorted(selected), dtype=int)
    return selected, strata, stratum_rows


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

    import numpy as np
    import pandas as pd

    psd_bundle = np.load(args.psd_npz)
    index = pd.read_csv(args.index_csv)
    decomp = np.load(args.decomp_npz)

    if len(index) != decomp["log_psd"].shape[0]:
        raise ValueError("Index and decomposition have different epoch counts.")

    params = np.stack([decomp["offset"], decomp["exponent"]], axis=-1)
    feature_sets = {
        "full_log_psd": flatten_spectral_features(decomp["log_psd"]),
        "aperiodic_spectrum": flatten_spectral_features(decomp["aperiodic_log_psd"]),
        "aperiodic_params": flatten_param_features(params),
        "periodic_residual": flatten_spectral_features(decomp["residual_log_psd"]),
    }

    # Match on mean offset and mean exponent across selected EEG channels.
    covariates = np.column_stack(
        [
            np.nanmean(decomp["offset"], axis=1),
            np.nanmean(decomp["exponent"], axis=1),
        ]
    )

    groups_all = index["subject"].to_numpy()
    task_labels = make_task_labels(index["stage"].to_numpy())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference = None
    if args.reference_summary_csv and args.reference_summary_csv.exists():
        reference = pd.read_csv(args.reference_summary_csv)

    all_fold_rows = []
    summary_rows = []
    match_rows = []
    selected_rows = []

    for task_name, labels in task_labels.items():
        mask, y_all, classes = encode_labels(labels)
        task_abs_idx = np.flatnonzero(mask)
        task_labels_clean = np.asarray(labels)[mask]
        task_covariates = covariates[mask]

        rel_selected, strata, stratum_rows = matched_indices(
            task_labels_clean,
            task_covariates,
            n_bins=args.n_bins,
            seed=args.seed,
        )
        abs_selected = task_abs_idx[rel_selected]
        selected_labels = task_labels_clean[rel_selected]
        selected_groups = groups_all[abs_selected]

        for row in stratum_rows:
            row["task"] = task_name
            match_rows.append(row)

        for rel_idx, abs_idx in zip(rel_selected, abs_selected):
            selected_rows.append(
                {
                    "task": task_name,
                    "psd_index": int(index.iloc[abs_idx]["psd_index"]),
                    "stage": str(index.iloc[abs_idx]["stage"]),
                    "subject": str(index.iloc[abs_idx]["subject"]),
                    "match_bin": str(strata[rel_idx]),
                    "mean_offset": float(covariates[abs_idx, 0]),
                    "mean_exponent": float(covariates[abs_idx, 1]),
                }
            )

        class_counts = {
            str(label): int(np.sum(selected_labels == label))
            for label in np.unique(selected_labels)
        }
        if len(abs_selected) < args.min_epochs or len(np.unique(selected_groups)) < 2:
            print(
                f"Skipping {task_name}: matched subset too small "
                f"({len(abs_selected)} epochs, {len(np.unique(selected_groups))} subjects)",
                file=sys.stderr,
            )
            continue

        _, y, _ = encode_labels(selected_labels)
        task_summaries = {}
        for feature_name, features in feature_sets.items():
            fold_rows = evaluate_feature_set(
                features[abs_selected],
                y,
                selected_groups,
                n_splits=args.n_splits,
                ridge_alpha=args.ridge_alpha,
            )
            for row in fold_rows:
                row.update(
                    {
                        "task": task_name,
                        "feature_set": feature_name,
                        "classes": "|".join(classes),
                        "matching": f"mean_offset_mean_exponent_q{args.n_bins}",
                        "matched_n_epochs": int(len(abs_selected)),
                        "matched_class_counts": json.dumps(class_counts),
                    }
                )
            all_fold_rows.extend(fold_rows)

            summary = summarize_fold_rows(fold_rows)
            task_summaries[feature_name] = summary

            ref_bacc = np.nan
            robustness = np.nan
            if reference is not None:
                ref_match = reference[
                    (reference["task"] == task_name)
                    & (reference["feature_set"] == feature_name)
                ]
                if len(ref_match):
                    ref_bacc = float(ref_match.iloc[0]["balanced_accuracy_mean"])
                    robustness = summary["balanced_accuracy_mean"] / ref_bacc if ref_bacc else np.nan

            summary_rows.append(
                {
                    "task": task_name,
                    "feature_set": feature_name,
                    "classes": "|".join(classes),
                    "matching": f"mean_offset_mean_exponent_q{args.n_bins}",
                    "original_n_epochs": int(mask.sum()),
                    "matched_n_epochs": int(len(abs_selected)),
                    "matched_n_subjects": int(len(np.unique(selected_groups))),
                    "matched_class_counts": json.dumps(class_counts),
                    "reference_balanced_accuracy_mean": ref_bacc,
                    "matching_robustness_balanced_accuracy": robustness,
                    **summary,
                }
            )

        full_bacc = task_summaries["full_log_psd"]["balanced_accuracy_mean"]
        if full_bacc > 0:
            for feature_name in ["aperiodic_spectrum", "aperiodic_params", "periodic_residual"]:
                summary_rows.append(
                    {
                        "task": task_name,
                        "feature_set": f"matched_retention::{feature_name}",
                        "classes": "|".join(classes),
                        "matching": f"mean_offset_mean_exponent_q{args.n_bins}",
                        "original_n_epochs": int(mask.sum()),
                        "matched_n_epochs": int(len(abs_selected)),
                        "matched_n_subjects": int(len(np.unique(selected_groups))),
                        "matched_class_counts": json.dumps(class_counts),
                        "reference_balanced_accuracy_mean": np.nan,
                        "matching_robustness_balanced_accuracy": np.nan,
                        "balanced_accuracy_mean": (
                            task_summaries[feature_name]["balanced_accuracy_mean"] / full_bacc
                        ),
                        "balanced_accuracy_std": np.nan,
                        "macro_f1_mean": (
                            task_summaries[feature_name]["macro_f1_mean"]
                            / max(task_summaries["full_log_psd"]["macro_f1_mean"], 1e-12)
                        ),
                        "macro_f1_std": np.nan,
                        "accuracy_mean": (
                            task_summaries[feature_name]["accuracy_mean"]
                            / max(task_summaries["full_log_psd"]["accuracy_mean"], 1e-12)
                        ),
                        "accuracy_std": np.nan,
                    }
                )

    write_csv(args.output_dir / "matched_fold_metrics.csv", all_fold_rows)
    write_csv(args.output_dir / "matched_summary_metrics.csv", summary_rows)
    write_csv(args.output_dir / "matched_strata.csv", match_rows)
    write_csv(args.output_dir / "matched_epoch_indices.csv", selected_rows)

    print(f"Wrote matched outputs to: {args.output_dir}")
    direct = pd.DataFrame(summary_rows)
    direct = direct[~direct["feature_set"].astype(str).str.startswith("matched_retention::")]
    print(
        direct[
            [
                "task",
                "feature_set",
                "matched_n_epochs",
                "balanced_accuracy_mean",
                "matching_robustness_balanced_accuracy",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

