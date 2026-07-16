#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_SRC))

from aperiodic_eeg.spectral import flatten_param_features, flatten_spectral_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run subject-held-out PhysioNet MI aperiodic audit baselines."
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=Path("results/physionet_mi/imagined_fists_psd_index.csv"),
    )
    parser.add_argument(
        "--decomp-npz",
        type=Path,
        default=Path("results/physionet_mi/specparam/imagined_fists_specparam_fixed.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/physionet_mi/baselines_specparam"),
    )
    parser.add_argument("--label-column", default="condition")
    parser.add_argument("--group-column", default="subject")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--max-trials", type=int, default=None)
    return parser.parse_args()


def make_group_folds(groups, n_splits: int):
    import numpy as np
    from sklearn.model_selection import GroupKFold

    unique_groups = np.unique(groups)
    n_splits = min(n_splits, len(unique_groups))
    if n_splits < 2:
        raise ValueError("Need at least two subject groups for group-held-out evaluation.")
    splitter = GroupKFold(n_splits=n_splits)
    placeholder_y = np.zeros(len(groups), dtype=int)
    return list(splitter.split(placeholder_y, groups=groups))


def evaluate_feature_set(features, labels, groups, n_splits: int, ridge_alpha: float):
    import numpy as np
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    folds = make_group_folds(groups, n_splits)
    rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
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
                "classes": ",".join(encoder.classes_),
                "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], pred)),
                "macro_f1": float(f1_score(y[test_idx], pred, average="macro")),
                "accuracy": float(accuracy_score(y[test_idx], pred)),
            }
        )
    return rows


def summarize_rows(rows):
    import numpy as np

    summary = {}
    for metric in ["balanced_accuracy", "macro_f1", "accuracy"]:
        values = np.array([row[metric] for row in rows], dtype=float)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    import csv

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

    index = pd.read_csv(args.index_csv)
    decomp = np.load(args.decomp_npz)
    required = ["log_psd", "aperiodic_log_psd", "residual_log_psd", "offset", "exponent"]
    for key in required:
        if key not in decomp:
            raise ValueError(f"Missing {key!r} from {args.decomp_npz}")

    n_trials = decomp["log_psd"].shape[0]
    if len(index) != n_trials:
        raise ValueError(f"Index rows {len(index)} != decomposition trials {n_trials}")
    if args.max_trials is not None:
        index = index.iloc[: args.max_trials].copy()
        n = len(index)
    else:
        n = n_trials

    params = np.stack([decomp["offset"][:n], decomp["exponent"][:n]], axis=-1)
    feature_sets = {
        "full_log_psd": flatten_spectral_features(decomp["log_psd"][:n]),
        "aperiodic_spectrum": flatten_spectral_features(decomp["aperiodic_log_psd"][:n]),
        "aperiodic_params": flatten_param_features(params),
        "periodic_residual": flatten_spectral_features(decomp["residual_log_psd"][:n]),
    }

    labels = index[args.label_column].astype(str).to_numpy()
    groups = index[args.group_column].astype(str).to_numpy()

    all_fold_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for feature_name, features in feature_sets.items():
        fold_rows = evaluate_feature_set(
            features=features,
            labels=labels,
            groups=groups,
            n_splits=args.n_splits,
            ridge_alpha=args.ridge_alpha,
        )
        for row in fold_rows:
            row["feature_set"] = feature_name
        all_fold_rows.extend(fold_rows)
        summary = summarize_rows(fold_rows)
        summary_rows.append({"feature_set": feature_name, **summary})
        print(f"{feature_name}: {summary}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "fold_metrics.csv", all_fold_rows)
    write_csv(args.output_dir / "summary_metrics.csv", summary_rows)

    r_squared = decomp["r_squared"] if "r_squared" in decomp else np.array([np.nan])
    metadata = {
        "index_csv": str(args.index_csv),
        "decomp_npz": str(args.decomp_npz),
        "n_trials": int(n),
        "n_subjects": int(len(np.unique(groups))),
        "labels": sorted(np.unique(labels).tolist()),
        "n_splits": int(args.n_splits),
        "ridge_alpha": float(args.ridge_alpha),
        "mean_r_squared": float(np.nanmean(r_squared)),
        "median_r_squared": float(np.nanmedian(r_squared)),
        "p10_r_squared": float(np.nanpercentile(r_squared, 10)),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote: {args.output_dir / 'fold_metrics.csv'}")
    print(f"Wrote: {args.output_dir / 'summary_metrics.csv'}")
    print(f"Wrote: {args.output_dir / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
