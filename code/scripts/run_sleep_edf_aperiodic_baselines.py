#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_SRC))

from aperiodic_eeg.spectral import (  # noqa: E402
    fit_fixed_aperiodic,
    flatten_param_features,
    flatten_spectral_features,
)


STAGE_ORDER = ["W", "N1", "N2", "N3", "REM"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run subject-level Sleep-EDF aperiodic audit baselines."
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
        "--output-dir",
        type=Path,
        default=Path("results/sleep_edf_subset/baselines"),
    )
    parser.add_argument(
        "--decomposition",
        choices=["fixed", "precomputed"],
        default="fixed",
        help="Use the built-in fixed 1/f fit or a precomputed decomposition NPZ.",
    )
    parser.add_argument(
        "--decomp-npz",
        type=Path,
        default=None,
        help="Precomputed decomposition NPZ from fit_sleep_edf_specparam.py.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--classifier",
        choices=["ridge", "logistic"],
        default="ridge",
        help="Linear classifier. Ridge is the fast default for repeated baseline sweeps.",
    )
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Optional smoke-test limit after loading PSD/index.",
    )
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


def evaluate_feature_set(
    features,
    y,
    groups,
    n_splits: int,
    seed: int,
    classifier: str,
    ridge_alpha: float,
):
    import numpy as np
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    folds = make_group_folds(groups, n_splits)
    rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        if classifier == "ridge":
            estimator = RidgeClassifier(
                alpha=ridge_alpha,
                class_weight="balanced",
                solver="lsqr",
            )
        else:
            estimator = LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2000,
                random_state=seed,
                solver="lbfgs",
            )

        clf = make_pipeline(
            StandardScaler(),
            estimator,
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

    metrics = ["balanced_accuracy", "macro_f1", "accuracy"]
    summary = {}
    for metric in metrics:
        values = np.array([row[metric] for row in rows], dtype=float)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    import numpy as np
    import pandas as pd

    psd_bundle = np.load(args.psd_npz)
    psd = psd_bundle["psd"]
    freqs = psd_bundle["freqs"]
    channels = psd_bundle["channels"].tolist()
    index = pd.read_csv(args.index_csv)

    if args.max_epochs is not None:
        psd = psd[: args.max_epochs]
        index = index.iloc[: args.max_epochs].copy()

    if len(index) != psd.shape[0]:
        raise ValueError(f"Index rows {len(index)} != PSD epochs {psd.shape[0]}")

    if args.decomposition == "fixed":
        fit = fit_fixed_aperiodic(psd, freqs)
        feature_sets = {
            "full_log_psd": flatten_spectral_features(fit.log_psd),
            "aperiodic_spectrum": flatten_spectral_features(fit.fitted_log_psd),
            "aperiodic_params": flatten_param_features(fit.params),
            "periodic_residual": flatten_spectral_features(fit.residual_log_psd),
        }
        fit_summary = {
            "decomposition": "fixed",
            "psd_shape": list(psd.shape),
            "freq_min": float(freqs.min()),
            "freq_max": float(freqs.max()),
            "channels": channels,
            "mean_r_squared": float(fit.r_squared.mean()),
            "median_r_squared": float(np.median(fit.r_squared)),
            "mean_exponent_by_channel": {
                channel: float(fit.exponent[:, idx].mean())
                for idx, channel in enumerate(channels)
            },
            "mean_offset_by_channel": {
                channel: float(fit.offset[:, idx].mean())
                for idx, channel in enumerate(channels)
            },
        }
    else:
        if args.decomp_npz is None:
            raise ValueError("--decomp-npz is required with --decomposition precomputed")
        decomp = np.load(args.decomp_npz)
        for key in ["log_psd", "aperiodic_log_psd", "residual_log_psd", "offset", "exponent"]:
            if key not in decomp:
                raise ValueError(f"Missing {key!r} from {args.decomp_npz}")
        if decomp["log_psd"].shape[0] != psd.shape[0]:
            raise ValueError(
                f"Decomposition epochs {decomp['log_psd'].shape[0]} != PSD epochs {psd.shape[0]}"
            )

        params = np.stack([decomp["offset"], decomp["exponent"]], axis=-1)
        feature_sets = {
            "full_log_psd": flatten_spectral_features(decomp["log_psd"]),
            "aperiodic_spectrum": flatten_spectral_features(decomp["aperiodic_log_psd"]),
            "aperiodic_params": flatten_param_features(params),
            "periodic_residual": flatten_spectral_features(decomp["residual_log_psd"]),
        }
        r_squared = decomp["r_squared"] if "r_squared" in decomp else np.array([np.nan])
        fit_summary = {
            "decomposition": "precomputed",
            "decomp_npz": str(args.decomp_npz),
            "psd_shape": list(psd.shape),
            "freq_min": float(freqs.min()),
            "freq_max": float(freqs.max()),
            "channels": channels,
            "mean_r_squared": float(np.nanmean(r_squared)),
            "median_r_squared": float(np.nanmedian(r_squared)),
            "mean_exponent_by_channel": {
                channel: float(np.nanmean(decomp["exponent"][:, idx]))
                for idx, channel in enumerate(channels)
            },
            "mean_offset_by_channel": {
                channel: float(np.nanmean(decomp["offset"][:, idx]))
                for idx, channel in enumerate(channels)
            },
            "mean_n_peaks": float(np.nanmean(decomp["n_peaks"])) if "n_peaks" in decomp else None,
        }

    stages = index["stage"].to_numpy()
    groups = index["subject"].to_numpy()
    task_labels = make_task_labels(stages)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    (args.output_dir / "aperiodic_fit_summary.json").write_text(
        json.dumps(fit_summary, indent=2)
    )

    all_fold_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for task_name, labels in task_labels.items():
        mask, y, classes = encode_labels(labels)
        task_groups = groups[mask]
        if len(np.unique(task_groups)) < 2:
            print(f"Skipping {task_name}: fewer than two subject groups", file=sys.stderr)
            continue

        task_summaries = {}
        for feature_name, features in feature_sets.items():
            task_features = features[mask]
            fold_rows = evaluate_feature_set(
                task_features,
                y,
                task_groups,
                n_splits=args.n_splits,
                seed=args.seed,
                classifier=args.classifier,
                ridge_alpha=args.ridge_alpha,
            )
            for row in fold_rows:
                row.update(
                    {
                        "task": task_name,
                        "feature_set": feature_name,
                        "classes": "|".join(classes),
                        "classifier": args.classifier,
                    }
                )
            all_fold_rows.extend(fold_rows)

            summary = summarize_fold_rows(fold_rows)
            task_summaries[feature_name] = summary
            summary_rows.append(
                {
                    "task": task_name,
                    "feature_set": feature_name,
                    "classes": "|".join(classes),
                    "classifier": args.classifier,
                    "n_epochs": int(mask.sum()),
                    "n_subjects": int(len(np.unique(task_groups))),
                    **summary,
                }
            )

        full_bacc = task_summaries["full_log_psd"]["balanced_accuracy_mean"]
        if full_bacc > 0:
            for feature_name in ["aperiodic_spectrum", "aperiodic_params", "periodic_residual"]:
                summary_rows.append(
                    {
                        "task": task_name,
                        "feature_set": f"retention::{feature_name}",
                        "classes": "|".join(classes),
                        "classifier": args.classifier,
                        "n_epochs": int(mask.sum()),
                        "n_subjects": int(len(np.unique(task_groups))),
                        "balanced_accuracy_mean": (
                            task_summaries[feature_name]["balanced_accuracy_mean"] / full_bacc
                        ),
                        "balanced_accuracy_std": "",
                        "macro_f1_mean": (
                            task_summaries[feature_name]["macro_f1_mean"]
                            / max(task_summaries["full_log_psd"]["macro_f1_mean"], 1e-12)
                        ),
                        "macro_f1_std": "",
                        "accuracy_mean": (
                            task_summaries[feature_name]["accuracy_mean"]
                            / max(task_summaries["full_log_psd"]["accuracy_mean"], 1e-12)
                        ),
                        "accuracy_std": "",
                    }
                )

    write_csv(args.output_dir / "fold_metrics.csv", all_fold_rows)
    write_csv(args.output_dir / "summary_metrics.csv", summary_rows)

    print(f"Wrote: {args.output_dir / 'fold_metrics.csv'}")
    print(f"Wrote: {args.output_dir / 'summary_metrics.csv'}")
    print(f"Wrote: {args.output_dir / 'aperiodic_fit_summary.json'}")

    summary_df = pd.DataFrame(summary_rows)
    direct = summary_df[~summary_df["feature_set"].astype(str).str.startswith("retention::")]
    print(direct[["task", "feature_set", "balanced_accuracy_mean", "macro_f1_mean"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
