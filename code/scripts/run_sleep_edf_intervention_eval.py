#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_SRC))

from aperiodic_eeg.spectral import flatten_spectral_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train on full Sleep-EDF log-PSD and evaluate counterfactual "
            "aperiodic/flattened test inputs."
        )
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
        "--output-dir",
        type=Path,
        default=Path("results/sleep_edf_subset/interventions_specparam"),
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--max-epochs", type=int, default=None)
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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(rows)
    out = []
    group_cols = ["task", "test_input", "classes", "train_input"]
    for key, group in df.groupby(group_cols, sort=False):
        task, test_input, classes, train_input = key
        item = {
            "task": task,
            "train_input": train_input,
            "test_input": test_input,
            "classes": classes,
            "n_folds": int(len(group)),
        }
        for metric in ["balanced_accuracy", "macro_f1", "accuracy"]:
            values = group[metric].to_numpy(dtype=float)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        out.append(item)

    full_by_task = {
        row["task"]: row
        for row in out
        if row["test_input"] == "full_log_psd"
    }
    derived_rows = []
    for row in out:
        if row["test_input"] == "full_log_psd":
            continue
        full = full_by_task[row["task"]]
        derived_rows.append(
            {
                "task": row["task"],
                "train_input": row["train_input"],
                "test_input": f"retention::{row['test_input']}",
                "classes": row["classes"],
                "n_folds": row["n_folds"],
                "balanced_accuracy_mean": (
                    row["balanced_accuracy_mean"] / max(full["balanced_accuracy_mean"], 1e-12)
                ),
                "balanced_accuracy_std": "",
                "macro_f1_mean": row["macro_f1_mean"] / max(full["macro_f1_mean"], 1e-12),
                "macro_f1_std": "",
                "accuracy_mean": row["accuracy_mean"] / max(full["accuracy_mean"], 1e-12),
                "accuracy_std": "",
            }
        )
        derived_rows.append(
            {
                "task": row["task"],
                "train_input": row["train_input"],
                "test_input": f"drop::{row['test_input']}",
                "classes": row["classes"],
                "n_folds": row["n_folds"],
                "balanced_accuracy_mean": (
                    full["balanced_accuracy_mean"] - row["balanced_accuracy_mean"]
                ),
                "balanced_accuracy_std": "",
                "macro_f1_mean": full["macro_f1_mean"] - row["macro_f1_mean"],
                "macro_f1_std": "",
                "accuracy_mean": full["accuracy_mean"] - row["accuracy_mean"],
                "accuracy_std": "",
            }
        )
    return out + derived_rows


def main() -> int:
    args = parse_args()

    import numpy as np
    import pandas as pd
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    psd_bundle = np.load(args.psd_npz)
    decomp = np.load(args.decomp_npz)
    index = pd.read_csv(args.index_csv)

    if args.max_epochs is not None:
        index = index.iloc[: args.max_epochs].copy()

    n_epochs = len(index)
    for key in ["log_psd", "aperiodic_log_psd", "residual_log_psd"]:
        if decomp[key].shape[0] < n_epochs:
            raise ValueError(f"{key} has fewer epochs than index rows.")

    arrays = {
        "full_log_psd": decomp["log_psd"][:n_epochs],
        "aperiodic_spectrum": decomp["aperiodic_log_psd"][:n_epochs],
        "flattened_log_psd": decomp["log_psd"][:n_epochs] - decomp["aperiodic_log_psd"][:n_epochs],
    }
    features = {name: flatten_spectral_features(array) for name, array in arrays.items()}

    stages = index["stage"].to_numpy()
    groups = index["subject"].to_numpy()
    task_labels = make_task_labels(stages)

    fold_rows: list[dict[str, object]] = []
    for task_name, labels in task_labels.items():
        mask, y, classes = encode_labels(labels)
        task_groups = groups[mask]
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
            clf.fit(features["full_log_psd"][mask][train_idx], y[train_idx])

            for test_input in ["full_log_psd", "aperiodic_spectrum", "flattened_log_psd"]:
                pred = clf.predict(features[test_input][mask][test_idx])
                fold_rows.append(
                    {
                        "task": task_name,
                        "train_input": "full_log_psd",
                        "test_input": test_input,
                        "fold": fold_idx,
                        "classes": "|".join(classes),
                        "n_train": int(len(train_idx)),
                        "n_test": int(len(test_idx)),
                        "train_subjects": ",".join(sorted(np.unique(task_groups[train_idx]))),
                        "test_subjects": ",".join(sorted(np.unique(task_groups[test_idx]))),
                        "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], pred)),
                        "macro_f1": float(f1_score(y[test_idx], pred, average="macro")),
                        "accuracy": float(accuracy_score(y[test_idx], pred)),
                    }
                )

    summary_rows = summarize(fold_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "intervention_fold_metrics.csv", fold_rows)
    write_csv(args.output_dir / "intervention_summary_metrics.csv", summary_rows)

    meta = {
        "psd_npz": str(args.psd_npz),
        "index_csv": str(args.index_csv),
        "decomp_npz": str(args.decomp_npz),
        "train_input": "full_log_psd",
        "test_inputs": ["full_log_psd", "aperiodic_spectrum", "flattened_log_psd"],
        "n_epochs": int(n_epochs),
        "channels": psd_bundle["channels"].tolist() if "channels" in psd_bundle else None,
        "ridge_alpha": float(args.ridge_alpha),
    }
    (args.output_dir / "intervention_metadata.json").write_text(json.dumps(meta, indent=2))

    direct = [row for row in summary_rows if not str(row["test_input"]).startswith("retention::")]
    print(f"Wrote outputs to: {args.output_dir}")
    print(
        pd.DataFrame(direct)[
            ["task", "train_input", "test_input", "balanced_accuracy_mean", "macro_f1_mean"]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
