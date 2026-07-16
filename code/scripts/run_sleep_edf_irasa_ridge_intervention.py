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


TEST_INPUTS = ["full_log_psd", "aperiodic_spectrum", "flattened_log_psd"]
METRICS = ["balanced_accuracy", "macro_f1", "accuracy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train ridge classifiers on full IRASA log-PSD features and evaluate "
            "IRASA-derived aperiodic-only and flattened-PSD counterfactuals."
        )
    )
    parser.add_argument(
        "--irasa-npz",
        type=Path,
        default=Path(
            "results/sleep_edf_full/irasa/"
            "irasa_aperiodic_stage_balanced_5k_volts.npz"
        ),
    )
    parser.add_argument(
        "--irasa-index-csv",
        type=Path,
        default=Path(
            "results/sleep_edf_full/irasa/"
            "irasa_aperiodic_stage_balanced_5k_volts.index.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/sleep_edf_full/irasa/"
            "irasa_ridge_interventions_stage_balanced_5k"
        ),
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--max-epochs", type=int, default=None)
    return parser.parse_args()


def make_task_labels(stages):
    import numpy as np

    stages = np.asarray(stages).astype(str)
    return {
        "wake_vs_sleep": np.where(stages == "W", "W", "Sleep"),
        "n2_vs_n3": np.where(np.isin(stages, ["N2", "N3"]), stages, None),
        "five_stage": stages,
    }


def encode_labels(labels):
    import numpy as np
    from sklearn.preprocessing import LabelEncoder

    labels = np.asarray(labels, dtype=object)
    mask = labels != None  # noqa: E711
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels[mask].astype(str))
    return mask, y, encoder.classes_.tolist()


def make_group_folds(groups, n_splits: int):
    import numpy as np
    from sklearn.model_selection import GroupKFold

    unique_groups = np.unique(groups)
    n_splits = min(n_splits, len(unique_groups))
    if n_splits < 2:
        raise ValueError("Need at least two subject groups for subject-held-out evaluation.")
    splitter = GroupKFold(n_splits=n_splits)
    placeholder_y = np.zeros(len(groups), dtype=int)
    return list(splitter.split(placeholder_y, groups=groups))


def balanced_accuracy(y_true, y_pred, n_classes: int) -> float:
    import numpy as np

    recalls = []
    for cls in range(n_classes):
        mask = y_true == cls
        if mask.sum() > 0:
            recalls.append(float((y_pred[mask] == cls).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def macro_f1(y_true, y_pred, n_classes: int) -> float:
    scores = []
    for cls in range(n_classes):
        tp = float(((y_true == cls) & (y_pred == cls)).sum())
        fp = float(((y_true != cls) & (y_pred == cls)).sum())
        fn = float(((y_true == cls) & (y_pred != cls)).sum())
        denom = 2 * tp + fp + fn
        scores.append((2 * tp / denom) if denom > 0 else 0.0)
    return float(sum(scores) / len(scores))


def accuracy(y_true, y_pred) -> float:
    import numpy as np

    return float(np.mean(y_true == y_pred))


def append_subject_rows(rows, base_row, subjects, y_true, y_pred, n_classes):
    import numpy as np

    subjects = np.asarray(subjects).astype(str)
    for subject in sorted(np.unique(subjects)):
        mask = subjects == subject
        support = {str(cls): int((y_true[mask] == cls).sum()) for cls in range(n_classes)}
        rows.append(
            {
                **base_row,
                "subject": str(subject),
                "n_subject": int(mask.sum()),
                "class_support_json": json.dumps(support, sort_keys=True),
                "balanced_accuracy": balanced_accuracy(y_true[mask], y_pred[mask], n_classes),
                "macro_f1": macro_f1(y_true[mask], y_pred[mask], n_classes),
                "accuracy": accuracy(y_true[mask], y_pred[mask]),
            }
        )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_folds(rows):
    import pandas as pd

    df = pd.DataFrame(rows)
    out = []
    for key, group in df.groupby(["task", "train_input", "test_input", "classes"], sort=False):
        task, train_input, test_input, classes = key
        item = {
            "task": task,
            "train_input": train_input,
            "test_input": test_input,
            "classes": classes,
            "n_folds": int(len(group)),
        }
        for metric in METRICS:
            vals = group[metric].to_numpy(dtype=float)
            item[f"{metric}_mean"] = float(vals.mean())
            item[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out.append(item)
    return out


def ci_bounds(values, ci: float):
    import numpy as np

    alpha = 1.0 - ci
    return (
        float(np.quantile(values, alpha / 2.0)),
        float(np.quantile(values, 1.0 - alpha / 2.0)),
    )


def subject_bootstrap(subject_rows, n_bootstrap: int, ci: float, seed: int):
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(subject_rows)
    rng = np.random.default_rng(seed)
    out = []

    for (task, train_input, test_input, classes), group in df.groupby(
        ["task", "train_input", "test_input", "classes"],
        sort=False,
    ):
        for metric in METRICS:
            subject_values = (
                group.groupby("subject", sort=False)[metric].mean().to_numpy(dtype=float)
            )
            sampled = rng.choice(
                subject_values,
                size=(n_bootstrap, len(subject_values)),
                replace=True,
            ).mean(axis=1)
            lower, upper = ci_bounds(sampled, ci)
            out.append(
                {
                    "task": task,
                    "train_input": train_input,
                    "test_input": test_input,
                    "metric": metric,
                    "estimate": "performance",
                    "point": float(subject_values.mean()),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "ci": float(ci),
                    "n_subjects": int(len(subject_values)),
                    "n_bootstrap": int(n_bootstrap),
                }
            )

    for task, group in df.groupby("task", sort=False):
        for test_input in ["aperiodic_spectrum", "flattened_log_psd"]:
            paired = group[group["test_input"].isin(["full_log_psd", test_input])]
            for metric in METRICS:
                pivot = paired.pivot_table(
                    index="subject",
                    columns="test_input",
                    values=metric,
                    aggfunc="mean",
                ).dropna(subset=["full_log_psd", test_input])
                if pivot.empty:
                    continue
                baseline = pivot["full_log_psd"].to_numpy(dtype=float)
                edited = pivot[test_input].to_numpy(dtype=float)
                diff = baseline - edited
                retention = edited / np.maximum(baseline, 1e-12)

                for estimate, values in [
                    (f"drop::{test_input}", diff),
                    (f"retention::{test_input}", retention),
                ]:
                    sampled = rng.choice(
                        values,
                        size=(n_bootstrap, len(values)),
                        replace=True,
                    ).mean(axis=1)
                    lower, upper = ci_bounds(sampled, ci)
                    out.append(
                        {
                            "task": task,
                            "train_input": "full_log_psd",
                            "test_input": test_input,
                            "metric": metric,
                            "estimate": estimate,
                            "point": float(values.mean()),
                            "ci_lower": lower,
                            "ci_upper": upper,
                            "ci": float(ci),
                            "n_subjects": int(len(values)),
                            "n_bootstrap": int(n_bootstrap),
                        }
                    )
    return out


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    focus = df[
        (df["metric"] == "balanced_accuracy")
        & (
            (df["estimate"] == "performance")
            | df["estimate"].isin(
                ["drop::aperiodic_spectrum", "drop::flattened_log_psd"]
            )
        )
    ].copy()
    focus = focus[
        [
            "task",
            "train_input",
            "test_input",
            "estimate",
            "point",
            "ci_lower",
            "ci_upper",
            "n_subjects",
        ]
    ]
    for col in ["point", "ci_lower", "ci_upper"]:
        focus[col] = focus[col].map(lambda value: f"{value:.3f}")

    lines = [
        "# Sleep-EDF IRASA Ridge Intervention Report",
        "",
        "Ridge classifiers were trained on full IRASA log-PSD features and evaluated on",
        "IRASA-derived aperiodic-only and flattened-PSD test inputs. Confidence",
        "intervals use held-out subject bootstrap.",
        "",
        "| " + " | ".join(focus.columns) + " |",
        "| " + " | ".join(["---"] * len(focus.columns)) + " |",
    ]
    for _, row in focus.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in focus.columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    args = parse_args()

    index = pd.read_csv(args.irasa_index_csv)
    decomp = np.load(args.irasa_npz)
    if args.max_epochs is not None:
        index = index.iloc[: args.max_epochs].copy()

    n_epochs = len(index)
    for key in ["log_psd", "aperiodic_log_psd", "residual_log_psd"]:
        if key not in decomp:
            raise KeyError(f"Missing {key} in {args.irasa_npz}")
        if decomp[key].shape[0] < n_epochs:
            raise ValueError(f"{key} has fewer rows than the IRASA index.")

    arrays = {
        "full_log_psd": decomp["log_psd"][:n_epochs],
        "aperiodic_spectrum": decomp["aperiodic_log_psd"][:n_epochs],
        "flattened_log_psd": decomp["residual_log_psd"][:n_epochs],
    }
    features = {
        name: flatten_spectral_features(array).astype("float32", copy=False)
        for name, array in arrays.items()
    }

    stages = index["stage"].astype(str).to_numpy()
    groups = index["subject"].astype(str).to_numpy()
    task_labels = make_task_labels(stages)

    fold_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    for task_name, labels in task_labels.items():
        mask, y, classes = encode_labels(labels)
        task_groups = groups[mask]
        folds = make_group_folds(task_groups, args.n_splits)
        n_classes = len(classes)

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

            train_subjects = ",".join(sorted(np.unique(task_groups[train_idx])))
            test_subjects = ",".join(sorted(np.unique(task_groups[test_idx])))
            for test_input in TEST_INPUTS:
                pred = clf.predict(features[test_input][mask][test_idx])
                base_row = {
                    "task": task_name,
                    "train_input": "full_log_psd",
                    "test_input": test_input,
                    "fold": fold_idx,
                    "classes": "|".join(classes),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "train_subjects": train_subjects,
                    "test_subjects": test_subjects,
                }
                fold_rows.append(
                    {
                        **base_row,
                        "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], pred)),
                        "macro_f1": float(f1_score(y[test_idx], pred, average="macro")),
                        "accuracy": float(accuracy_score(y[test_idx], pred)),
                    }
                )
                append_subject_rows(
                    subject_rows,
                    base_row,
                    task_groups[test_idx],
                    y[test_idx],
                    pred,
                    n_classes,
                )
            print(f"{task_name} fold {fold_idx} done", flush=True)

    summary_rows = summarize_folds(fold_rows)
    bootstrap_rows = subject_bootstrap(
        subject_rows,
        n_bootstrap=args.n_bootstrap,
        ci=args.ci,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "irasa_ridge_intervention_fold_metrics.csv", fold_rows)
    write_csv(args.output_dir / "irasa_ridge_intervention_subject_metrics.csv", subject_rows)
    write_csv(args.output_dir / "irasa_ridge_intervention_summary_metrics.csv", summary_rows)
    write_csv(args.output_dir / "irasa_ridge_intervention_subject_bootstrap.csv", bootstrap_rows)
    write_markdown(args.output_dir / "irasa_ridge_intervention_subject_bootstrap.md", bootstrap_rows)

    metadata = {
        "irasa_npz": str(args.irasa_npz),
        "irasa_index_csv": str(args.irasa_index_csv),
        "n_epochs": int(n_epochs),
        "n_subjects": int(len(np.unique(groups))),
        "stage_counts": index["stage"].value_counts().sort_index().astype(int).to_dict(),
        "channels": decomp["channels"].tolist() if "channels" in decomp else None,
        "freqs": decomp["freqs"].tolist() if "freqs" in decomp else None,
        "train_input": "full_log_psd",
        "test_inputs": TEST_INPUTS,
        "n_splits": int(args.n_splits),
        "ridge_alpha": float(args.ridge_alpha),
        "n_bootstrap": int(args.n_bootstrap),
        "ci": float(args.ci),
        "seed": int(args.seed),
    }
    (args.output_dir / "irasa_ridge_intervention_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    focus = pd.DataFrame(bootstrap_rows)
    focus = focus[
        (focus["metric"] == "balanced_accuracy")
        & focus["estimate"].isin(["performance", "drop::flattened_log_psd"])
    ]
    print(f"Wrote outputs to: {args.output_dir}")
    print(
        focus[
            ["task", "test_input", "estimate", "point", "ci_lower", "ci_upper"]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
