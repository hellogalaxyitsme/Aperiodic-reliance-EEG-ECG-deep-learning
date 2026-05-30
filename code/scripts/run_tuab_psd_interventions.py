#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_SRC))

from aperiodic_eeg.spectral import fit_fixed_aperiodic, flatten_spectral_features  # noqa: E402


TRAIN_INPUTS = ["full_log_psd", "aperiodic_spectrum", "flattened_log_psd"]
TEST_INPUTS = ["full_log_psd", "aperiodic_spectrum", "flattened_log_psd"]
METRICS = ["balanced_accuracy", "macro_f1", "accuracy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the first official-split TUAB PSD aperiodic intervention baseline."
        )
    )
    parser.add_argument(
        "--psd-npz",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/results/tuab_subset_200/psd_20s_multitaper.npz"),
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/results/tuab_subset_200/psd_20s_multitaper_index.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/results/tuab_subset_200/psd_interventions_fixed"),
    )
    parser.add_argument(
        "--decomposition",
        choices=["fixed", "precomputed"],
        default="fixed",
        help="Use vectorized fixed 1/f decomposition or a precomputed specparam NPZ.",
    )
    parser.add_argument(
        "--decomp-npz",
        type=Path,
        default=None,
        help="Precomputed decomposition NPZ with log_psd, aperiodic_log_psd, residual_log_psd.",
    )
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument(
        "--subject-filter-csv",
        type=Path,
        default=None,
        help=(
            "Optional subject manifest. If it has subject_id, official_split, and "
            "label columns, filtering uses all three fields; otherwise it filters "
            "by subject_id/subject only."
        ),
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def make_subject_filter_mask(index, subject_filter_csv: Path):
    import numpy as np

    rows = read_csv(subject_filter_csv)
    if not rows:
        raise ValueError(f"No rows found in subject filter: {subject_filter_csv}")
    columns = set(rows[0].keys())
    subject_col = "subject_id" if "subject_id" in columns else "subject"
    if subject_col not in columns:
        raise ValueError(
            f"Subject filter must contain subject_id or subject: {subject_filter_csv}"
        )

    if {"official_split", "label", subject_col}.issubset(columns):
        allowed = {
            (str(row["official_split"]), str(row["label"]), str(row[subject_col]))
            for row in rows
        }
        values = zip(
            index["official_split"].astype(str),
            index["label"].astype(str),
            index["subject"].astype(str),
        )
        mask = np.array([value in allowed for value in values], dtype=bool)
        mode = "official_split_label_subject"
    else:
        allowed_subjects = {str(row[subject_col]) for row in rows}
        mask = index["subject"].astype(str).isin(allowed_subjects).to_numpy(dtype=bool)
        mode = "subject"

    if int(mask.sum()) == 0:
        raise ValueError(f"Subject filter selected zero epochs: {subject_filter_csv}")
    return mask, {
        "subject_filter_csv": str(subject_filter_csv),
        "subject_filter_rows": int(len(rows)),
        "subject_filter_mode": mode,
        "subject_filter_selected_epochs": int(mask.sum()),
        "subject_filter_selected_subjects": int(index.loc[mask, "subject"].nunique()),
    }


def encode_labels(labels):
    import numpy as np

    labels = np.asarray(labels).astype(str)
    classes = sorted(np.unique(labels).tolist())
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    y = np.array([class_to_idx[label] for label in labels], dtype=int)
    return y, classes


def balanced_accuracy(y_true, y_pred, n_classes: int) -> float:
    import numpy as np

    recalls = []
    for cls in range(n_classes):
        mask = y_true == cls
        if mask.sum() > 0:
            recalls.append(float((y_pred[mask] == cls).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def macro_f1(y_true, y_pred, n_classes: int) -> float:
    import numpy as np

    scores = []
    for cls in range(n_classes):
        tp = float(((y_true == cls) & (y_pred == cls)).sum())
        fp = float(((y_true != cls) & (y_pred == cls)).sum())
        fn = float(((y_true == cls) & (y_pred != cls)).sum())
        denom = 2 * tp + fp + fn
        scores.append((2 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(scores))


def accuracy(y_true, y_pred) -> float:
    import numpy as np

    return float(np.mean(y_true == y_pred))


def metric_value(metric: str, y_true, y_pred, n_classes: int) -> float:
    if metric == "balanced_accuracy":
        return balanced_accuracy(y_true, y_pred, n_classes)
    if metric == "macro_f1":
        return macro_f1(y_true, y_pred, n_classes)
    if metric == "accuracy":
        return accuracy(y_true, y_pred)
    raise ValueError(metric)


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


def ci_bounds(values, ci: float):
    import numpy as np

    alpha = 1.0 - ci
    return (
        float(np.quantile(values, alpha / 2.0)),
        float(np.quantile(values, 1.0 - alpha / 2.0)),
    )


def subject_stratified_bootstrap(prediction_rows, n_bootstrap: int, ci: float, seed: int):
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(prediction_rows)
    rng = np.random.default_rng(seed)
    out: list[dict[str, object]] = []

    def subject_confusions(group):
        subjects = sorted(group["subject"].unique())
        n_classes = int(group["n_classes"].iloc[0])
        matrices = np.zeros((len(subjects), n_classes, n_classes), dtype=np.int64)
        labels = []
        for subject_idx, subject in enumerate(subjects):
            sub = group[group["subject"] == subject]
            labels.append(str(sub["label"].iloc[0]))
            for true, pred in zip(sub["y_true"].to_numpy(dtype=int), sub["y_pred"].to_numpy(dtype=int)):
                matrices[subject_idx, true, pred] += 1
        return np.array(subjects, dtype=object), np.array(labels, dtype=object), matrices

    def metric_from_confusion(metric: str, cm: np.ndarray) -> float:
        total = float(cm.sum())
        if total <= 0:
            return 0.0
        if metric == "accuracy":
            return float(np.trace(cm) / total)
        recalls = []
        f1s = []
        for cls in range(cm.shape[0]):
            tp = float(cm[cls, cls])
            fn = float(cm[cls, :].sum() - tp)
            fp = float(cm[:, cls].sum() - tp)
            support = tp + fn
            if support > 0:
                recalls.append(tp / support)
            denom = 2 * tp + fp + fn
            f1s.append((2 * tp / denom) if denom > 0 else 0.0)
        if metric == "balanced_accuracy":
            return float(np.mean(recalls)) if recalls else 0.0
        if metric == "macro_f1":
            return float(np.mean(f1s)) if f1s else 0.0
        raise ValueError(metric)

    def stratified_sample_indices(labels):
        sampled_indices = []
        for label in sorted(np.unique(labels)):
            label_indices = np.flatnonzero(labels == label)
            sampled_indices.extend(
                rng.choice(label_indices, size=len(label_indices), replace=True).tolist()
            )
        return np.array(sampled_indices, dtype=int)

    for (train_input, test_input, classes), group in df.groupby(
        ["train_input", "test_input", "classes"],
        sort=False,
    ):
        subjects, subject_labels, matrices = subject_confusions(group)
        for metric in METRICS:
            boot = np.empty(n_bootstrap, dtype=float)
            for boot_idx in range(n_bootstrap):
                sampled = stratified_sample_indices(subject_labels)
                boot[boot_idx] = metric_from_confusion(metric, matrices[sampled].sum(axis=0))
            point = metric_from_confusion(metric, matrices.sum(axis=0))
            lower, upper = ci_bounds(boot, ci)
            out.append(
                {
                    "task": "tuab_normal_vs_abnormal",
                    "train_input": train_input,
                    "test_input": test_input,
                    "classes": classes,
                    "metric": metric,
                    "estimate": "performance",
                    "point": point,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "ci": float(ci),
                    "n_eval_subjects": int(subjects.shape[0]),
                    "n_bootstrap": int(n_bootstrap),
                }
            )

    full_train = df[df["train_input"] == "full_log_psd"]
    for test_input in ["aperiodic_spectrum", "flattened_log_psd"]:
        paired = full_train[full_train["test_input"].isin(["full_log_psd", test_input])]
        if paired.empty:
            continue
        for metric in METRICS:
            baseline_group = paired[paired["test_input"] == "full_log_psd"]
            edited_group = paired[paired["test_input"] == test_input]
            subjects, subject_labels, baseline_matrices = subject_confusions(baseline_group)
            edited_subjects, edited_labels, edited_matrices = subject_confusions(edited_group)
            if subjects.tolist() != edited_subjects.tolist() or subject_labels.tolist() != edited_labels.tolist():
                raise ValueError("Paired bootstrap subject ordering mismatch.")
            boot_drop = np.empty(n_bootstrap, dtype=float)
            boot_retention = np.empty(n_bootstrap, dtype=float)
            for boot_idx in range(n_bootstrap):
                sampled = stratified_sample_indices(subject_labels)
                base_metric = metric_from_confusion(
                    metric,
                    baseline_matrices[sampled].sum(axis=0),
                )
                edit_metric = metric_from_confusion(
                    metric,
                    edited_matrices[sampled].sum(axis=0),
                )
                boot_drop[boot_idx] = base_metric - edit_metric
                boot_retention[boot_idx] = edit_metric / max(base_metric, 1e-12)

            base_point = metric_from_confusion(metric, baseline_matrices.sum(axis=0))
            edit_point = metric_from_confusion(metric, edited_matrices.sum(axis=0))
            for estimate, values, point in [
                (f"drop::{test_input}", boot_drop, base_point - edit_point),
                (
                    f"retention::{test_input}",
                    boot_retention,
                    edit_point / max(base_point, 1e-12),
                ),
            ]:
                lower, upper = ci_bounds(values, ci)
                out.append(
                    {
                        "task": "tuab_normal_vs_abnormal",
                        "train_input": "full_log_psd",
                        "test_input": test_input,
                        "classes": "|".join(sorted(baseline_group["label"].unique())),
                        "metric": metric,
                        "estimate": estimate,
                        "point": point,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "ci": float(ci),
                        "n_eval_subjects": int(subjects.shape[0]),
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
    cols = [
        "train_input",
        "test_input",
        "estimate",
        "point",
        "ci_lower",
        "ci_upper",
        "n_eval_subjects",
    ]
    focus = focus[cols]
    for col in ["point", "ci_lower", "ci_upper"]:
        focus[col] = focus[col].map(lambda value: f"{value:.3f}")

    lines = [
        "# TUAB PSD Intervention Report",
        "",
        "Official train/eval split. Confidence intervals use stratified eval-subject bootstrap.",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in focus.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import RidgeClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    args = parse_args()

    index = pd.read_csv(args.index_csv)
    if args.max_epochs is not None:
        index = index.iloc[: args.max_epochs].copy()

    subject_filter_summary: dict[str, object] = {}
    subject_filter_mask = None
    if args.subject_filter_csv is not None:
        subject_filter_mask, subject_filter_summary = make_subject_filter_mask(
            index,
            args.subject_filter_csv,
        )

    if args.decomposition == "fixed":
        psd_bundle = np.load(args.psd_npz)
        psd = psd_bundle["psd"]
        freqs = psd_bundle["freqs"]
        if args.max_epochs is not None:
            psd = psd[: args.max_epochs]
        if subject_filter_mask is not None:
            psd = psd[subject_filter_mask]
            index = index.loc[subject_filter_mask].reset_index(drop=True)
        fit = fit_fixed_aperiodic(psd, freqs)
        arrays = {
            "full_log_psd": fit.log_psd,
            "aperiodic_spectrum": fit.fitted_log_psd,
            "flattened_log_psd": fit.residual_log_psd,
        }
        decomp_summary = {
            "decomposition": "fixed",
            "psd_npz": str(args.psd_npz),
            "psd_shape": [int(v) for v in psd.shape],
            "mean_r_squared": float(np.nanmean(fit.r_squared)),
            "median_r_squared": float(np.nanmedian(fit.r_squared)),
            "mean_exponent": float(np.nanmean(fit.exponent)),
            "median_exponent": float(np.nanmedian(fit.exponent)),
        }
    else:
        if args.decomp_npz is None:
            raise ValueError("--decomp-npz is required with --decomposition precomputed")
        decomp = np.load(args.decomp_npz)
        n_epochs = len(index)
        arrays = {
            "full_log_psd": decomp["log_psd"][:n_epochs],
            "aperiodic_spectrum": decomp["aperiodic_log_psd"][:n_epochs],
            "flattened_log_psd": decomp["residual_log_psd"][:n_epochs],
        }
        if subject_filter_mask is not None:
            arrays = {name: array[subject_filter_mask] for name, array in arrays.items()}
            index = index.loc[subject_filter_mask].reset_index(drop=True)
        r_squared = decomp["r_squared"] if "r_squared" in decomp else np.array([np.nan])
        decomp_summary = {
            "decomposition": "precomputed",
            "decomp_npz": str(args.decomp_npz),
            "mean_r_squared": float(np.nanmean(r_squared)),
            "median_r_squared": float(np.nanmedian(r_squared)),
        }

    if len(index) != arrays["full_log_psd"].shape[0]:
        raise ValueError(
            f"Index rows {len(index)} != feature epochs {arrays['full_log_psd'].shape[0]}"
        )

    features = {
        name: flatten_spectral_features(array).astype("float32", copy=False)
        for name, array in arrays.items()
    }
    labels = index["label"].astype(str).to_numpy()
    subjects = index["subject"].astype(str).to_numpy()
    splits = index["official_split"].astype(str).to_numpy()
    y, classes = encode_labels(labels)
    n_classes = len(classes)

    train_idx = np.flatnonzero(splits == "train")
    eval_idx = np.flatnonzero(splits == "eval")
    if len(train_idx) == 0 or len(eval_idx) == 0:
        raise ValueError("Expected both train and eval rows in TUAB index.")

    fold_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for train_input in TRAIN_INPUTS:
        clf = make_pipeline(
            StandardScaler(),
            RidgeClassifier(
                alpha=args.ridge_alpha,
                class_weight="balanced",
                solver="lsqr",
            ),
        )
        clf.fit(features[train_input][train_idx], y[train_idx])
        print(f"Fitted train_input={train_input}", flush=True)

        for test_input in TEST_INPUTS:
            pred = clf.predict(features[test_input][eval_idx])
            base_row = {
                "task": "tuab_normal_vs_abnormal",
                "train_input": train_input,
                "test_input": test_input,
                "split": "official_eval",
                "classes": "|".join(classes),
                "n_train": int(len(train_idx)),
                "n_eval": int(len(eval_idx)),
                "n_train_subjects": int(len(np.unique(subjects[train_idx]))),
                "n_eval_subjects": int(len(np.unique(subjects[eval_idx]))),
            }
            fold_rows.append(
                {
                    **base_row,
                    "balanced_accuracy": balanced_accuracy(y[eval_idx], pred, n_classes),
                    "macro_f1": macro_f1(y[eval_idx], pred, n_classes),
                    "accuracy": accuracy(y[eval_idx], pred),
                }
            )
            append_subject_rows(
                subject_rows,
                base_row,
                subjects[eval_idx],
                y[eval_idx],
                pred,
                n_classes,
            )
            for idx, y_pred in zip(eval_idx, pred):
                prediction_rows.append(
                    {
                        "task": "tuab_normal_vs_abnormal",
                        "train_input": train_input,
                        "test_input": test_input,
                        "classes": "|".join(classes),
                        "n_classes": n_classes,
                        "row_index": int(idx),
                        "subject": subjects[idx],
                        "label": labels[idx],
                        "y_true": int(y[idx]),
                        "y_pred": int(y_pred),
                    }
                )
            print(f"Evaluated train={train_input} test={test_input}", flush=True)

    bootstrap_rows = subject_stratified_bootstrap(
        prediction_rows,
        n_bootstrap=args.n_bootstrap,
        ci=args.ci,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "tuab_psd_intervention_eval_metrics.csv", fold_rows)
    write_csv(args.output_dir / "tuab_psd_intervention_subject_metrics.csv", subject_rows)
    write_csv(args.output_dir / "tuab_psd_intervention_predictions.csv", prediction_rows)
    write_csv(args.output_dir / "tuab_psd_intervention_subject_bootstrap.csv", bootstrap_rows)
    write_markdown(args.output_dir / "tuab_psd_intervention_subject_bootstrap.md", bootstrap_rows)

    metadata = {
        "index_csv": str(args.index_csv),
        "psd_npz": str(args.psd_npz),
        "output_dir": str(args.output_dir),
        "n_epochs": int(len(index)),
        "n_train_epochs": int(len(train_idx)),
        "n_eval_epochs": int(len(eval_idx)),
        "n_train_subjects": int(len(np.unique(subjects[train_idx]))),
        "n_eval_subjects": int(len(np.unique(subjects[eval_idx]))),
        "classes": classes,
        "train_inputs": TRAIN_INPUTS,
        "test_inputs": TEST_INPUTS,
        "ridge_alpha": float(args.ridge_alpha),
        "n_bootstrap": int(args.n_bootstrap),
        **subject_filter_summary,
        **decomp_summary,
    }
    (args.output_dir / "tuab_psd_intervention_metadata.json").write_text(
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
            ["train_input", "test_input", "estimate", "point", "ci_lower", "ci_upper"]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
