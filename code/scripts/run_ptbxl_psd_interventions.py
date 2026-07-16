#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPRESENTATIONS = ["full_log_psd", "aperiodic_spectrum", "flattened_log_psd"]
ARRAY_KEY = {
    "full_log_psd": "log_psd",
    "aperiodic_spectrum": "aperiodic_log_psd",
    "flattened_log_psd": "flattened_log_psd",
}
METRICS = ["balanced_accuracy", "macro_f1", "accuracy"]


def parse_args():
    parser = argparse.ArgumentParser(description="PTB-XL PSD ridge spectral intervention audit.")
    parser.add_argument(
        "--psd-npz",
        type=Path,
        default=Path("results/ptbxl_1f_demo/ptbxl_records100_normal_abnormal_psd_fixed.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ptbxl_1f_demo/psd_interventions"),
    )
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260529)
    return parser.parse_args()


def metric_value(metric: str, y_true, y_pred):
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    if metric == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    raise ValueError(metric)


def features(array: np.ndarray) -> np.ndarray:
    return array.reshape(array.shape[0], -1).astype("float32", copy=False)


def ci_bounds(values):
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def sign_p(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    nonpos = int(np.count_nonzero(values <= 0.0))
    nonneg = int(np.count_nonzero(values >= 0.0))
    return {
        "p_one_sided_positive": nonpos / n,
        "p_two_sided_zero": min(1.0, 2.0 * min(nonpos / n, nonneg / n)),
        "n_bootstrap_nonpositive": nonpos,
        "n_bootstrap_nonnegative": nonneg,
        "n_bootstrap_valid": n,
    }


def bootstrap_subject_metrics(subject_rows, n_bootstrap: int, seed: int):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(subject_rows)
    rows = []
    for (train_input, test_input), group in df.groupby(["train_input", "test_input"], sort=False):
        for metric in METRICS:
            values = group.groupby("subject", sort=False)[metric].mean().to_numpy(dtype=float)
            boot = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
            lo, hi = ci_bounds(boot)
            rows.append(
                {
                    "task": "ptbxl_normal_vs_abnormal",
                    "train_input": train_input,
                    "test_input": test_input,
                    "metric": metric,
                    "estimate": "performance",
                    "point": float(values.mean()),
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "ci": 0.95,
                    "n_subjects": int(len(values)),
                    "n_bootstrap": int(n_bootstrap),
                    **sign_p(boot),
                }
            )

    full = df[df["train_input"] == "full_log_psd"]
    for test_input in ["aperiodic_spectrum", "flattened_log_psd"]:
        paired = full[full["test_input"].isin(["full_log_psd", test_input])]
        for metric in METRICS:
            pivot = paired.pivot_table(index="subject", columns="test_input", values=metric, aggfunc="mean")
            pivot = pivot.dropna(subset=["full_log_psd", test_input])
            drop_values = pivot["full_log_psd"].to_numpy(dtype=float) - pivot[test_input].to_numpy(dtype=float)
            boot = rng.choice(drop_values, size=(n_bootstrap, len(drop_values)), replace=True).mean(axis=1)
            lo, hi = ci_bounds(boot)
            rows.append(
                {
                    "task": "ptbxl_normal_vs_abnormal",
                    "train_input": "full_log_psd",
                    "test_input": test_input,
                    "metric": metric,
                    "estimate": f"drop::{test_input}",
                    "point": float(drop_values.mean()),
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "ci": 0.95,
                    "n_subjects": int(len(drop_values)),
                    "n_bootstrap": int(n_bootstrap),
                    **sign_p(boot),
                }
            )
    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.psd_npz, allow_pickle=True)
    y = data["y"].astype(int)
    patient = data["patient_id"].astype(str)
    fold = data["strat_fold"].astype(int)
    train_mask = fold <= 8
    test_mask = fold == 10

    arrays = {name: features(data[ARRAY_KEY[name]]) for name in REPRESENTATIONS}
    pred_rows = []
    subject_rows = []

    for train_input in REPRESENTATIONS:
        clf = make_pipeline(
            StandardScaler(),
            RidgeClassifier(alpha=args.ridge_alpha),
        )
        clf.fit(arrays[train_input][train_mask], y[train_mask])
        for test_input in REPRESENTATIONS:
            pred = clf.predict(arrays[test_input][test_mask])
            y_true = y[test_mask]
            subjects = patient[test_mask]
            for subject in sorted(np.unique(subjects)):
                mask = subjects == subject
                row = {
                    "task": "ptbxl_normal_vs_abnormal",
                    "train_input": train_input,
                    "test_input": test_input,
                    "subject": subject,
                    "n_subject": int(mask.sum()),
                }
                for metric in METRICS:
                    row[metric] = metric_value(metric, y_true[mask], pred[mask])
                subject_rows.append(row)
            for idx, (subj, yt, yp) in enumerate(zip(subjects, y_true, pred)):
                pred_rows.append(
                    {
                        "task": "ptbxl_normal_vs_abnormal",
                        "train_input": train_input,
                        "test_input": test_input,
                        "subject": subj,
                        "record_index": int(np.flatnonzero(test_mask)[idx]),
                        "y_true": int(yt),
                        "y_pred": int(yp),
                    }
                )

    write_csv(args.output_dir / "ptbxl_psd_intervention_subject_metrics.csv", subject_rows)
    write_csv(args.output_dir / "ptbxl_psd_intervention_predictions.csv", pred_rows)
    boot_rows = bootstrap_subject_metrics(subject_rows, args.n_bootstrap, args.seed)
    write_csv(args.output_dir / "ptbxl_psd_intervention_subject_bootstrap.csv", boot_rows)
    (args.output_dir / "ptbxl_psd_intervention_metadata.json").write_text(
        json.dumps(
            {
                "psd_npz": str(args.psd_npz),
                "train_folds": "1-8",
                "validation_fold": "9 unused",
                "test_fold": 10,
                "n_train_records": int(train_mask.sum()),
                "n_test_records": int(test_mask.sum()),
                "n_test_patients": int(len(np.unique(patient[test_mask]))),
                "ridge_alpha": float(args.ridge_alpha),
                "n_bootstrap": int(args.n_bootstrap),
            },
            indent=2,
        )
    )
    focus = pd.DataFrame(boot_rows)
    print(
        focus[
            (focus["metric"] == "balanced_accuracy")
            & (focus["train_input"] == "full_log_psd")
        ][["test_input", "estimate", "point", "ci_lower", "ci_upper", "p_one_sided_positive"]].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
