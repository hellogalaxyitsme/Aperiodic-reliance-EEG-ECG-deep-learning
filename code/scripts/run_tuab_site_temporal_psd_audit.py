#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_SRC))

from aperiodic_eeg.spectral import flatten_spectral_features  # noqa: E402


TRAIN_INPUTS = ["full_log_psd", "aperiodic_spectrum", "flattened_log_psd"]
TEST_INPUTS = ["full_log_psd", "aperiodic_spectrum", "flattened_log_psd"]
METRICS = ["balanced_accuracy", "macro_f1", "accuracy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "TUAB temporal acquisition-proxy PSD audit: within-bin and cross-bin "
            "ridge intervention analyses using cached full-TUAB specparam arrays."
        )
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=Path(
            "/mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/"
            "preprocess_20s_100hz/psd_20s_multitaper_index.csv"
        ),
    )
    parser.add_argument(
        "--decomp-npz",
        type=Path,
        default=Path(
            "/mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/"
            "preprocess_20s_100hz/specparam/specparam_fixed_20s.npz"
        ),
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path(
            "/mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/age_matched/"
            "tuab_full_age_sex_matched_caliper5_header_metadata_files.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/"
            "site_temporal_psd_audit"
        ),
    )
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument(
        "--include-sentinel-years",
        action="store_true",
        help="Include 1899/2000/2007 date outliers in temporal binning.",
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


def encode_labels(labels):
    import numpy as np

    labels = np.asarray(labels).astype(str)
    classes = sorted(np.unique(labels).tolist())
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    y = np.array([class_to_idx[label] for label in labels], dtype=int)
    return y, classes


def ci_bounds(values, ci: float):
    import numpy as np

    alpha = 1.0 - ci
    return (
        float(np.quantile(values, alpha / 2.0)),
        float(np.quantile(values, 1.0 - alpha / 2.0)),
    )


def year_from_header(value: object) -> float:
    import pandas as pd

    if pd.notna(value):
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.notna(parsed):
            return float(parsed.year)
    return float("nan")


def year_from_recording_header(value: object) -> float:
    if value is None:
        return float("nan")
    match = re.search(r"\b(\d{4})\b", str(value))
    return float(match.group(1)) if match else float("nan")


def add_temporal_bins(index, metadata, include_sentinel_years: bool):
    import numpy as np
    import pandas as pd

    meta = metadata.copy()
    meta["recording_year"] = meta["meas_date"].map(year_from_header)
    missing = meta["recording_year"].isna()
    if missing.any():
        meta.loc[missing, "recording_year"] = meta.loc[missing, "recording_header"].map(
            year_from_recording_header
        )
    meta = meta[
        [
            "remote_rel_path",
            "recording_year",
            "meas_date",
            "recording_header",
            "n_channels",
            "sfreq",
        ]
    ].drop_duplicates("remote_rel_path")

    merged = index.merge(meta, on="remote_rel_path", how="left", validate="many_to_one")
    if "montage" not in merged:
        merged["montage"] = merged["remote_rel_path"].astype(str).str.split("/").str[2]

    valid_year = merged["recording_year"].notna()
    if not include_sentinel_years:
        valid_year &= merged["recording_year"].between(2009, 2013)
    merged["temporal_valid"] = valid_year
    merged["temporal_primary"] = np.where(
        valid_year & (merged["recording_year"] <= 2011),
        "early_2009_2011",
        np.where(valid_year & (merged["recording_year"] >= 2012), "late_2012_2013", ""),
    )
    merged["temporal_tercile"] = np.where(
        valid_year & (merged["recording_year"] <= 2010),
        "tercile_early_2009_2010",
        np.where(
            valid_year & (merged["recording_year"] <= 2012),
            "tercile_middle_2011_2012",
            np.where(valid_year, "tercile_late_2013", ""),
        ),
    )
    return merged


def subject_confusions(group):
    import numpy as np

    subjects = sorted(group["subject"].unique())
    n_classes = int(group["n_classes"].iloc[0])
    matrices = np.zeros((len(subjects), n_classes, n_classes), dtype=np.int64)
    labels = []
    for subject_idx, subject in enumerate(subjects):
        sub = group[group["subject"] == subject]
        labels.append(str(sub["label"].iloc[0]))
        for true, pred in zip(
            sub["y_true"].to_numpy(dtype=int),
            sub["y_pred"].to_numpy(dtype=int),
        ):
            matrices[subject_idx, true, pred] += 1
    return np.array(subjects, dtype=object), np.array(labels, dtype=object), matrices


def metric_from_confusion(metric: str, cm) -> float:
    import numpy as np

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


def subject_stratified_bootstrap(prediction_rows, n_bootstrap: int, ci: float, seed: int):
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(prediction_rows)
    rng = np.random.default_rng(seed)
    out: list[dict[str, object]] = []

    def stratified_sample_indices(labels):
        sampled_indices = []
        for label in sorted(np.unique(labels)):
            label_indices = np.flatnonzero(labels == label)
            sampled_indices.extend(
                rng.choice(label_indices, size=len(label_indices), replace=True).tolist()
            )
        return np.array(sampled_indices, dtype=int)

    group_cols = ["scenario", "train_input", "test_input", "classes"]
    for (scenario, train_input, test_input, classes), group in df.groupby(
        group_cols,
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
                    "scenario": scenario,
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
    for scenario, scenario_group in full_train.groupby("scenario", sort=False):
        for test_input in ["aperiodic_spectrum", "flattened_log_psd"]:
            paired = scenario_group[
                scenario_group["test_input"].isin(["full_log_psd", test_input])
            ]
            if paired.empty:
                continue
            for metric in METRICS:
                baseline_group = paired[paired["test_input"] == "full_log_psd"]
                edited_group = paired[paired["test_input"] == test_input]
                subjects, subject_labels, baseline_matrices = subject_confusions(
                    baseline_group
                )
                edited_subjects, edited_labels, edited_matrices = subject_confusions(
                    edited_group
                )
                if (
                    subjects.tolist() != edited_subjects.tolist()
                    or subject_labels.tolist() != edited_labels.tolist()
                ):
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
                            "scenario": scenario,
                            "task": "tuab_normal_vs_abnormal",
                            "train_input": "full_log_psd",
                            "test_input": test_input,
                            "classes": "|".join(
                                sorted(baseline_group["label"].unique())
                            ),
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
            "scenario",
            "train_input",
            "test_input",
            "estimate",
            "point",
            "ci_lower",
            "ci_upper",
            "n_eval_subjects",
        ]
    ]
    for col in ["point", "ci_lower", "ci_upper"]:
        focus[col] = focus[col].map(lambda value: f"{value:.3f}")

    lines = [
        "# TUAB Site-Level Temporal PSD Audit",
        "",
        "Recording year is used as an acquisition-condition proxy. Sentinel/outlier "
        "years are excluded unless noted in metadata. Confidence intervals use "
        "stratified eval-subject bootstrap.",
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
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(args.index_csv)
    metadata = pd.read_csv(args.metadata_csv)
    index = add_temporal_bins(index, metadata, args.include_sentinel_years)
    index = index[index["temporal_valid"]].reset_index(drop=False).rename(
        columns={"index": "source_row_index"}
    )

    decomp = np.load(args.decomp_npz)
    source_rows = index["source_row_index"].to_numpy(dtype=int)
    arrays = {
        "full_log_psd": decomp["log_psd"][source_rows],
        "aperiodic_spectrum": decomp["aperiodic_log_psd"][source_rows],
        "flattened_log_psd": decomp["residual_log_psd"][source_rows],
    }
    features = {
        name: flatten_spectral_features(array).astype("float32", copy=False)
        for name, array in arrays.items()
    }

    labels = index["label"].astype(str).to_numpy()
    subjects = index["subject"].astype(str).to_numpy()
    splits = index["official_split"].astype(str).to_numpy()
    y, classes = encode_labels(labels)
    n_classes = len(classes)

    scenarios: list[tuple[str, object, object]] = [
        (
            "within_primary_early_2009_2011",
            (index["official_split"] == "train")
            & (index["temporal_primary"] == "early_2009_2011"),
            (index["official_split"] == "eval")
            & (index["temporal_primary"] == "early_2009_2011"),
        ),
        (
            "within_primary_late_2012_2013",
            (index["official_split"] == "train")
            & (index["temporal_primary"] == "late_2012_2013"),
            (index["official_split"] == "eval")
            & (index["temporal_primary"] == "late_2012_2013"),
        ),
        (
            "cross_primary_train_early_eval_late",
            (index["official_split"] == "train")
            & (index["temporal_primary"] == "early_2009_2011"),
            (index["official_split"] == "eval")
            & (index["temporal_primary"] == "late_2012_2013"),
        ),
        (
            "cross_primary_train_late_eval_early",
            (index["official_split"] == "train")
            & (index["temporal_primary"] == "late_2012_2013"),
            (index["official_split"] == "eval")
            & (index["temporal_primary"] == "early_2009_2011"),
        ),
    ]
    for bin_name in [
        "tercile_early_2009_2010",
        "tercile_middle_2011_2012",
        "tercile_late_2013",
    ]:
        scenarios.append(
            (
                f"within_{bin_name}",
                (index["official_split"] == "train") & (index["temporal_tercile"] == bin_name),
                (index["official_split"] == "eval") & (index["temporal_tercile"] == bin_name),
            )
        )

    scenario_counts: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for scenario, train_mask, eval_mask in scenarios:
        train_idx = np.flatnonzero(np.asarray(train_mask))
        eval_idx = np.flatnonzero(np.asarray(eval_mask))
        row = {
            "scenario": scenario,
            "n_train_epochs": int(len(train_idx)),
            "n_eval_epochs": int(len(eval_idx)),
            "n_train_subjects": int(len(np.unique(subjects[train_idx]))),
            "n_eval_subjects": int(len(np.unique(subjects[eval_idx]))),
        }
        for split_name, idxs in [("train", train_idx), ("eval", eval_idx)]:
            for label in classes:
                label_mask = labels[idxs] == label
                row[f"n_{split_name}_{label}_epochs"] = int(label_mask.sum())
                row[f"n_{split_name}_{label}_subjects"] = int(
                    len(np.unique(subjects[idxs][label_mask]))
                )
        scenario_counts.append(row)
        if len(train_idx) == 0 or len(eval_idx) == 0:
            print(f"Skipping {scenario}: empty train/eval", flush=True)
            continue
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[eval_idx])) < 2:
            print(f"Skipping {scenario}: missing class in train/eval", flush=True)
            continue

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
            print(f"Fitted scenario={scenario} train_input={train_input}", flush=True)

            for test_input in TEST_INPUTS:
                pred = clf.predict(features[test_input][eval_idx])
                base_row = {
                    "scenario": scenario,
                    "task": "tuab_normal_vs_abnormal",
                    "train_input": train_input,
                    "test_input": test_input,
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
                            "scenario": scenario,
                            "task": "tuab_normal_vs_abnormal",
                            "train_input": train_input,
                            "test_input": test_input,
                            "classes": "|".join(classes),
                            "n_classes": n_classes,
                            "source_row_index": int(index.iloc[idx]["source_row_index"]),
                            "subject": subjects[idx],
                            "label": labels[idx],
                            "recording_year": int(index.iloc[idx]["recording_year"]),
                            "temporal_primary": index.iloc[idx]["temporal_primary"],
                            "temporal_tercile": index.iloc[idx]["temporal_tercile"],
                            "y_true": int(y[idx]),
                            "y_pred": int(y_pred),
                        }
                    )
                print(
                    f"Evaluated scenario={scenario} train={train_input} test={test_input}",
                    flush=True,
                )

    bootstrap_rows = subject_stratified_bootstrap(
        prediction_rows,
        n_bootstrap=args.n_bootstrap,
        ci=args.ci,
        seed=args.seed,
    )

    write_csv(args.output_dir / "tuab_site_temporal_scenario_counts.csv", scenario_counts)
    write_csv(args.output_dir / "tuab_site_temporal_eval_metrics.csv", fold_rows)
    write_csv(args.output_dir / "tuab_site_temporal_subject_metrics.csv", subject_rows)
    write_csv(args.output_dir / "tuab_site_temporal_predictions.csv", prediction_rows)
    write_csv(args.output_dir / "tuab_site_temporal_subject_bootstrap.csv", bootstrap_rows)
    write_markdown(args.output_dir / "tuab_site_temporal_subject_bootstrap.md", bootstrap_rows)

    metadata_years = metadata.copy()
    metadata_years["recording_year"] = metadata_years["meas_date"].map(year_from_header)
    valid_years = metadata_years["recording_year"].dropna()
    metadata_out = {
        "index_csv": str(args.index_csv),
        "decomp_npz": str(args.decomp_npz),
        "metadata_csv": str(args.metadata_csv),
        "output_dir": str(args.output_dir),
        "date_proxy": "EDF meas_date / Startdate recording header, year-level in TUAB headers",
        "sentinel_year_policy": (
            "Excluded years outside 2009--2013"
            if not args.include_sentinel_years
            else "Included all parsed years"
        ),
        "primary_bins": {
            "early_2009_2011": "recording_year <= 2011",
            "late_2012_2013": "recording_year >= 2012",
        },
        "tercile_bins": {
            "tercile_early_2009_2010": "recording_year <= 2010",
            "tercile_middle_2011_2012": "2011 <= recording_year <= 2012",
            "tercile_late_2013": "recording_year == 2013",
        },
        "n_epochs_used": int(len(index)),
        "n_recordings_metadata": int(len(metadata)),
        "recording_year_counts_all_metadata": {
            str(int(k)): int(v) for k, v in valid_years.value_counts().sort_index().items()
        },
        "montage_counts_used_epochs": {
            str(k): int(v) for k, v in index["montage"].value_counts().sort_index().items()
        },
        "recording_header_note": (
            "The recording header contains Startdate and file/session tokens; the "
            "technician/equipment-like suffix is mostly anonymized as 'XXX X'."
        ),
        "ridge_alpha": float(args.ridge_alpha),
        "n_bootstrap": int(args.n_bootstrap),
        "ci": float(args.ci),
        "seed": int(args.seed),
        "classes": classes,
    }
    (args.output_dir / "tuab_site_temporal_metadata.json").write_text(
        json.dumps(metadata_out, indent=2) + "\n"
    )

    focus = pd.DataFrame(bootstrap_rows)
    focus = focus[
        (focus["metric"] == "balanced_accuracy")
        & focus["estimate"].isin(["performance", "drop::flattened_log_psd"])
    ]
    print(f"Wrote outputs to: {args.output_dir}")
    print(
        focus[
            [
                "scenario",
                "train_input",
                "test_input",
                "estimate",
                "point",
                "ci_lower",
                "ci_upper",
                "n_eval_subjects",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
