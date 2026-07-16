#!/usr/bin/env python
from __future__ import annotations

import argparse
import bisect
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
    parser = argparse.ArgumentParser(description="PTB-XL age/sex-matched PSD intervention control.")
    parser.add_argument("--index-csv", type=Path, default=Path("results/ptbxl_1f_demo/ptbxl_records100_normal_abnormal_index.csv"))
    parser.add_argument("--psd-npz", type=Path, default=Path("results/ptbxl_1f_demo/ptbxl_records100_normal_abnormal_psd_fixed.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/ptbxl_1f_demo/age_sex_matched_psd"))
    parser.add_argument("--caliper-years", type=float, default=5.0)
    parser.add_argument("--max-age", type=float, default=90.0, help="Exclude PTB-XL de-identified oldest-age bucket, encoded as 300.")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260529)
    return parser.parse_args()


def split_name(fold: int) -> str:
    if fold <= 8:
        return "train"
    if fold == 9:
        return "validation"
    if fold == 10:
        return "test"
    return "other"


def matched_pairs_for_split(df: pd.DataFrame, caliper: float):
    normals = df[df["label"] == "normal"].copy().sort_values(["sex", "age", "ecg_id"])
    abnormal_by_sex = {}
    for sex, group in df[df["label"] == "abnormal"].copy().sort_values(["age", "ecg_id"]).groupby("sex"):
        records = []
        ages = []
        for item in group.itertuples(index=False):
            records.append(item)
            ages.append(float(item.age))
        abnormal_by_sex[int(sex)] = {"records": records, "ages": ages}
    used_normal_patients = set()
    used_abnormal_patients = set()
    pairs = []
    for normal in normals.itertuples(index=False):
        if normal.patient_id in used_normal_patients:
            continue
        pool = abnormal_by_sex.get(int(normal.sex))
        if not pool or not pool["records"]:
            continue
        ages = pool["ages"]
        records = pool["records"]
        center = bisect.bisect_left(ages, float(normal.age))
        best_idx = None
        best_key = None
        left = center - 1
        while left >= 0 and abs(ages[left] - float(normal.age)) <= caliper:
            candidate = records[left]
            if str(candidate.patient_id) not in used_abnormal_patients:
                key = (abs(float(candidate.age) - float(normal.age)), float(candidate.age), int(candidate.ecg_id))
                if best_key is None or key < best_key:
                    best_key = key
                    best_idx = left
            left -= 1
        right = center
        while right < len(records) and abs(ages[right] - float(normal.age)) <= caliper:
            candidate = records[right]
            if str(candidate.patient_id) not in used_abnormal_patients:
                key = (abs(float(candidate.age) - float(normal.age)), float(candidate.age), int(candidate.ecg_id))
                if best_key is None or key < best_key:
                    best_key = key
                    best_idx = right
            right += 1
        if best_idx is None:
            continue
        abnormal = records.pop(best_idx)
        ages.pop(best_idx)
        pair_id = f"{normal.split_group}_pair_{len(pairs):05d}"
        pairs.append(
            {
                "pair_id": pair_id,
                "split_group": normal.split_group,
                "normal_ecg_id": int(normal.ecg_id),
                "abnormal_ecg_id": int(abnormal.ecg_id),
                "normal_patient_id": str(normal.patient_id),
                "abnormal_patient_id": str(abnormal.patient_id),
                "normal_age": float(normal.age),
                "abnormal_age": float(abnormal.age),
                "age_abs_diff": float(abs(abnormal.age - normal.age)),
                "sex": int(normal.sex),
            }
        )
        used_normal_patients.add(str(normal.patient_id))
        used_abnormal_patients.add(str(abnormal.patient_id))
    return pairs


def features(array: np.ndarray) -> np.ndarray:
    return array.reshape(array.shape[0], -1).astype("float32", copy=False)


def metric_value(metric: str, y_true, y_pred):
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    if metric == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    raise ValueError(metric)


def pooled_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    return {metric: metric_value(metric, y_true, y_pred) for metric in METRICS}


def ci_bounds(values):
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def sign_p(values):
    values = np.asarray(values, dtype=float)
    n = int(len(values))
    nonpos = int(np.count_nonzero(values <= 0.0))
    nonneg = int(np.count_nonzero(values >= 0.0))
    return {
        "p_one_sided_positive": nonpos / n,
        "p_two_sided_zero": min(1.0, 2.0 * min(nonpos / n, nonneg / n)),
        "n_bootstrap_nonpositive": nonpos,
        "n_bootstrap_nonnegative": nonneg,
        "n_bootstrap_valid": n,
    }


def bootstrap_prediction_rows(pred_rows, n_bootstrap: int, seed: int):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(pred_rows)
    rows = []

    def counts_by_pair(group: pd.DataFrame, pair_ids: np.ndarray):
        pair_index = {pair: idx for idx, pair in enumerate(pair_ids)}
        counts = np.zeros((len(pair_ids), 4), dtype=np.int64)
        for row in group.itertuples(index=False):
            idx = pair_index[row.pair_id]
            if row.y_true == 0 and row.y_pred == 0:
                counts[idx, 0] += 1
            elif row.y_true == 0 and row.y_pred == 1:
                counts[idx, 1] += 1
            elif row.y_true == 1 and row.y_pred == 0:
                counts[idx, 2] += 1
            elif row.y_true == 1 and row.y_pred == 1:
                counts[idx, 3] += 1
        return counts

    def metrics_from_counts(counts: np.ndarray):
        tn = counts[..., 0].astype(float)
        fp = counts[..., 1].astype(float)
        fn = counts[..., 2].astype(float)
        tp = counts[..., 3].astype(float)
        total = np.maximum(tn + fp + fn + tp, 1.0)
        accuracy = (tn + tp) / total
        recall0 = tn / np.maximum(tn + fp, 1e-12)
        recall1 = tp / np.maximum(tp + fn, 1e-12)
        balanced_accuracy = 0.5 * (recall0 + recall1)
        f1_0 = 2.0 * tn / np.maximum(2.0 * tn + fn + fp, 1e-12)
        f1_1 = 2.0 * tp / np.maximum(2.0 * tp + fp + fn, 1e-12)
        macro_f1 = 0.5 * (f1_0 + f1_1)
        return {"balanced_accuracy": balanced_accuracy, "macro_f1": macro_f1, "accuracy": accuracy}

    for (train_input, test_input), group in df.groupby(["train_input", "test_input"], sort=False):
        pair_ids = np.array(sorted(group["pair_id"].unique()))
        counts = counts_by_pair(group, pair_ids)
        point = metrics_from_counts(counts.sum(axis=0))
        sample_idx = rng.integers(0, len(pair_ids), size=(n_bootstrap, len(pair_ids)))
        pooled = counts[sample_idx].sum(axis=1)
        boot_metrics = metrics_from_counts(pooled)
        for metric in METRICS:
            boot = np.asarray(boot_metrics[metric], dtype=float)
            lo, hi = ci_bounds(boot)
            rows.append(
                {
                    "task": "ptbxl_normal_vs_abnormal",
                    "train_input": train_input,
                    "test_input": test_input,
                    "metric": metric,
                    "estimate": "performance",
                    "point": float(point[metric]),
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "ci": 0.95,
                    "n_pairs": int(len(pair_ids)),
                    "n_bootstrap": int(n_bootstrap),
                    **sign_p(boot),
                }
            )

    full = df[df["train_input"] == "full_log_psd"]
    for test_input in ["aperiodic_spectrum", "flattened_log_psd"]:
        paired = full[full["test_input"].isin(["full_log_psd", test_input])]
        pair_ids = np.array(sorted(paired["pair_id"].unique()))
        base = paired[paired["test_input"] == "full_log_psd"]
        other = paired[paired["test_input"] == test_input]
        base_counts = counts_by_pair(base, pair_ids)
        other_counts = counts_by_pair(other, pair_ids)
        point_base = metrics_from_counts(base_counts.sum(axis=0))
        point_other = metrics_from_counts(other_counts.sum(axis=0))
        sample_idx = rng.integers(0, len(pair_ids), size=(n_bootstrap, len(pair_ids)))
        boot_base = metrics_from_counts(base_counts[sample_idx].sum(axis=1))
        boot_other = metrics_from_counts(other_counts[sample_idx].sum(axis=1))
        for metric in METRICS:
            boot = np.asarray(boot_base[metric] - boot_other[metric], dtype=float)
            lo, hi = ci_bounds(boot)
            rows.append(
                {
                    "task": "ptbxl_normal_vs_abnormal",
                    "train_input": "full_log_psd",
                    "test_input": test_input,
                    "metric": metric,
                    "estimate": f"drop::{test_input}",
                    "point": float(point_base[metric] - point_other[metric]),
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "ci": 0.95,
                    "n_pairs": int(len(pair_ids)),
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
    index = pd.read_csv(args.index_csv)
    index["record_index"] = np.arange(len(index))
    index["split_group"] = index["strat_fold"].map(split_name)
    age_valid = index["age"].notna() & (index["age"] <= args.max_age)
    sex_valid = index["sex"].isin([0, 1])
    match_base = index[age_valid & sex_valid & index["split_group"].isin(["train", "validation", "test"])].copy()

    all_pairs = []
    for split in ["train", "validation", "test"]:
        all_pairs.extend(matched_pairs_for_split(match_base[match_base["split_group"] == split], args.caliper_years))
    pairs = pd.DataFrame(all_pairs)
    if pairs.empty:
        raise RuntimeError("No matched PTB-XL pairs found.")

    selected = []
    pair_lookup = {}
    for pair in pairs.itertuples(index=False):
        for label, ecg_id in [("normal", pair.normal_ecg_id), ("abnormal", pair.abnormal_ecg_id)]:
            row = index[index["ecg_id"] == ecg_id].iloc[0].to_dict()
            row["pair_id"] = pair.pair_id
            selected.append(row)
            pair_lookup[int(row["record_index"])] = pair.pair_id
    selected_df = pd.DataFrame(selected)

    data = np.load(args.psd_npz, allow_pickle=True)
    y = data["y"].astype(int)
    fold = data["strat_fold"].astype(int)
    arrays = {name: features(data[ARRAY_KEY[name]]) for name in REPRESENTATIONS}
    train_indices = selected_df[selected_df["split_group"] == "train"]["record_index"].to_numpy(dtype=int)
    test_indices = selected_df[selected_df["split_group"] == "test"]["record_index"].to_numpy(dtype=int)
    pred_rows = []

    for train_input in REPRESENTATIONS:
        clf = make_pipeline(StandardScaler(), RidgeClassifier(alpha=args.ridge_alpha))
        clf.fit(arrays[train_input][train_indices], y[train_indices])
        for test_input in REPRESENTATIONS:
            pred = clf.predict(arrays[test_input][test_indices])
            for rec_idx, yt, yp in zip(test_indices, y[test_indices], pred):
                pred_rows.append(
                    {
                        "task": "ptbxl_normal_vs_abnormal",
                        "train_input": train_input,
                        "test_input": test_input,
                        "pair_id": pair_lookup[int(rec_idx)],
                        "subject": str(index.iloc[int(rec_idx)]["patient_id"]),
                        "record_index": int(rec_idx),
                        "ecg_id": int(index.iloc[int(rec_idx)]["ecg_id"]),
                        "y_true": int(yt),
                        "y_pred": int(yp),
                    }
                )

    boot_rows = bootstrap_prediction_rows(pred_rows, args.n_bootstrap, args.seed)
    pairs.to_csv(args.output_dir / "ptbxl_age_sex_matched_pairs.csv", index=False)
    selected_df.to_csv(args.output_dir / "ptbxl_age_sex_matched_records.csv", index=False)
    write_csv(args.output_dir / "ptbxl_age_sex_matched_psd_predictions.csv", pred_rows)
    write_csv(args.output_dir / "ptbxl_age_sex_matched_psd_bootstrap.csv", boot_rows)

    summary = {
        "caliper_years": float(args.caliper_years),
        "max_age": float(args.max_age),
        "n_records_total": int(len(index)),
        "n_records_age_gt_max_or_missing": int((~age_valid).sum()),
        "n_records_eligible_for_matching": int(len(match_base)),
        "n_pairs_by_split": {split: int((pairs["split_group"] == split).sum()) for split in ["train", "validation", "test"]},
        "n_selected_records_by_split": {split: int((selected_df["split_group"] == split).sum()) for split in ["train", "validation", "test"]},
        "train_indices": int(len(train_indices)),
        "test_indices": int(len(test_indices)),
        "ridge_alpha": float(args.ridge_alpha),
        "n_bootstrap": int(args.n_bootstrap),
    }
    (args.output_dir / "ptbxl_age_sex_matched_summary.json").write_text(json.dumps(summary, indent=2))
    focus = pd.DataFrame(boot_rows)
    print(json.dumps(summary, indent=2))
    print(
        focus[
            (focus["metric"] == "balanced_accuracy")
            & (focus["train_input"] == "full_log_psd")
        ][["test_input", "estimate", "point", "ci_lower", "ci_upper", "p_one_sided_positive"]].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
