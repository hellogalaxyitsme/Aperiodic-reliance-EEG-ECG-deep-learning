#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path


TEST_INPUTS = ["raw_eeg", "phase_sham", "phase_aperiodic", "phase_flattened"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a raw-EEG CNN and evaluate phase-preserving aperiodic "
            "spectral interventions on held-out Sleep-EDF subjects."
        )
    )
    parser.add_argument(
        "--raw-npz",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/results/sleep_edf_subset/raw_epochs_fpz_pz_100hz.npz"),
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/results/sleep_edf_subset/raw_epochs_index.csv"),
    )
    parser.add_argument(
        "--decomp-npz",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/results/sleep_edf_subset/specparam/specparam_fixed.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/results/sleep_edf_subset/raw_cnn_interventions"),
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--band-min", type=float, default=1.0)
    parser.add_argument("--band-max", type=float, default=45.0)
    parser.add_argument(
        "--no-match-rms",
        action="store_true",
        help="Do not rescale edited epochs back to the original per-channel RMS.",
    )
    parser.add_argument("--max-epochs-data", type=int, default=None)
    parser.add_argument("--device", default="cuda")
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

    labels = np.asarray(labels)
    mask = labels != None  # noqa: E711
    classes = sorted(np.unique(labels[mask]).tolist())
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    y = np.array([class_to_idx[label] for label in labels[mask]], dtype=np.int64)
    return mask, y, classes


def make_subject_folds(groups, n_splits: int):
    import numpy as np

    groups = np.asarray(groups)
    subjects, counts = np.unique(groups, return_counts=True)
    order = np.argsort(-counts)
    fold_subjects = [[] for _ in range(min(n_splits, len(subjects)))]
    fold_counts = [0 for _ in fold_subjects]
    for idx in order:
        fold_idx = int(np.argmin(fold_counts))
        fold_subjects[fold_idx].append(subjects[idx])
        fold_counts[fold_idx] += int(counts[idx])

    folds = []
    all_idx = np.arange(len(groups))
    for test_subjects in fold_subjects:
        test_mask = np.isin(groups, test_subjects)
        test_idx = all_idx[test_mask]
        train_idx = all_idx[~test_mask]
        folds.append((train_idx, test_idx, sorted(str(s) for s in test_subjects)))
    return folds


def balanced_accuracy(y_true, y_pred, n_classes: int) -> float:
    import numpy as np

    recalls = []
    for cls in range(n_classes):
        mask = y_true == cls
        if mask.sum() == 0:
            continue
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


def append_subject_metric_rows(
    subject_rows: list[dict[str, object]],
    base_row: dict[str, object],
    subjects,
    y_true,
    y_pred,
    n_classes: int,
) -> None:
    import numpy as np

    subjects = np.asarray(subjects)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    for subject in sorted(np.unique(subjects)):
        subject_mask = subjects == subject
        support = {
            str(cls): int((y_true[subject_mask] == cls).sum())
            for cls in range(n_classes)
        }
        subject_rows.append(
            {
                **base_row,
                "subject": str(subject),
                "n_subject": int(subject_mask.sum()),
                "class_support_json": json.dumps(support, sort_keys=True),
                "balanced_accuracy": balanced_accuracy(
                    y_true[subject_mask],
                    y_pred[subject_mask],
                    n_classes,
                ),
                "macro_f1": macro_f1(y_true[subject_mask], y_pred[subject_mask], n_classes),
                "accuracy": accuracy(y_true[subject_mask], y_pred[subject_mask]),
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


def summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
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
        for metric in ["balanced_accuracy", "macro_f1", "accuracy"]:
            vals = group[metric].to_numpy(dtype=float)
            item[f"{metric}_mean"] = float(vals.mean())
            item[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out.append(item)

    full_by_task = {row["task"]: row for row in out if row["test_input"] == "raw_eeg"}
    derived = []
    for row in out:
        if row["test_input"] == "raw_eeg":
            continue
        full = full_by_task[row["task"]]
        derived.append(
            {
                "task": row["task"],
                "train_input": row["train_input"],
                "test_input": f"retention::{row['test_input']}",
                "classes": row["classes"],
                "n_folds": row["n_folds"],
                "balanced_accuracy_mean": row["balanced_accuracy_mean"]
                / max(full["balanced_accuracy_mean"], 1e-12),
                "balanced_accuracy_std": "",
                "macro_f1_mean": row["macro_f1_mean"] / max(full["macro_f1_mean"], 1e-12),
                "macro_f1_std": "",
                "accuracy_mean": row["accuracy_mean"] / max(full["accuracy_mean"], 1e-12),
                "accuracy_std": "",
            }
        )
        derived.append(
            {
                "task": row["task"],
                "train_input": row["train_input"],
                "test_input": f"drop::{row['test_input']}",
                "classes": row["classes"],
                "n_folds": row["n_folds"],
                "balanced_accuracy_mean": full["balanced_accuracy_mean"]
                - row["balanced_accuracy_mean"],
                "balanced_accuracy_std": "",
                "macro_f1_mean": full["macro_f1_mean"] - row["macro_f1_mean"],
                "macro_f1_std": "",
                "accuracy_mean": full["accuracy_mean"] - row["accuracy_mean"],
                "accuracy_std": "",
            }
        )
    return out + derived


class RawSleepCNN:
    def __new__(cls, n_channels: int, n_classes: int, dropout: float):
        from torch import nn

        return nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=25, stride=2, padding=12, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(64, 128, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, n_classes),
        )


def make_train_val_split(y, seed: int, val_fraction: float = 0.15):
    import numpy as np

    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        rng.shuffle(cls_idx)
        n_val = max(1, int(round(len(cls_idx) * val_fraction)))
        val_idx.extend(cls_idx[:n_val].tolist())
        train_idx.extend(cls_idx[n_val:].tolist())
    return np.array(train_idx, dtype=int), np.array(val_idx, dtype=int)


def interpolate_aperiodic(ap_log_psd, decomp_freqs, target_freqs):
    import numpy as np

    out = np.empty(ap_log_psd.shape[:2] + (len(target_freqs),), dtype="float32")
    for epoch_idx in range(ap_log_psd.shape[0]):
        for channel_idx in range(ap_log_psd.shape[1]):
            out[epoch_idx, channel_idx] = np.interp(
                target_freqs,
                decomp_freqs,
                ap_log_psd[epoch_idx, channel_idx],
            ).astype("float32")
    return out


def make_phase_preserving_inputs(
    x_test,
    ap_log_psd,
    decomp_freqs,
    sfreq: float,
    band_min: float,
    band_max: float,
    match_rms: bool,
):
    import numpy as np

    n_times = x_test.shape[-1]
    fft_freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    band_mask = (fft_freqs >= band_min) & (fft_freqs <= band_max)
    band_freqs = fft_freqs[band_mask]

    fft = np.fft.rfft(x_test, axis=-1)
    band_fft = fft[:, :, band_mask]
    band_amp = np.maximum(np.abs(band_fft), 1e-12)
    band_phase = band_fft / band_amp

    ap_interp = interpolate_aperiodic(ap_log_psd, decomp_freqs, band_freqs)
    ap_centered = ap_interp - ap_interp.mean(axis=-1, keepdims=True)
    ap_amp_shape = np.power(10.0, 0.5 * ap_centered).astype("float32")

    flattened_fft = fft.copy()
    flattened_fft[:, :, band_mask] = band_fft / np.maximum(ap_amp_shape, 1e-6)

    aperiodic_fft = fft.copy()
    geom_amp = np.exp(np.mean(np.log(band_amp), axis=-1, keepdims=True))
    aperiodic_fft[:, :, band_mask] = band_phase * geom_amp * ap_amp_shape

    sham_fft = fft.copy()
    sham_fft[:, :, band_mask] = band_phase * band_amp

    edited = {
        "phase_sham": np.fft.irfft(sham_fft, n=n_times, axis=-1),
        "phase_aperiodic": np.fft.irfft(aperiodic_fft, n=n_times, axis=-1),
        "phase_flattened": np.fft.irfft(flattened_fft, n=n_times, axis=-1),
    }

    original_std = x_test.std(axis=-1, keepdims=True)
    for name, array in edited.items():
        array = array.real.astype("float32", copy=False)
        array -= array.mean(axis=-1, keepdims=True)
        if match_rms:
            edited_std = array.std(axis=-1, keepdims=True)
            array *= original_std / np.maximum(edited_std, 1e-6)
        edited[name] = array.astype("float32", copy=False)
    return edited


def fit_predict_fold(
    x_train,
    y_train,
    test_inputs,
    args,
    n_classes: int,
    fold_seed: int,
):
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = x_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x_train_scaled = ((x_train - mean) / std).astype("float32")

    sub_train_idx, val_idx = make_train_val_split(y_train, seed=fold_seed)
    class_counts = np.bincount(y_train[sub_train_idx], minlength=n_classes).astype("float32")
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.mean()

    train_ds = TensorDataset(
        torch.from_numpy(x_train_scaled[sub_train_idx]),
        torch.from_numpy(y_train[sub_train_idx]),
    )
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    val_x = torch.from_numpy(x_train_scaled[val_idx]).to(device)
    val_y = torch.from_numpy(y_train[val_idx]).to(device)

    model = RawSleepCNN(x_train.shape[1], n_classes, args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_val = math.inf
    best_epoch = 0
    patience_left = args.patience
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(val_x), val_y).item())
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    predictions = {}
    with torch.no_grad():
        for name, x_test in test_inputs.items():
            x_test_scaled = ((x_test - mean) / std).astype("float32")
            preds = []
            for start in range(0, len(x_test_scaled), args.batch_size):
                xb = torch.from_numpy(x_test_scaled[start : start + args.batch_size]).to(device)
                preds.append(model(xb).argmax(dim=1).cpu().numpy())
            predictions[name] = np.concatenate(preds)

    return predictions, {
        "device": str(device),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
    }


def main() -> int:
    import numpy as np
    import pandas as pd
    import torch

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    raw = np.load(args.raw_npz)
    decomp = np.load(args.decomp_npz)
    x = raw["x"].astype("float32", copy=False)
    ap_log_psd = decomp["aperiodic_log_psd"].astype("float32", copy=False)
    decomp_freqs = decomp["freqs"].astype("float32", copy=False)
    index = pd.read_csv(args.index_csv)

    if args.max_epochs_data is not None:
        x = x[: args.max_epochs_data]
        ap_log_psd = ap_log_psd[: args.max_epochs_data]
        index = index.iloc[: args.max_epochs_data].copy()
    if len(index) != len(x):
        raise ValueError(f"Index rows ({len(index)}) do not match raw epochs ({len(x)})")
    if len(ap_log_psd) < len(x):
        raise ValueError(f"Specparam rows ({len(ap_log_psd)}) do not cover raw epochs ({len(x)})")

    task_labels = make_task_labels(index["stage"].to_numpy())
    groups_all = index["subject"].to_numpy()
    fold_rows = []
    subject_rows = []
    train_rows = []
    match_rms = not args.no_match_rms

    for task_name, labels in task_labels.items():
        mask, y, classes = encode_labels(labels)
        task_x = x[mask]
        task_ap = ap_log_psd[mask]
        task_groups = groups_all[mask]
        if len(set(task_groups.tolist())) < 2:
            print(f"Skipping {task_name}: fewer than two subject groups")
            continue
        n_classes = len(classes)
        folds = make_subject_folds(task_groups, args.n_splits)

        for fold_idx, (train_idx, test_idx, test_subjects) in enumerate(folds):
            if len(train_idx) == 0 or len(test_idx) == 0:
                print(f"Skipping {task_name} fold {fold_idx}: empty train/test split")
                continue
            train_subjects = sorted(str(s) for s in np.unique(task_groups[train_idx]))
            edited = make_phase_preserving_inputs(
                task_x[test_idx],
                task_ap[test_idx],
                decomp_freqs,
                sfreq=float(raw["sfreq"]),
                band_min=args.band_min,
                band_max=args.band_max,
                match_rms=match_rms,
            )
            test_inputs = {"raw_eeg": task_x[test_idx], **edited}
            predictions, meta = fit_predict_fold(
                task_x[train_idx],
                y[train_idx],
                test_inputs,
                args,
                n_classes=n_classes,
                fold_seed=args.seed + fold_idx,
            )
            y_test = y[test_idx]
            train_rows.append(
                {
                    "task": task_name,
                    "seed": int(args.seed),
                    "fold": fold_idx,
                    "classes": "|".join(classes),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "train_subjects": ",".join(train_subjects),
                    "test_subjects": ",".join(test_subjects),
                    **meta,
                }
            )
            for test_input in TEST_INPUTS:
                pred = predictions[test_input]
                base_row = {
                    "task": task_name,
                    "seed": int(args.seed),
                    "train_input": "raw_eeg",
                    "test_input": test_input,
                    "fold": fold_idx,
                    "classes": "|".join(classes),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "train_subjects": ",".join(train_subjects),
                    "test_subjects": ",".join(test_subjects),
                }
                fold_rows.append(
                    {
                        **base_row,
                        "balanced_accuracy": balanced_accuracy(y_test, pred, n_classes),
                        "macro_f1": macro_f1(y_test, pred, n_classes),
                        "accuracy": accuracy(y_test, pred),
                    }
                )
                append_subject_metric_rows(
                    subject_rows,
                    base_row,
                    task_groups[test_idx],
                    y_test,
                    pred,
                    n_classes,
                )
            print(f"{task_name} fold {fold_idx} done", flush=True)

    summary_rows = summarize_rows(fold_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "raw_cnn_intervention_fold_metrics.csv", fold_rows)
    write_csv(args.output_dir / "raw_cnn_intervention_subject_metrics.csv", subject_rows)
    write_csv(args.output_dir / "raw_cnn_intervention_summary_metrics.csv", summary_rows)
    write_csv(args.output_dir / "raw_cnn_intervention_training_log.csv", train_rows)
    (args.output_dir / "raw_cnn_intervention_metadata.json").write_text(
        json.dumps(
            {
                "raw_npz": str(args.raw_npz),
                "index_csv": str(args.index_csv),
                "decomp_npz": str(args.decomp_npz),
                "raw_shape": list(x.shape),
                "sfreq": float(raw["sfreq"]),
                "channels": raw["channels"].tolist(),
                "train_input": "raw_eeg",
                "test_inputs": TEST_INPUTS,
                "intervention": "phase-preserving FFT amplitude edits",
                "band_min": float(args.band_min),
                "band_max": float(args.band_max),
                "match_rms": bool(match_rms),
                "n_splits": int(args.n_splits),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "dropout": float(args.dropout),
                "learning_rate": float(args.learning_rate),
                "weight_decay": float(args.weight_decay),
                "device_requested": args.device,
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
            indent=2,
        )
    )

    direct = [row for row in summary_rows if not str(row["test_input"]).startswith(("retention::", "drop::"))]
    print(
        pd.DataFrame(direct)[
            ["task", "test_input", "balanced_accuracy_mean", "macro_f1_mean", "accuracy_mean"]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
