#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path


TRAIN_INPUTS = ["full_log_psd", "aperiodic_spectrum", "flattened_log_psd"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train matched MLP controls directly on full, aperiodic-only, and "
            "aperiodic-flattened PSD features. This estimates how much "
            "residual class information is learnable after aperiodic removal."
        )
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/results/sleep_edf_full/psd_index.csv"),
    )
    parser.add_argument(
        "--decomp-npz",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/results/sleep_edf_full/specparam/specparam_fixed.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/results/sleep_edf_full/psd_train_input_controls"),
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
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
        folds.append(
            (
                all_idx[~test_mask],
                all_idx[test_mask],
                sorted(str(s) for s in test_subjects),
            )
        )
    return folds


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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_subject_rows(rows, base_row, subjects, y_true, y_pred, n_classes):
    import numpy as np

    subjects = np.asarray(subjects)
    for subject in sorted(np.unique(subjects)):
        subject_mask = subjects == subject
        support = {
            str(cls): int((y_true[subject_mask] == cls).sum())
            for cls in range(n_classes)
        }
        rows.append(
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


class MLP:
    def __new__(cls, input_dim, hidden_dim, n_classes, dropout):
        from torch import nn

        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
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


def fit_predict(x_train, y_train, x_test, args, n_classes: int, fold_seed: int):
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x_train = ((x_train - mean) / std).astype("float32")
    x_test = ((x_test - mean) / std).astype("float32")

    sub_train_idx, val_idx = make_train_val_split(y_train, seed=fold_seed)
    class_counts = np.bincount(y_train[sub_train_idx], minlength=n_classes).astype("float32")
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.mean()

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train[sub_train_idx]),
            torch.from_numpy(y_train[sub_train_idx]),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_x = torch.from_numpy(x_train[val_idx]).to(device)
    val_y = torch.from_numpy(y_train[val_idx]).to(device)
    model = MLP(x_train.shape[1], args.hidden_dim, n_classes, args.dropout).to(device)
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
            xb = xb.to(device)
            yb = yb.to(device)
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
    preds = []
    with torch.no_grad():
        for start in range(0, len(x_test), args.batch_size):
            xb = torch.from_numpy(x_test[start : start + args.batch_size]).to(device)
            preds.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds), {
        "device": str(device),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
    }


def summarize(rows):
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
    return out


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

    index = pd.read_csv(args.index_csv)
    decomp = np.load(args.decomp_npz)
    if args.max_epochs_data is not None:
        index = index.iloc[: args.max_epochs_data].copy()
    n_epochs = len(index)
    arrays = {
        "full_log_psd": decomp["log_psd"][:n_epochs],
        "aperiodic_spectrum": decomp["aperiodic_log_psd"][:n_epochs],
        "flattened_log_psd": decomp["log_psd"][:n_epochs] - decomp["aperiodic_log_psd"][:n_epochs],
    }
    features = {
        name: array.reshape(array.shape[0], -1).astype("float32")
        for name, array in arrays.items()
    }

    task_labels = make_task_labels(index["stage"].to_numpy())
    groups_all = index["subject"].to_numpy()
    fold_rows = []
    subject_rows = []
    train_rows = []

    for task_name, labels in task_labels.items():
        mask, y, classes = encode_labels(labels)
        task_groups = groups_all[mask]
        if len(set(task_groups.tolist())) < 2:
            continue
        folds = make_subject_folds(task_groups, args.n_splits)
        n_classes = len(classes)
        for train_input in TRAIN_INPUTS:
            task_features = features[train_input][mask]
            for fold_idx, (train_idx, test_idx, test_subjects) in enumerate(folds):
                train_subjects = sorted(str(s) for s in np.unique(task_groups[train_idx]))
                pred, meta = fit_predict(
                    task_features[train_idx],
                    y[train_idx],
                    task_features[test_idx],
                    args,
                    n_classes=n_classes,
                    fold_seed=args.seed + fold_idx + 1000 * TRAIN_INPUTS.index(train_input),
                )
                y_test = y[test_idx]
                base_row = {
                    "task": task_name,
                    "seed": int(args.seed),
                    "train_input": train_input,
                    "test_input": train_input,
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
                append_subject_rows(
                    subject_rows,
                    base_row,
                    task_groups[test_idx],
                    y_test,
                    pred,
                    n_classes,
                )
                train_rows.append({**base_row, **meta})
                print(f"{task_name} {train_input} fold {fold_idx} done", flush=True)

    summary_rows = summarize(fold_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "psd_train_input_control_fold_metrics.csv", fold_rows)
    write_csv(args.output_dir / "psd_train_input_control_subject_metrics.csv", subject_rows)
    write_csv(args.output_dir / "psd_train_input_control_summary_metrics.csv", summary_rows)
    write_csv(args.output_dir / "psd_train_input_control_training_log.csv", train_rows)
    (args.output_dir / "psd_train_input_control_metadata.json").write_text(
        json.dumps(
            {
                "index_csv": str(args.index_csv),
                "decomp_npz": str(args.decomp_npz),
                "train_inputs": TRAIN_INPUTS,
                "n_epochs": int(n_epochs),
                "seed": int(args.seed),
                "n_splits": int(args.n_splits),
                "model": "matched MLP trained directly on each PSD representation",
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
            },
            indent=2,
        )
    )
    print(
        pd.DataFrame(summary_rows)[
            ["task", "train_input", "balanced_accuracy_mean", "macro_f1_mean"]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
