#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path


TEST_INPUTS = ["raw_eeg", "phase_sham", "phase_aperiodic", "phase_flattened"]
METRICS = ["balanced_accuracy", "macro_f1", "accuracy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train TUAB raw EEG neural models with the official train/eval "
            "split, then evaluate phase-preserving aperiodic interventions."
        )
    )
    parser.add_argument(
        "--raw-npz",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/results/tuab_subset_200/raw_epochs_20s_100hz.npz"),
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/results/tuab_subset_200/raw_epochs_20s_100hz_index.csv"),
    )
    parser.add_argument(
        "--decomp-npz",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/results/tuab_subset_200/specparam/specparam_fixed_20s.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/results/tuab_subset_200/braindecode_eegnet_interventions_specparam"),
    )
    parser.add_argument(
        "--subject-filter-csv",
        type=Path,
        default=None,
        help=(
            "Optional subject manifest with official_split, label, and subject_id "
            "columns. Rows outside this subject set are excluded before training/eval."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--band-min", type=float, default=1.0)
    parser.add_argument("--band-max", type=float, default=45.0)
    parser.add_argument("--f1", type=int, default=8)
    parser.add_argument("--depth-multiplier", type=int, default=2)
    parser.add_argument("--kernel-length", type=int, default=64)
    parser.add_argument("--depthwise-kernel-length", type=int, default=16)
    parser.add_argument(
        "--model",
        choices=["raw_cnn", "eegnet", "shallow_fbcsp", "deep4"],
        default="eegnet",
        help="Raw neural classifier. Braindecode models use the installed Braindecode package.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Filename prefix for outputs. Defaults to tuab_<model>.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument(
        "--no-match-rms",
        action="store_true",
        help="Do not rescale edited epochs back to the original per-channel RMS.",
    )
    parser.add_argument(
        "--max-train-epochs-data",
        type=int,
        default=None,
        help="Optional smoke-test cap within official train rows.",
    )
    parser.add_argument(
        "--max-eval-epochs-data",
        type=int,
        default=None,
        help="Optional smoke-test cap within official eval rows.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def apply_subject_filter(index, subject_filter_csv: Path | None):
    if subject_filter_csv is None:
        return index, {}

    import pandas as pd

    filt = pd.read_csv(subject_filter_csv)
    required = {"official_split", "label"}
    if "subject_id" not in filt.columns and "subject" not in filt.columns:
        raise ValueError("Subject filter must contain subject_id or subject column.")
    missing = required.difference(filt.columns)
    if missing:
        raise ValueError(f"Subject filter is missing columns: {sorted(missing)}")
    subject_col = "subject_id" if "subject_id" in filt.columns else "subject"
    allowed = {
        (str(row.official_split), str(row.label), str(getattr(row, subject_col)))
        for row in filt.itertuples(index=False)
    }
    keep = [
        (str(row.official_split), str(row.label), str(row.subject)) in allowed
        for row in index.itertuples(index=False)
    ]
    filtered = index.loc[keep].copy()
    if filtered.empty:
        raise ValueError(f"Subject filter selected zero rows: {subject_filter_csv}")
    summary = {
        "subject_filter_csv": str(subject_filter_csv),
        "subject_filter_subject_rows": int(len(filt)),
        "subject_filter_epoch_rows": int(len(filtered)),
        "subject_filter_train_subjects": int(
            filtered.loc[filtered["official_split"].astype(str) == "train", "subject"].nunique()
        ),
        "subject_filter_eval_subjects": int(
            filtered.loc[filtered["official_split"].astype(str) == "eval", "subject"].nunique()
        ),
    }
    return filtered, summary


def encode_labels(labels):
    import numpy as np

    labels = np.asarray(labels).astype(str)
    classes = sorted(np.unique(labels).tolist())
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    y = np.array([class_to_idx[label] for label in labels], dtype=np.int64)
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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
    x_eval,
    ap_log_psd,
    decomp_freqs,
    sfreq: float,
    band_min: float,
    band_max: float,
    match_rms: bool,
):
    import numpy as np

    n_times = x_eval.shape[-1]
    fft_freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    band_mask = (fft_freqs >= band_min) & (fft_freqs <= band_max)
    band_freqs = fft_freqs[band_mask]

    fft = np.fft.rfft(x_eval, axis=-1)
    band_fft = fft[:, :, band_mask]
    band_amp = np.maximum(np.abs(band_fft), 1e-12)
    band_phase = band_fft / band_amp

    ap_interp = interpolate_aperiodic(ap_log_psd, decomp_freqs, band_freqs)
    ap_centered = ap_interp - ap_interp.mean(axis=-1, keepdims=True)
    ap_amp_shape = np.power(10.0, 0.5 * ap_centered).astype("float32")

    sham_fft = fft.copy()
    sham_fft[:, :, band_mask] = band_phase * band_amp

    aperiodic_fft = fft.copy()
    geom_amp = np.exp(np.mean(np.log(band_amp), axis=-1, keepdims=True))
    aperiodic_fft[:, :, band_mask] = band_phase * geom_amp * ap_amp_shape

    flattened_fft = fft.copy()
    flattened_fft[:, :, band_mask] = band_fft / np.maximum(ap_amp_shape, 1e-6)

    edited = {
        "phase_sham": np.fft.irfft(sham_fft, n=n_times, axis=-1),
        "phase_aperiodic": np.fft.irfft(aperiodic_fft, n=n_times, axis=-1),
        "phase_flattened": np.fft.irfft(flattened_fft, n=n_times, axis=-1),
    }

    original_std = x_eval.std(axis=-1, keepdims=True)
    for name, array in edited.items():
        array = array.real.astype("float32", copy=False)
        array -= array.mean(axis=-1, keepdims=True)
        if match_rms:
            edited_std = array.std(axis=-1, keepdims=True)
            array *= original_std / np.maximum(edited_std, 1e-6)
        edited[name] = array.astype("float32", copy=False)
    return edited


class RawCNN:
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


def build_raw_model(n_channels: int, n_times: int, n_classes: int, sfreq: float, args):
    import torch

    if args.model == "raw_cnn":
        model = RawCNN(n_channels=n_channels, n_classes=n_classes, dropout=args.dropout)
    elif args.model == "eegnet":
        from braindecode.models import EEGNet

        model = EEGNet(
            n_chans=n_channels,
            n_outputs=n_classes,
            n_times=n_times,
            final_conv_length="auto",
            F1=args.f1,
            D=args.depth_multiplier,
            F2=args.f1 * args.depth_multiplier,
            kernel_length=args.kernel_length,
            depthwise_kernel_length=args.depthwise_kernel_length,
            drop_prob=args.dropout,
            sfreq=sfreq,
        )
    elif args.model == "shallow_fbcsp":
        from braindecode.models import ShallowFBCSPNet

        model = ShallowFBCSPNet(
            n_chans=n_channels,
            n_outputs=n_classes,
            n_times=n_times,
            final_conv_length="auto",
            drop_prob=args.dropout,
            sfreq=sfreq,
        )
    elif args.model == "deep4":
        from braindecode.models import Deep4Net

        model = Deep4Net(
            n_chans=n_channels,
            n_outputs=n_classes,
            n_times=n_times,
            final_conv_length="auto",
            drop_prob=args.dropout,
            sfreq=sfreq,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    with torch.no_grad():
        out = model(torch.zeros(2, n_channels, n_times, dtype=torch.float32))
    if out.ndim != 2 or out.shape[1] != n_classes:
        raise RuntimeError(f"Unexpected {args.model} output shape: {tuple(out.shape)}")
    return model


def fit_predict(
    x_train,
    y_train,
    test_inputs,
    sfreq: float,
    args,
    n_classes: int,
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

    sub_train_idx, val_idx = make_train_val_split(y_train, seed=args.seed)
    class_counts = np.bincount(y_train[sub_train_idx], minlength=n_classes).astype("float32")
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.mean()

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train_scaled[sub_train_idx]),
            torch.from_numpy(y_train[sub_train_idx]),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    val_x_cpu = torch.from_numpy(x_train_scaled[val_idx])
    val_y_cpu = torch.from_numpy(y_train[val_idx])

    model = build_raw_model(
        n_channels=x_train.shape[1],
        n_times=x_train.shape[2],
        n_classes=n_classes,
        sfreq=sfreq,
        args=args,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_log = []
    best_state = None
    best_val = math.inf
    best_epoch = 0
    patience_left = args.patience
    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        train_seen = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * int(len(yb))
            train_seen += int(len(yb))

        model.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            val_seen = 0
            for start in range(0, len(val_idx), args.batch_size):
                xb = val_x_cpu[start : start + args.batch_size].to(device, non_blocking=True)
                yb = val_y_cpu[start : start + args.batch_size].to(device, non_blocking=True)
                loss = criterion(model(xb), yb)
                val_loss_sum += float(loss.item()) * int(len(yb))
                val_seen += int(len(yb))
            val_loss = val_loss_sum / max(val_seen, 1)
        train_loss = train_loss_sum / max(train_seen, 1)
        train_log.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
        )
        print(f"epoch={epoch} train_loss={train_loss:.5f} val_loss={val_loss:.5f}", flush=True)
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch={epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    predictions = {}
    with torch.no_grad():
        for name, x_eval in test_inputs.items():
            x_scaled = ((x_eval - mean) / std).astype("float32")
            preds = []
            for start in range(0, len(x_scaled), args.batch_size):
                xb = torch.from_numpy(x_scaled[start : start + args.batch_size]).to(device)
                preds.append(model(xb).argmax(dim=1).cpu().numpy())
            predictions[name] = np.concatenate(preds)
            print(f"Predicted {name}: n={len(predictions[name])}", flush=True)

    meta = {
        "device": str(device),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "n_subtrain": int(len(sub_train_idx)),
        "n_val": int(len(val_idx)),
    }
    return predictions, train_log, meta


def compute_channel_standardization(x, indices, batch_size: int):
    import numpy as np

    channel_sum = None
    channel_sumsq = None
    n_values = 0
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        batch = x[batch_idx].astype("float64", copy=False)
        batch_sum = batch.sum(axis=(0, 2), keepdims=True)
        batch_sumsq = np.square(batch).sum(axis=(0, 2), keepdims=True)
        if channel_sum is None:
            channel_sum = batch_sum
            channel_sumsq = batch_sumsq
        else:
            channel_sum += batch_sum
            channel_sumsq += batch_sumsq
        n_values += int(batch.shape[0] * batch.shape[2])

    if channel_sum is None or channel_sumsq is None or n_values == 0:
        raise ValueError("Cannot compute normalization from an empty training set.")
    mean = channel_sum / n_values
    variance = np.maximum(channel_sumsq / n_values - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype("float32"), std.astype("float32")


def scale_batch(batch, mean, std):
    return ((batch - mean) / std).astype("float32", copy=False)


def predict_array_batches(model, array, mean, std, batch_size: int, device):
    import numpy as np
    import torch

    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(array), batch_size):
            batch = scale_batch(array[start : start + batch_size], mean, std)
            xb = torch.from_numpy(batch).to(device, non_blocking=True)
            preds.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=int)


def predict_raw_indexed(model, x, indices, mean, std, batch_size: int, device):
    import numpy as np
    import torch

    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            batch = scale_batch(x[batch_idx], mean, std)
            xb = torch.from_numpy(batch).to(device, non_blocking=True)
            preds.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=int)


def fit_model_indexed(
    x,
    y,
    train_idx,
    sfreq: float,
    args,
    n_classes: int,
):
    import numpy as np
    import torch
    from torch import nn

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    mean, std = compute_channel_standardization(x, train_idx, batch_size=args.batch_size)

    split_train_idx, split_val_idx = make_train_val_split(y[train_idx], seed=args.seed)
    sub_train_idx = train_idx[split_train_idx]
    val_idx = train_idx[split_val_idx]

    class_counts = np.bincount(y[sub_train_idx], minlength=n_classes).astype("float32")
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.mean()

    model = build_raw_model(
        n_channels=x.shape[1],
        n_times=x.shape[2],
        n_classes=n_classes,
        sfreq=sfreq,
        args=args,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    rng = np.random.default_rng(args.seed)
    train_log = []
    best_state = None
    best_val = math.inf
    best_epoch = 0
    patience_left = args.patience

    for epoch in range(args.epochs):
        model.train()
        shuffled = sub_train_idx.copy()
        rng.shuffle(shuffled)
        train_loss_sum = 0.0
        train_seen = 0
        for start in range(0, len(shuffled), args.batch_size):
            batch_idx = shuffled[start : start + args.batch_size]
            xb_np = scale_batch(x[batch_idx], mean, std)
            yb_np = y[batch_idx]
            xb = torch.from_numpy(xb_np).to(device, non_blocking=True)
            yb = torch.from_numpy(yb_np).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * int(len(yb_np))
            train_seen += int(len(yb_np))

        model.eval()
        val_loss_sum = 0.0
        val_seen = 0
        with torch.no_grad():
            for start in range(0, len(val_idx), args.batch_size):
                batch_idx = val_idx[start : start + args.batch_size]
                xb_np = scale_batch(x[batch_idx], mean, std)
                yb_np = y[batch_idx]
                xb = torch.from_numpy(xb_np).to(device, non_blocking=True)
                yb = torch.from_numpy(yb_np).to(device, non_blocking=True)
                loss = criterion(model(xb), yb)
                val_loss_sum += float(loss.item()) * int(len(yb_np))
                val_seen += int(len(yb_np))

        train_loss = train_loss_sum / max(train_seen, 1)
        val_loss = val_loss_sum / max(val_seen, 1)
        train_log.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
        )
        print(f"epoch={epoch} train_loss={train_loss:.5f} val_loss={val_loss:.5f}", flush=True)
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch={epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    meta = {
        "device": str(device),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "n_subtrain": int(len(sub_train_idx)),
        "n_val": int(len(val_idx)),
        "normalization": "chunked_train_channel_time_standardization",
    }
    return model, train_log, meta, mean, std, device


def predict_interventions_indexed(
    model,
    x,
    ap_log_psd,
    indices,
    decomp_freqs,
    sfreq: float,
    mean,
    std,
    args,
    device,
):
    import numpy as np

    predictions = {
        "raw_eeg": predict_raw_indexed(
            model,
            x,
            indices,
            mean,
            std,
            batch_size=args.batch_size,
            device=device,
        )
    }
    edited_preds = {name: [] for name in ["phase_sham", "phase_aperiodic", "phase_flattened"]}
    match_rms = not args.no_match_rms
    for start in range(0, len(indices), args.batch_size):
        batch_idx = indices[start : start + args.batch_size]
        edited = make_phase_preserving_inputs(
            x[batch_idx],
            ap_log_psd[batch_idx],
            decomp_freqs,
            sfreq=sfreq,
            band_min=args.band_min,
            band_max=args.band_max,
            match_rms=match_rms,
        )
        for name, array in edited.items():
            edited_preds[name].append(
                predict_array_batches(
                    model,
                    array,
                    mean,
                    std,
                    batch_size=args.batch_size,
                    device=device,
                )
            )
        print(
            f"Predicted intervention batch {start}:{min(start + args.batch_size, len(indices))}",
            flush=True,
        )

    for name, chunks in edited_preds.items():
        predictions[name] = np.concatenate(chunks) if chunks else np.array([], dtype=int)
        print(f"Predicted {name}: n={len(predictions[name])}", flush=True)
    print(f"Predicted raw_eeg: n={len(predictions['raw_eeg'])}", flush=True)
    return predictions


def append_subject_rows(rows, base_row, subjects, y_true, y_pred, labels, n_classes):
    import numpy as np

    subjects = np.asarray(subjects).astype(str)
    labels = np.asarray(labels).astype(str)
    for subject in sorted(np.unique(subjects)):
        mask = subjects == subject
        support = {str(cls): int((y_true[mask] == cls).sum()) for cls in range(n_classes)}
        rows.append(
            {
                **base_row,
                "subject": str(subject),
                "label": str(labels[mask][0]),
                "n_subject": int(mask.sum()),
                "class_support_json": json.dumps(support, sort_keys=True),
                "balanced_accuracy": balanced_accuracy(y_true[mask], y_pred[mask], n_classes),
                "macro_f1": macro_f1(y_true[mask], y_pred[mask], n_classes),
                "accuracy": accuracy(y_true[mask], y_pred[mask]),
            }
        )


def make_prediction_rows(base_row, row_indices, subjects, labels, y_true, y_pred, n_classes):
    rows = []
    for idx, subject, label, true, pred in zip(row_indices, subjects, labels, y_true, y_pred):
        rows.append(
            {
                **base_row,
                "n_classes": int(n_classes),
                "row_index": int(idx),
                "subject": str(subject),
                "label": str(label),
                "y_true": int(true),
                "y_pred": int(pred),
            }
        )
    return rows


def subject_confusions(group, n_classes: int):
    import numpy as np

    subjects = sorted(group["subject"].unique())
    matrices = np.zeros((len(subjects), n_classes, n_classes), dtype=np.int64)
    labels = []
    for subject_idx, subject in enumerate(subjects):
        sub = group[group["subject"] == subject]
        labels.append(str(sub["label"].iloc[0]))
        for true, pred in zip(sub["y_true"].to_numpy(dtype=int), sub["y_pred"].to_numpy(dtype=int)):
            matrices[subject_idx, true, pred] += 1
    return np.array(subjects, dtype=object), np.array(labels, dtype=object), matrices


def subject_stratified_bootstrap(
    prediction_rows,
    n_classes: int,
    n_bootstrap: int,
    ci: float,
    seed: int,
    train_input_name: str,
):
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(prediction_rows)
    rng = np.random.default_rng(seed)
    out = []

    def sample_indices(labels):
        sampled = []
        for label in sorted(np.unique(labels)):
            label_idx = np.flatnonzero(labels == label)
            sampled.extend(rng.choice(label_idx, size=len(label_idx), replace=True).tolist())
        return np.array(sampled, dtype=int)

    def ci_bounds(values):
        alpha = 1.0 - ci
        return (
            float(np.quantile(values, alpha / 2.0)),
            float(np.quantile(values, 1.0 - alpha / 2.0)),
        )

    for test_input, group in df.groupby("test_input", sort=False):
        subjects, subject_labels, matrices = subject_confusions(group, n_classes)
        for metric in METRICS:
            boot = np.empty(n_bootstrap, dtype=float)
            for boot_idx in range(n_bootstrap):
                sampled = sample_indices(subject_labels)
                boot[boot_idx] = metric_from_confusion(metric, matrices[sampled].sum(axis=0))
            lower, upper = ci_bounds(boot)
            out.append(
                {
                    "task": "tuab_normal_vs_abnormal",
                    "train_input": train_input_name,
                    "test_input": test_input,
                    "classes": str(group["classes"].iloc[0]),
                    "metric": metric,
                    "estimate": "performance",
                    "point": metric_from_confusion(metric, matrices.sum(axis=0)),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "ci": float(ci),
                    "n_eval_subjects": int(subjects.shape[0]),
                    "n_bootstrap": int(n_bootstrap),
                }
            )

    baseline_group = df[df["test_input"] == "raw_eeg"]
    base_subjects, base_labels, base_matrices = subject_confusions(baseline_group, n_classes)
    for test_input in ["phase_sham", "phase_aperiodic", "phase_flattened"]:
        edited_group = df[df["test_input"] == test_input]
        edit_subjects, edit_labels, edit_matrices = subject_confusions(edited_group, n_classes)
        if base_subjects.tolist() != edit_subjects.tolist() or base_labels.tolist() != edit_labels.tolist():
            raise ValueError(f"Subject ordering mismatch for {test_input}")
        for metric in METRICS:
            boot_drop = np.empty(n_bootstrap, dtype=float)
            boot_retention = np.empty(n_bootstrap, dtype=float)
            for boot_idx in range(n_bootstrap):
                sampled = sample_indices(base_labels)
                base_metric = metric_from_confusion(metric, base_matrices[sampled].sum(axis=0))
                edit_metric = metric_from_confusion(metric, edit_matrices[sampled].sum(axis=0))
                boot_drop[boot_idx] = base_metric - edit_metric
                boot_retention[boot_idx] = edit_metric / max(base_metric, 1e-12)
            base_point = metric_from_confusion(metric, base_matrices.sum(axis=0))
            edit_point = metric_from_confusion(metric, edit_matrices.sum(axis=0))
            for estimate, values, point in [
                (f"drop::{test_input}", boot_drop, base_point - edit_point),
                (f"retention::{test_input}", boot_retention, edit_point / max(base_point, 1e-12)),
            ]:
                lower, upper = ci_bounds(values)
                out.append(
                    {
                        "task": "tuab_normal_vs_abnormal",
                        "train_input": train_input_name,
                        "test_input": test_input,
                        "classes": str(baseline_group["classes"].iloc[0]),
                        "metric": metric,
                        "estimate": estimate,
                        "point": point,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "ci": float(ci),
                        "n_eval_subjects": int(base_subjects.shape[0]),
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
                ["drop::phase_sham", "drop::phase_aperiodic", "drop::phase_flattened"]
            )
        )
    ].copy()
    cols = ["test_input", "estimate", "point", "ci_lower", "ci_upper", "n_eval_subjects"]
    focus = focus[cols]
    for col in ["point", "ci_lower", "ci_upper"]:
        focus[col] = focus[col].map(lambda value: f"{value:.3f}")
    lines = [
        "# TUAB Raw Neural Intervention Report",
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
    import torch

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    raw = np.load(args.raw_npz)
    decomp = np.load(args.decomp_npz)
    index = pd.read_csv(args.index_csv)
    x = raw["x"].astype("float32", copy=False)
    ap_log_psd = decomp["aperiodic_log_psd"].astype("float32", copy=False)
    decomp_freqs = decomp["freqs"].astype("float32", copy=False)

    if len(index) != len(x):
        raise ValueError(f"Index rows ({len(index)}) do not match raw epochs ({len(x)})")
    if len(ap_log_psd) < len(x):
        raise ValueError(f"Specparam rows ({len(ap_log_psd)}) do not cover raw epochs ({len(x)})")
    index, filter_summary = apply_subject_filter(index, args.subject_filter_csv)
    if args.subject_filter_csv is not None:
        selected_rows = index.index.to_numpy(dtype=int)
        x = x[selected_rows]
        ap_log_psd = ap_log_psd[selected_rows]
    index = index.reset_index(drop=True)

    splits = index["official_split"].astype(str).to_numpy()
    labels = index["label"].astype(str).to_numpy()
    subjects = index["subject"].astype(str).to_numpy()
    y, classes = encode_labels(labels)
    n_classes = len(classes)

    train_idx = np.flatnonzero(splits == "train")
    eval_idx = np.flatnonzero(splits == "eval")
    if args.max_train_epochs_data is not None:
        train_idx = train_idx[: args.max_train_epochs_data]
    if args.max_eval_epochs_data is not None:
        eval_idx = eval_idx[: args.max_eval_epochs_data]
    if len(train_idx) == 0 or len(eval_idx) == 0:
        raise ValueError("Expected non-empty official train and eval rows.")

    sfreq = float(raw["sfreq"])
    print(
        f"TUAB raw neural run: train_epochs={len(train_idx)} eval_epochs={len(eval_idx)} "
        f"train_subjects={len(np.unique(subjects[train_idx]))} "
        f"eval_subjects={len(np.unique(subjects[eval_idx]))} model={args.model} classes={classes}",
        flush=True,
    )

    model, train_log, meta, mean, std, device = fit_model_indexed(
        x,
        y,
        train_idx,
        sfreq=sfreq,
        args=args,
        n_classes=n_classes,
    )
    predictions = predict_interventions_indexed(
        model,
        x,
        ap_log_psd,
        eval_idx,
        decomp_freqs,
        sfreq=sfreq,
        mean=mean,
        std=std,
        args=args,
        device=device,
    )

    eval_rows = []
    subject_rows = []
    prediction_rows = []
    train_input_name = f"{args.model}_raw_eeg"
    for test_input in TEST_INPUTS:
        pred = predictions[test_input]
        base_row = {
            "task": "tuab_normal_vs_abnormal",
            "seed": int(args.seed),
            "train_input": train_input_name,
            "test_input": test_input,
            "split": "official_eval",
            "classes": "|".join(classes),
            "n_train": int(len(train_idx)),
            "n_eval": int(len(eval_idx)),
            "n_train_subjects": int(len(np.unique(subjects[train_idx]))),
            "n_eval_subjects": int(len(np.unique(subjects[eval_idx]))),
        }
        eval_rows.append(
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
            labels[eval_idx],
            n_classes,
        )
        prediction_rows.extend(
            make_prediction_rows(
                base_row,
                eval_idx,
                subjects[eval_idx],
                labels[eval_idx],
                y[eval_idx],
                pred,
                n_classes,
            )
        )

    bootstrap_rows = subject_stratified_bootstrap(
        prediction_rows,
        n_classes=n_classes,
        n_bootstrap=args.n_bootstrap,
        ci=args.ci,
        seed=args.seed,
        train_input_name=train_input_name,
    )

    output_prefix = args.output_prefix or f"tuab_{args.model}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / f"{output_prefix}_intervention_eval_metrics.csv", eval_rows)
    write_csv(args.output_dir / f"{output_prefix}_intervention_subject_metrics.csv", subject_rows)
    write_csv(args.output_dir / f"{output_prefix}_intervention_predictions.csv", prediction_rows)
    write_csv(args.output_dir / f"{output_prefix}_intervention_subject_bootstrap.csv", bootstrap_rows)
    write_csv(args.output_dir / f"{output_prefix}_intervention_training_log.csv", train_log)
    write_markdown(args.output_dir / f"{output_prefix}_intervention_subject_bootstrap.md", bootstrap_rows)
    braindecode_version = None
    if args.model != "raw_cnn":
        import braindecode

        braindecode_version = braindecode.__version__
    (args.output_dir / f"{output_prefix}_intervention_metadata.json").write_text(
        json.dumps(
            {
                "raw_npz": str(args.raw_npz),
                "index_csv": str(args.index_csv),
                "decomp_npz": str(args.decomp_npz),
                "raw_shape": [int(v) for v in x.shape],
                "sfreq": sfreq,
                "channels": raw["channels"].tolist(),
                "model": args.model if args.model == "raw_cnn" else f"braindecode:{args.model}",
                "braindecode_version": braindecode_version,
                "train_input": "raw_eeg",
                "test_inputs": TEST_INPUTS,
                "intervention": "phase-preserving FFT amplitude edits",
                "band_min": float(args.band_min),
                "band_max": float(args.band_max),
                "match_rms": bool(not args.no_match_rms),
                "seed": int(args.seed),
                "epochs_requested": int(args.epochs),
                "batch_size": int(args.batch_size),
                "dropout": float(args.dropout),
                "learning_rate": float(args.learning_rate),
                "weight_decay": float(args.weight_decay),
                "F1": int(args.f1),
                "D": int(args.depth_multiplier),
                "F2": int(args.f1 * args.depth_multiplier),
                "kernel_length": int(args.kernel_length),
                "depthwise_kernel_length": int(args.depthwise_kernel_length),
                "n_train_epochs": int(len(train_idx)),
                "n_eval_epochs": int(len(eval_idx)),
                "n_train_subjects": int(len(np.unique(subjects[train_idx]))),
                "n_eval_subjects": int(len(np.unique(subjects[eval_idx]))),
                "classes": classes,
                "device_requested": args.device,
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                **filter_summary,
                **meta,
            },
            indent=2,
        )
    )

    print(pd.DataFrame(eval_rows)[["test_input", "balanced_accuracy", "macro_f1", "accuracy"]].to_string(index=False))
    print(f"Wrote outputs to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
