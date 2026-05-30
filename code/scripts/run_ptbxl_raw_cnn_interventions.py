#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


TEST_INPUTS = ["raw_eeg", "phase_sham", "phase_aperiodic", "phase_flattened"]
METRICS = ["balanced_accuracy", "macro_f1", "accuracy"]


def parse_args():
    parser = argparse.ArgumentParser(description="PTB-XL raw ECG architecture Fourier intervention audit.")
    parser.add_argument(
        "--raw-npz",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/results/ptbxl_1f_demo/ptbxl_records100_normal_abnormal_raw.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        default="small_cnn",
        choices=["small_cnn", "resnet1d_wang", "xresnet1d101", "inception1d"],
        help="Raw ECG architecture to train.",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--record-filter-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV with record_index and split_group columns. When provided, "
            "train/validation/test records are restricted to that matched subset."
        ),
    )
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def intervention_batch(x: np.ndarray, sfreq: float):
    freqs = np.fft.rfftfreq(x.shape[-1], d=1.0 / sfreq)
    fft = np.fft.rfft(x, axis=-1)
    amp = np.abs(fft)
    phase = np.angle(fft)
    mask = (freqs >= 1.0) & (freqs <= min(45.0, sfreq / 2.0 - 1e-6))
    logf = np.log(freqs[mask]).astype("float64")
    design = np.column_stack([np.ones_like(logf), -logf])
    pinv = np.linalg.pinv(design)

    out = {}
    for mode in TEST_INPUTS:
        if mode == "raw_eeg":
            out[mode] = x.astype("float32", copy=True)
            continue

        new_amp = amp.copy()
        if mode != "phase_sham":
            log_amp = np.log(np.maximum(amp[:, :, mask], 1e-12)).astype("float64")
            obs = log_amp.transpose(2, 0, 1).reshape(len(logf), -1)
            coef = pinv @ obs
            fitted = (design @ coef).reshape(len(logf), x.shape[0], x.shape[1]).transpose(1, 2, 0)
            if mode == "phase_aperiodic":
                new_amp[:, :, mask] = np.exp(fitted)
            elif mode == "phase_flattened":
                residual = log_amp - fitted
                anchor = fitted.mean(axis=-1, keepdims=True)
                new_amp[:, :, mask] = np.exp(residual + anchor)
            else:
                raise ValueError(mode)

        recon = np.fft.irfft(new_amp * np.exp(1j * phase), n=x.shape[-1], axis=-1).astype("float32")
        orig_rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True))
        recon_rms = np.sqrt(np.mean(recon**2, axis=-1, keepdims=True))
        recon = recon * (orig_rms / np.maximum(recon_rms, 1e-8))
        recon = recon - recon.mean(axis=-1, keepdims=True)
        out[mode] = recon.astype("float32")
    return out


class EcgDataset(Dataset):
    def __init__(self, x, y, mean=None, std=None):
        self.x = x.astype("float32")
        self.y = y.astype("int64")
        if mean is None:
            mean = self.x.mean(axis=(0, 2), keepdims=True)
        if std is None:
            std = self.x.std(axis=(0, 2), keepdims=True)
        self.mean = mean.astype("float32")
        self.std = np.maximum(std.astype("float32"), 1e-6)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = (self.x[idx] - self.mean[0]) / self.std[0]
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)


class SmallEcgCNN(nn.Module):
    def __init__(self, n_channels: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, 2)

    def forward(self, x):
        z = self.net(x).squeeze(-1)
        return self.head(z)


class SamePadConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, bias=False):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=bias)
        total_pad = kernel_size - 1
        self.left_pad = total_pad // 2
        self.right_pad = total_pad - self.left_pad

    def forward(self, x):
        if self.left_pad or self.right_pad:
            x = torch.nn.functional.pad(x, (self.left_pad, self.right_pad))
        return super().forward(x)


def align_time(a: torch.Tensor, b: torch.Tensor):
    if a.shape[-1] == b.shape[-1]:
        return a, b
    n = min(a.shape[-1], b.shape[-1])
    return a[..., :n], b[..., :n]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, act=True):
        super().__init__()
        layers = [
            SamePadConv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm1d(out_channels),
        ]
        if act:
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class WangResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size=8),
            ConvBlock(out_channels, out_channels, kernel_size=5),
            ConvBlock(out_channels, out_channels, kernel_size=3, act=False),
        )
        self.shortcut = (
            nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm1d(out_channels))
            if in_channels != out_channels
            else nn.BatchNorm1d(out_channels)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out, shortcut = align_time(self.conv(x), self.shortcut(x))
        return self.act(out + shortcut)


class ResNet1dWang(nn.Module):
    """Time-series ResNet architecture used in Wang et al.-style baselines."""

    def __init__(self, n_channels: int = 12, n_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            WangResBlock(n_channels, 64),
            WangResBlock(64, 128),
            WangResBlock(128, 128),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        return self.head(self.features(x).squeeze(-1))


class XResNetBottleneck1d(nn.Module):
    expansion = 4

    def __init__(self, in_channels, planes, stride=1):
        super().__init__()
        out_channels = planes * self.expansion
        self.conv = nn.Sequential(
            ConvBlock(in_channels, planes, kernel_size=1),
            ConvBlock(planes, planes, kernel_size=5, stride=stride),
            ConvBlock(planes, out_channels, kernel_size=1, act=False),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.AvgPool1d(kernel_size=stride, stride=stride, ceil_mode=True) if stride > 1 else nn.Identity(),
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out, shortcut = align_time(self.conv(x), self.shortcut(x))
        return self.act(out + shortcut)


class XResNet1d101(nn.Module):
    """1D XResNet-101 variant following the PTB-XL benchmark model family."""

    def __init__(self, n_channels: int = 12, n_classes: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(n_channels, 32, kernel_size=5, stride=2),
            ConvBlock(32, 32, kernel_size=5),
            ConvBlock(32, 64, kernel_size=5),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.in_channels = 64
        self.layer1 = self._make_layer(64, blocks=3, stride=1)
        self.layer2 = self._make_layer(128, blocks=4, stride=2)
        self.layer3 = self._make_layer(256, blocks=23, stride=2)
        self.layer4 = self._make_layer(512, blocks=3, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(512 * XResNetBottleneck1d.expansion, n_classes))

    def _make_layer(self, planes, blocks, stride):
        layers = [XResNetBottleneck1d(self.in_channels, planes, stride=stride)]
        self.in_channels = planes * XResNetBottleneck1d.expansion
        for _ in range(1, blocks):
            layers.append(XResNetBottleneck1d(self.in_channels, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(self.pool(x).squeeze(-1))


class InceptionBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels=32, bottleneck_channels=32, kernel_sizes=(39, 19, 9)):
        super().__init__()
        bottleneck = bottleneck_channels if in_channels > 1 else in_channels
        self.bottleneck = nn.Conv1d(in_channels, bottleneck, kernel_size=1, bias=False)
        self.branches = nn.ModuleList(
            [SamePadConv1d(bottleneck, out_channels, kernel_size=k) for k in kernel_sizes]
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
        )
        self.bn = nn.BatchNorm1d(out_channels * (len(kernel_sizes) + 1))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        z = self.bottleneck(x)
        branches = [branch(z) for branch in self.branches]
        branches.append(self.pool_branch(x))
        return self.act(self.bn(torch.cat(branches, dim=1)))


class InceptionResidual1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, residual):
        out, shortcut = align_time(x, self.shortcut(residual))
        return self.act(out + shortcut)


class Inception1d(nn.Module):
    """InceptionTime-style multiscale 1D convolutional classifier."""

    def __init__(self, n_channels: int = 12, n_classes: int = 2, depth: int = 6, branch_channels: int = 32):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.residuals = nn.ModuleDict()
        in_channels = n_channels
        block_out = branch_channels * 4
        for idx in range(depth):
            self.blocks.append(InceptionBlock1d(in_channels, out_channels=branch_channels))
            if idx in {2, 5}:
                res_in = n_channels if idx == 2 else block_out
                self.residuals[str(idx)] = InceptionResidual1d(res_in, block_out)
            in_channels = block_out
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(block_out, n_classes)

    def forward(self, x):
        residual = x
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if str(idx) in self.residuals:
                x = self.residuals[str(idx)](x, residual)
                residual = x
        return self.head(self.pool(x).squeeze(-1))


def build_model(name: str, n_channels: int):
    if name == "small_cnn":
        return SmallEcgCNN(n_channels)
    if name == "resnet1d_wang":
        return ResNet1dWang(n_channels)
    if name == "xresnet1d101":
        return XResNet1d101(n_channels)
    if name == "inception1d":
        return Inception1d(n_channels)
    raise ValueError(f"Unknown PTB-XL model: {name}")


def confusion(y_true, y_pred):
    cm = np.zeros((2, 2), dtype=np.int64)
    for yt, yp in zip(y_true, y_pred):
        cm[int(yt), int(yp)] += 1
    return cm


def metrics_from_cm(cm):
    total = cm.sum()
    acc = float(np.trace(cm) / max(total, 1))
    recalls = []
    f1s = []
    for cls in range(2):
        tp = float(cm[cls, cls])
        fn = float(cm[cls].sum() - tp)
        fp = float(cm[:, cls].sum() - tp)
        recalls.append(tp / max(tp + fn, 1e-12))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return {
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "accuracy": acc,
    }


def predictions_from_scores(scores: np.ndarray, threshold: float):
    return (scores >= threshold).astype("int64")


def best_threshold_for_balanced_accuracy(y_true: np.ndarray, scores: np.ndarray):
    thresholds = np.unique(np.concatenate([[0.0, 1.0], np.linspace(0.05, 0.95, 181), scores]))
    best_threshold = 0.5
    best_score = -math.inf
    for threshold in thresholds:
        score = metrics_from_cm(confusion(y_true, predictions_from_scores(scores, threshold)))["balanced_accuracy"]
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, float(best_score)


def predict_scores(model, x, y, mean, std, device, batch_size):
    ds = EcgDataset(x, y, mean=mean, std=std)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    scores = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device))
            scores.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(scores)


def evaluate(model, x, y, patient, mean, std, device, batch_size, threshold):
    scores = predict_scores(model, x, y, mean, std, device, batch_size)
    y_pred = predictions_from_scores(scores, threshold)
    rows = []
    for subject in sorted(np.unique(patient)):
        mask = patient == subject
        cm = confusion(y[mask], y_pred[mask])
        rows.append((subject, int(mask.sum()), cm, metrics_from_cm(cm)))
    return y_pred, scores, rows


def train_model(x_train, y_train, x_val, y_val, seed, args, device):
    train_ds = EcgDataset(x_train, y_train)
    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=gen,
    )
    model = build_model(args.model, x_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    class_counts = np.bincount(y_train, minlength=2).astype("float32")
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    best_state = None
    best_val = -math.inf
    best_threshold = 0.5
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            losses.append(float(loss.item()))
        scheduler.step()

        val_scores = predict_scores(model, x_val, y_val, train_ds.mean, train_ds.std, device, args.batch_size)
        threshold, val_ba = best_threshold_for_balanced_accuracy(y_val, val_scores)
        y_pred = predictions_from_scores(val_scores, threshold)
        y_true = y_val
        val_ba = metrics_from_cm(confusion(y_true, y_pred))["balanced_accuracy"]
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_balanced_accuracy": val_ba,
                "val_threshold": threshold,
                "lr": float(scheduler.get_last_lr()[0]),
            }
        )
        if val_ba > best_val:
            best_val = val_ba
            best_threshold = threshold
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, train_ds.mean, train_ds.std, log_rows, best_threshold, best_val


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    data = np.load(args.raw_npz, allow_pickle=True)
    x = data["x"].astype("float32")
    y = data["y"].astype("int64")
    patient = data["patient_id"].astype(str)
    folds = data["strat_fold"].astype(int)
    sfreq = float(data["sfreq"])

    pair_by_record_index: dict[int, str] = {}
    if args.record_filter_csv is not None:
        import pandas as pd

        record_filter = pd.read_csv(args.record_filter_csv)
        if "record_index" not in record_filter.columns or "split_group" not in record_filter.columns:
            raise ValueError("--record-filter-csv must include record_index and split_group columns")
        train_indices = record_filter.loc[record_filter["split_group"] == "train", "record_index"].to_numpy(dtype=int)
        val_indices = record_filter.loc[record_filter["split_group"] == "validation", "record_index"].to_numpy(dtype=int)
        test_indices = record_filter.loc[record_filter["split_group"] == "test", "record_index"].to_numpy(dtype=int)
        if "pair_id" in record_filter.columns:
            pair_by_record_index = {
                int(row.record_index): str(row.pair_id) for row in record_filter.itertuples(index=False)
            }
    else:
        train_indices = np.flatnonzero(folds <= 8)
        val_indices = np.flatnonzero(folds == 9)
        test_indices = np.flatnonzero(folds == 10)

    if len(train_indices) == 0 or len(val_indices) == 0 or len(test_indices) == 0:
        raise ValueError(
            f"Empty PTB-XL split after filtering: train={len(train_indices)}, "
            f"validation={len(val_indices)}, test={len(test_indices)}"
        )

    model, mean, std, training_log, threshold, best_val = train_model(
        x[train_indices], y[train_indices], x[val_indices], y[val_indices], args.seed, args, device
    )

    interventions = intervention_batch(x[test_indices], sfreq)
    subject_rows = []
    prediction_rows = []
    for test_input, x_test in interventions.items():
        y_pred, scores, subject_metrics = evaluate(
            model, x_test, y[test_indices], patient[test_indices], mean, std, device, args.batch_size, threshold
        )
        for subject, n_subject, cm, metrics in subject_metrics:
            subject_rows.append(
                {
                    "task": "ptbxl_normal_vs_abnormal",
                    "seed": int(args.seed),
                    "test_input": test_input,
                    "subject": subject,
                    "n_subject": n_subject,
                    "class_support_json": json.dumps(
                        {str(cls): int(cm[cls].sum()) for cls in range(2)}, sort_keys=True
                    ),
                    **metrics,
                }
            )
        for rec_idx, (subj, yt, yp, score) in enumerate(zip(patient[test_indices], y[test_indices], y_pred, scores)):
            source_record_index = int(test_indices[rec_idx])
            prediction_rows.append(
                {
                    "task": "ptbxl_normal_vs_abnormal",
                    "seed": int(args.seed),
                    "test_input": test_input,
                    "subject": subj,
                    "pair_id": pair_by_record_index.get(source_record_index, ""),
                    "record_index": source_record_index,
                    "y_true": int(yt),
                    "y_pred": int(yp),
                    "score_abnormal": float(score),
                }
            )

    write_csv(args.output_dir / "ptbxl_raw_cnn_intervention_subject_metrics.csv", subject_rows)
    write_csv(args.output_dir / "ptbxl_raw_cnn_intervention_predictions.csv", prediction_rows)
    write_csv(args.output_dir / "ptbxl_raw_cnn_training_log.csv", training_log)
    (args.output_dir / "ptbxl_raw_cnn_metadata.json").write_text(
        json.dumps(
            {
                "raw_npz": str(args.raw_npz),
                "model": args.model,
                "seed": int(args.seed),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "record_filter_csv": None if args.record_filter_csv is None else str(args.record_filter_csv),
                "val_threshold": float(threshold),
                "best_val_balanced_accuracy": float(best_val),
                "device": str(device),
                "train_folds": "1-8",
                "val_fold": 9,
                "test_fold": 10,
                "n_train_records": int(len(train_indices)),
                "n_val_records": int(len(val_indices)),
                "n_test_records": int(len(test_indices)),
            },
            indent=2,
        )
    )
    print(f"Wrote PTB-XL {args.model} seed {args.seed}: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
