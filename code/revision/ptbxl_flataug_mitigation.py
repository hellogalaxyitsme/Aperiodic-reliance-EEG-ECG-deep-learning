#!/usr/bin/env python
"""Training-time mitigation on PTB-XL.

Identical to ptbxl_revision_interventions.py (original recipe: AdamW 1e-3,
wd 1e-3, cosine schedule, 35 epochs, batch 192, gradient clipping 1.0,
validation-BA checkpoint/threshold selection on fold 9) except that each training
record is replaced, with probability --aug-prob, by its phase-preserving
flattened version (fixed log-amplitude linear fit; flataug_torch variant "ecg",
identical to the paper's evaluation operator). Validation records stay unedited.

Evaluation covers the training-time operator (phase_flattened) and the
held-out envelope estimators of ptbxl_revision_interventions.py (harmonic-masked, SpecParam,
filter-bank). The unaugmented baseline is results/ptbxl_revision (same seeds).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ptbxl_raw_cnn_interventions as base  # noqa: E402
import ptbxl_revision_interventions as rev  # noqa: E402
import flataug_torch  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-npz", type=Path, required=True)
    p.add_argument("--psd-npz", type=Path, required=True)
    p.add_argument("--specparam-npz", type=Path, required=True, help="fit_ptbxl_specparam_test.py output")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="resnet1d_wang", choices=["resnet1d_wang", "inception1d", "xresnet1d101"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--aug-prob", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch-size", type=int, default=192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--harmonic-halfwidth", type=float, default=0.25)
    return p.parse_args()


def train_model_flataug(x_train, y_train, x_val, y_val, sfreq, args, device):
    """base.train_model with on-the-fly flattening of training records."""
    mean = x_train.mean(axis=(0, 2), keepdims=True).astype("float32")
    std = np.maximum(x_train.std(axis=(0, 2), keepdims=True).astype("float32"), 1e-6)
    xt = torch.from_numpy(x_train.astype("float32")).to(device)
    yt = torch.from_numpy(y_train.astype("int64")).to(device)
    mean_t, std_t = torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)
    gen = torch.Generator().manual_seed(args.seed)
    aug_gen = torch.Generator().manual_seed(args.seed + 1000)
    model = base.build_model(args.model, x_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    counts = np.bincount(y_train, minlength=2).astype("float32")
    w = counts.sum() / np.maximum(counts, 1.0)
    w = w / w.mean()
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    best_state, best_val, best_thr, log_rows = None, -math.inf, 0.5, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, n_aug = [], 0
        perm = torch.randperm(len(yt), generator=gen).to(device)
        for start in range(0, len(perm), args.batch_size):
            bi = perm[start:start + args.batch_size]
            xb, k = flataug_torch.augment_batch(xt[bi], sfreq, args.aug_prob, aug_gen, variant="ecg")
            n_aug += k
            xb = (xb - mean_t[0]) / std_t[0]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yt[bi])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            losses.append(float(loss.item()))
        sched.step()
        val_scores = base.predict_scores(model, x_val, y_val, mean, std, device, args.batch_size)
        thr, _ = base.best_threshold_for_balanced_accuracy(y_val, val_scores)
        val_ba = base.metrics_from_cm(base.confusion(y_val, base.predictions_from_scores(val_scores, thr)))["balanced_accuracy"]
        log_rows.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_balanced_accuracy": val_ba,
                         "val_threshold": thr, "aug_fraction": n_aug / len(yt)})
        print(f"epoch={epoch} loss={np.mean(losses):.4f} val_BA={val_ba:.4f} aug_frac={n_aug / len(yt):.3f}", flush=True)
        if val_ba > best_val:
            best_val, best_thr = val_ba, thr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, mean, std, log_rows, best_thr, best_val


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base.set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    data = np.load(args.raw_npz, allow_pickle=True)
    x = data["x"].astype("float32")
    y = data["y"].astype("int64")
    patient = data["patient_id"].astype(str)
    folds = data["strat_fold"].astype(int)
    sfreq = float(data["sfreq"])
    psd = np.load(args.psd_npz, allow_pickle=True)
    assert np.array_equal(psd["strat_fold"].astype(int), folds)
    train_idx, val_idx, test_idx = (np.flatnonzero(folds <= 8), np.flatnonzero(folds == 9),
                                    np.flatnonzero(folds == 10))

    ckpt = args.output_dir / "model.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = base.build_model(args.model, x.shape[1]).to(device)
        model.load_state_dict(state["state_dict"])
        mean, std, threshold, best_val = state["mean"], state["std"], state["threshold"], state["best_val"]
        training_log = state["training_log"]
    else:
        model, mean, std, training_log, threshold, best_val = train_model_flataug(
            x[train_idx], y[train_idx], x[val_idx], y[val_idx], sfreq, args, device)
        torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std, "threshold": threshold,
                    "best_val": best_val, "training_log": training_log, "model": args.model,
                    "seed": args.seed, "aug_prob": args.aug_prob}, ckpt)
        print(f"Trained in {time.time() - t0:.0f}s; val BA={best_val:.4f}", flush=True)

    sp_ap, sp_freqs = rev.load_specparam_test(args.specparam_npz, test_idx)
    conds, _ = rev.revision_interventions(x[test_idx], sfreq, sp_ap, sp_freqs, args.harmonic_halfwidth)
    pred_rows, summary = [], {}
    for name in rev.CONDITIONS:
        scores = base.predict_scores(model, conds[name], y[test_idx], mean, std, device, args.batch_size)
        y_pred = base.predictions_from_scores(scores, threshold)
        summary[name] = base.metrics_from_cm(base.confusion(y[test_idx], y_pred))["balanced_accuracy"]
        for rec, subj, yt, yp, sc in zip(test_idx, patient[test_idx], y[test_idx], y_pred, scores):
            pred_rows.append({"task": "ptbxl_normal_vs_abnormal", "seed": args.seed, "test_input": name,
                              "subject": subj, "record_index": int(rec), "y_true": int(yt),
                              "y_pred": int(yp), "score_abnormal": float(sc)})
        print(f"{name:22s} BA={summary[name]:.4f}", flush=True)
    base.write_csv(args.output_dir / "predictions.csv", pred_rows)
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "model": args.model, "seed": args.seed, "aug_prob": args.aug_prob, "epochs": args.epochs,
        "batch_size": args.batch_size, "lr": args.lr, "threshold": float(threshold),
        "best_val_balanced_accuracy": float(best_val), "balanced_accuracy": summary,
        "training_log": training_log, "torch": torch.__version__,
        "elapsed_sec": round(time.time() - t0, 1)}, indent=2))


if __name__ == "__main__":
    main()
