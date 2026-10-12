#!/usr/bin/env python
"""Training-time mitigation on TUAB (conventional model).

Identical to tuab_fit_procedure_sensitivity.py (original EEGNet recipe, same
seeds, same evaluation conditions) except that during training each window is
replaced, with probability --aug-prob, by its phase-preserving flattened version
computed with the fast fixed linear-fit operator (flataug_torch.flatten_linear,
variant "eeg"). Validation windows (early stopping) stay unedited.

Evaluation reports the raw input, the sham, and flattening with BOTH the
training-time operator (lin_*) and the held-out SpecParam operator (phase_*),
so invariance is not credited merely for memorising the training operator.
The unaugmented baseline for the paired comparison is results/tuab_fit_sensitivity.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_tuab_braindecode_eegnet_intervention as conv  # noqa: E402
import run_tuab_biot_intervention as fm  # noqa: E402
import flataug_torch  # noqa: E402

CONDITIONS = ["raw_eeg", "phase_sham", "phase_aperiodic", "phase_flattened", "lin_aperiodic", "lin_flattened"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-npz", type=Path, required=True)
    p.add_argument("--index-csv", type=Path, required=True)
    p.add_argument("--specparam-eval-npz", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="eegnet", choices=["eegnet", "shallow_fbcsp", "deep4"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--aug-prob", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def fit_model_flataug(x, y, train_idx, sfreq, args, n_classes, aug_prob, device):
    """conv.fit_model_indexed with on-the-fly flattening of training windows."""
    import numpy as np
    import torch
    from torch import nn

    mean, std = conv.compute_channel_standardization(x, train_idx, batch_size=args.batch_size)
    split_train_idx, split_val_idx = conv.make_train_val_split(y[train_idx], seed=args.seed)
    sub_train_idx = train_idx[split_train_idx]
    val_idx = train_idx[split_val_idx]
    class_counts = np.bincount(y[sub_train_idx], minlength=n_classes).astype("float32")
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.mean()

    model = conv.build_raw_model(n_channels=x.shape[1], n_times=x.shape[2], n_classes=n_classes,
                                 sfreq=sfreq, args=args).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    mean_t = torch.from_numpy(mean).to(device)
    std_t = torch.from_numpy(std).to(device)
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator().manual_seed(args.seed + 1000)
    train_log, best_state, best_val, best_epoch = [], None, math.inf, 0
    patience_left = args.patience
    n_aug_total = 0
    start_epoch = 0
    state_path = args.state_path
    if state_path.exists():  # resume after interruption (per-epoch checkpoint below)
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        best_state, best_val, best_epoch = st["best_state"], st["best_val"], st["best_epoch"]
        patience_left, train_log, n_aug_total = st["patience_left"], st["train_log"], st["n_aug_total"]
        rng.bit_generator.state = st["np_rng"]
        gen.set_state(st["aug_rng"])
        start_epoch = args.epochs if st["stopped"] else st["epoch"] + 1
        print(f"resumed from epoch {st['epoch']} (stopped={st['stopped']})", flush=True)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        shuffled = sub_train_idx.copy()
        rng.shuffle(shuffled)
        loss_sum, seen, n_aug = 0.0, 0, 0
        for start in range(0, len(shuffled), args.batch_size):
            batch_idx = np.sort(shuffled[start:start + args.batch_size])  # sorted for memmap locality
            xb = torch.from_numpy(np.asarray(x[batch_idx], dtype="float32")).to(device, non_blocking=True)
            xb, k = flataug_torch.augment_batch(xb, sfreq, aug_prob, gen, variant="eeg")
            n_aug += k
            xb = (xb - mean_t) / std_t
            yb = torch.from_numpy(y[batch_idx]).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(batch_idx)
            seen += len(batch_idx)
        model.eval()
        vsum, vseen = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(val_idx), args.batch_size):
                bi = val_idx[start:start + args.batch_size]
                xb = torch.from_numpy(conv.scale_batch(x[bi], mean, std)).to(device, non_blocking=True)
                yb = torch.from_numpy(y[bi]).to(device, non_blocking=True)
                vsum += float(criterion(model(xb), yb).item()) * len(bi)
                vseen += len(bi)
        tl, vl = loss_sum / max(seen, 1), vsum / max(vseen, 1)
        n_aug_total += n_aug
        train_log.append({"epoch": epoch, "train_loss": tl, "val_loss": vl, "aug_fraction": n_aug / max(seen, 1)})
        print(f"epoch={epoch} train_loss={tl:.5f} val_loss={vl:.5f} aug_frac={n_aug / max(seen, 1):.3f}", flush=True)
        stopped = False
        if vl < best_val - 1e-5:
            best_val, best_epoch, patience_left = vl, epoch, args.patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch={epoch}", flush=True)
                stopped = True
        tmp = state_path.with_suffix(".tmp")
        torch.save({"epoch": epoch, "stopped": stopped, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "best_state": best_state, "best_val": best_val,
                    "best_epoch": best_epoch, "patience_left": patience_left, "train_log": train_log,
                    "n_aug_total": n_aug_total, "np_rng": rng.bit_generator.state, "aug_rng": gen.get_state()}, tmp)
        tmp.replace(state_path)
        if stopped:
            break
    model.load_state_dict(best_state)
    meta = {"device": str(device), "best_epoch": best_epoch, "best_val_loss": best_val,
            "n_subtrain": int(len(sub_train_idx)), "n_val": int(len(val_idx)), "aug_prob": aug_prob,
            "n_augmented_windows_total": int(n_aug_total),
            "augmentation": "fixed linear-fit phase-preserving flattening (flataug_torch, variant eeg)",
            "normalization": "chunked_train_channel_time_standardization (unaugmented training data)"}
    return model, train_log, meta, mean, std


def main():
    import random

    import numpy as np
    import pandas as pd
    import torch

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cargs = argparse.Namespace(
        model=args.model, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=1e-3, weight_decay=1e-4, dropout=0.25, patience=args.patience,
        band_min=1.0, band_max=45.0, f1=8, depth_multiplier=2, kernel_length=64,
        depthwise_kernel_length=16, device=args.device, no_match_rms=False,
        state_path=args.output_dir / "train_state.pt",
    )
    index =pd.read_csv(args.index_csv, usecols=["official_split", "label", "subject"])
    raw = np.load(args.raw_npz)
    x = raw["x"]
    sfreq = float(raw["sfreq"])
    splits = index["official_split"].astype(str).to_numpy()
    labels = index["label"].astype(str).to_numpy()
    subjects = index["subject"].astype(str).to_numpy()
    y, classes = conv.encode_labels(labels)
    train_idx = np.flatnonzero(splits == "train")
    eval_idx = np.flatnonzero(splits == "eval")
    print(f"loaded raw {x.shape} ({time.time() - t0:.0f}s)", flush=True)

    ckpt = args.output_dir / "model.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = conv.build_raw_model(x.shape[1], x.shape[2], len(classes), sfreq, cargs)
        model.load_state_dict(state["state_dict"])
        model = model.to(device).eval()
        mean, std, meta, train_log = state["mean"], state["std"], state["meta"], state["train_log"]
        print("loaded checkpoint", flush=True)
    else:
        model, train_log, meta, mean, std = fit_model_flataug(
            x, y, train_idx, sfreq, cargs, len(classes), args.aug_prob, device)
        torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std, "meta": meta,
                    "train_log": train_log, "args": vars(cargs)}, ckpt)
        print(f"trained in {time.time() - t0:.0f}s best_epoch={meta['best_epoch']}", flush=True)

    sp = np.load(args.specparam_eval_npz)
    if not np.array_equal(sp["raw_row_index"], eval_idx):
        raise ValueError("SpecParam eval rows do not match official eval rows")
    ap_eval, sp_freqs = sp["aperiodic_log_psd"], sp["freqs"]
    preds = {c: [] for c in CONDITIONS}
    bs = args.batch_size
    for start in range(0, len(eval_idx), bs):
        rows = eval_idx[start:start + bs]
        xb = np.asarray(x[rows], dtype="float32")
        preds["raw_eeg"].append(conv.predict_array_batches(model, xb, mean, std, bs, device))
        sp_edit = conv.make_phase_preserving_inputs(
            xb, ap_eval[start:start + bs], sp_freqs, sfreq=sfreq, band_min=1.0, band_max=45.0, match_rms=True)
        lin_edit = fm.make_phase_preserving_inputs(xb, sfreq=sfreq, band_min=1.0, band_max=45.0)
        for name, arr in [("phase_sham", sp_edit["phase_sham"]), ("phase_aperiodic", sp_edit["phase_aperiodic"]),
                          ("phase_flattened", sp_edit["phase_flattened"]),
                          ("lin_aperiodic", lin_edit["phase_aperiodic"]), ("lin_flattened", lin_edit["phase_flattened"])]:
            preds[name].append(conv.predict_array_batches(model, arr, mean, std, bs, device))
    preds = {k: np.concatenate(v) for k, v in preds.items()}

    pred_rows, subject_rows, summary = [], [], {}
    for name in CONDITIONS:
        base_row = {"task": "tuab_normal_vs_abnormal", "seed": args.seed,
                    "train_input": f"{args.model}_flataug_p{args.aug_prob:g}", "test_input": name,
                    "classes": "|".join(classes)}
        summary[name] = conv.balanced_accuracy(y[eval_idx], preds[name], len(classes))
        conv.append_subject_rows(subject_rows, base_row, subjects[eval_idx], y[eval_idx], preds[name],
                                 labels[eval_idx], len(classes))
        pred_rows.extend(conv.make_prediction_rows(base_row, eval_idx, subjects[eval_idx], labels[eval_idx],
                                                   y[eval_idx], preds[name], len(classes)))
        print(f"{name:16s} window BA={summary[name]:.4f}", flush=True)
    conv.write_csv(args.output_dir / "subject_metrics.csv", subject_rows)
    conv.write_csv(args.output_dir / "predictions.csv", pred_rows)
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "model": args.model, "seed": args.seed, "aug_prob": args.aug_prob, "patience": args.patience,
        "epochs": args.epochs, "batch_size": args.batch_size, "window_balanced_accuracy": summary,
        "train_meta": meta, "train_log": train_log, "n_eval_windows": int(len(eval_idx)),
        "torch": torch.__version__, "elapsed_sec": round(time.time() - t0, 1)}, indent=2, default=str))


if __name__ == "__main__":
    main()
