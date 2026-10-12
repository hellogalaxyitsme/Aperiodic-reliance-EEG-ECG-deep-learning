"""Resumable copy of run_tuab_bendr_intervention.fit_bendr (per-epoch checkpoints for long runs).

Training logic, data order, loss, optimiser, early stopping and threshold choice are
identical to the original; the only addition is a per-epoch checkpoint (model,
optimiser, early-stopping state and all RNG states) so an interrupted run resumes
at the next epoch with the same shuffling sequence.
"""
from __future__ import annotations

import math
import random
from pathlib import Path


def fit_bendr_resumable(B, x, y, subjects, channel_names, train_indices, args, state_path: Path):
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    common = B.common

    class IndexedBENDRDataset(Dataset):
        def __init__(self, x_array, y_array, indices):
            self.x_array, self.y_array = x_array, y_array
            self.indices = np.asarray(indices, dtype=int)

        def __len__(self):
            return int(len(self.indices))

        def __getitem__(self, item):
            idx = int(self.indices[item])
            xb = B.make_bendr_input(np.array(self.x_array[idx], dtype=np.float32, copy=True), args)
            yb = np.array([float(self.y_array[idx])], dtype=np.float32)
            return torch.from_numpy(xb), torch.from_numpy(yb)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    subtrain_idx, val_idx = common.make_subject_val_split(
        train_indices, y, subjects, val_fraction=args.val_fraction, seed=args.seed)
    model, load_summary = B.load_bendr_model(args, channel_names, x.shape[2], device)
    class_counts = np.bincount(y[subtrain_idx], minlength=2).astype("float32")
    pos_weight = torch.tensor([class_counts[0] / max(class_counts[1], 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(IndexedBENDRDataset(x, y, subtrain_idx), batch_size=args.batch_size, shuffle=True,
                        drop_last=False, pin_memory=torch.cuda.is_available(), num_workers=0)

    best_state, best_val_loss, best_epoch, patience_left, train_log, start_epoch = None, math.inf, 0, args.patience, [], 0
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        best_state, best_val_loss, best_epoch = st["best_state"], st["best_val_loss"], st["best_epoch"]
        patience_left, train_log, start_epoch = st["patience_left"], st["train_log"], st["epoch"] + 1
        torch.set_rng_state(st["torch_rng"])
        if torch.cuda.is_available() and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        np.random.set_state(st["np_rng"])
        random.setstate(st["py_rng"])
        print(f"resumed from epoch {st['epoch']} (stopped={st['stopped']})", flush=True)
        if st["stopped"]:
            start_epoch = args.epochs

    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_sum, seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            if args.gradient_clip_norm and args.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optimizer.step()
            loss_sum += float(loss.item()) * int(len(yb))
            seen += int(len(yb))
        model.eval()
        val_losses, val_probs = [], []
        with torch.no_grad():
            for start in range(0, len(val_idx), args.batch_size):
                bi = val_idx[start:start + args.batch_size]
                xb = torch.from_numpy(B.make_bendr_input(np.array(x[bi], dtype=np.float32, copy=True), args)).to(device)
                yb = torch.from_numpy(y[bi].astype("float32")[:, None]).to(device)
                logits = model(xb)
                vl = criterion(logits, yb)
                if not torch.isfinite(vl):
                    raise FloatingPointError(f"Non-finite validation loss at epoch {epoch}")
                val_losses.append(float(vl.item()) * int(len(bi)))
                val_probs.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
        val_prob = np.concatenate(val_probs)
        val_loss = float(sum(val_losses) / max(len(val_idx), 1))
        threshold = B.choose_threshold(val_prob, y[val_idx])
        val_bacc = common.balanced_accuracy(y[val_idx], (val_prob >= threshold).astype("int64"), 2)
        train_loss = loss_sum / max(seen, 1)
        train_log.append({"epoch": int(epoch), "train_loss": float(train_loss), "val_loss": val_loss,
                          "val_balanced_accuracy": float(val_bacc), "threshold": float(threshold)})
        print(f"epoch={epoch} train_loss={train_loss:.5f} val_loss={val_loss:.5f} val_bacc={val_bacc:.4f}", flush=True)
        stopped = False
        if val_loss < best_val_loss - 1e-5:
            best_val_loss, best_epoch, patience_left = val_loss, epoch, args.patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch={epoch}", flush=True)
                stopped = True
        tmp = state_path.with_suffix(".tmp")
        torch.save({"epoch": epoch, "stopped": stopped, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "best_state": best_state, "best_val_loss": best_val_loss, "best_epoch": best_epoch,
                    "patience_left": patience_left, "train_log": train_log, "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "np_rng": np.random.get_state(), "py_rng": random.getstate()}, tmp)
        tmp.replace(state_path)
        if stopped:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    val_prob = B.predict_prob(model, x, args, indices=val_idx)
    threshold = B.choose_threshold(val_prob, y[val_idx])
    train_meta = {"device": str(device), "best_epoch": int(best_epoch), "best_val_loss": float(best_val_loss),
                  "n_subtrain_windows": int(len(subtrain_idx)), "n_val_windows": int(len(val_idx)),
                  "n_subtrain_subjects": int(len(set(subjects[subtrain_idx]))),
                  "n_val_subjects": int(len(set(subjects[val_idx]))), "threshold": float(threshold),
                  "resumable_trainer": True, **load_summary}
    return model, threshold, train_log, train_meta
