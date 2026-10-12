#!/usr/bin/env python
"""SpecParam vs fixed linear-fit envelope on the same conventional model.

Retrains a full-TUAB conventional model (EEGNet by default) with the original
recipe (run_tuab_braindecode_eegnet_intervention.fit_model_indexed: official
split, AdamW 1e-3/1e-4, inverse-frequency CE, 80 epochs, early stopping on
validation loss with patience 12, batch 512), saves the checkpoint, and
evaluates the official-eval windows under two envelope estimators that differ
only in how the aperiodic shape is estimated:

  phase_aperiodic / phase_flattened  SpecParam fixed-mode aperiodic fit to the
                                     multitaper PSD (conventional-model procedure)
  lin_aperiodic / lin_flattened      fixed log-power linear fit to the epoch's
                                     own FFT periodogram over 1-45 Hz
                                     (foundation-model procedure,
                                     run_tuab_biot_intervention.make_phase_preserving_inputs)

The sham reconstruction is shared. Everything downstream of the envelope
(centring, division, phase preservation, std matching) is identical.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_tuab_braindecode_eegnet_intervention as conv  # noqa: E402
import run_tuab_biot_intervention as fm  # noqa: E402

CONDITIONS = ["raw_eeg", "phase_sham", "phase_aperiodic", "phase_flattened", "lin_aperiodic", "lin_flattened"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-npz", type=Path, required=True)
    p.add_argument("--index-csv", type=Path, required=True)
    p.add_argument("--specparam-eval-npz", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="eegnet", choices=["eegnet", "shallow_fbcsp", "deep4"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    import numpy as np
    import pandas as pd
    import random
    import torch

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # recipe defaults of the original runner (see parse_args there)
    cargs = argparse.Namespace(
        model=args.model, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=1e-3, weight_decay=1e-4, dropout=0.25, patience=args.patience,
        band_min=1.0, band_max=45.0, f1=8, depth_multiplier=2, kernel_length=64,
        depthwise_kernel_length=16, device=args.device, no_match_rms=False,
    )

    index = pd.read_csv(args.index_csv, usecols=["official_split", "label", "subject"])
    raw = np.load(args.raw_npz)
    x = raw["x"]
    sfreq = float(raw["sfreq"])
    print(f"loaded raw {x.shape} ({time.time() - t0:.0f}s)", flush=True)
    splits = index["official_split"].astype(str).to_numpy()
    labels = index["label"].astype(str).to_numpy()
    subjects = index["subject"].astype(str).to_numpy()
    y, classes = conv.encode_labels(labels)
    train_idx = np.flatnonzero(splits == "train")
    eval_idx = np.flatnonzero(splits == "eval")

    ckpt = args.output_dir / "model.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = conv.build_raw_model(x.shape[1], x.shape[2], len(classes), sfreq, cargs)
        model.load_state_dict(state["state_dict"])
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        mean, std, meta, train_log = state["mean"], state["std"], state["meta"], state["train_log"]
        print("loaded checkpoint", flush=True)
    else:
        model, train_log, meta, mean, std, device = conv.fit_model_indexed(
            x, y, train_idx, sfreq=sfreq, args=cargs, n_classes=len(classes))
        torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std, "meta": meta,
                    "train_log": train_log, "args": vars(cargs)}, ckpt)
        print(f"trained in {time.time() - t0:.0f}s best_epoch={meta['best_epoch']}", flush=True)

    # SpecParam eval fits are produced by fit_tuab_specparam_eval.py; wait if still running.
    while not args.specparam_eval_npz.with_suffix(".summary.json").exists():
        print("waiting for SpecParam eval fits ...", flush=True)
        time.sleep(120)
    sp = np.load(args.specparam_eval_npz)
    if not np.array_equal(sp["raw_row_index"], eval_idx):
        raise ValueError("SpecParam eval rows do not match official eval rows")
    ap_eval = sp["aperiodic_log_psd"]
    sp_freqs = sp["freqs"]

    preds = {c: [] for c in CONDITIONS}
    bs = args.batch_size
    for start in range(0, len(eval_idx), bs):
        rows = eval_idx[start:start + bs]
        xb = np.asarray(x[rows], dtype="float32")
        preds["raw_eeg"].append(conv.predict_array_batches(model, xb, mean, std, bs, device))
        sp_edit = conv.make_phase_preserving_inputs(
            xb, ap_eval[start:start + bs], sp_freqs, sfreq=sfreq, band_min=1.0, band_max=45.0, match_rms=True)
        lin_edit = fm.make_phase_preserving_inputs(xb, sfreq=sfreq, band_min=1.0, band_max=45.0)
        for name, arr in [("phase_sham", sp_edit["phase_sham"]),
                          ("phase_aperiodic", sp_edit["phase_aperiodic"]),
                          ("phase_flattened", sp_edit["phase_flattened"]),
                          ("lin_aperiodic", lin_edit["phase_aperiodic"]),
                          ("lin_flattened", lin_edit["phase_flattened"])]:
            preds[name].append(conv.predict_array_batches(model, arr, mean, std, bs, device))
    preds = {k: np.concatenate(v) for k, v in preds.items()}

    pred_rows, subject_rows, summary = [], [], {}
    for name in CONDITIONS:
        base_row = {"task": "tuab_normal_vs_abnormal", "seed": args.seed, "train_input": f"{args.model}_raw_eeg",
                    "test_input": name, "classes": "|".join(classes)}
        summary[name] = conv.balanced_accuracy(y[eval_idx], preds[name], len(classes))
        conv.append_subject_rows(subject_rows, base_row, subjects[eval_idx], y[eval_idx], preds[name],
                                 labels[eval_idx], len(classes))
        pred_rows.extend(conv.make_prediction_rows(base_row, eval_idx, subjects[eval_idx], labels[eval_idx],
                                                   y[eval_idx], preds[name], len(classes)))
        print(f"{name:16s} window BA={summary[name]:.4f}", flush=True)
    conv.write_csv(args.output_dir / "subject_metrics.csv", subject_rows)
    conv.write_csv(args.output_dir / "predictions.csv", pred_rows)
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "model": args.model, "seed": args.seed, "patience": args.patience, "epochs": args.epochs,
        "batch_size": args.batch_size, "window_balanced_accuracy": summary, "train_meta": meta,
        "n_eval_windows": int(len(eval_idx)), "n_eval_subjects": int(len(np.unique(subjects[eval_idx]))),
        "torch": torch.__version__, "elapsed_sec": round(time.time() - t0, 1)}, indent=2, default=str))


if __name__ == "__main__":
    main()
