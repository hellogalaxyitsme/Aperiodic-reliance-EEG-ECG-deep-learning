#!/usr/bin/env python
"""BENDR full-TUAB intervention evaluation with corrected input scaling.

In the original version of run_tuab_bendr_intervention.py (now corrected in code/scripts), edited
inputs were built from ``make_bendr_input(x)`` (clip to +/-500 uV, x1e-6 to V)
and then passed to ``predict_prob``, which applies ``make_bendr_input`` again.
Sham/aperiodic/flattened inputs therefore reached the network at 1e-12 of the
signal scale (an effectively all-zero input), while raw inputs were scaled
once. Every edited condition consequently produced a constant output.

Correction: interventions are computed on the clipped signal in uV, and the
uV->V scaling is applied exactly once, immediately before the forward pass,
identically for raw and edited inputs. Training, checkpoint initialisation,
hyper-parameters, thresholding and bootstrap are reused unchanged from the
original module. The fine-tuned weights are saved, and sham-integrity
diagnostics (relative reconstruction error, raw-vs-sham prediction agreement)
are recorded.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_tuab_bendr_intervention as B  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bendr_resumable import fit_bendr_resumable  # noqa: E402  (same logic as B.fit_bendr + per-epoch checkpoint)

common = B.common


def predict_prob_prepared(model, x_prepared, batch_size):
    """Forward pass on inputs that are already in model units (no re-scaling)."""
    import numpy as np
    import torch

    device = next(model.parameters()).device
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x_prepared), batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(x_prepared[start:start + batch_size], dtype=np.float32)).to(device)
            out.append(torch.sigmoid(model(xb)).cpu().numpy().reshape(-1))
    return np.concatenate(out)


def main() -> int:
    import numpy as np
    import torch

    args = B.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    bundle = B.load_cache(args)
    x, y = bundle["x"], bundle["y"]
    subjects, splits, labels = bundle["subjects"], bundle["splits"], bundle["labels"]
    channel_names = bundle["channels"]
    train_idx = np.flatnonzero(splits == "train")
    eval_idx = np.flatnonzero(splits == "eval")
    train_idx = common.subset_indices(train_idx, args.max_train_windows, args.seed)
    eval_idx = common.subset_indices(eval_idx, args.max_eval_windows, args.seed + 1)

    ckpt = args.output_dir / "model.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        model, _ = B.load_bendr_model(args, channel_names, x.shape[2], device)
        model.load_state_dict(state["state_dict"])
        threshold, train_log, train_meta = state["threshold"], state["train_log"], state["train_meta"]
        print("loaded checkpoint", flush=True)
    else:
        model, threshold, train_log, train_meta = fit_bendr_resumable(
            B, x, y, subjects, channel_names, train_idx, args, args.output_dir / "train_state.pt")
        torch.save({"state_dict": model.state_dict(), "threshold": threshold, "train_log": train_log,
                    "train_meta": train_meta}, ckpt)
        print(f"fine-tuned in {time.time() - t0:.0f}s", flush=True)

    scale = float(args.input_scale)
    clip = float(args.clip_uv)
    conds = ["raw_eeg", "phase_sham", "phase_aperiodic", "phase_flattened"]
    probs = {c: [] for c in conds}
    rel_err, max_abs_err = [], []
    for start in range(0, len(eval_idx), args.intervention_batch_size):
        rows = eval_idx[start:start + args.intervention_batch_size]
        x_uv = np.clip(np.asarray(x[rows], dtype=np.float32), -clip, clip)
        edited = common.make_phase_preserving_inputs(x_uv, sfreq=float(args.sampling_rate),
                                                     band_min=float(args.band_min), band_max=float(args.band_max))
        probs["raw_eeg"].append(predict_prob_prepared(model, x_uv * scale, args.batch_size))
        for c in conds[1:]:
            probs[c].append(predict_prob_prepared(model, edited[c] * scale, args.batch_size))
        d = edited["phase_sham"] - x_uv
        rel_err.append(np.sqrt((d ** 2).mean(axis=-1)) / np.maximum(np.sqrt((x_uv ** 2).mean(axis=-1)), 1e-6))
        max_abs_err.append(np.abs(d).max(axis=(1, 2)))
        if (start // args.intervention_batch_size) % 20 == 0:
            print(f"eval {start + len(rows)}/{len(eval_idx)}", flush=True)
    probs = {c: np.concatenate(v) for c, v in probs.items()}
    rel_err = np.concatenate(rel_err)
    max_abs_err = np.concatenate(max_abs_err)

    pred_rows, eval_rows = [], []
    for c in conds:
        pred = (probs[c] >= threshold).astype("int64")
        eval_rows.append({"task": "tuab_normal_vs_abnormal", "model": "BENDR", "train_input": "raw_eeg",
                          "test_input": c, "n_eval_windows": int(len(eval_idx)),
                          "n_eval_subjects": int(len(np.unique(subjects[eval_idx]))), "threshold": float(threshold),
                          "balanced_accuracy": common.balanced_accuracy(y[eval_idx], pred, 2),
                          "macro_f1": common.macro_f1(y[eval_idx], pred, 2),
                          "accuracy": common.accuracy(y[eval_idx], pred),
                          "prob_std": float(probs[c].std())})
        for ri, s, lab, t, sc, yp in zip(eval_idx, subjects[eval_idx], labels[eval_idx], y[eval_idx], probs[c], pred):
            pred_rows.append({"task": "tuab_normal_vs_abnormal", "model": "BENDR", "train_input": "raw_eeg",
                              "test_input": c, "row_index": int(ri), "subject": str(s), "label": str(lab),
                              "y_true": int(t), "y_pred": int(yp), "prob_abnormal": float(sc)})
        print(f"{c:16s} BA={eval_rows[-1]['balanced_accuracy']:.4f} prob_std={probs[c].std():.4f}", flush=True)

    raw_pred = (probs["raw_eeg"] >= threshold)
    sham_pred = (probs["phase_sham"] >= threshold)
    integrity = {
        "sham_relative_rms_error_median": float(np.median(rel_err)),
        "sham_relative_rms_error_p99": float(np.percentile(rel_err, 99)),
        "sham_max_abs_error_uv_median": float(np.median(max_abs_err)),
        "raw_vs_sham_decision_agreement": float((raw_pred == sham_pred).mean()),
        "raw_vs_sham_prob_mean_abs_diff": float(np.abs(probs["raw_eeg"] - probs["phase_sham"]).mean()),
    }
    print(json.dumps(integrity, indent=2), flush=True)

    out = args.output_dir
    B.write_csv(out / "tuab_bendr_intervention_eval_metrics.csv", eval_rows)
    B.write_csv(out / "tuab_bendr_intervention_predictions.csv", pred_rows)
    boot = B.bootstrap_rows(pred_rows, args.n_bootstrap, args.ci, args.seed)
    B.write_csv(out / "tuab_bendr_intervention_subject_bootstrap.csv", boot)
    B.write_markdown(out / "tuab_bendr_intervention_subject_bootstrap.md", boot)
    B.write_csv(out / "tuab_bendr_training_log.csv", train_log)
    (out / "tuab_bendr_intervention_metadata.json").write_text(json.dumps({
        "seed": int(args.seed), "epochs": int(args.epochs), "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate), "weight_decay": float(args.weight_decay),
        "input_scale": scale, "clip_uv": clip, "encoder_only": bool(args.encoder_only),
        "fix": "edited inputs scaled once (uV->V) immediately before forward; original version scaled twice",
        "sham_integrity": integrity, "train_meta": train_meta, "threshold": float(threshold),
        "torch": torch.__version__, "elapsed_sec": round(time.time() - t0, 1)}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
