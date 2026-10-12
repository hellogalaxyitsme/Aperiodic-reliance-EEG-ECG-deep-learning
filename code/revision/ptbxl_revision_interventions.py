#!/usr/bin/env python
"""PTB-XL envelope-estimator robustness.

Retrains the PTB-XL architectures with the original recipe
(run_ptbxl_raw_cnn_interventions.train_model), saves the checkpoint, and
evaluates the held-out fold under the original conditions plus three
alternative estimates of the broadband spectral envelope that is removed by
flattening:

  phase_flattened            original log-amplitude linear 1/f fit, 1-45 Hz
  flat_harmonic_masked       linear 1/f fit restricted to inter-harmonic bins
                             (bins near multiples of the per-record heart-rate
                             fundamental are excluded from the fit)
  flat_specparam             SpecParam fixed-mode aperiodic fit (with peaks,
                             EEG settings) of the original test-fold
                             periodograms (fit_ptbxl_specparam_test.py; log10
                             power). The original psd_fixed file holds a
                             peak-free natural-log fit and must not be used here.
  flat_filterbank            non-parametric filter-bank equalisation: the mean
                             log-amplitude of each band in a fixed bank is set
                             to the across-band mean (no 1/f model at all)

All flattening variants share the same reconstruction as the original:
log|X| -> log|X| - (E(f) - <E>), original phase, inverse FFT, RMS matching to
the original record/lead, mean centring. Only the envelope estimate E(f)
differs, so differences between conditions isolate the estimator.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_ptbxl_raw_cnn_interventions as base  # noqa: E402

BAND = (1.0, 45.0)
FILTERBANK_EDGES = [1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 45.0]
CONDITIONS = [
    "raw_eeg",
    "phase_sham",
    "phase_aperiodic",
    "phase_flattened",
    "flat_harmonic_masked",
    "flat_specparam",
    "flat_filterbank",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-npz", type=Path, required=True)
    p.add_argument("--psd-npz", type=Path, required=True, help="Original PSD file (fold alignment check)")
    p.add_argument("--specparam-npz", type=Path, required=True, help="fit_ptbxl_specparam_test.py output")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", required=True, choices=["resnet1d_wang", "inception1d", "xresnet1d101"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch-size", type=int, default=192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--harmonic-halfwidth", type=float, default=0.25,
                   help="Exclusion half-width around k*f0, as a fraction of f0")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Envelope estimators (all return natural-log amplitude envelopes on band bins)
# ----------------------------------------------------------------------------

def estimate_fundamental(x: np.ndarray, sfreq: float) -> np.ndarray:
    """Per-record heart-rate fundamental (Hz) from the autocorrelation of the
    lead-summed squared derivative (QRS energy). Search range 40-180 bpm."""
    d = np.diff(x, axis=-1)
    e = (d ** 2).sum(axis=1)
    k = max(1, int(round(0.05 * sfreq)))
    kernel = np.ones(k) / k
    e = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 1, e)
    e = e - e.mean(axis=1, keepdims=True)
    n = e.shape[1]
    spec = np.fft.rfft(e, n=2 * n, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :n]
    lo, hi = int(sfreq * 60 / 180), int(sfreq * 60 / 40)
    lag = lo + np.argmax(ac[:, lo:hi + 1], axis=1)
    return sfreq / lag.astype(float)


def harmonic_keep_mask(freqs: np.ndarray, f0: np.ndarray, halfwidth: float) -> np.ndarray:
    """Boolean (n_records, n_bins): True for inter-harmonic bins."""
    ratio = freqs[None, :] / f0[:, None]
    dist = np.abs(ratio - np.round(ratio))  # distance to nearest harmonic, in units of f0
    return dist >= halfwidth


def linear_fit(log_amp: np.ndarray, logf: np.ndarray, weights: np.ndarray | None = None):
    """Least-squares a - b*log f per (record, lead). weights: (n_rec, n_bins) 0/1."""
    n_rec, n_lead, n_bin = log_amp.shape
    design = np.column_stack([np.ones_like(logf), -logf])
    if weights is None:
        coef = np.linalg.pinv(design) @ log_amp.reshape(-1, n_bin).T
        fitted = (design @ coef).T.reshape(n_rec, n_lead, n_bin)
        return fitted
    fitted = np.empty_like(log_amp)
    for r in range(n_rec):
        w = weights[r]
        pinv = np.linalg.pinv(design[w])
        coef = pinv @ log_amp[r][:, w].T
        fitted[r] = (design @ coef).T
    return fitted


def r_squared(obs: np.ndarray, fit: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is not None:
        out = np.empty(obs.shape[:2])
        for r in range(obs.shape[0]):
            o, f = obs[r][:, mask[r]], fit[r][:, mask[r]]
            ss_res = ((o - f) ** 2).sum(-1)
            ss_tot = ((o - o.mean(-1, keepdims=True)) ** 2).sum(-1)
            out[r] = 1 - ss_res / np.maximum(ss_tot, 1e-12)
        return out
    ss_res = ((obs - fit) ** 2).sum(-1)
    ss_tot = ((obs - obs.mean(-1, keepdims=True)) ** 2).sum(-1)
    return 1 - ss_res / np.maximum(ss_tot, 1e-12)


def filterbank_envelope(log_amp: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    env = np.empty_like(log_amp)
    edges = FILTERBANK_EDGES
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        sel = (freqs >= lo) & ((freqs < hi) if i < len(edges) - 2 else (freqs <= hi))
        env[..., sel] = log_amp[..., sel].mean(axis=-1, keepdims=True)
    return env


def reconstruct(x, amp, phase, mask, new_log_amp_band):
    new_amp = amp.copy()
    new_amp[:, :, mask] = np.exp(new_log_amp_band)
    recon = np.fft.irfft(new_amp * np.exp(1j * phase), n=x.shape[-1], axis=-1).astype("float32")
    orig_rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True))
    recon_rms = np.sqrt(np.mean(recon ** 2, axis=-1, keepdims=True))
    recon = recon * (orig_rms / np.maximum(recon_rms, 1e-8))
    return (recon - recon.mean(axis=-1, keepdims=True)).astype("float32")


def load_specparam_test(path, test_idx):
    """(aperiodic log10 power, freqs) for the test records, in test_idx order."""
    sp = np.load(path)
    if not np.array_equal(sp["record_index"], test_idx):
        raise ValueError("SpecParam test-fold rows do not match the test records")
    return sp["aperiodic_log10_psd"].astype("float64"), sp["freqs"].astype("float64")


def revision_interventions(x, sfreq, specparam_ap_log10, specparam_freqs, halfwidth):
    """Return (dict condition -> array, diagnostics dict)."""
    out = base.intervention_batch(x, sfreq)  # raw, sham, aperiodic, flattened (original)
    freqs = np.fft.rfftfreq(x.shape[-1], d=1.0 / sfreq)
    fft = np.fft.rfft(x, axis=-1)
    amp, phase = np.abs(fft), np.angle(fft)
    mask = (freqs >= BAND[0]) & (freqs <= min(BAND[1], sfreq / 2.0 - 1e-6))
    bf = freqs[mask]
    logf = np.log(bf)
    log_amp = np.log(np.maximum(amp[:, :, mask], 1e-12)).astype("float64")

    def flatten_with(env):
        return reconstruct(x, amp, phase, mask, log_amp - (env - env.mean(axis=-1, keepdims=True)))

    # original estimator, for fit-quality reporting
    fit_lin = linear_fit(log_amp, logf)

    # harmonic-masked linear fit
    f0 = estimate_fundamental(x, sfreq)
    keep = harmonic_keep_mask(bf, f0, halfwidth)
    fit_harm = linear_fit(log_amp, logf, weights=keep)
    out["flat_harmonic_masked"] = flatten_with(fit_harm)

    # SpecParam fixed-mode aperiodic (log10 power) -> natural-log amplitude
    sp = np.empty_like(log_amp)
    for r in range(x.shape[0]):
        for c in range(x.shape[1]):
            sp[r, c] = np.interp(bf, specparam_freqs, specparam_ap_log10[r, c])
    sp = 0.5 * np.log(10.0) * sp
    out["flat_specparam"] = flatten_with(sp)

    # filter-bank equalisation
    out["flat_filterbank"] = flatten_with(filterbank_envelope(log_amp, bf))

    diag = {
        "f0_hz": f0,
        "r2_linear_allbins": r_squared(log_amp, fit_lin),
        "r2_linear_interharmonic": r_squared(log_amp, fit_lin, keep),
        "r2_harmonic_masked_interharmonic": r_squared(log_amp, fit_harm, keep),
        "frac_bins_kept": keep.mean(axis=1),
    }
    return out, diag


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

    train_idx = np.flatnonzero(folds <= 8)
    val_idx = np.flatnonzero(folds == 9)
    test_idx = np.flatnonzero(folds == 10)

    ckpt = args.output_dir / "model.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = base.build_model(args.model, x.shape[1]).to(device)
        model.load_state_dict(state["state_dict"])
        mean, std, threshold, best_val = state["mean"], state["std"], state["threshold"], state["best_val"]
        training_log = state["training_log"]
        print("Loaded checkpoint", ckpt, flush=True)
    else:
        bargs = argparse.Namespace(model=args.model, epochs=args.epochs, batch_size=args.batch_size,
                                   lr=args.lr, num_workers=args.num_workers)
        model, mean, std, training_log, threshold, best_val = base.train_model(
            x[train_idx], y[train_idx], x[val_idx], y[val_idx], args.seed, bargs, device)
        torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std, "threshold": threshold,
                    "best_val": best_val, "training_log": training_log, "model": args.model,
                    "seed": args.seed}, ckpt)
        print(f"Trained {args.model} seed {args.seed} in {time.time() - t0:.0f}s; val BA={best_val:.4f}", flush=True)

    sp_ap, sp_freqs = load_specparam_test(args.specparam_npz, test_idx)
    conds, diag = revision_interventions(x[test_idx], sfreq, sp_ap, sp_freqs, args.harmonic_halfwidth)
    pred_rows = []
    summary = {}
    for name in CONDITIONS:
        scores = base.predict_scores(model, conds[name], y[test_idx], mean, std, device, args.batch_size)
        y_pred = base.predictions_from_scores(scores, threshold)
        m = base.metrics_from_cm(base.confusion(y[test_idx], y_pred))
        summary[name] = m["balanced_accuracy"]
        for rec, subj, yt, yp, sc in zip(test_idx, patient[test_idx], y[test_idx], y_pred, scores):
            pred_rows.append({"task": "ptbxl_normal_vs_abnormal", "seed": args.seed, "test_input": name,
                              "subject": subj, "record_index": int(rec), "y_true": int(yt),
                              "y_pred": int(yp), "score_abnormal": float(sc)})
        print(f"{name:22s} BA={m['balanced_accuracy']:.4f}", flush=True)

    base.write_csv(args.output_dir / "predictions.csv", pred_rows)
    np.savez_compressed(args.output_dir / "envelope_fit_diagnostics.npz",
                        record_index=test_idx, specparam_r2=np.load(args.specparam_npz)["r_squared"], **diag)
    meta = {
        "model": args.model, "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": args.lr, "threshold": float(threshold), "best_val_balanced_accuracy": float(best_val),
        "harmonic_halfwidth_f0_fraction": args.harmonic_halfwidth, "filterbank_edges_hz": FILTERBANK_EDGES,
        "n_test_records": int(len(test_idx)), "balanced_accuracy": summary,
        "envelope_fit_median": {k: float(np.median(v)) for k, v in diag.items()},
        "specparam_r2_median_test": float(np.nanmedian(np.load(args.specparam_npz)["r_squared"])),
        "torch": torch.__version__, "device": str(device), "elapsed_sec": round(time.time() - t0, 1),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["envelope_fit_median"], indent=2))


if __name__ == "__main__":
    main()
