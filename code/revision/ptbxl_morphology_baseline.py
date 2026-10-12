#!/usr/bin/env python
"""Morphological (median-beat feature) ECG baseline under the interventions.

Per record: R peaks on lead II, median beat (+/-300 ms) per lead, and classical
morphology features per lead (R, Q, S, T amplitude; ST level; QRS width) plus
RR mean / SD. A gradient-boosting classifier is trained on folds 1-8 (raw),
threshold-tuned for balanced accuracy on fold 9, and evaluated on fold 10 under
raw, sham, original flattening, harmonic-masked flattening and filter-bank
equalisation. Output predictions use the same layout as the neural runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ptbxl_revision_interventions as rev  # noqa: E402

CONDS = ["raw_eeg", "phase_sham", "phase_flattened", "flat_harmonic_masked", "flat_filterbank"]


def r_peaks(sig, sf):
    b, a = butter(2, [5 / (sf / 2), 15 / (sf / 2)], btype="band")
    env = np.convolve(filtfilt(b, a, sig) ** 2, np.ones(int(0.08 * sf)) / int(0.08 * sf), mode="same")
    pk, _ = find_peaks(env, distance=int(0.33 * sf), height=0.3 * np.percentile(env, 99))
    w = int(0.05 * sf)
    return np.array([max(0, p - w) + np.abs(sig[max(0, p - w):p + w + 1] - np.median(sig)).argmax() for p in pk], int)


def features(x, sf):
    """x: (n_rec, 12, T) -> (n_rec, 12*6 + 3) feature matrix."""
    half = int(0.3 * sf)
    ms = lambda v: int(round(v * sf / 1000.0))
    out = np.full((x.shape[0], x.shape[1] * 6 + 3), np.nan, dtype="float32")
    for i in range(x.shape[0]):
        pk = r_peaks(x[i, 1], sf)
        pk = pk[(pk - half >= 0) & (pk + half <= x.shape[-1])]
        if len(pk) >= 2:
            rr = np.diff(pk) / sf
            out[i, -3:] = [rr.mean(), rr.std(), 60.0 / rr.mean()]
        if len(pk) < 1:
            continue
        beats = np.stack([x[i, :, p - half:p + half] for p in pk])  # (n_beats, 12, 2*half)
        mb = np.median(beats, axis=0)
        c = half
        for l in range(x.shape[1]):
            b = mb[l]
            base = np.median(b[c - ms(300):c - ms(200)]) if c - ms(300) >= 0 else np.median(b[:ms(100)])
            b = b - base
            win = b[c - ms(50):c + ms(50) + 1]
            r_amp = win.max()
            q_amp = b[c - ms(80):c + 1].min()
            s_amp = b[c:c + ms(80) + 1].min()
            st = b[c + ms(80):c + ms(120) + 1].mean()
            t_seg = b[c + ms(150):c + ms(300)]
            t_amp = t_seg[np.abs(t_seg).argmax()] if len(t_seg) else np.nan
            thr = 0.3 * np.abs(win).max()
            qrs_w = (np.abs(b[c - ms(80):c + ms(80) + 1]) > thr).sum() * 1000.0 / sf
            out[i, l * 6:(l + 1) * 6] = [r_amp, q_amp, s_amp, st, t_amp, qrs_w]
    return out


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-npz", type=Path, required=True)
    p.add_argument("--psd-npz", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(args.raw_npz, allow_pickle=True)
    psd = np.load(args.psd_npz, allow_pickle=True)
    x, y = d["x"].astype("float32"), d["y"].astype(int)
    folds, patient, sf = d["strat_fold"].astype(int), d["patient_id"].astype(str), float(d["sfreq"])
    tr, va, te = np.flatnonzero(folds <= 8), np.flatnonzero(folds == 9), np.flatnonzero(folds == 10)

    f_tr, f_va = features(x[tr], sf), features(x[va], sf)
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, class_weight="balanced", random_state=0)
    clf.fit(f_tr, y[tr])
    s_va = clf.predict_proba(f_va)[:, 1]
    grid = np.linspace(0.05, 0.95, 181)
    ba = lambda yt, yp: 0.5 * (((yp == 0) & (yt == 0)).sum() / max((yt == 0).sum(), 1)
                                + ((yp == 1) & (yt == 1)).sum() / max((yt == 1).sum(), 1))
    thr = float(grid[np.argmax([ba(y[va], (s_va >= t).astype(int)) for t in grid])])

    # flat_specparam is not evaluated here (CONDS), so the SpecParam input is a placeholder of the right shape
    conds, _ = rev.revision_interventions(x[te], sf, np.zeros(psd["aperiodic_log_psd"][te].shape),
                                          psd["freqs"].astype("float64"), 0.25)
    rows, summary = [], {}
    for c in CONDS:
        fe = features(conds[c], sf)
        sc = clf.predict_proba(fe)[:, 1]
        yp = (sc >= thr).astype(int)
        summary[c] = float(ba(y[te], yp))
        for rec, subj, yt, ypp, s in zip(te, patient[te], y[te], yp, sc):
            rows.append({"task": "ptbxl_normal_vs_abnormal", "seed": 0, "test_input": c, "subject": subj,
                         "record_index": int(rec), "y_true": int(yt), "y_pred": int(ypp), "score_abnormal": float(s)})
        print(f"{c:22s} BA={summary[c]:.4f}  missing-feature frac={np.isnan(fe).mean():.3f}", flush=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "predictions.csv", index=False)
    (args.output_dir / "metadata.json").write_text(json.dumps(
        {"classifier": "HistGradientBoosting(max_iter=400, lr=0.05, balanced)", "threshold": thr,
         "val_balanced_accuracy": float(ba(y[va], (s_va >= thr).astype(int))), "n_features": int(f_tr.shape[1]),
         "test_balanced_accuracy": summary}, indent=2))


if __name__ == "__main__":
    main()
