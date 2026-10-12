#!/usr/bin/env python
"""Model-free diagnostic: how much broadband envelope each ECG operator removes.

For the PTB-XL test fold, measures on the reconstructed signals:
  residual_slope       slope of log|X| vs log f over 1-45 Hz (0 = fully flat)
  band_envelope_sd     SD of the filter-bank band means of log|X| (coarse envelope)
  harmonic_prominence  median (harmonic-bin minus inter-harmonic-bin) log|X| over
                       1-25 Hz, i.e. preservation of the cardiac harmonic comb
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ptbxl_revision_interventions as rev  # noqa: E402


def measures(sig, sfreq, f0):
    freqs = np.fft.rfftfreq(sig.shape[-1], d=1.0 / sfreq)
    mask = (freqs >= 1.0) & (freqs <= 45.0)
    bf = freqs[mask]
    la = np.log(np.maximum(np.abs(np.fft.rfft(sig, axis=-1))[:, :, mask], 1e-12))
    design = np.column_stack([np.ones_like(bf), -np.log(bf)])
    coef = np.linalg.pinv(design) @ la.reshape(-1, la.shape[-1]).T
    slope = coef[1].reshape(la.shape[:2])
    env = rev.filterbank_envelope(la, bf)
    edges = rev.FILTERBANK_EDGES
    band_means = np.stack([env[..., np.argmax(bf >= lo)] for lo in edges[:-1]], -1)
    band_sd = band_means.std(-1)
    low = bf <= 25.0
    keep = rev.harmonic_keep_mask(bf[low], f0, 0.25)
    near = rev.harmonic_keep_mask(bf[low], f0, 0.10) == False  # noqa: E712  (within 0.1 f0 of a harmonic)
    prom = np.empty(la.shape[:2])
    for r in range(la.shape[0]):
        a = la[r][:, low]
        prom[r] = a[:, near[r]].mean(-1) - a[:, keep[r]].mean(-1)
    return slope, band_sd, prom


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-npz", type=Path, required=True)
    p.add_argument("--psd-npz", type=Path, required=True)
    p.add_argument("--specparam-npz", type=Path, required=True, help="fit_ptbxl_specparam_test.py output")
    p.add_argument("--output-csv", type=Path, required=True)
    args = p.parse_args()
    data = np.load(args.raw_npz, allow_pickle=True)
    psd = np.load(args.psd_npz, allow_pickle=True)
    test = np.flatnonzero(data["strat_fold"] == 10)
    x = data["x"][test].astype("float32")
    sfreq = float(data["sfreq"])
    sp_ap, sp_freqs = rev.load_specparam_test(args.specparam_npz, test)
    conds, diag = rev.revision_interventions(x, sfreq, sp_ap, sp_freqs, 0.25)
    f0 = diag["f0_hz"]
    rows = []
    for name in ["raw_eeg", "phase_sham", "phase_flattened", "flat_harmonic_masked", "flat_specparam", "flat_filterbank"]:
        slope, band_sd, prom = measures(conds[name], sfreq, f0)
        for metric, v in [("residual_slope", slope), ("band_envelope_sd", band_sd), ("harmonic_prominence", prom)]:
            v = v.ravel()
            rows.append({"condition": name, "metric": metric, "median": float(np.median(v)),
                         "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75))})
    for k in ["r2_linear_allbins", "r2_linear_interharmonic", "r2_harmonic_masked_interharmonic"]:
        v = diag[k].ravel()
        rows.append({"condition": "fit_quality", "metric": k, "median": float(np.median(v)),
                     "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75))})
    for metric, v in [("specparam_r2", np.load(args.specparam_npz)["r_squared"].ravel()),
                      ("peak_free_log_fit_r2", psd["r_squared"][test].ravel())]:
        rows.append({"condition": "fit_quality", "metric": metric, "median": float(np.nanmedian(v)),
                     "q25": float(np.nanpercentile(v, 25)), "q75": float(np.nanpercentile(v, 75))})
    rows.append({"condition": "fit_quality", "metric": "heart_rate_bpm", "median": float(np.median(60 * f0)),
                 "q25": float(np.percentile(60 * f0, 25)), "q75": float(np.percentile(60 * f0, 75))})
    out = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
