#!/usr/bin/env python
"""SpecParam (fixed aperiodic mode, with peaks) for the PTB-XL test fold.

The original PTB-XL PSD file (ptbxl_records100_normal_abnormal_psd_fixed.npz)
holds a peak-free least-squares fit of natural-log power on log frequency
(aperiodic_eeg.spectral.fit_fixed_aperiodic), not a SpecParam fit. For the
SpecParam envelope variant, the test-fold periodograms in that file are
fitted here with SpecParam using the settings of the EEG analyses (fixed mode,
max 6 peaks, min peak height 0.1, peak threshold 2.0, peak width 0.5-8 Hz,
1-45 Hz) and the same per-spectrum routine as the TUAB fits
(fit_tuab_specparam_full.fit_one_spectrum). The stored aperiodic component is
log10 power.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fit_tuab_specparam_full as fitmod  # noqa: E402


def main():
    import numpy as np
    from joblib import Parallel, delayed

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--psd-npz", type=Path, required=True)
    p.add_argument("--output-npz", type=Path, required=True)
    p.add_argument("--n-jobs", type=int, default=24)
    args = p.parse_args()
    t0 = time.time()

    bundle = np.load(args.psd_npz, allow_pickle=True)
    test = np.flatnonzero(bundle["strat_fold"].astype(int) == 10)
    freqs = bundle["freqs"].astype("float64")
    freqs_fit = np.linspace(float(freqs[0]), float(freqs[-1]), len(freqs))
    sub = bundle["psd"][test].astype("float64")
    n_rec, n_lead, n_f = sub.shape
    settings = {"freq_min": 1.0, "freq_max": 45.0, "max_n_peaks": 6, "min_peak_height": 0.1,
                "peak_threshold": 2.0, "peak_width_min": 0.5, "peak_width_max": 8.0, "eps": 1e-30}
    results = Parallel(n_jobs=args.n_jobs, batch_size=128, verbose=5)(
        delayed(fitmod.fit_one_spectrum)(freqs_fit, s, settings) for s in sub.reshape(-1, n_f))
    ap = np.stack([r[2] for r in results]).reshape(n_rec, n_lead, n_f).astype("float32")
    expo = np.array([r[5] for r in results], dtype="float32").reshape(n_rec, n_lead)
    r2 = np.array([r[6] for r in results], dtype="float32").reshape(n_rec, n_lead)
    npk = np.array([r[8] for r in results], dtype="float32").reshape(n_rec, n_lead)
    ok = np.array([r[0] for r in results]).reshape(n_rec, n_lead)
    import specparam
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, aperiodic_log10_psd=ap, exponent=expo, r_squared=r2, n_peaks=npk, ok=ok,
             freqs=freqs_fit.astype("float32"), record_index=test)
    summary = {"n_records": int(n_rec), "n_leads": int(n_lead), "ok_fraction": float(ok.mean()),
               "median_r_squared": float(np.nanmedian(r2)), "p25_r_squared": float(np.nanpercentile(r2, 25)),
               "p75_r_squared": float(np.nanpercentile(r2, 75)), "median_exponent": float(np.nanmedian(expo)),
               "mean_n_peaks": float(np.nanmean(npk)),
               "original_peak_free_fit_median_r_squared": float(np.median(bundle["r_squared"][test])),
               "specparam_version": specparam.__version__, "settings": settings,
               "elapsed_sec": round(time.time() - t0, 1)}
    args.output_npz.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
