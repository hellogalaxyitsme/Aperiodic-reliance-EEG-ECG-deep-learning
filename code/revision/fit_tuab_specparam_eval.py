#!/usr/bin/env python
"""Refit full-TUAB SpecParam (fixed mode) for the official-eval windows only.

Raw-signal interventions are applied only to evaluation windows, so the
evaluation subset (18,417 windows x 21 channels) is fitted here with the
identical settings and the same per-spectrum routine
(``fit_tuab_specparam_full.fit_one_spectrum``) from the intact multitaper PSD
cache.
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
    import pandas as pd
    from joblib import Parallel, delayed

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--psd-npz", type=Path, required=True)
    p.add_argument("--psd-index-csv", type=Path, required=True)
    p.add_argument("--raw-index-csv", type=Path, required=True)
    p.add_argument("--output-npz", type=Path, required=True)
    p.add_argument("--n-jobs", type=int, default=24)
    args = p.parse_args()
    t0 = time.time()

    psd_index = pd.read_csv(args.psd_index_csv)
    raw_index = pd.read_csv(args.raw_index_csv)
    if len(psd_index) != len(raw_index):
        raise ValueError("PSD and raw index lengths differ")
    key_cols = [c for c in ["subject", "official_split", "label", "remote_rel_path", "onset_sec"]
                if c in psd_index.columns and c in raw_index.columns]
    if not (psd_index[key_cols].astype(str).values == raw_index[key_cols].astype(str).values).all():
        raise ValueError(f"PSD and raw index rows are not aligned on {key_cols}")
    eval_rows = np.flatnonzero(raw_index["official_split"].astype(str).to_numpy() == "eval")
    print(f"eval windows={len(eval_rows)} aligned on {key_cols} ({time.time() - t0:.0f}s)", flush=True)

    bundle = np.load(args.psd_npz)
    freqs = bundle["freqs"]
    psd = bundle["psd"]
    print(f"loaded psd {psd.shape} ({time.time() - t0:.0f}s)", flush=True)
    fmask = (freqs >= 1.0) & (freqs <= 45.0)
    fsel = freqs[fmask]
    freqs_fit = np.linspace(float(fsel[0]), float(fsel[-1]), int(fsel.shape[0]), dtype="float64")
    sub = psd[eval_rows][:, :, fmask]
    del psd
    n_ep, n_ch, n_f = sub.shape
    settings = {"freq_min": 1.0, "freq_max": 45.0, "max_n_peaks": 6, "min_peak_height": 0.1,
                "peak_threshold": 2.0, "peak_width_min": 0.5, "peak_width_max": 8.0, "eps": 1e-30}
    flat = sub.reshape(-1, n_f)
    results = Parallel(n_jobs=args.n_jobs, batch_size=256, verbose=5)(
        delayed(fitmod.fit_one_spectrum)(freqs_fit, s, settings) for s in flat)
    ap = np.stack([r[2] for r in results]).reshape(n_ep, n_ch, n_f).astype("float32")
    r2 = np.array([r[6] for r in results], dtype="float32").reshape(n_ep, n_ch)
    expo = np.array([r[5] for r in results], dtype="float32").reshape(n_ep, n_ch)
    ok = np.array([r[0] for r in results]).reshape(n_ep, n_ch)
    import specparam
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, aperiodic_log_psd=ap, r_squared=r2, exponent=expo, ok=ok,
             freqs=freqs_fit.astype("float32"), raw_row_index=eval_rows)
    summary = {"n_eval_windows": int(n_ep), "n_channels": int(n_ch), "n_freqs": int(n_f),
               "ok_fraction": float(ok.mean()), "median_r_squared": float(np.nanmedian(r2)),
               "p10_r_squared": float(np.nanpercentile(r2, 10)), "median_exponent": float(np.nanmedian(expo)),
               "specparam_version": specparam.__version__, "settings": settings,
               "elapsed_sec": round(time.time() - t0, 1)}
    args.output_npz.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
