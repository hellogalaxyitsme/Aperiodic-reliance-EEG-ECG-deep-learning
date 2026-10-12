#!/usr/bin/env python
"""Decompose the (small) deviation of the foundation-model sham from identity.

The sham operator = FFT -> identical amplitude/phase -> iFFT -> per-window mean
removal -> std matching. On zero-mean inputs it is the identity. Foundation-model
inputs (LaBraM/BENDR 0.1-75 Hz cache, BIOT quantile-normalised cache) are not
mean-centred per window, so the only effective change is DC removal (plus the
std rescale that follows it). This script measures, on 2,000 random eval windows:
  dc_fraction            |window mean| / window std
  sham_rel_err           ||sham - x|| / ||x||
  sham_rel_err_vs_dc_removed  ||sham - std_match(x - mean(x))|| / ||x||  (should be ~0)
"""
import json
import sys
import os
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_tuab_biot_intervention as fm  # noqa: E402

R = Path(os.environ["REVISION_DIR"])
M = Path(os.environ["RESULTS_DIR"]) / "tuab_full_v3_0_1"
caches = {"LaBraM/BENDR cache (uV, 200 Hz)": R / "results/tuab_full/labram_10s_200hz_cache",
          "BIOT cache (quantile-normalised, 200 Hz)": M / "biot_10s_200hz_cache"}
rng = np.random.default_rng(7)
out = {}
for name, stem in caches.items():
    x_all = np.load(str(stem) + ".x.npy", mmap_mode="r")
    splits = np.load(str(stem) + ".splits.npy", allow_pickle=True).astype(str)
    ev = np.flatnonzero(splits == "eval")
    rows = np.sort(rng.choice(ev, size=2000, replace=False))
    x = np.asarray(x_all[rows], dtype="float32")
    if "BENDR" in name:
        x = np.clip(x, -500, 500)
    sham = fm.make_phase_preserving_inputs(x, sfreq=200.0, band_min=1.0, band_max=45.0)["phase_sham"]
    xc = x - x.mean(-1, keepdims=True)
    xc = xc * (x.std(-1, keepdims=True) / np.maximum(xc.std(-1, keepdims=True), 1e-6))
    norm = np.linalg.norm(x, axis=-1)
    dc = np.abs(x.mean(-1)) / np.maximum(x.std(-1), 1e-6)
    out[name] = {
        "dc_fraction_median": float(np.median(dc)),
        "dc_fraction_p90": float(np.percentile(dc, 90)),
        "sham_rel_err_median": float(np.median(np.linalg.norm(sham - x, axis=-1) / np.maximum(norm, 1e-9))),
        "sham_rel_err_vs_dc_removed_median": float(np.median(np.linalg.norm(sham - xc, axis=-1) / np.maximum(norm, 1e-9))),
        "sham_rel_err_vs_dc_removed_max": float(np.max(np.linalg.norm(sham - xc, axis=-1) / np.maximum(norm, 1e-9))),
    }
print(json.dumps(out, indent=2))
(R / "results/sham_integrity").mkdir(parents=True, exist_ok=True)
(R / "results/sham_integrity/fm_sham_dc_check.json").write_text(json.dumps(out, indent=2))
