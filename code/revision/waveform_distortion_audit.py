#!/usr/bin/env python
"""Time-domain effect of the spectral interventions.

Applies the exact operators used for the reported results to a random sample
of windows from each dataset and quantifies local temporal morphology change:

  pearson_r            waveform correlation, edited vs. original
  p2p_ratio            peak-to-peak amplitude, edited / original
  transient_amp_ratio  |edited| / |original| at the largest-|x| sample (main transient)
  transient_shift_ms   latency shift of that transient (argmax |edited| within +/-100 ms)
  ECG only (lead II):  R-peak timing preservation (+/-20 ms), R-wave amplitude ratio,
                       median-beat correlation

Example traces are saved for the overlay figure.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import zipfile
import os
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_sleep_edf_braindecode_eegnet_intervention as eeg_op  # noqa: E402
import run_ptbxl_raw_cnn_interventions as ecg_op  # noqa: E402

M = Path(os.environ["RESULTS_DIR"])
CONDS = ["phase_sham", "phase_aperiodic", "phase_flattened"]


def npz_member_memmap(path: Path, member: str):
    zf = zipfile.ZipFile(path)
    info = zf.getinfo(member)
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"{member} is compressed; cannot memmap")
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        hdr = f.read(30)
        n, m = struct.unpack("<HH", hdr[26:30])
        f.seek(info.header_offset + 30 + n + m)
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        offset = f.tell()
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=offset, order="F" if fortran else "C")


def generic_metrics(x, e, sfreq):
    """x, e: (n, ch, t). Returns dict of (n, ch) arrays."""
    xc = x - x.mean(-1, keepdims=True)
    ec = e - e.mean(-1, keepdims=True)
    r = (xc * ec).sum(-1) / np.maximum(np.sqrt((xc ** 2).sum(-1) * (ec ** 2).sum(-1)), 1e-12)
    p2p = np.ptp(e, axis=-1) / np.maximum(np.ptp(x, axis=-1), 1e-12)
    t_star = np.abs(xc).argmax(-1)
    amp_x = np.take_along_axis(np.abs(xc), t_star[..., None], -1)[..., 0]
    amp_e = np.take_along_axis(np.abs(ec), t_star[..., None], -1)[..., 0]
    w = int(round(0.1 * sfreq))
    shift = np.empty(t_star.shape)
    T = x.shape[-1]
    for idx in np.ndindex(t_star.shape):
        lo, hi = max(0, t_star[idx] - w), min(T, t_star[idx] + w + 1)
        shift[idx] = (lo + np.abs(ec[idx][lo:hi]).argmax() - t_star[idx]) * 1000.0 / sfreq
    return {"pearson_r": r, "p2p_ratio": p2p, "transient_amp_ratio": amp_e / np.maximum(amp_x, 1e-12),
            "transient_shift_ms": shift}


def detect_r_peaks(sig, sfreq):
    from scipy.signal import butter, filtfilt, find_peaks
    b, a = butter(2, [5 / (sfreq / 2), 15 / (sfreq / 2)], btype="band")
    f = filtfilt(b, a, sig)
    env = np.convolve(f ** 2, np.ones(int(0.08 * sfreq)) / int(0.08 * sfreq), mode="same")
    pk, _ = find_peaks(env, distance=int(0.33 * sfreq), height=0.3 * np.percentile(env, 99))
    # refine to the local extremum of the original signal (+/-50 ms)
    w = int(0.05 * sfreq)
    out = []
    for p in pk:
        lo, hi = max(0, p - w), min(len(sig), p + w + 1)
        out.append(lo + np.abs(sig[lo:hi] - np.median(sig)).argmax())
    return np.array(out, dtype=int)


def ecg_metrics(x, e, sfreq, lead=1):
    rows = []
    tol = int(round(0.02 * sfreq))
    half = int(round(0.3 * sfreq))
    for i in range(x.shape[0]):
        xo, eo = x[i, lead], e[i, lead]
        ro, re_ = detect_r_peaks(xo, sfreq), detect_r_peaks(eo, sfreq)
        if len(ro) < 3:
            continue
        matched = [np.min(np.abs(re_ - p)) <= tol if len(re_) else False for p in ro]
        amp_ratio = np.median(np.abs(eo[ro] - np.median(eo)) / np.maximum(np.abs(xo[ro] - np.median(xo)), 1e-9))
        beats_o = [xo[p - half:p + half] for p in ro if p - half >= 0 and p + half <= len(xo)]
        beats_e = [eo[p - half:p + half] for p in ro if p - half >= 0 and p + half <= len(eo)]
        mb_r = np.nan
        if len(beats_o) >= 2:
            mo, me = np.median(beats_o, 0), np.median(beats_e, 0)
            mb_r = np.corrcoef(mo, me)[0, 1]
        rows.append({"r_peak_preserved_frac": float(np.mean(matched)), "r_amp_ratio": float(amp_ratio),
                     "median_beat_r": float(mb_r), "n_beats": int(len(ro))})
    return pd.DataFrame(rows)


def load_dataset(name, n, rng, tuab_specparam_eval):
    if name == "sleep_edf":
        raw = np.load(M / "sleep_edf_full/raw_epochs_fpz_pz_100hz.npz")
        sp = np.load(M / "sleep_edf_full/specparam/specparam_fixed.npz")
        idx_df = pd.read_csv(M / "sleep_edf_full/raw_epochs_index.csv", usecols=["stage"])
        x_all = raw["x"]
        rows = np.sort(rng.choice(len(x_all), size=n, replace=False))
        return dict(x=x_all[rows].astype("float32"), ap=sp["aperiodic_log_psd"][rows], f=sp["freqs"],
                    sfreq=float(raw["sfreq"]), band=(1.0, 45.0), op="eeg", meta=idx_df["stage"].to_numpy()[rows],
                    channels=list(raw["channels"]) if "channels" in raw.files else ["Fpz-Cz", "Pz-Oz"])
    if name == "tuab":
        P = M / "tuab_full_v3_0_1/preprocess_20s_100hz/raw_epochs_20s_100hz.npz"
        x_all = npz_member_memmap(P, "x.npy")
        sp = np.load(tuab_specparam_eval)
        eval_rows = sp["raw_row_index"]
        pick = np.sort(rng.choice(len(eval_rows), size=n, replace=False))
        lab = pd.read_csv(M / "tuab_full_v3_0_1/preprocess_20s_100hz/raw_epochs_20s_100hz_index.csv",
                          usecols=["label"])["label"].to_numpy()
        return dict(x=np.asarray(x_all[eval_rows[pick]], dtype="float32"), ap=sp["aperiodic_log_psd"][pick],
                    f=sp["freqs"], sfreq=100.0, band=(1.0, 45.0), op="eeg", meta=lab[eval_rows[pick]],
                    channels=list(np.load(P)["channels"]))
    if name == "physionet_mi":
        raw = np.load(M / "physionet_mi/imagined_fists_raw_160hz.npz", allow_pickle=True)
        sp = np.load(M / "physionet_mi/specparam/imagined_fists_specparam_fixed.npz", allow_pickle=True)
        x_all = raw["x"]
        rows = np.sort(rng.choice(len(x_all), size=min(n, len(x_all)), replace=False))
        return dict(x=x_all[rows].astype("float32"), ap=sp["aperiodic_log_psd"][rows], f=sp["freqs"],
                    sfreq=float(raw["sfreq"]), band=(2.0, 45.0), op="eeg", meta=None,
                    channels=[str(c) for c in raw["channels"]] if "channels" in raw.files else None)
    if name == "ptbxl":
        raw = np.load(M / "ptbxl_1f_demo/ptbxl_records100_normal_abnormal_raw.npz", allow_pickle=True)
        test = np.flatnonzero(raw["strat_fold"] == 10)
        rows = np.sort(rng.choice(test, size=min(n, len(test)), replace=False))
        return dict(x=raw["x"][rows].astype("float32"), sfreq=float(raw["sfreq"]), op="ecg",
                    meta=raw["y"][rows], channels=[str(c) for c in raw["leads"]])
    raise ValueError(name)


def apply_op(d):
    if d["op"] == "eeg":
        return eeg_op.make_phase_preserving_inputs(d["x"], d["ap"], d["f"], d["sfreq"], d["band"][0],
                                                   d["band"][1], match_rms=True)
    out = ecg_op.intervention_batch(d["x"], d["sfreq"])
    return {c: out[c] for c in CONDS}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--tuab-specparam-eval", type=Path, required=True)
    p.add_argument("--n-windows", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20261004)
    p.add_argument("--datasets", default="sleep_edf,tuab,physionet_mi,ptbxl")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summary_rows, examples = [], {}
    for ds in args.datasets.split(","):
        d = load_dataset(ds, args.n_windows, rng, args.tuab_specparam_eval)
        edited = apply_op(d)
        per = []
        for c in CONDS:
            m = generic_metrics(d["x"], edited[c], d["sfreq"])
            for k, v in m.items():
                v = v.ravel()
                summary_rows.append({"dataset": ds, "condition": c, "metric": k, "median": float(np.median(v)),
                                     "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)),
                                     "p5": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)),
                                     "mean_abs_dev_from_identity": float(np.mean(np.abs(v - (0.0 if k == "transient_shift_ms" else 1.0)))),
                                     "n": int(v.size)})
            if d["op"] == "ecg":
                e = ecg_metrics(d["x"], edited[c], d["sfreq"])
                for k in ["r_peak_preserved_frac", "r_amp_ratio", "median_beat_r"]:
                    v = e[k].dropna().to_numpy()
                    summary_rows.append({"dataset": ds, "condition": c, "metric": k, "median": float(np.median(v)),
                                         "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)),
                                         "p5": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)),
                                         "mean_abs_dev_from_identity": float(np.mean(np.abs(v - 1.0))),
                                         "n": int(v.size)})
        # example: the window/channel closest to the median flattened-vs-original correlation
        r_flat = generic_metrics(d["x"], edited["phase_flattened"], d["sfreq"])["pearson_r"]
        ch = 1 if d["op"] == "ecg" else 0
        wi = int(np.argmin(np.abs(r_flat[:, ch] - np.median(r_flat[:, ch]))))
        examples[ds] = {"sfreq": d["sfreq"], "channel": (d["channels"] or ["ch0", "ch1"])[ch],
                        "original": d["x"][wi, ch], **{c: edited[c][wi, ch] for c in CONDS},
                        "r_flattened": float(r_flat[wi, ch]),
                        "meta": None if d["meta"] is None else str(d["meta"][wi])}
        print(f"{ds}: done ({d['x'].shape})", flush=True)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "waveform_distortion_summary.csv", index=False)
    np.savez(args.output_dir / "overlay_examples.npz",
             **{f"{ds}__{k}": np.asarray(v) for ds, ex in examples.items() for k, v in ex.items() if v is not None})
    (args.output_dir / "overlay_examples_meta.json").write_text(json.dumps(
        {ds: {k: v for k, v in ex.items() if not isinstance(v, np.ndarray)} for ds, ex in examples.items()}, indent=2))
    show = summary[summary.metric.isin(["pearson_r", "p2p_ratio", "transient_amp_ratio", "transient_shift_ms",
                                        "r_peak_preserved_frac", "r_amp_ratio", "median_beat_r"])]
    print(show[["dataset", "condition", "metric", "median", "q25", "q75"]].to_string(index=False))


if __name__ == "__main__":
    main()
