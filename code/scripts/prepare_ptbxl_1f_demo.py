#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_SRC))

from aperiodic_eeg.spectral import fit_fixed_aperiodic  # noqa: E402


LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare PTB-XL records100 normal-vs-abnormal 1/f demo arrays."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/data/ptbxl/1.0.3"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/results/ptbxl_1f_demo"),
    )
    parser.add_argument("--sfreq", type=float, default=100.0)
    parser.add_argument("--low-hz", type=float, default=0.5)
    parser.add_argument("--high-hz", type=float, default=40.0)
    parser.add_argument("--psd-fmin", type=float, default=1.0)
    parser.add_argument("--psd-fmax", type=float, default=45.0)
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args()


def parse_signal_field(field: str):
    # WFDB gain/baseline/units examples: 1000.0(0)/mV
    match = re.match(r"(?P<gain>[-+0-9.]+)(?:\((?P<baseline>[-+0-9.]+)\))?(?:/(?P<unit>.+))?", field)
    if not match:
        return 1.0, 0.0, ""
    gain = float(match.group("gain") or 1.0)
    baseline = float(match.group("baseline") or 0.0)
    unit = match.group("unit") or ""
    return gain, baseline, unit


def read_wfdb_record(base_path: Path) -> np.ndarray:
    header = base_path.with_suffix(".hea")
    if not header.exists():
        raise FileNotFoundError(header)
    lines = header.read_text().strip().splitlines()
    first = lines[0].split()
    n_sig = int(first[1])
    n_samples = int(first[3])

    gains = []
    baselines = []
    units = []
    dat_name = None
    lead_names = []
    for line in lines[1 : 1 + n_sig]:
        parts = line.split()
        if dat_name is None:
            dat_name = parts[0]
        gain, baseline, unit = parse_signal_field(parts[2])
        gains.append(gain)
        baselines.append(baseline)
        units.append(unit)
        lead_names.append(parts[-1].upper())

    if dat_name is None:
        raise ValueError(f"No signal lines in {header}")
    raw = np.fromfile(base_path.parent / dat_name, dtype="<i2")
    raw = raw.reshape(n_samples, n_sig).astype("float32")
    gains = np.asarray(gains, dtype="float32")
    baselines = np.asarray(baselines, dtype="float32")
    values = (raw - baselines[None, :]) / np.maximum(gains[None, :], 1e-12)

    # Convert millivolts to microvolts for consistency with the EEG pipeline.
    unit_scale = np.asarray([1000.0 if unit.lower() == "mv" else 1.0 for unit in units], dtype="float32")
    values = values * unit_scale[None, :]

    if [name.upper() for name in lead_names] != LEADS:
        order = [lead_names.index(lead) for lead in LEADS]
        values = values[:, order]
    return values.T.astype("float32", copy=False)


def diagnostic_classes(scp_codes, scp_table: pd.DataFrame) -> set[str]:
    classes = set()
    for code, likelihood in scp_codes.items():
        try:
            likelihood = float(likelihood)
        except (TypeError, ValueError):
            likelihood = 0.0
        if likelihood <= 0 or code not in scp_table.index:
            continue
        item = scp_table.loc[code]
        if bool(item.get("diagnostic", 0)):
            diagnostic_class = str(item.get("diagnostic_class", "")).strip()
            if diagnostic_class and diagnostic_class.lower() != "nan":
                classes.add(diagnostic_class)
    return classes


def load_manifest(data_root: Path) -> pd.DataFrame:
    meta = pd.read_csv(data_root / "ptbxl_database.csv")
    scp = pd.read_csv(data_root / "scp_statements.csv", index_col=0)
    rows = []
    for _, row in meta.iterrows():
        codes = ast.literal_eval(row["scp_codes"])
        classes = diagnostic_classes(codes, scp)
        nonnorm = sorted(cls for cls in classes if cls != "NORM")
        if nonnorm:
            label = "abnormal"
        elif classes == {"NORM"}:
            label = "normal"
        else:
            continue
        rows.append(
            {
                "ecg_id": int(row["ecg_id"]),
                "patient_id": str(row["patient_id"]),
                "strat_fold": int(row["strat_fold"]),
                "age": row.get("age", ""),
                "sex": row.get("sex", ""),
                "filename_lr": str(row["filename_lr"]),
                "label": label,
                "diagnostic_classes": "|".join(sorted(classes)),
            }
        )
    return pd.DataFrame(rows)


def bandpass_filter(x: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
    sos = signal.butter(4, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1).astype("float32")


def compute_psd(x: np.ndarray, sfreq: float, fmin: float, fmax: float):
    freqs, psd = signal.welch(
        x,
        fs=sfreq,
        window="hann",
        nperseg=x.shape[-1],
        noverlap=0,
        axis=-1,
        scaling="density",
    )
    mask = (freqs >= fmin) & (freqs <= fmax)
    return freqs[mask].astype("float32"), psd[:, :, mask].astype("float32")


def write_index(path: Path, rows: list[dict[str, object]]):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.data_root)
    if args.max_records:
        manifest = manifest.head(args.max_records).copy()

    x_records = []
    out_rows = []
    for _, row in manifest.iterrows():
        base = args.data_root / row["filename_lr"]
        x = read_wfdb_record(base)
        x = bandpass_filter(x, args.sfreq, args.low_hz, args.high_hz)
        x = x - x.mean(axis=-1, keepdims=True)
        x_records.append(x)
        out_rows.append(row.to_dict())

    x = np.stack(x_records).astype("float32")
    y = np.asarray([1 if row["label"] == "abnormal" else 0 for row in out_rows], dtype="int64")
    patient = np.asarray([row["patient_id"] for row in out_rows], dtype=object)
    folds = np.asarray([row["strat_fold"] for row in out_rows], dtype="int64")

    freqs, psd = compute_psd(x, args.sfreq, args.psd_fmin, args.psd_fmax)
    fit = fit_fixed_aperiodic(psd, freqs)

    np.savez_compressed(
        args.output_dir / "ptbxl_records100_normal_abnormal_raw.npz",
        x=x,
        y=y,
        patient_id=patient,
        strat_fold=folds,
        sfreq=np.asarray(args.sfreq, dtype="float32"),
        leads=np.asarray(LEADS, dtype=object),
    )
    np.savez_compressed(
        args.output_dir / "ptbxl_records100_normal_abnormal_psd_fixed.npz",
        psd=psd,
        freqs=freqs,
        log_psd=fit.log_psd,
        aperiodic_log_psd=fit.fitted_log_psd,
        flattened_log_psd=fit.residual_log_psd,
        offset=fit.offset,
        exponent=fit.exponent,
        r_squared=fit.r_squared,
        y=y,
        patient_id=patient,
        strat_fold=folds,
    )
    write_index(args.output_dir / "ptbxl_records100_normal_abnormal_index.csv", out_rows)

    summary = {
        "n_records": int(len(out_rows)),
        "n_patients": int(len(set(patient.tolist()))),
        "n_normal": int((y == 0).sum()),
        "n_abnormal": int((y == 1).sum()),
        "fold_counts": {str(k): int(v) for k, v in pd.Series(folds).value_counts().sort_index().items()},
        "sfreq": float(args.sfreq),
        "filter_hz": [float(args.low_hz), float(args.high_hz)],
        "psd_frequency_hz": [float(freqs[0]), float(freqs[-1])],
        "specparam_fixed_mean_r2": float(np.mean(fit.r_squared)),
        "specparam_fixed_median_r2": float(np.median(fit.r_squared)),
        "specparam_fixed_p10_r2": float(np.quantile(fit.r_squared, 0.10)),
    }
    (args.output_dir / "ptbxl_records100_normal_abnormal_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
