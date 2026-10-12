#!/usr/bin/env python
"""Window-level sham integrity for every raw-signal model.

For each saved prediction file, compares raw and sham-reconstructed inputs at
the level of individual windows/records: decision agreement, and (when scores
are stored) mean absolute probability change, probability correlation and the
spread of sham probabilities (a constant output indicates a broken input path).
"""
from __future__ import annotations

import argparse
import re
import os
from pathlib import Path

import numpy as np
import pandas as pd

M = Path(os.environ["RESULTS_DIR"])


def discover(extra):
    files = []
    pats = [
        ("TUAB", "conventional", "tuab_full_v3_0_1/multiseed_neural/*_seed*/*_predictions.csv"),
        ("TUAB", "foundation", "tuab_full_v3_0_1/*_full/tuab_*_intervention_predictions.csv"),
        ("TUAB", "foundation", "tuab_full_v3_0_1/foundation_multiseed/*_seed*/tuab_*_intervention_predictions.csv"),
        ("PTB-XL", "conventional", "ptbxl_1f_demo/*_seed*/ptbxl_raw_cnn_intervention_predictions.csv"),
        ("PTB-XL matched", "conventional", "ptbxl_1f_demo/age_sex_matched_neural/*_seed*/ptbxl_raw_cnn_intervention_predictions.csv"),
    ]
    for dataset, family, pat in pats:
        for f in sorted(M.glob(pat)):
            files.append((dataset, family, f, "submitted"))
    for item in extra:
        dataset, family, path = item.split("|")
        files.append((dataset, family, Path(path), "revision_fixed"))
    return files


def model_name(path: Path):
    m = re.search(r"tuab_(\w+?)_intervention", path.name)
    if m and m.group(1) not in ("full",):
        return m.group(1)
    parent = path.parent.name
    return re.sub(r"_seed\d+$", "", parent).replace("tuab_full_", "")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extra", action="append", default=[], help="dataset|family|path for revision runs")
    p.add_argument("--output-csv", type=Path, required=True)
    args = p.parse_args()
    rows = []
    for dataset, family, f, version in discover(args.extra):
        d = pd.read_csv(f)
        key = "row_index" if "row_index" in d.columns else "record_index"
        raw = d[d.test_input == "raw_eeg"].set_index(key)
        sham = d[d.test_input == "phase_sham"].set_index(key).reindex(raw.index)
        score = next((c for c in ["prob_abnormal", "score_abnormal"] if c in d.columns), None)
        row = {"dataset": dataset, "family": family, "model": model_name(f),
               "run": f.parent.name, "version": version, "n_windows": int(len(raw)),
               "decision_agreement": float((raw.y_pred.to_numpy() == sham.y_pred.to_numpy()).mean())}
        if score:
            a, b = raw[score].to_numpy(float), sham[score].to_numpy(float)
            row.update({"prob_mean_abs_change": float(np.abs(a - b).mean()),
                        "prob_corr": float(np.corrcoef(a, b)[0, 1]) if b.std() > 0 else float("nan"),
                        "raw_prob_std": float(a.std()), "sham_prob_std": float(b.std())})
        rows.append(row)
    out = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    pd.set_option("display.width", 200)
    print(out.drop(columns=["run"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
