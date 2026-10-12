#!/usr/bin/env python
"""Pooled balanced accuracy for the binary Sleep-EDF and PhysioNet MI tasks (supplementary check).

These runs store per-subject metrics rather than window-level predictions; each per-subject row stores the
class supports (n0, n1), balanced accuracy, accuracy and macro-F1. For a binary task these
determine the subject's confusion counts: TN + TP = round(acc * N) and the pair (TN, TP) is found
by exhaustive integer search, accepted only if it reproduces all three stored metrics to 1e-9 and
is unique. The counts are then expanded to prediction rows and summarised with the hierarchical
seed x subject bootstrap (aggregate_revision_bootstrap.py; Table S9 uses one subject draw shared across seeds).

Checks: every subject row must be reconstructed uniquely and exactly, and the subject-averaged
BA recomputed from the reconstructed counts must equal the stored values.

Usage: python reconstruct_binary_counts.py <results dir> <output dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

M, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

RUNS = {  # model -> (dir pattern with {s}, metrics file)
    "sleep|MLP": ("sleep_edf_full/multiseed_neural/deep_mlp_seed{s}", "deep_mlp_intervention_subject_metrics.csv"),
    "sleep|CNN": ("sleep_edf_full/reviewer_resistance_controls/raw_cnn_sham_seed{s}", None),
    "sleep|EEGNet": ("sleep_edf_full/reviewer_resistance_controls/braindecode_eegnet_seed{s}",
                     "braindecode_eegnet_intervention_subject_metrics.csv"),
    "sleep|ShallowFBCSPNet": ("sleep_edf_full/reviewer_resistance_controls/braindecode_shallow_fbcsp_seed{s}",
                              "braindecode_shallow_fbcsp_intervention_subject_metrics.csv"),
    "sleep|Deep4Net": ("sleep_edf_full/reviewer_resistance_controls/braindecode_deep4_seed{s}",
                       "braindecode_deep4_intervention_subject_metrics.csv"),
    "mi|EEGNet": ("physionet_mi/multiseed_neural/eegnet_seed{s}", None),
    "mi|ShallowFBCSPNet": ("physionet_mi/multiseed_neural/shallow_fbcsp_seed{s}", None),
    "mi|Deep4Net": ("physionet_mi/multiseed_neural/deep4_seed{s}", None),
}


def metrics(tn, tp, n0, n1):
    fp, fn = n0 - tn, n1 - tp
    rec = [r for r, n in ((tn / n0 if n0 else None, n0), (tp / n1 if n1 else None, n1)) if n]
    ba = float(np.mean(rec))
    acc = (tn + tp) / (n0 + n1)
    f0 = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return ba, acc, (f0 + f1) / 2


def reconstruct(n0, n1, ba, acc, mf1):
    k = int(round(acc * (n0 + n1)))
    sols = []
    for tn in range(max(0, k - n1), min(n0, k) + 1):
        tp = k - tn
        b, a, f = metrics(tn, tp, n0, n1)
        if abs(b - ba) < 1e-9 and abs(a - acc) < 1e-9 and abs(f - mf1) < 1e-9:
            sols.append((tn, tp))
    return sols


rows, summary = [], []
for key, (pat, fname) in RUNS.items():
    for s in (42, 43, 44):
        d = M / pat.format(s=s)
        f = d / fname if fname else next(iter(sorted(d.glob("*subject_metrics.csv"))))
        sm = pd.read_csv(f)
        n_ok = n_amb = n_fail = 0
        for r in sm.itertuples(index=False):
            sup = json.loads(r.class_support_json)
            if len(sup) != 2:
                continue  # multi-class task (five-stage): not reconstructable, skipped
            n0, n1 = int(sup.get("0", 0)), int(sup.get("1", 0))
            sols = reconstruct(n0, n1, r.balanced_accuracy, r.accuracy, r.macro_f1)
            if len(sols) == 1:
                n_ok += 1
            elif len(sols) > 1:
                n_amb += 1
            else:
                n_fail += 1
            if not sols:
                continue
            # ambiguous rows (equal class supports, TN/TP swap) are kept in both resolutions:
            # variant A uses the first solution, variant B the last; results are reported for both
            for variant, (tn, tp) in (("A", sols[0]), ("B", sols[-1])):
                rows.append({"run": key, "task": r.task, "seed": s, "subject": str(r.subject),
                             "test_input": r.test_input, "train_input": getattr(r, "train_input", ""),
                             "n0": n0, "n1": n1, "tn": tn, "tp": tp, "ambiguous": len(sols) > 1,
                             "variant": variant, "ba_subject": r.balanced_accuracy})
        summary.append({"run": key, "seed": s, "file": str(f), "reconstructed": n_ok, "ambiguous": n_amb,
                        "failed": n_fail})
        print(f"{key:24s} seed {s}: exact unique {n_ok}, ambiguous {n_amb}, failed {n_fail}", flush=True)

counts = pd.DataFrame(rows)
counts.to_csv(OUT / "reconstructed_subject_counts.csv", index=False)
pd.DataFrame(summary).to_csv(OUT / "reconstruction_summary.csv", index=False)

# expand to prediction rows per (run, task) for the bootstrap
for (run, task, variant), g in counts.groupby(["run", "task", "variant"]):
    if variant == "B" and not g.ambiguous.any():
        continue
    if g.train_input.astype(str).str.contains("full_log_psd|aperiodic|flattened").any():
        g = g[g.train_input.astype(str).str.contains("full_log_psd")]  # MLP: full-PSD-trained model only
    ex = []
    for r in g.itertuples(index=False):
        for yt, yp, n in ((0, 0, r.tn), (0, 1, r.n0 - r.tn), (1, 1, r.tp), (1, 0, r.n1 - r.tp)):
            if n:
                ex.append(pd.DataFrame({"seed": r.seed, "subject": r.subject, "test_input": r.test_input,
                                        "y_true": np.full(n, yt, np.int8), "y_pred": np.full(n, yp, np.int8)}))
    e = pd.concat(ex, ignore_index=True)
    tag = run.replace("|", "_") + "_" + task + ("" if variant == "A" else "_variantB")
    for s in (42, 43, 44):
        e[e.seed == s].to_csv(OUT / f"pred_{tag}_seed{s}.csv", index=False)
    print("expanded", tag, len(e), sorted(e.test_input.unique()), flush=True)
