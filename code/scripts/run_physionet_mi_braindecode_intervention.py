#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_sleep_edf_braindecode_eegnet_intervention import (  # noqa: E402
    TEST_INPUTS,
    append_subject_metric_rows,
    balanced_accuracy,
    encode_labels,
    fit_predict_fold,
    macro_f1,
    make_phase_preserving_inputs,
    make_subject_folds,
    summarize_rows,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Braindecode raw EEG models on PhysioNet MI trials and evaluate "
            "phase-preserving aperiodic interventions on held-out subjects."
        )
    )
    parser.add_argument(
        "--raw-npz",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/results/physionet_mi/imagined_fists_raw_160hz.npz"),
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/results/physionet_mi/imagined_fists_raw_index.csv"),
    )
    parser.add_argument(
        "--decomp-npz",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/results/physionet_mi/specparam/imagined_fists_specparam_fixed.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/results/physionet_mi/raw_braindecode_interventions"),
    )
    parser.add_argument("--label-column", default="condition")
    parser.add_argument("--group-column", default="subject")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--band-min", type=float, default=2.0)
    parser.add_argument("--band-max", type=float, default=45.0)
    parser.add_argument("--f1", type=int, default=8)
    parser.add_argument("--depth-multiplier", type=int, default=2)
    parser.add_argument("--kernel-length", type=int, default=64)
    parser.add_argument("--depthwise-kernel-length", type=int, default=16)
    parser.add_argument(
        "--model",
        choices=["eegnet", "shallow_fbcsp", "deep4", "usleep", "eegconformer"],
        default="eegnet",
    )
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--no-match-rms", action="store_true")
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import braindecode
    import numpy as np
    import pandas as pd
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    raw = np.load(args.raw_npz)
    decomp = np.load(args.decomp_npz)
    x = raw["x"].astype("float32", copy=False)
    ap_log_psd = decomp["aperiodic_log_psd"].astype("float32", copy=False)
    decomp_freqs = decomp["freqs"].astype("float32", copy=False)
    index = pd.read_csv(args.index_csv)

    if args.max_trials is not None:
        x = x[: args.max_trials]
        ap_log_psd = ap_log_psd[: args.max_trials]
        index = index.iloc[: args.max_trials].copy()
    if len(index) != len(x):
        raise ValueError(f"Index rows ({len(index)}) do not match raw trials ({len(x)})")
    if len(ap_log_psd) < len(x):
        raise ValueError(f"Specparam rows ({len(ap_log_psd)}) do not cover raw trials ({len(x)})")

    labels = index[args.label_column].astype(str).to_numpy()
    groups = index[args.group_column].astype(str).to_numpy()
    mask, y, classes = encode_labels(labels)
    task_x = x[mask]
    task_ap = ap_log_psd[mask]
    task_groups = groups[mask]
    n_classes = len(classes)
    folds = make_subject_folds(task_groups, args.n_splits)
    sfreq = float(raw["sfreq"])
    match_rms = not args.no_match_rms

    fold_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    train_rows: list[dict[str, object]] = []
    task_name = "imagined_left_vs_right_fist"

    for fold_idx, (train_idx, test_idx, test_subjects) in enumerate(folds):
        if len(train_idx) == 0 or len(test_idx) == 0:
            print(f"Skipping fold {fold_idx}: empty train/test split")
            continue
        train_subjects = sorted(str(s) for s in np.unique(task_groups[train_idx]))
        edited = make_phase_preserving_inputs(
            task_x[test_idx],
            task_ap[test_idx],
            decomp_freqs,
            sfreq=sfreq,
            band_min=args.band_min,
            band_max=args.band_max,
            match_rms=match_rms,
        )
        test_inputs = {"raw_eeg": task_x[test_idx], **edited}
        predictions, meta = fit_predict_fold(
            task_x[train_idx],
            y[train_idx],
            test_inputs,
            sfreq=sfreq,
            args=args,
            n_classes=n_classes,
            fold_seed=args.seed + fold_idx,
        )
        y_test = y[test_idx]
        train_rows.append(
            {
                "task": task_name,
                "seed": int(args.seed),
                "fold": fold_idx,
                "classes": "|".join(classes),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "train_subjects": ",".join(train_subjects),
                "test_subjects": ",".join(test_subjects),
                **meta,
            }
        )
        for test_input in TEST_INPUTS:
            pred = predictions[test_input]
            base_row = {
                "task": task_name,
                "seed": int(args.seed),
                "train_input": f"braindecode_{args.model}_raw_eeg",
                "test_input": test_input,
                "fold": fold_idx,
                "classes": "|".join(classes),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "train_subjects": ",".join(train_subjects),
                "test_subjects": ",".join(test_subjects),
            }
            fold_rows.append(
                {
                    **base_row,
                    "balanced_accuracy": balanced_accuracy(y_test, pred, n_classes),
                    "macro_f1": macro_f1(y_test, pred, n_classes),
                    "accuracy": float(np.mean(y_test == pred)),
                }
            )
            append_subject_metric_rows(
                subject_rows,
                base_row,
                task_groups[test_idx],
                y_test,
                pred,
                n_classes,
            )
        print(f"{args.model} fold {fold_idx} done", flush=True)

    summary_rows = summarize_rows(fold_rows)
    output_prefix = args.output_prefix or f"braindecode_{args.model}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / f"{output_prefix}_intervention_fold_metrics.csv", fold_rows)
    write_csv(args.output_dir / f"{output_prefix}_intervention_subject_metrics.csv", subject_rows)
    write_csv(args.output_dir / f"{output_prefix}_intervention_summary_metrics.csv", summary_rows)
    write_csv(args.output_dir / f"{output_prefix}_intervention_training_log.csv", train_rows)

    channels = raw["channels"] if "channels" in raw else np.array([])
    edf_channels = raw["edf_channels"] if "edf_channels" in raw else channels
    if len(channels) != x.shape[1] and len(edf_channels) == x.shape[1]:
        channels = edf_channels
    metadata = {
        "raw_npz": str(args.raw_npz),
        "index_csv": str(args.index_csv),
        "decomp_npz": str(args.decomp_npz),
        "raw_shape": list(x.shape),
        "sfreq": sfreq,
        "channels": channels.tolist(),
        "model": f"braindecode:{args.model}",
        "braindecode_version": braindecode.__version__,
        "task": task_name,
        "classes": classes,
        "train_input": "raw_eeg",
        "test_inputs": TEST_INPUTS,
        "intervention": "phase-preserving FFT amplitude edits",
        "band_min": float(args.band_min),
        "band_max": float(args.band_max),
        "match_rms": bool(match_rms),
        "n_splits": int(args.n_splits),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "dropout": float(args.dropout),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "F1": int(args.f1),
        "D": int(args.depth_multiplier),
        "F2": int(args.f1 * args.depth_multiplier),
        "kernel_length": int(args.kernel_length),
        "depthwise_kernel_length": int(args.depthwise_kernel_length),
        "device_requested": args.device,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (args.output_dir / f"{output_prefix}_intervention_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    direct = [
        row
        for row in summary_rows
        if not str(row["test_input"]).startswith(("retention::", "drop::"))
    ]
    print(
        pd.DataFrame(direct)[
            ["task", "test_input", "balanced_accuracy_mean", "macro_f1_mean", "accuracy_mean"]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
