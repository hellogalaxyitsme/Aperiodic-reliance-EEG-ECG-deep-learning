#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import types
from pathlib import Path


TEST_INPUTS = ["raw_eeg", "phase_sham", "phase_aperiodic", "phase_flattened"]
METRICS = ["balanced_accuracy", "macro_f1", "accuracy"]
LABRAM_CHANNELS = [
    "EEG FP1-REF",
    "EEG FP2-REF",
    "EEG F3-REF",
    "EEG F4-REF",
    "EEG C3-REF",
    "EEG C4-REF",
    "EEG P3-REF",
    "EEG P4-REF",
    "EEG O1-REF",
    "EEG O2-REF",
    "EEG F7-REF",
    "EEG F8-REF",
    "EEG T3-REF",
    "EEG T4-REF",
    "EEG T5-REF",
    "EEG T6-REF",
    "EEG A1-REF",
    "EEG A2-REF",
    "EEG FZ-REF",
    "EEG CZ-REF",
    "EEG PZ-REF",
    "EEG T1-REF",
    "EEG T2-REF",
]
LABRAM_COMPACT_CHANNELS = [name.split(" ")[-1].split("-")[0] for name in LABRAM_CHANNELS]
STANDARD_1020 = [
    "FP1",
    "FPZ",
    "FP2",
    "AF9",
    "AF7",
    "AF5",
    "AF3",
    "AF1",
    "AFZ",
    "AF2",
    "AF4",
    "AF6",
    "AF8",
    "AF10",
    "F9",
    "F7",
    "F5",
    "F3",
    "F1",
    "FZ",
    "F2",
    "F4",
    "F6",
    "F8",
    "F10",
    "FT9",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "FCZ",
    "FC2",
    "FC4",
    "FC6",
    "FT8",
    "FT10",
    "T9",
    "T7",
    "C5",
    "C3",
    "C1",
    "CZ",
    "C2",
    "C4",
    "C6",
    "T8",
    "T10",
    "TP9",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "CPZ",
    "CP2",
    "CP4",
    "CP6",
    "TP8",
    "TP10",
    "P9",
    "P7",
    "P5",
    "P3",
    "P1",
    "PZ",
    "P2",
    "P4",
    "P6",
    "P8",
    "P10",
    "PO9",
    "PO7",
    "PO5",
    "PO3",
    "PO1",
    "POZ",
    "PO2",
    "PO4",
    "PO6",
    "PO8",
    "PO10",
    "O1",
    "OZ",
    "O2",
    "O9",
    "CB1",
    "CB2",
    "IZ",
    "O10",
    "T3",
    "T5",
    "T4",
    "T6",
    "M1",
    "M2",
    "A1",
    "A2",
    "CFC1",
    "CFC2",
    "CFC3",
    "CFC4",
    "CFC5",
    "CFC6",
    "CFC7",
    "CFC8",
    "CCP1",
    "CCP2",
    "CCP3",
    "CCP4",
    "CCP5",
    "CCP6",
    "CCP7",
    "CCP8",
    "T1",
    "T2",
    "FTT9h",
    "TTP7h",
    "TPP9h",
    "FTT10h",
    "TPP8h",
    "TPP10h",
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune official LaBraM on TUAB-format 10s referential windows and "
            "evaluate phase-preserving aperiodic interventions."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/data/tuab/v3.0.1_random_stratified_200"),
    )
    parser.add_argument(
        "--selected-files-csv",
        type=Path,
        default=Path(
            "/mnt/data/aperiodic_confounds/results/tuab_subset_200/"
            "tuab_v3_0_1_random_stratified_200_files.csv"
        ),
    )
    parser.add_argument(
        "--subject-filter-csv",
        type=Path,
        default=None,
        help="Optional subject_id/official_split/label manifest, e.g. age-matched subjects.",
    )
    parser.add_argument(
        "--cache-npz",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/results/tuab_subset_200/labram_10s_200hz_cache.npz"),
    )
    parser.add_argument(
        "--cache-format",
        choices=["npz", "npy"],
        default="npz",
        help=(
            "npz keeps the original compact in-memory cache used for TUAB-200; "
            "npy writes memmap-loadable arrays for full-TUAB scale."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/results/tuab_subset_200/labram_interventions_base"),
    )
    parser.add_argument(
        "--labram-repo",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/external/LaBraM"),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/external/LaBraM/checkpoints/labram-base.pth"),
    )
    parser.add_argument("--sampling-rate", type=int, default=200)
    parser.add_argument("--sample-length-sec", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--band-min", type=float, default=1.0)
    parser.add_argument("--band-max", type=float, default=45.0)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-eval-windows", type=int, default=None)
    parser.add_argument("--intervention-batch-size", type=int, default=256)
    parser.add_argument(
        "--max-cache-files",
        type=int,
        default=None,
        help="Debug-only limit for cache construction; use a separate cache path when set.",
    )
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--balanced-loss", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def binary_label(label: str) -> int:
    normalized = label.strip().lower()
    if normalized == "normal":
        return 0
    if normalized == "abnormal":
        return 1
    raise ValueError(f"Unexpected label: {label!r}")


def local_edf_path(data_root: Path, remote_rel_path: str) -> Path:
    return data_root / "edf" / remote_rel_path


def apply_subject_filter(rows: list[dict[str, str]], subject_filter_csv: Path | None):
    if subject_filter_csv is None:
        return rows, {}
    filter_rows = read_csv(subject_filter_csv)
    if not filter_rows:
        raise ValueError(f"No rows in subject filter: {subject_filter_csv}")
    allowed = {
        (row.get("official_split", ""), row.get("label", ""), row.get("subject_id", row.get("subject", "")))
        for row in filter_rows
    }
    kept = [
        row
        for row in rows
        if (row["official_split"], row["label"], row["subject_id"]) in allowed
    ]
    if not kept:
        raise ValueError(f"Subject filter selected zero files: {subject_filter_csv}")
    return kept, {
        "subject_filter_csv": str(subject_filter_csv),
        "subject_filter_rows": int(len(filter_rows)),
        "subject_filter_kept_files": int(len(kept)),
        "subject_filter_kept_subjects": int(len({row["subject_id"] for row in kept})),
    }


def npy_cache_paths(cache_npz: Path) -> dict[str, Path]:
    return {
        "x": cache_npz.with_suffix(".x.npy"),
        "y": cache_npz.with_suffix(".y.npy"),
        "subjects": cache_npz.with_suffix(".subjects.npy"),
        "splits": cache_npz.with_suffix(".splits.npy"),
        "labels": cache_npz.with_suffix(".labels.npy"),
        "index": cache_npz.with_suffix(".index.csv"),
        "summary": cache_npz.with_suffix(".summary.json"),
    }


def estimate_labram_windows(args: argparse.Namespace, selected: list[dict[str, str]]):
    import mne

    n_times = int(round(args.sampling_rate * args.sample_length_sec))
    total_windows = 0
    file_summaries = []
    skipped = []
    for file_idx, row in enumerate(selected):
        path = local_edf_path(args.data_root, row["remote_rel_path"])
        try:
            raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
            missing = [ch for ch in LABRAM_CHANNELS if ch not in raw.ch_names]
            if missing:
                raise ValueError(f"Missing LaBraM channels: {missing}")
            duration = float(raw.n_times) / float(raw.info["sfreq"])
            n_windows = int(duration * args.sampling_rate) // n_times
        except Exception as exc:
            skipped.append({"remote_rel_path": row["remote_rel_path"], "reason": repr(exc)})
            continue
        file_summaries.append(
            {
                "file_idx": int(file_idx),
                "row": row,
                "n_windows": int(n_windows),
                "start_window": int(total_windows),
            }
        )
        total_windows += int(n_windows)
        if (file_idx + 1) % 100 == 0:
            print(
                f"LaBraM cache planned {file_idx + 1}/{len(selected)} EDF files; "
                f"windows={total_windows}",
                flush=True,
            )
    return total_windows, file_summaries, skipped


def build_labram_cache_npy(args: argparse.Namespace) -> None:
    import mne
    import numpy as np

    selected, filter_summary = apply_subject_filter(read_csv(args.selected_files_csv), args.subject_filter_csv)
    if args.max_cache_files is not None:
        selected = selected[: args.max_cache_files]
    n_times = int(round(args.sampling_rate * args.sample_length_sec))
    paths = npy_cache_paths(args.cache_npz)
    args.cache_npz.parent.mkdir(parents=True, exist_ok=True)

    total_windows, file_summaries, skipped = estimate_labram_windows(args, selected)
    if total_windows <= 0:
        raise ValueError("No LaBraM windows were planned.")

    x = np.lib.format.open_memmap(
        paths["x"],
        mode="w+",
        dtype="float32",
        shape=(total_windows, len(LABRAM_CHANNELS), n_times),
    )
    y = np.lib.format.open_memmap(paths["y"], mode="w+", dtype="int64", shape=(total_windows,))
    subjects = np.empty(total_windows, dtype=object)
    splits = np.empty(total_windows, dtype=object)
    labels = np.empty(total_windows, dtype=object)
    meta_rows = []
    write_pos = 0

    for plan_idx, plan in enumerate(file_summaries):
        row = plan["row"]
        if plan["n_windows"] <= 0:
            continue
        path = local_edf_path(args.data_root, row["remote_rel_path"])
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        drop_channels = [ch for ch in raw.ch_names if ch not in LABRAM_CHANNELS]
        if drop_channels:
            raw.drop_channels(drop_channels)
        missing = [ch for ch in LABRAM_CHANNELS if ch not in raw.ch_names]
        if missing:
            raise ValueError(f"Missing LaBraM channels after planning: {missing}")
        raw.reorder_channels(LABRAM_CHANNELS)
        raw.filter(l_freq=0.1, h_freq=75.0, verbose="ERROR")
        raw.notch_filter(50.0, verbose="ERROR")
        raw.resample(args.sampling_rate, n_jobs=5, verbose="ERROR")
        data = raw.get_data(units="uV").astype("float32", copy=False)

        n_windows = min(int(plan["n_windows"]), data.shape[1] // n_times)
        for window_idx in range(n_windows):
            start = window_idx * n_times
            stop = start + n_times
            x[write_pos] = data[:, start:stop]
            y[write_pos] = binary_label(row["label"])
            subjects[write_pos] = row["subject_id"]
            splits[write_pos] = row["official_split"]
            labels[write_pos] = row["label"]
            meta_rows.append(
                {
                    "window_index": int(write_pos),
                    "file_index": int(plan["file_idx"]),
                    "recording_window_index": int(window_idx),
                    "subject": row["subject_id"],
                    "official_split": row["official_split"],
                    "label": row["label"],
                    "target": binary_label(row["label"]),
                    "remote_rel_path": row["remote_rel_path"],
                    "onset_sec": f"{window_idx * args.sample_length_sec:.6f}",
                    "duration_sec": f"{args.sample_length_sec:.6f}",
                }
            )
            write_pos += 1
        if (plan_idx + 1) % 20 == 0:
            print(
                f"LaBraM cache wrote {plan_idx + 1}/{len(file_summaries)} usable EDF files; "
                f"windows={write_pos}/{total_windows}",
                flush=True,
            )

    if write_pos != total_windows:
        raise RuntimeError(
            f"Planned {total_windows} LaBraM windows but wrote {write_pos}. "
            "This indicates a resampling/window-count mismatch; rebuild with a corrected planner."
        )

    x.flush()
    y.flush()
    np.save(paths["subjects"], subjects)
    np.save(paths["splits"], splits)
    np.save(paths["labels"], labels)
    write_csv(paths["index"], meta_rows)
    summary = {
        "cache_format": "npy",
        "cache_npz_arg": str(args.cache_npz),
        "x_npy": str(paths["x"]),
        "selected_files_csv": str(args.selected_files_csv),
        "n_input_files": int(len(selected)),
        "max_cache_files": args.max_cache_files,
        "n_usable_files": int(len(file_summaries)),
        "n_skipped_files": int(len(skipped)),
        "skipped_files": skipped[:30],
        "n_windows": int(write_pos),
        "shape": [int(write_pos), int(len(LABRAM_CHANNELS)), int(n_times)],
        "sampling_rate": int(args.sampling_rate),
        "sample_length_sec": float(args.sample_length_sec),
        "channels": LABRAM_CHANNELS,
        "preprocessing": "official LaBraM TUAB style: 0.1-75 Hz bandpass, 50 Hz notch, 200 Hz resample, uV units",
        **filter_summary,
    }
    paths["summary"].write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def build_labram_cache(args: argparse.Namespace) -> None:
    import mne
    import numpy as np

    selected, filter_summary = apply_subject_filter(read_csv(args.selected_files_csv), args.subject_filter_csv)
    if args.max_cache_files is not None:
        selected = selected[: args.max_cache_files]
    n_times = int(round(args.sampling_rate * args.sample_length_sec))
    x_rows = []
    meta_rows = []
    skipped = []

    for file_idx, row in enumerate(selected):
        path = local_edf_path(args.data_root, row["remote_rel_path"])
        try:
            raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
            drop_channels = [ch for ch in raw.ch_names if ch not in LABRAM_CHANNELS]
            if drop_channels:
                raw.drop_channels(drop_channels)
            missing = [ch for ch in LABRAM_CHANNELS if ch not in raw.ch_names]
            if missing:
                raise ValueError(f"Missing LaBraM channels: {missing}")
            raw.reorder_channels(LABRAM_CHANNELS)
            raw.filter(l_freq=0.1, h_freq=75.0, verbose="ERROR")
            raw.notch_filter(50.0, verbose="ERROR")
            raw.resample(args.sampling_rate, n_jobs=5, verbose="ERROR")
            data = raw.get_data(units="uV").astype("float32", copy=False)
        except Exception as exc:
            skipped.append({"remote_rel_path": row["remote_rel_path"], "reason": repr(exc)})
            continue

        n_windows = data.shape[1] // n_times
        for window_idx in range(n_windows):
            start = window_idx * n_times
            x_rows.append(data[:, start : start + n_times])
            meta_rows.append(
                {
                    "window_index": len(meta_rows),
                    "file_index": file_idx,
                    "recording_window_index": window_idx,
                    "subject": row["subject_id"],
                    "official_split": row["official_split"],
                    "label": row["label"],
                    "target": binary_label(row["label"]),
                    "remote_rel_path": row["remote_rel_path"],
                    "onset_sec": f"{window_idx * args.sample_length_sec:.6f}",
                    "duration_sec": f"{args.sample_length_sec:.6f}",
                }
            )
        if (file_idx + 1) % 20 == 0:
            print(f"LaBraM cache read {file_idx + 1}/{len(selected)} EDF files", flush=True)

    if not x_rows:
        raise ValueError("No LaBraM windows were extracted.")
    x = np.stack(x_rows).astype("float32", copy=False)
    y = np.array([int(row["target"]) for row in meta_rows], dtype="int64")
    subjects = np.array([row["subject"] for row in meta_rows], dtype=object)
    splits = np.array([row["official_split"] for row in meta_rows], dtype=object)
    labels = np.array([row["label"] for row in meta_rows], dtype=object)

    args.cache_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.cache_npz,
        x=x,
        y=y,
        subjects=subjects,
        splits=splits,
        labels=labels,
        sampling_rate=np.array(args.sampling_rate),
        sample_length_sec=np.array(args.sample_length_sec),
        channels=np.array(LABRAM_CHANNELS, dtype=object),
    )
    write_csv(args.cache_npz.with_suffix(".index.csv"), meta_rows)
    summary = {
        "cache_npz": str(args.cache_npz),
        "selected_files_csv": str(args.selected_files_csv),
        "n_input_files": int(len(selected)),
        "max_cache_files": args.max_cache_files,
        "n_skipped_files": int(len(skipped)),
        "skipped_files": skipped[:30],
        "n_windows": int(x.shape[0]),
        "shape": [int(v) for v in x.shape],
        "sampling_rate": int(args.sampling_rate),
        "sample_length_sec": float(args.sample_length_sec),
        "channels": LABRAM_CHANNELS,
        "preprocessing": "official LaBraM TUAB style: 0.1-75 Hz bandpass, 50 Hz notch, 200 Hz resample, uV units",
        **filter_summary,
    }
    args.cache_npz.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def load_labram_cache(args: argparse.Namespace):
    import numpy as np

    if args.cache_format == "npy":
        paths = npy_cache_paths(args.cache_npz)
        if args.force_rebuild_cache or not paths["x"].exists():
            build_labram_cache_npy(args)
        return {
            "x": np.load(paths["x"], mmap_mode="r"),
            "y": np.load(paths["y"], mmap_mode="r").astype("int64", copy=False),
            "subjects": np.load(paths["subjects"], allow_pickle=True).astype(str),
            "splits": np.load(paths["splits"], allow_pickle=True).astype(str),
            "labels": np.load(paths["labels"], allow_pickle=True).astype(str),
        }

    if args.force_rebuild_cache or not args.cache_npz.exists():
        build_labram_cache(args)
    bundle = np.load(args.cache_npz, allow_pickle=True)
    return {
        "x": bundle["x"].astype("float32", copy=False),
        "y": bundle["y"].astype("int64", copy=False),
        "subjects": bundle["subjects"].astype(str),
        "splits": bundle["splits"].astype(str),
        "labels": bundle["labels"].astype(str),
    }


def subset_indices(indices, max_count: int | None, seed: int):
    import numpy as np

    if max_count is None or len(indices) <= max_count:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=max_count, replace=False))


def make_subject_val_split(train_indices, y, subjects, val_fraction: float, seed: int):
    import numpy as np

    rng = np.random.default_rng(seed)
    val_subjects = []
    train_subjects = []
    for cls in np.unique(y[train_indices]):
        cls_subjects = np.unique(subjects[train_indices][y[train_indices] == cls])
        rng.shuffle(cls_subjects)
        n_val = max(1, int(round(len(cls_subjects) * val_fraction)))
        val_subjects.extend(cls_subjects[:n_val].tolist())
        train_subjects.extend(cls_subjects[n_val:].tolist())
    val_subjects = set(val_subjects)
    train_subjects = set(train_subjects)
    subtrain = np.array([idx for idx in train_indices if subjects[idx] in train_subjects], dtype=int)
    val = np.array([idx for idx in train_indices if subjects[idx] in val_subjects], dtype=int)
    return subtrain, val


def install_timm_shim() -> None:
    import torch

    def drop_path(x, drop_prob: float = 0.0, training: bool = False):
        if drop_prob == 0.0 or not training:
            return x
        keep_prob = 1.0 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

    def to_2tuple(x):
        return x if isinstance(x, tuple) else (x, x)

    def register_model(fn):
        return fn

    timm_mod = types.ModuleType("timm")
    models_mod = types.ModuleType("timm.models")
    layers_mod = types.ModuleType("timm.models.layers")
    registry_mod = types.ModuleType("timm.models.registry")
    layers_mod.drop_path = drop_path
    layers_mod.to_2tuple = to_2tuple
    layers_mod.trunc_normal_ = torch.nn.init.trunc_normal_
    registry_mod.register_model = register_model
    models_mod.layers = layers_mod
    models_mod.registry = registry_mod
    timm_mod.models = models_mod
    sys.modules["timm"] = timm_mod
    sys.modules["timm.models"] = models_mod
    sys.modules["timm.models.layers"] = layers_mod
    sys.modules["timm.models.registry"] = registry_mod


def labram_input_chans() -> list[int]:
    return [0] + [STANDARD_1020.index(name) + 1 for name in LABRAM_COMPACT_CHANNELS]


def make_labram_batch(x, device):
    import torch

    tensor = torch.from_numpy(x).to(device, non_blocking=True)
    return tensor.reshape(tensor.shape[0], tensor.shape[1], tensor.shape[2] // 200, 200)


def load_labram_model(args: argparse.Namespace, device):
    import torch

    install_timm_shim()
    sys.path.insert(0, str(args.labram_repo))
    import modeling_finetune

    model = modeling_finetune.labram_base_patch200_200(
        num_classes=1,
        EEG_size=int(round(args.sampling_rate * args.sample_length_sec)),
        drop_rate=0.0,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        use_mean_pooling=True,
        init_scale=0.001,
        use_rel_pos_bias=False,
        use_abs_pos_emb=True,
        init_values=0.1,
        qkv_bias=False,
    )
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    cleaned = {}
    model_state = model.state_dict()
    for key, value in state.items():
        key = key.removeprefix("module.").removeprefix("student.")
        if key in {"head.weight", "head.bias"}:
            continue
        if "relative_position_index" in key:
            continue
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
            cleaned[key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(
        f"Loaded LaBraM checkpoint keys={len(cleaned)} missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    if missing:
        print(f"Missing keys sample: {missing[:10]}", flush=True)
    if unexpected:
        print(f"Unexpected keys sample: {unexpected[:10]}", flush=True)
    return model.to(device), labram_input_chans()


def choose_threshold(prob, y_true):
    import numpy as np

    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives <= 0 or negatives <= 0:
        return 0.5
    sorted_prob = np.sort(prob)
    return float(sorted_prob[-positives])


def fit_labram(x, y, subjects, train_indices, args):
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    class IndexedArrayDataset(Dataset):
        def __init__(self, x_array, y_array, indices):
            self.x_array = x_array
            self.y_array = y_array
            self.indices = np.asarray(indices, dtype=int)

        def __len__(self):
            return int(len(self.indices))

        def __getitem__(self, item):
            idx = int(self.indices[item])
            xb = np.array(self.x_array[idx], dtype=np.float32, copy=True)
            yb = np.array([float(self.y_array[idx])], dtype=np.float32)
            return torch.from_numpy(xb), torch.from_numpy(yb)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    subtrain_idx, val_idx = make_subject_val_split(
        train_indices,
        y,
        subjects,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    model, input_chans = load_labram_model(args, device)
    if args.balanced_loss:
        class_counts = np.bincount(y[subtrain_idx], minlength=2).astype("float32")
        pos_weight = torch.tensor([class_counts[0] / max(class_counts[1], 1.0)], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loader = DataLoader(
        IndexedArrayDataset(x, y, subtrain_idx),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )

    best_state = None
    best_val_loss = math.inf
    best_epoch = 0
    patience_left = args.patience
    train_log = []

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        seen = 0
        for xb, yb in loader:
            xb = make_labram_batch(xb.numpy(), device)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, input_chans=input_chans)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * int(len(yb))
            seen += int(len(yb))

        model.eval()
        val_losses = []
        val_probs = []
        with torch.no_grad():
            for start in range(0, len(val_idx), args.batch_size):
                idx = val_idx[start : start + args.batch_size]
                xb = make_labram_batch(np.array(x[idx], dtype=np.float32, copy=True), device)
                yb = torch.from_numpy(y[idx].astype("float32")[:, None]).to(device)
                logits = model(xb, input_chans=input_chans)
                val_losses.append(float(criterion(logits, yb).item()) * int(len(idx)))
                val_probs.append(torch.sigmoid(logits).detach().cpu().numpy().reshape(-1))
        val_loss = float(np.sum(val_losses) / max(len(val_idx), 1))
        val_prob = np.concatenate(val_probs)
        threshold = choose_threshold(val_prob, y[val_idx])
        val_pred = (val_prob >= threshold).astype("int64")
        val_bacc = balanced_accuracy(y[val_idx], val_pred, 2)
        train_loss = loss_sum / max(seen, 1)
        train_log.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_balanced_accuracy": float(val_bacc),
                "threshold": float(threshold),
            }
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f} val_bacc={val_bacc:.4f}",
            flush=True,
        )
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch={epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    val_prob = predict_prob(model, x, input_chans, args, indices=val_idx)
    threshold = choose_threshold(val_prob, y[val_idx])
    return model, input_chans, threshold, train_log, {
        "device": str(device),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "n_subtrain_windows": int(len(subtrain_idx)),
        "n_val_windows": int(len(val_idx)),
        "n_subtrain_subjects": int(len(set(subjects[subtrain_idx]))),
        "n_val_subjects": int(len(set(subjects[val_idx]))),
        "threshold": float(threshold),
        "balanced_loss": bool(args.balanced_loss),
    }


def interpolate_fixed_aperiodic(log_power, freqs, target_freqs):
    import numpy as np

    x = np.log10(freqs)
    design = np.column_stack([np.ones_like(x), -x])
    coef = np.linalg.pinv(design) @ log_power.T
    fitted = (design @ coef).T
    if np.array_equal(freqs, target_freqs):
        return fitted.astype("float32")

    out = np.empty((log_power.shape[0], len(target_freqs)), dtype="float32")
    for channel_idx in range(log_power.shape[0]):
        out[channel_idx] = np.interp(target_freqs, freqs, fitted[channel_idx]).astype("float32")
    return out


def make_phase_preserving_inputs(x_eval, sfreq: float, band_min: float, band_max: float):
    import numpy as np

    n_times = x_eval.shape[-1]
    fft_freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    band_mask = (fft_freqs >= band_min) & (fft_freqs <= band_max)
    band_freqs = fft_freqs[band_mask]

    fft = np.fft.rfft(x_eval, axis=-1)
    band_fft = fft[:, :, band_mask]
    band_amp = np.maximum(np.abs(band_fft), 1e-12)
    band_phase = band_fft / band_amp
    band_log_power = np.log10(np.maximum(band_amp**2, 1e-24))

    ap_fit = np.empty_like(band_log_power, dtype="float32")
    for epoch_idx in range(x_eval.shape[0]):
        ap_fit[epoch_idx] = interpolate_fixed_aperiodic(
            band_log_power[epoch_idx],
            band_freqs,
            band_freqs,
        )
    ap_centered = ap_fit - ap_fit.mean(axis=-1, keepdims=True)
    ap_amp_shape = np.power(10.0, 0.5 * ap_centered).astype("float32")

    sham_fft = fft.copy()
    sham_fft[:, :, band_mask] = band_phase * band_amp

    aperiodic_fft = fft.copy()
    geom_amp = np.exp(np.mean(np.log(band_amp), axis=-1, keepdims=True))
    aperiodic_fft[:, :, band_mask] = band_phase * geom_amp * ap_amp_shape

    flattened_fft = fft.copy()
    flattened_fft[:, :, band_mask] = band_fft / np.maximum(ap_amp_shape, 1e-6)

    edited = {
        "phase_sham": np.fft.irfft(sham_fft, n=n_times, axis=-1).real,
        "phase_aperiodic": np.fft.irfft(aperiodic_fft, n=n_times, axis=-1).real,
        "phase_flattened": np.fft.irfft(flattened_fft, n=n_times, axis=-1).real,
    }
    original_std = x_eval.std(axis=-1, keepdims=True)
    for key, value in edited.items():
        value = value.astype("float32", copy=False)
        value -= value.mean(axis=-1, keepdims=True)
        edited_std = value.std(axis=-1, keepdims=True)
        value *= original_std / np.maximum(edited_std, 1e-6)
        edited[key] = value.astype("float32", copy=False)
    return edited


def predict_prob(model, x, input_chans, args, indices=None):
    import numpy as np
    import torch

    device = next(model.parameters()).device
    if indices is None:
        indices = np.arange(len(x))
    probs = []
    with torch.no_grad():
        for start in range(0, len(indices), args.batch_size):
            batch_indices = indices[start : start + args.batch_size]
            xb = make_labram_batch(np.array(x[batch_indices], dtype=np.float32, copy=True), device)
            logits = model(xb, input_chans=input_chans)
            probs.append(torch.sigmoid(logits).detach().cpu().numpy().reshape(-1))
    return np.concatenate(probs)


def balanced_accuracy(y_true, y_pred, n_classes: int) -> float:
    import numpy as np

    recalls = []
    for cls in range(n_classes):
        mask = y_true == cls
        if mask.sum() > 0:
            recalls.append(float((y_pred[mask] == cls).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def macro_f1(y_true, y_pred, n_classes: int) -> float:
    import numpy as np

    f1s = []
    for cls in range(n_classes):
        tp = float(((y_true == cls) & (y_pred == cls)).sum())
        fp = float(((y_true != cls) & (y_pred == cls)).sum())
        fn = float(((y_true == cls) & (y_pred != cls)).sum())
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s))


def accuracy(y_true, y_pred) -> float:
    import numpy as np

    return float(np.mean(y_true == y_pred))


def metric_from_confusion(metric: str, cm) -> float:
    import numpy as np

    total = float(cm.sum())
    if total <= 0:
        return 0.0
    if metric == "accuracy":
        return float(np.trace(cm) / total)
    recalls = []
    f1s = []
    for cls in range(cm.shape[0]):
        tp = float(cm[cls, cls])
        fn = float(cm[cls, :].sum() - tp)
        fp = float(cm[:, cls].sum() - tp)
        support = tp + fn
        if support > 0:
            recalls.append(tp / support)
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    if metric == "balanced_accuracy":
        return float(np.mean(recalls)) if recalls else 0.0
    if metric == "macro_f1":
        return float(np.mean(f1s))
    raise ValueError(metric)


def subject_confusions(prediction_rows, test_input: str):
    import numpy as np
    import pandas as pd

    group = pd.DataFrame([row for row in prediction_rows if row["test_input"] == test_input])
    subjects = sorted(group["subject"].unique())
    matrices = np.zeros((len(subjects), 2, 2), dtype=np.int64)
    labels = []
    for subject_idx, subject in enumerate(subjects):
        sub = group[group["subject"] == subject]
        labels.append(str(sub["label"].iloc[0]))
        for true, pred in zip(sub["y_true"].to_numpy(dtype=int), sub["y_pred"].to_numpy(dtype=int)):
            matrices[subject_idx, true, pred] += 1
    return np.array(subjects, dtype=object), np.array(labels, dtype=object), matrices


def bootstrap_rows(prediction_rows, n_bootstrap: int, ci: float, seed: int):
    import numpy as np

    rng = np.random.default_rng(seed)
    out = []

    def sample_indices(labels):
        sampled = []
        for label in sorted(np.unique(labels)):
            label_idx = np.flatnonzero(labels == label)
            sampled.extend(rng.choice(label_idx, size=len(label_idx), replace=True).tolist())
        return np.array(sampled, dtype=int)

    def ci_bounds(values):
        alpha = 1.0 - ci
        return (
            float(np.quantile(values, alpha / 2.0)),
            float(np.quantile(values, 1.0 - alpha / 2.0)),
        )

    baseline_subjects, baseline_labels, baseline_matrices = subject_confusions(
        prediction_rows,
        "raw_eeg",
    )
    for test_input in TEST_INPUTS:
        subjects, labels, matrices = subject_confusions(prediction_rows, test_input)
        if subjects.tolist() != baseline_subjects.tolist() or labels.tolist() != baseline_labels.tolist():
            raise ValueError(f"Subject mismatch for {test_input}")
        for metric in METRICS:
            boot = np.empty(n_bootstrap, dtype=float)
            for boot_idx in range(n_bootstrap):
                sampled = sample_indices(labels)
                boot[boot_idx] = metric_from_confusion(metric, matrices[sampled].sum(axis=0))
            lower, upper = ci_bounds(boot)
            out.append(
                {
                    "task": "tuab_normal_vs_abnormal",
                    "model": "LaBraM",
                    "train_input": "raw_eeg",
                    "test_input": test_input,
                    "metric": metric,
                    "estimate": "performance",
                    "point": metric_from_confusion(metric, matrices.sum(axis=0)),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "ci": float(ci),
                    "n_eval_subjects": int(subjects.shape[0]),
                    "n_bootstrap": int(n_bootstrap),
                }
            )

    for test_input in ["phase_sham", "phase_aperiodic", "phase_flattened"]:
        subjects, labels, edited_matrices = subject_confusions(prediction_rows, test_input)
        for metric in METRICS:
            boot_drop = np.empty(n_bootstrap, dtype=float)
            boot_retention = np.empty(n_bootstrap, dtype=float)
            for boot_idx in range(n_bootstrap):
                sampled = sample_indices(baseline_labels)
                base_metric = metric_from_confusion(metric, baseline_matrices[sampled].sum(axis=0))
                edit_metric = metric_from_confusion(metric, edited_matrices[sampled].sum(axis=0))
                boot_drop[boot_idx] = base_metric - edit_metric
                boot_retention[boot_idx] = edit_metric / max(base_metric, 1e-12)
            base_point = metric_from_confusion(metric, baseline_matrices.sum(axis=0))
            edit_point = metric_from_confusion(metric, edited_matrices.sum(axis=0))
            for estimate, values, point in [
                (f"drop::{test_input}", boot_drop, base_point - edit_point),
                (f"retention::{test_input}", boot_retention, edit_point / max(base_point, 1e-12)),
            ]:
                lower, upper = ci_bounds(values)
                out.append(
                    {
                        "task": "tuab_normal_vs_abnormal",
                        "model": "LaBraM",
                        "train_input": "raw_eeg",
                        "test_input": test_input,
                        "metric": metric,
                        "estimate": estimate,
                        "point": point,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "ci": float(ci),
                        "n_eval_subjects": int(subjects.shape[0]),
                        "n_bootstrap": int(n_bootstrap),
                    }
                )
    return out


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    focus = df[
        (df["metric"] == "balanced_accuracy")
        & (
            (df["estimate"] == "performance")
            | df["estimate"].isin(
                ["drop::phase_sham", "drop::phase_aperiodic", "drop::phase_flattened"]
            )
        )
    ].copy()
    cols = ["test_input", "estimate", "point", "ci_lower", "ci_upper", "n_eval_subjects"]
    focus = focus[cols]
    for col in ["point", "ci_lower", "ci_upper"]:
        focus[col] = focus[col].map(lambda value: f"{value:.3f}")

    lines = [
        "# TUAB LaBraM Intervention Report",
        "",
        "LaBraM initialized from the official labram-base checkpoint and fine-tuned on",
        "TUAB train windows. Confidence intervals use stratified eval-subject bootstrap.",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in focus.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    import numpy as np
    import pandas as pd

    args = parse_args()
    bundle = load_labram_cache(args)
    x = bundle["x"]
    y = bundle["y"]
    subjects = bundle["subjects"]
    splits = bundle["splits"]
    labels = bundle["labels"]

    train_idx = np.flatnonzero(splits == "train")
    eval_idx = np.flatnonzero(splits == "eval")
    train_idx = subset_indices(train_idx, args.max_train_windows, args.seed)
    eval_idx = subset_indices(eval_idx, args.max_eval_windows, args.seed + 1)
    if len(train_idx) == 0 or len(eval_idx) == 0:
        raise ValueError("Need both train and eval LaBraM windows.")

    model, input_chans, threshold, train_log, train_meta = fit_labram(x, y, subjects, train_idx, args)
    prediction_rows = []
    eval_rows = []

    def append_condition(test_input: str, prob):
        pred = (prob >= threshold).astype("int64")
        eval_rows.append(
            {
                "task": "tuab_normal_vs_abnormal",
                "model": "LaBraM",
                "train_input": "raw_eeg",
                "test_input": test_input,
                "n_eval_windows": int(len(eval_idx)),
                "n_eval_subjects": int(len(np.unique(subjects[eval_idx]))),
                "threshold": float(threshold),
                "balanced_accuracy": balanced_accuracy(y[eval_idx], pred, 2),
                "macro_f1": macro_f1(y[eval_idx], pred, 2),
                "accuracy": accuracy(y[eval_idx], pred),
            }
        )
        for row_index, subject, label, true, score, y_pred in zip(
            eval_idx,
            subjects[eval_idx],
            labels[eval_idx],
            y[eval_idx],
            prob,
            pred,
        ):
            prediction_rows.append(
                {
                    "task": "tuab_normal_vs_abnormal",
                    "model": "LaBraM",
                    "train_input": "raw_eeg",
                    "test_input": test_input,
                    "row_index": int(row_index),
                    "subject": str(subject),
                    "label": str(label),
                    "y_true": int(true),
                    "y_pred": int(y_pred),
                    "prob_abnormal": float(score),
                }
            )
        print(f"Evaluated {test_input}: bacc={eval_rows[-1]['balanced_accuracy']:.4f}", flush=True)

    raw_prob = predict_prob(model, x, input_chans, args, indices=eval_idx)
    append_condition("raw_eeg", raw_prob)

    edited_probs = {key: [] for key in ["phase_sham", "phase_aperiodic", "phase_flattened"]}
    for start in range(0, len(eval_idx), args.intervention_batch_size):
        batch_indices = eval_idx[start : start + args.intervention_batch_size]
        x_batch = np.array(x[batch_indices], dtype=np.float32, copy=True)
        edited = make_phase_preserving_inputs(
            x_batch,
            sfreq=float(args.sampling_rate),
            band_min=float(args.band_min),
            band_max=float(args.band_max),
        )
        for test_input, x_test in edited.items():
            edited_probs[test_input].append(
                predict_prob(model, x_test.astype("float32", copy=False), input_chans, args)
            )
        print(
            f"Intervention eval chunk {min(start + args.intervention_batch_size, len(eval_idx))}/"
            f"{len(eval_idx)}",
            flush=True,
        )
    for test_input in ["phase_sham", "phase_aperiodic", "phase_flattened"]:
        append_condition(test_input, np.concatenate(edited_probs[test_input]))

    boot = bootstrap_rows(prediction_rows, args.n_bootstrap, args.ci, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "tuab_labram_intervention_eval_metrics.csv", eval_rows)
    write_csv(args.output_dir / "tuab_labram_intervention_predictions.csv", prediction_rows)
    write_csv(args.output_dir / "tuab_labram_intervention_subject_bootstrap.csv", boot)
    write_csv(args.output_dir / "tuab_labram_training_log.csv", train_log)
    write_markdown(args.output_dir / "tuab_labram_intervention_subject_bootstrap.md", boot)

    metadata = {
        "labram_repo": str(args.labram_repo),
        "checkpoint_path": str(args.checkpoint_path),
        "cache_npz": str(args.cache_npz),
        "output_dir": str(args.output_dir),
        "n_train_windows": int(len(train_idx)),
        "n_eval_windows": int(len(eval_idx)),
        "n_train_subjects": int(len(np.unique(subjects[train_idx]))),
        "n_eval_subjects": int(len(np.unique(subjects[eval_idx]))),
        "sampling_rate": int(args.sampling_rate),
        "sample_length_sec": float(args.sample_length_sec),
        "channels": LABRAM_CHANNELS,
        "input_chans": input_chans,
        "preprocessing": "official LaBraM TUAB style: 0.1-75 Hz bandpass, 50 Hz notch, 200 Hz resample, uV units",
        "intervention": "phase-preserving FFT amplitude edit after LaBraM preprocessing and before model forward",
        "aperiodic_fit": "fixed log-power linear aperiodic fit over intervention band",
        "band_min": float(args.band_min),
        "band_max": float(args.band_max),
        "threshold_source": "validation subjects from TUAB train split",
        **train_meta,
    }
    (args.output_dir / "tuab_labram_intervention_metadata.json").write_text(json.dumps(metadata, indent=2))

    focus = pd.DataFrame(boot)
    focus = focus[
        (focus["metric"] == "balanced_accuracy")
        & focus["estimate"].isin(["performance", "drop::phase_flattened"])
    ]
    print(f"Wrote outputs to: {args.output_dir}")
    print(
        focus[
            ["test_input", "estimate", "point", "ci_lower", "ci_upper"]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
