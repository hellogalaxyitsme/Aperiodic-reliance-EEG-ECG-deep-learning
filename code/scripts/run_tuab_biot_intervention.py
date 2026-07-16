#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path


TEST_INPUTS = ["raw_eeg", "phase_sham", "phase_aperiodic", "phase_flattened"]
METRICS = ["balanced_accuracy", "macro_f1", "accuracy"]
BIOT_BIPOLAR_PAIRS = [
    ("EEG FP1-REF", "EEG F7-REF"),
    ("EEG F7-REF", "EEG T3-REF"),
    ("EEG T3-REF", "EEG T5-REF"),
    ("EEG T5-REF", "EEG O1-REF"),
    ("EEG FP2-REF", "EEG F8-REF"),
    ("EEG F8-REF", "EEG T4-REF"),
    ("EEG T4-REF", "EEG T6-REF"),
    ("EEG T6-REF", "EEG O2-REF"),
    ("EEG FP1-REF", "EEG F3-REF"),
    ("EEG F3-REF", "EEG C3-REF"),
    ("EEG C3-REF", "EEG P3-REF"),
    ("EEG P3-REF", "EEG O1-REF"),
    ("EEG FP2-REF", "EEG F4-REF"),
    ("EEG F4-REF", "EEG C4-REF"),
    ("EEG C4-REF", "EEG P4-REF"),
    ("EEG P4-REF", "EEG O2-REF"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune official BIOT on TUAB-format 10s bipolar windows and "
            "evaluate phase-preserving aperiodic interventions."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/tuab/v3.0.1_random_stratified_200"),
    )
    parser.add_argument(
        "--selected-files-csv",
        type=Path,
        default=Path(
            "results/tuab_subset_200/"
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
        default=Path("results/tuab_subset_200/biot_10s_200hz_cache.npz"),
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
        default=Path("results/tuab_subset_200/biot_interventions_prest"),
    )
    parser.add_argument(
        "--biot-repo",
        type=Path,
        default=Path("external/BIOT"),
    )
    parser.add_argument(
        "--pretrain-model-path",
        type=Path,
        default=Path(
            "external/BIOT/pretrained-models/"
            "EEG-PREST-16-channels.ckpt"
        ),
    )
    parser.add_argument("--sampling-rate", type=int, default=200)
    parser.add_argument("--sample-length-sec", type=float, default=10.0)
    parser.add_argument("--n-fft", type=int, default=200)
    parser.add_argument("--hop-length", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
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
    parser.add_argument("--intervention-batch-size", type=int, default=1024)
    parser.add_argument("--force-rebuild-cache", action="store_true")
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
        "subject_filter_kept_subjects": int(len({row['subject_id'] for row in kept})),
    }


def normalize_biot_windows(x):
    import numpy as np

    scale = np.quantile(np.abs(x), q=0.95, method="linear", axis=-1, keepdims=True)
    return (x / (scale + 1e-8)).astype("float32", copy=False)


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


def estimate_biot_windows(args: argparse.Namespace, selected: list[dict[str, str]]):
    import mne

    n_times = int(round(args.sampling_rate * args.sample_length_sec))
    total_windows = 0
    file_summaries = []
    skipped = []
    required_channels = sorted({ch for pair in BIOT_BIPOLAR_PAIRS for ch in pair})
    for file_idx, row in enumerate(selected):
        path = local_edf_path(args.data_root, row["remote_rel_path"])
        try:
            raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
            missing = [ch for ch in required_channels if ch not in raw.ch_names]
            if missing:
                raise ValueError(f"missing channels: {missing[:5]}")
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
                f"BIOT cache planned {file_idx + 1}/{len(selected)} EDF files; "
                f"windows={total_windows}",
                flush=True,
            )
    return total_windows, file_summaries, skipped


def build_biot_cache_npy(args: argparse.Namespace):
    import mne
    import numpy as np

    selected, filter_summary = apply_subject_filter(read_csv(args.selected_files_csv), args.subject_filter_csv)
    n_times = int(round(args.sampling_rate * args.sample_length_sec))
    paths = npy_cache_paths(args.cache_npz)
    args.cache_npz.parent.mkdir(parents=True, exist_ok=True)

    total_windows, file_summaries, skipped = estimate_biot_windows(args, selected)
    if total_windows <= 0:
        raise ValueError("No BIOT windows were planned.")

    x = np.lib.format.open_memmap(
        paths["x"],
        mode="w+",
        dtype="float32",
        shape=(total_windows, len(BIOT_BIPOLAR_PAIRS), n_times),
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
        raw.resample(args.sampling_rate, verbose="ERROR")
        ch_names = raw.ch_names
        raw_data = raw.get_data().astype("float32", copy=False)
        bipolar = np.empty((len(BIOT_BIPOLAR_PAIRS), raw_data.shape[1]), dtype="float32")
        for channel_idx, (left, right) in enumerate(BIOT_BIPOLAR_PAIRS):
            bipolar[channel_idx] = raw_data[ch_names.index(left)] - raw_data[ch_names.index(right)]

        n_windows = min(int(plan["n_windows"]), bipolar.shape[1] // n_times)
        for window_idx in range(n_windows):
            start = window_idx * n_times
            stop = start + n_times
            x[write_pos] = normalize_biot_windows(bipolar[:, start:stop][None, ...])[0]
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
                f"BIOT cache wrote {plan_idx + 1}/{len(file_summaries)} usable EDF files; "
                f"windows={write_pos}/{total_windows}",
                flush=True,
            )

    if write_pos != total_windows:
        raise RuntimeError(
            f"Planned {total_windows} BIOT windows but wrote {write_pos}. "
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
        "n_usable_files": int(len(file_summaries)),
        "n_skipped_files": int(len(skipped)),
        "skipped_files": skipped[:20],
        "n_windows": int(write_pos),
        "shape": [int(write_pos), int(len(BIOT_BIPOLAR_PAIRS)), int(n_times)],
        "sampling_rate": int(args.sampling_rate),
        "sample_length_sec": float(args.sample_length_sec),
        "normalization": "per-window per-channel 0.95 absolute quantile, matching BIOT TUABLoader",
        **filter_summary,
    }
    paths["summary"].write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def build_biot_cache(args: argparse.Namespace):
    import mne
    import numpy as np

    selected, filter_summary = apply_subject_filter(read_csv(args.selected_files_csv), args.subject_filter_csv)
    n_times = int(round(args.sampling_rate * args.sample_length_sec))
    x_rows = []
    meta_rows = []
    skipped = []

    for file_idx, row in enumerate(selected):
        path = local_edf_path(args.data_root, row["remote_rel_path"])
        try:
            raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
            raw.resample(args.sampling_rate, verbose="ERROR")
            ch_names = raw.ch_names
            raw_data = raw.get_data().astype("float32", copy=False)
            bipolar = np.empty((len(BIOT_BIPOLAR_PAIRS), raw_data.shape[1]), dtype="float32")
            for channel_idx, (left, right) in enumerate(BIOT_BIPOLAR_PAIRS):
                bipolar[channel_idx] = raw_data[ch_names.index(left)] - raw_data[ch_names.index(right)]
        except Exception as exc:
            skipped.append({"remote_rel_path": row["remote_rel_path"], "reason": repr(exc)})
            continue

        n_windows = bipolar.shape[1] // n_times
        for window_idx in range(n_windows):
            start = window_idx * n_times
            x_rows.append(bipolar[:, start : start + n_times])
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
            print(f"BIOT cache read {file_idx + 1}/{len(selected)} EDF files", flush=True)

    if not x_rows:
        raise ValueError("No BIOT windows were extracted.")
    x = normalize_biot_windows(np.stack(x_rows).astype("float32", copy=False))
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
        bipolar_pairs=np.array([f"{left}-{right}" for left, right in BIOT_BIPOLAR_PAIRS], dtype=object),
    )
    write_csv(args.cache_npz.with_suffix(".index.csv"), meta_rows)
    summary = {
        "cache_npz": str(args.cache_npz),
        "selected_files_csv": str(args.selected_files_csv),
        "n_input_files": int(len(selected)),
        "n_skipped_files": int(len(skipped)),
        "skipped_files": skipped[:20],
        "n_windows": int(x.shape[0]),
        "shape": [int(v) for v in x.shape],
        "sampling_rate": int(args.sampling_rate),
        "sample_length_sec": float(args.sample_length_sec),
        "normalization": "per-window per-channel 0.95 absolute quantile, matching BIOT TUABLoader",
        **filter_summary,
    }
    args.cache_npz.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def load_biot_cache(args: argparse.Namespace):
    import numpy as np

    if args.cache_format == "npy":
        paths = npy_cache_paths(args.cache_npz)
        if args.force_rebuild_cache or not paths["x"].exists():
            build_biot_cache_npy(args)
        return {
            "x": np.load(paths["x"], mmap_mode="r"),
            "y": np.load(paths["y"], mmap_mode="r").astype("int64", copy=False),
            "subjects": np.load(paths["subjects"], allow_pickle=True).astype(str),
            "splits": np.load(paths["splits"], allow_pickle=True).astype(str),
            "labels": np.load(paths["labels"], allow_pickle=True).astype(str),
        }

    if args.force_rebuild_cache or not args.cache_npz.exists():
        build_biot_cache(args)
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


def load_biot_model(args: argparse.Namespace, device):
    import torch

    sys.path.insert(0, str(args.biot_repo))
    from model.biot import BIOTClassifier

    model = BIOTClassifier(
        n_classes=1,
        n_channels=16,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )
    state = torch.load(args.pretrain_model_path, map_location="cpu")
    model.biot.load_state_dict(state)
    return model.to(device)


def choose_threshold(prob, y_true):
    import numpy as np

    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives <= 0 or negatives <= 0:
        return 0.5
    sorted_prob = np.sort(prob)
    return float(sorted_prob[-positives])


def fit_biot(x, y, subjects, train_indices, args):
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
    model = load_biot_model(args, device)
    class_counts = np.bincount(y[subtrain_idx], minlength=2).astype("float32")
    pos_weight = torch.tensor(
        [class_counts[0] / max(class_counts[1], 1.0)],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
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
    val_y_np = y[val_idx].astype("float32")[:, None]

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
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * int(len(yb))
            seen += int(len(yb))

        model.eval()
        with torch.no_grad():
            val_logits_parts = []
            val_loss_sum = 0.0
            val_seen = 0
            for start in range(0, len(val_idx), args.batch_size):
                batch_indices = val_idx[start : start + args.batch_size]
                xb = torch.from_numpy(np.array(x[batch_indices], dtype=np.float32, copy=True)).to(device)
                yb = torch.from_numpy(val_y_np[start : start + len(batch_indices)]).to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_logits_parts.append(logits.detach().cpu())
                val_loss_sum += float(loss.item()) * int(len(batch_indices))
                val_seen += int(len(batch_indices))
            val_logits = torch.cat(val_logits_parts, dim=0)
            val_loss = val_loss_sum / max(val_seen, 1)
            val_prob = torch.sigmoid(val_logits).numpy().reshape(-1)
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

    with torch.no_grad():
        val_logits = []
        for start in range(0, len(val_idx), args.batch_size):
            xb = torch.from_numpy(
                np.array(x[val_idx[start : start + args.batch_size]], dtype=np.float32, copy=True)
            ).to(device)
            val_logits.append(model(xb).detach().cpu())
        val_prob = torch.sigmoid(torch.cat(val_logits, dim=0)).numpy().reshape(-1)
    threshold = choose_threshold(val_prob, y[val_idx])
    return model, threshold, train_log, {
        "device": str(device),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "n_subtrain_windows": int(len(subtrain_idx)),
        "n_val_windows": int(len(val_idx)),
        "n_subtrain_subjects": int(len(set(subjects[subtrain_idx]))),
        "n_val_subjects": int(len(set(subjects[val_idx]))),
        "threshold": float(threshold),
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


def predict_prob(model, x, args):
    import numpy as np
    import torch

    device = next(model.parameters()).device
    probs = []
    with torch.no_grad():
        for start in range(0, len(x), args.batch_size):
            xb = torch.from_numpy(x[start : start + args.batch_size]).to(device)
            logits = model(xb)
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
                    "model": "BIOT",
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
                        "model": "BIOT",
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
        "# TUAB BIOT Intervention Report",
        "",
        "BIOT initialized from the official encoder checkpoint and fine-tuned on",
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
    import torch

    args = parse_args()
    bundle = load_biot_cache(args)
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
        raise ValueError("Need both train and eval BIOT windows.")

    model, threshold, train_log, train_meta = fit_biot(x, y, subjects, train_idx, args)
    prediction_rows = []
    eval_rows = []

    def append_condition(test_input: str, prob):
        pred = (prob >= threshold).astype("int64")
        eval_rows.append(
            {
                "task": "tuab_normal_vs_abnormal",
                "model": "BIOT",
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
                    "model": "BIOT",
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

    raw_probs = []
    for start in range(0, len(eval_idx), args.batch_size):
        batch_indices = eval_idx[start : start + args.batch_size]
        x_batch = np.array(x[batch_indices], dtype=np.float32, copy=True)
        raw_probs.append(predict_prob(model, x_batch, args))
    append_condition("raw_eeg", np.concatenate(raw_probs))

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
            edited_probs[test_input].append(predict_prob(model, x_test.astype("float32", copy=False), args))
        print(
            f"Intervention eval chunk {min(start + args.intervention_batch_size, len(eval_idx))}/"
            f"{len(eval_idx)}",
            flush=True,
        )
    for test_input in ["phase_sham", "phase_aperiodic", "phase_flattened"]:
        append_condition(test_input, np.concatenate(edited_probs[test_input]))

    boot = bootstrap_rows(prediction_rows, args.n_bootstrap, args.ci, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "tuab_biot_intervention_eval_metrics.csv", eval_rows)
    write_csv(args.output_dir / "tuab_biot_intervention_predictions.csv", prediction_rows)
    write_csv(args.output_dir / "tuab_biot_intervention_subject_bootstrap.csv", boot)
    write_csv(args.output_dir / "tuab_biot_training_log.csv", train_log)
    write_markdown(args.output_dir / "tuab_biot_intervention_subject_bootstrap.md", boot)

    metadata = {
        "biot_repo": str(args.biot_repo),
        "pretrain_model_path": str(args.pretrain_model_path),
        "cache_npz": str(args.cache_npz),
        "output_dir": str(args.output_dir),
        "n_train_windows": int(len(train_idx)),
        "n_eval_windows": int(len(eval_idx)),
        "n_train_subjects": int(len(np.unique(subjects[train_idx]))),
        "n_eval_subjects": int(len(np.unique(subjects[eval_idx]))),
        "sampling_rate": int(args.sampling_rate),
        "sample_length_sec": float(args.sample_length_sec),
        "n_fft": int(args.n_fft),
        "hop_length": int(args.hop_length),
        "normalization": "BIOT q95 abs per-channel per-window normalization before intervention/model",
        "intervention": "phase-preserving FFT amplitude edit after BIOT preprocessing/normalization",
        "aperiodic_fit": "fixed log-power linear aperiodic fit over intervention band",
        "band_min": float(args.band_min),
        "band_max": float(args.band_max),
        "threshold_source": "validation subjects from TUAB train split",
        **train_meta,
    }
    (args.output_dir / "tuab_biot_intervention_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

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
