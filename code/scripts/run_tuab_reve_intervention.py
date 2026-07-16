#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_tuab_biot_intervention as common  # noqa: E402


TEST_INPUTS = ["raw_eeg", "phase_sham", "phase_aperiodic", "phase_flattened"]
METRICS = ["balanced_accuracy", "macro_f1", "accuracy"]

REVE_CHANNELS = [
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
]
REVE_COMPACT_CHANNELS = [name.split(" ")[-1].split("-")[0] for name in REVE_CHANNELS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune REVE-base on TUAB-format 10s/200Hz windows and evaluate "
            "phase-preserving aperiodic interventions."
        )
    )
    parser.add_argument(
        "--cache-npz",
        type=Path,
        default=Path("results/tuab_subset_200/labram_10s_200hz_cache.npz"),
        help="TUAB 23-channel 10s/200Hz cache; same cache used for LaBraM/EEGPT/CBraMod.",
    )
    parser.add_argument(
        "--cache-format",
        choices=["npz", "npy"],
        default="npz",
        help=(
            "npz keeps the original compact in-memory cache used for TUAB-200; "
            "npy reads memmap-loadable arrays for full-TUAB scale."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tuab_subset_200/reve_base_interventions"),
    )
    parser.add_argument("--pretrained-repo", default="brain-bzh/reve-base")
    parser.add_argument("--positions-repo", default="brain-bzh/reve-positions")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze REVE-base and train only the TUAB classifier head.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["val_loss", "val_balanced_accuracy"],
        default="val_balanced_accuracy",
        help="Validation metric used for checkpoint selection and early stopping.",
    )
    parser.add_argument("--sampling-rate", type=int, default=200)
    parser.add_argument("--sample-length-sec", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--band-min", type=float, default=1.0)
    parser.add_argument("--band-max", type=float, default=45.0)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-eval-windows", type=int, default=None)
    parser.add_argument("--intervention-batch-size", type=int, default=256)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def npy_cache_paths(cache_npz: Path) -> dict[str, Path]:
    return {
        "x": cache_npz.with_suffix(".x.npy"),
        "y": cache_npz.with_suffix(".y.npy"),
        "subjects": cache_npz.with_suffix(".subjects.npy"),
        "splits": cache_npz.with_suffix(".splits.npy"),
        "labels": cache_npz.with_suffix(".labels.npy"),
        "summary": cache_npz.with_suffix(".summary.json"),
    }


def load_cache(args: argparse.Namespace):
    import numpy as np

    if args.cache_format == "npy":
        paths = npy_cache_paths(args.cache_npz)
        missing = [str(path) for key, path in paths.items() if key != "summary" and not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing full-scale REVE/LaBraM cache files: " + ", ".join(missing)
            )
        summary = json.loads(paths["summary"].read_text())
        return {
            "x": np.load(paths["x"], mmap_mode="r"),
            "y": np.load(paths["y"], mmap_mode="r").astype("int64", copy=False),
            "subjects": np.load(paths["subjects"], allow_pickle=True).astype(str),
            "splits": np.load(paths["splits"], allow_pickle=True).astype(str),
            "labels": np.load(paths["labels"], allow_pickle=True).astype(str),
            "channels": np.array(summary["channels"], dtype=str),
        }

    bundle = np.load(args.cache_npz, allow_pickle=True)
    return {
        "x": bundle["x"].astype("float32", copy=False),
        "y": bundle["y"].astype("int64", copy=False),
        "subjects": bundle["subjects"].astype(str),
        "splits": bundle["splits"].astype(str),
        "labels": bundle["labels"].astype(str),
        "channels": bundle["channels"].astype(str) if "channels" in bundle.files else None,
    }


def make_reve_tuab_input(x, channel_names):
    import numpy as np

    if channel_names is None:
        raise ValueError("REVE runner needs channel names in the cache.")
    name_to_idx = {str(name): idx for idx, name in enumerate(channel_names)}
    missing = [name for name in REVE_CHANNELS if name not in name_to_idx]
    if missing:
        raise ValueError(f"Missing channels needed for REVE TUAB input: {missing}")
    out = np.stack([x[:, name_to_idx[name], :] for name in REVE_CHANNELS], axis=1).astype(
        "float32", copy=False
    )
    mean = out.mean(axis=-1, keepdims=True)
    std = out.std(axis=-1, keepdims=True)
    out = ((out - mean) / np.maximum(std, 1e-6)).astype("float32", copy=False)
    return np.clip(out, -15.0, 15.0).astype("float32", copy=False)


def load_reve_model(args: argparse.Namespace, n_chans: int, device):
    import torch
    from torch import nn
    from transformers import AutoModel

    if n_chans != len(REVE_COMPACT_CHANNELS):
        raise ValueError(f"Expected {len(REVE_COMPACT_CHANNELS)} REVE channels, got {n_chans}")

    pos_bank = AutoModel.from_pretrained(args.positions_repo, trust_remote_code=True)
    positions = pos_bank(REVE_COMPACT_CHANNELS)
    if positions.shape[0] != n_chans:
        raise ValueError(
            f"REVE position bank returned {positions.shape[0]} positions for {n_chans} channels."
        )

    backbone = AutoModel.from_pretrained(args.pretrained_repo, trust_remote_code=True)
    embed_dim = int(getattr(backbone, "embed_dim", 512))

    class Model(nn.Module):
        def __init__(self, backbone_module, position_tensor):
            super().__init__()
            self.backbone = backbone_module
            self.register_buffer("positions", position_tensor.float())
            self.classifier = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Dropout(args.dropout),
                nn.Linear(embed_dim, 1),
            )

        def forward(self, x):
            pos = self.positions.unsqueeze(0).expand(x.shape[0], -1, -1)
            features = self.backbone(x, pos)
            pooled = self.backbone.attention_pooling(features)
            return self.classifier(pooled)

    model = Model(backbone, positions)
    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
    load_summary = {
        "model_source": "Hugging Face AutoModel with trust_remote_code",
        "pretrained_repo": str(args.pretrained_repo),
        "positions_repo": str(args.positions_repo),
        "classifier": "REVE attention_pooling tokens -> LayerNorm/Dropout/Linear binary head",
        "n_reve_channels": int(n_chans),
        "reve_channels": REVE_COMPACT_CHANNELS,
        "freeze_backbone": bool(args.freeze_backbone),
        "embed_dim": int(embed_dim),
    }
    return model.to(device), load_summary


def choose_threshold(prob, y_true):
    import numpy as np

    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives <= 0 or negatives <= 0:
        return 0.5
    sorted_prob = np.sort(prob)
    return float(sorted_prob[-positives])


def predict_prob(model, x, args, channel_names=None, indices=None):
    import numpy as np
    import torch

    device = next(model.parameters()).device
    if indices is None:
        indices = np.arange(len(x))
    probs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), args.batch_size):
            batch_idx = indices[start : start + args.batch_size]
            xb_np = np.array(x[batch_idx], dtype=np.float32, copy=True)
            if channel_names is not None:
                xb_np = make_reve_tuab_input(xb_np, channel_names)
            xb = torch.from_numpy(xb_np).to(device)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).detach().cpu().numpy().reshape(-1))
    return np.concatenate(probs)


def fit_reve(x, y, subjects, channel_names, train_indices, args):
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    class IndexedREVEDataset(Dataset):
        def __init__(self, x_array, y_array, indices):
            self.x_array = x_array
            self.y_array = y_array
            self.indices = np.asarray(indices, dtype=int)

        def __len__(self):
            return int(len(self.indices))

        def __getitem__(self, item):
            idx = int(self.indices[item])
            xb_ref = np.array(self.x_array[idx : idx + 1], dtype=np.float32, copy=True)
            xb = make_reve_tuab_input(xb_ref, channel_names)[0]
            yb = np.array([float(self.y_array[idx])], dtype=np.float32)
            return torch.from_numpy(xb), torch.from_numpy(yb)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    subtrain_idx, val_idx = common.make_subject_val_split(
        train_indices,
        y,
        subjects,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    model, load_summary = load_reve_model(args, len(REVE_CHANNELS), device)

    class_counts = np.bincount(y[subtrain_idx], minlength=2).astype("float32")
    pos_weight = torch.tensor(
        [class_counts[0] / max(class_counts[1], 1.0)],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable REVE parameters remain.")
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loader = DataLoader(
        IndexedREVEDataset(x, y, subtrain_idx),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )

    best_state = None
    best_val_loss = math.inf
    best_val_bacc = -math.inf
    best_score = math.inf if args.selection_metric == "val_loss" else -math.inf
    best_epoch = 0
    patience_left = args.patience
    train_log = []

    for epoch in range(args.epochs):
        model.train()
        if args.freeze_backbone:
            model.backbone.eval()
        loss_sum = 0.0
        seen = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            if args.gradient_clip_norm and args.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optimizer.step()
            loss_sum += float(loss.item()) * int(len(yb))
            seen += int(len(yb))

        model.eval()
        val_losses = []
        val_probs = []
        with torch.no_grad():
            for start in range(0, len(val_idx), args.batch_size):
                batch_idx = val_idx[start : start + args.batch_size]
                xb_np = make_reve_tuab_input(
                    np.array(x[batch_idx], dtype=np.float32, copy=True),
                    channel_names,
                )
                xb = torch.from_numpy(xb_np).to(device)
                yb = torch.from_numpy(y[batch_idx].astype("float32")[:, None]).to(device)
                logits = model(xb)
                val_loss_tensor = criterion(logits, yb)
                if not torch.isfinite(val_loss_tensor):
                    raise FloatingPointError(f"Non-finite validation loss at epoch {epoch}")
                val_losses.append(float(val_loss_tensor.item()) * int(len(batch_idx)))
                val_probs.append(torch.sigmoid(logits).detach().cpu().numpy().reshape(-1))
        val_prob = np.concatenate(val_probs)
        val_loss = float(sum(val_losses) / max(len(val_idx), 1))
        threshold = choose_threshold(val_prob, y[val_idx])
        val_pred = (val_prob >= threshold).astype("int64")
        val_bacc = common.balanced_accuracy(y[val_idx], val_pred, 2)
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
        current_score = val_loss if args.selection_metric == "val_loss" else val_bacc
        improved = (
            current_score < best_score - 1e-5
            if args.selection_metric == "val_loss"
            else current_score > best_score + 1e-5
        )
        if improved:
            best_score = current_score
            best_val_loss = val_loss
            best_val_bacc = val_bacc
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
    val_prob = predict_prob(model, x, args, channel_names=channel_names, indices=val_idx)
    threshold = choose_threshold(val_prob, y[val_idx])
    train_meta = {
        "device": str(device),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_balanced_accuracy": float(best_val_bacc),
        "selection_metric": str(args.selection_metric),
        "n_subtrain_windows": int(len(subtrain_idx)),
        "n_val_windows": int(len(val_idx)),
        "n_subtrain_subjects": int(len(set(subjects[subtrain_idx]))),
        "n_val_subjects": int(len(set(subjects[val_idx]))),
        "threshold": float(threshold),
        "n_trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "n_total_parameters": int(sum(p.numel() for p in model.parameters())),
        **load_summary,
    }
    return model, threshold, train_log, train_meta


def bootstrap_rows(prediction_rows, n_bootstrap: int, ci: float, seed: int):
    rows = common.bootstrap_rows(prediction_rows, n_bootstrap, ci, seed)
    for row in rows:
        row["model"] = "REVE-base"
    return rows


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
        "# TUAB REVE-base Intervention Report",
        "",
        "REVE-base uses the `brain-bzh/reve-base` Hugging Face encoder with",
        "`brain-bzh/reve-positions`, 21 recognized referential TUAB channels,",
        "REVE-style z-score/clipping, attention pooling, and a binary head.",
        "Confidence intervals use stratified eval-subject bootstrap.",
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
    bundle = load_cache(args)
    x = bundle["x"]
    y = bundle["y"]
    subjects = bundle["subjects"]
    splits = bundle["splits"]
    labels = bundle["labels"]
    channel_names = bundle.get("channels")

    train_idx = np.flatnonzero(splits == "train")
    eval_idx = np.flatnonzero(splits == "eval")
    train_idx = common.subset_indices(train_idx, args.max_train_windows, args.seed)
    eval_idx = common.subset_indices(eval_idx, args.max_eval_windows, args.seed + 1)
    if len(train_idx) == 0 or len(eval_idx) == 0:
        raise ValueError("Need both train and eval REVE windows.")

    model, threshold, train_log, train_meta = fit_reve(
        x,
        y,
        subjects,
        channel_names,
        train_idx,
        args,
    )
    prediction_rows = []
    eval_rows = []

    def append_condition(test_input: str, prob):
        pred = (prob >= threshold).astype("int64")
        eval_rows.append(
            {
                "task": "tuab_normal_vs_abnormal",
                "model": "REVE-base",
                "train_input": "raw_eeg",
                "test_input": test_input,
                "n_eval_windows": int(len(eval_idx)),
                "n_eval_subjects": int(len(np.unique(subjects[eval_idx]))),
                "threshold": float(threshold),
                "balanced_accuracy": common.balanced_accuracy(y[eval_idx], pred, 2),
                "macro_f1": common.macro_f1(y[eval_idx], pred, 2),
                "accuracy": common.accuracy(y[eval_idx], pred),
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
                    "model": "REVE-base",
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

    raw_prob = predict_prob(model, x, args, channel_names=channel_names, indices=eval_idx)
    append_condition("raw_eeg", raw_prob)

    edited_probs = {key: [] for key in ["phase_sham", "phase_aperiodic", "phase_flattened"]}
    for start in range(0, len(eval_idx), args.intervention_batch_size):
        batch_idx = eval_idx[start : start + args.intervention_batch_size]
        x_batch = make_reve_tuab_input(
            np.array(x[batch_idx], dtype=np.float32, copy=True),
            channel_names,
        )
        edited = common.make_phase_preserving_inputs(
            x_batch,
            sfreq=float(args.sampling_rate),
            band_min=float(args.band_min),
            band_max=float(args.band_max),
        )
        for test_input, x_test in edited.items():
            edited_probs[test_input].append(
                predict_prob(model, x_test.astype("float32", copy=False), args)
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
    write_csv(args.output_dir / "tuab_reve_intervention_eval_metrics.csv", eval_rows)
    write_csv(args.output_dir / "tuab_reve_intervention_predictions.csv", prediction_rows)
    write_csv(args.output_dir / "tuab_reve_intervention_subject_bootstrap.csv", boot)
    write_csv(args.output_dir / "tuab_reve_training_log.csv", train_log)
    write_markdown(args.output_dir / "tuab_reve_intervention_subject_bootstrap.md", boot)

    metadata = {
        "cache_npz": str(args.cache_npz),
        "cache_format": str(args.cache_format),
        "output_dir": str(args.output_dir),
        "n_train_windows": int(len(train_idx)),
        "n_eval_windows": int(len(eval_idx)),
        "n_train_subjects": int(len(np.unique(subjects[train_idx]))),
        "n_eval_subjects": int(len(np.unique(subjects[eval_idx]))),
        "sampling_rate": int(args.sampling_rate),
        "sample_length_sec": float(args.sample_length_sec),
        "normalization": "per-window/channel z-score then clip to +/-15, matching REVE guidance",
        "preprocessing": "selected 21 TUAB referential channels recognized by brain-bzh/reve-positions; dropped T1/T2 because the REVE position bank did not include them",
        "intervention": "phase-preserving FFT amplitude edit after REVE preprocessing and before model forward",
        "aperiodic_fit": "fixed log-power linear aperiodic fit over intervention band",
        "band_min": float(args.band_min),
        "band_max": float(args.band_max),
        "threshold_source": "validation subjects from TUAB train split",
        **train_meta,
    }
    (args.output_dir / "tuab_reve_intervention_metadata.json").write_text(
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
