#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SCENARIOS = [
    "aperiodic_only",
    "oscillatory_only",
    "mixed",
    "train_confound_test_unconfounded",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ground-truth simulation checks for the aperiodic audit pipeline. "
            "Synthetic PSDs are generated with known aperiodic and oscillatory "
            "label structure."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/simulations/aperiodic_validation"),
    )
    parser.add_argument("--n-subjects", type=int, default=40)
    parser.add_argument("--epochs-per-subject", type=int, default=120)
    parser.add_argument("--n-channels", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    return parser.parse_args()


def balanced_accuracy(y_true, y_pred, n_classes: int = 2) -> float:
    import numpy as np

    recalls = []
    for cls in range(n_classes):
        mask = y_true == cls
        if mask.sum() > 0:
            recalls.append(float((y_pred[mask] == cls).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def make_subject_folds(subjects, n_splits=5):
    import numpy as np

    unique_subjects = np.array(sorted(np.unique(subjects)))
    folds = np.array_split(unique_subjects, n_splits)
    all_idx = np.arange(len(subjects))
    out = []
    for test_subjects in folds:
        test_mask = np.isin(subjects, test_subjects)
        out.append((all_idx[~test_mask], all_idx[test_mask]))
    return out


def gaussian(freqs, center, width):
    import numpy as np

    return np.exp(-0.5 * ((freqs - center) / width) ** 2)


def generate_psd(
    scenario: str,
    args,
    rng,
    unconfounded_test: bool = False,
    base_subjects=None,
    base_y=None,
):
    import numpy as np

    freqs = np.linspace(1.0, 45.0, 177, dtype="float32")
    n = args.n_subjects * args.epochs_per_subject
    if base_subjects is None:
        subjects = np.repeat(np.arange(args.n_subjects), args.epochs_per_subject)
    else:
        subjects = np.asarray(base_subjects)
    if base_y is None:
        y = np.tile(np.arange(args.epochs_per_subject) % 2, args.n_subjects).astype(int)
        rng.shuffle(y)
    else:
        y = np.asarray(base_y).astype(int)

    subject_offset = rng.normal(0.0, 0.12, size=(args.n_subjects, 1, 1))
    subject_exponent = rng.normal(0.0, 0.08, size=(args.n_subjects, 1, 1))
    label_sign = (2 * y - 1).reshape(n, 1, 1)

    offset_effect = 0.0
    exponent_effect = 0.0
    peak_effect = 0.0
    if scenario == "aperiodic_only":
        offset_effect = 0.18
        exponent_effect = 0.22
    elif scenario == "oscillatory_only":
        peak_effect = 0.45
    elif scenario == "mixed":
        offset_effect = 0.14
        exponent_effect = 0.16
        peak_effect = 0.30
    elif scenario == "train_confound_test_unconfounded":
        if not unconfounded_test:
            offset_effect = 0.22
            exponent_effect = 0.24
        peak_effect = 0.22
    else:
        raise ValueError(scenario)

    log_freqs = np.log10(freqs).reshape(1, 1, -1)
    offset = 0.5 + subject_offset[subjects] + offset_effect * label_sign
    exponent = 1.0 + subject_exponent[subjects] + exponent_effect * label_sign
    aperiodic = offset - exponent * log_freqs

    alpha_peak = gaussian(freqs, 10.0, 1.2).reshape(1, 1, -1)
    sigma_peak = gaussian(freqs, 13.5, 1.5).reshape(1, 1, -1)
    peak_template = alpha_peak - 0.65 * sigma_peak
    oscillatory = peak_effect * label_sign * peak_template
    channel_gain = rng.normal(1.0, 0.08, size=(n, args.n_channels, 1))
    noise = rng.normal(0.0, 0.08, size=(n, args.n_channels, len(freqs)))

    log_psd = (aperiodic + oscillatory) * channel_gain + noise
    aperiodic_log_psd = aperiodic * channel_gain
    return {
        "freqs": freqs,
        "subjects": subjects,
        "y": y,
        "full_log_psd": log_psd.astype("float32"),
        "aperiodic_spectrum": aperiodic_log_psd.astype("float32"),
        "flattened_log_psd": (log_psd - aperiodic_log_psd).astype("float32"),
    }


def fit_eval_classifier(x_train, y_train, x_test):
    import numpy as np

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    classes = np.array(sorted(np.unique(y_train)))
    y_onehot = np.column_stack([(y_train == cls).astype(float) for cls in classes])
    design = np.column_stack([np.ones(len(x_train)), x_train])
    reg = np.eye(design.shape[1]) * 1.0
    reg[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + reg, design.T @ y_onehot)
    test_design = np.column_stack([np.ones(len(x_test)), x_test])
    return classes[np.argmax(test_design @ weights, axis=1)]


def run_scenario(scenario: str, args, rng):
    import numpy as np

    data = generate_psd(scenario, args, rng, unconfounded_test=False)
    test_data = (
        generate_psd(
            scenario,
            args,
            rng,
            unconfounded_test=True,
            base_subjects=data["subjects"],
            base_y=data["y"],
        )
        if scenario == "train_confound_test_unconfounded"
        else data
    )
    rows = []
    for train_input in ["full_log_psd", "aperiodic_spectrum", "flattened_log_psd"]:
        for test_input in ["full_log_psd", "aperiodic_spectrum", "flattened_log_psd"]:
            fold_scores = []
            for fold_idx, (train_idx, test_idx) in enumerate(make_subject_folds(data["subjects"])):
                x_train = data[train_input][train_idx].reshape(len(train_idx), -1)
                y_train = data["y"][train_idx]
                x_test = test_data[test_input][test_idx].reshape(len(test_idx), -1)
                y_test = test_data["y"][test_idx]
                pred = fit_eval_classifier(x_train, y_train, x_test)
                fold_scores.append(balanced_accuracy(y_test, pred))
            rows.append(
                {
                    "scenario": scenario,
                    "train_input": train_input,
                    "test_input": test_input,
                    "balanced_accuracy_mean": float(np.mean(fold_scores)),
                    "balanced_accuracy_std": float(np.std(fold_scores, ddof=1)),
                    "n_folds": len(fold_scores),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "| scenario | train_input | test_input | balanced_accuracy_mean | balanced_accuracy_std |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {train_input} | {test_input} | {balanced_accuracy_mean:.3f} | {balanced_accuracy_std:.3f} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    import numpy as np

    args = parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    for scenario in SCENARIOS:
        rows.extend(run_scenario(scenario, args, rng))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "simulation_validation_metrics.csv"
    md_path = args.output_dir / "simulation_validation_metrics.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    (args.output_dir / "simulation_validation_metadata.json").write_text(
        json.dumps(vars(args), indent=2, default=str)
    )
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
