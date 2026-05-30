#!/usr/bin/env python
from __future__ import annotations

import argparse
import zlib
from pathlib import Path


METRICS = ["balanced_accuracy", "macro_f1", "accuracy"]
TEST_INPUTS = ["raw_eeg", "phase_sham", "phase_aperiodic", "phase_flattened"]
ESTIMATE_BY_TEST_INPUT = {
    "raw_eeg": "baseline",
    "phase_sham": "sham",
    "phase_aperiodic": "aperiodic",
    "phase_flattened": "flattened",
}
EDITED_INPUTS = ["phase_sham", "phase_aperiodic", "phase_flattened"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate full-TUAB foundation-model intervention prediction CSVs "
            "with a seed/subject hierarchical bootstrap."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input in the form model:seed=/path/to/*_predictions.csv",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-batch-size", type=int, default=1000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stable_seed_offset(*parts) -> int:
    text = "::".join(str(part) for part in parts)
    return zlib.crc32(text.encode("utf-8")) % 1_000_000


def parse_input(text: str) -> tuple[str, int, Path]:
    if "=" not in text or ":" not in text.split("=", 1)[0]:
        raise ValueError(f"Expected model:seed=path input, got: {text}")
    left, path = text.split("=", 1)
    model, seed = left.split(":", 1)
    model = model.strip()
    if not model:
        raise ValueError(f"Missing model name in input: {text}")
    return model, int(seed), Path(path)


def confusion(y_true, y_pred):
    import numpy as np

    cm = np.zeros((2, 2), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        cm[int(true), int(pred)] += 1
    return cm


def metric_from_confusion(metric: str, cm):
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


def ci_bounds(values, ci: float):
    import numpy as np

    alpha = 1.0 - ci
    return (
        float(np.quantile(values, alpha / 2.0)),
        float(np.quantile(values, 1.0 - alpha / 2.0)),
    )


def bootstrap_sign_p_values(values):
    import numpy as np

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(len(values))
    if n == 0:
        return {
            "p_one_sided_positive": float("nan"),
            "p_two_sided_zero": float("nan"),
            "n_nonpositive": 0,
            "n_nonnegative": 0,
            "n_bootstrap_valid": 0,
        }

    n_nonpositive = int(np.count_nonzero(values <= 0.0))
    n_nonnegative = int(np.count_nonzero(values >= 0.0))
    p_one = n_nonpositive / n
    p_two = min(1.0, 2.0 * min(n_nonpositive / n, n_nonnegative / n))
    return {
        "p_one_sided_positive": float(p_one),
        "p_two_sided_zero": float(p_two),
        "n_nonpositive": n_nonpositive,
        "n_nonnegative": n_nonnegative,
        "n_bootstrap_valid": n,
    }


def load_prediction_input(model: str, seed: int, path: Path):
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"task", "test_input", "subject", "label", "y_true", "y_pred"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame = frame[frame["test_input"].isin(TEST_INPUTS)].copy()
    if frame.empty:
        raise ValueError(f"{path} has no recognized test inputs.")
    frame["model"] = model
    frame["seed"] = seed
    return frame


def build_arrays(group):
    import numpy as np
    import pandas as pd

    seeds = np.array(sorted(group["seed"].unique()), dtype=int)
    subjects = sorted(group["subject"].astype(str).unique())
    labels = []
    values = np.zeros((len(seeds), len(subjects), len(TEST_INPUTS), 2, 2), dtype=np.int64)
    seed_index = {seed: idx for idx, seed in enumerate(seeds)}
    subject_index = {subject: idx for idx, subject in enumerate(subjects)}
    input_index = {name: idx for idx, name in enumerate(TEST_INPUTS)}

    label_by_subject = {}
    for subject, sub in group.groupby("subject", sort=False):
        labels_for_subject = sorted(sub["label"].astype(str).unique())
        if len(labels_for_subject) != 1:
            raise ValueError(f"Subject {subject} has multiple labels: {labels_for_subject}")
        label_by_subject[str(subject)] = labels_for_subject[0]
    labels = np.array([label_by_subject[subject] for subject in subjects], dtype=object)

    complete = np.zeros((len(seeds), len(subjects), len(TEST_INPUTS)), dtype=bool)
    for (seed, subject, test_input), sub in group.groupby(["seed", "subject", "test_input"], sort=False):
        values[
            seed_index[int(seed)],
            subject_index[str(subject)],
            input_index[str(test_input)],
        ] = confusion(sub["y_true"].to_numpy(dtype=int), sub["y_pred"].to_numpy(dtype=int))
        complete[
            seed_index[int(seed)],
            subject_index[str(subject)],
            input_index[str(test_input)],
        ] = True

    complete_subject = complete.all(axis=(0, 2))
    if not complete_subject.any():
        raise ValueError("No subjects are complete across all seeds and interventions.")
    values = values[:, complete_subject, :, :, :]
    labels = labels[complete_subject]
    subjects = np.array(subjects, dtype=object)[complete_subject]
    return seeds, subjects, labels, values


def point_estimates(values, metric: str) -> dict[str, float]:
    out = {}
    matrices = values.sum(axis=(0, 1))
    for idx, test_input in enumerate(TEST_INPUTS):
        estimate = ESTIMATE_BY_TEST_INPUT[test_input]
        out[estimate] = metric_from_confusion(metric, matrices[idx])
    baseline = out["baseline"]
    for test_input in EDITED_INPUTS:
        estimate = ESTIMATE_BY_TEST_INPUT[test_input]
        out[f"drop_{estimate}"] = baseline - out[estimate]
        out[f"retention_{estimate}"] = out[estimate] / max(baseline, 1e-12)
    return out


def bootstrap(values, labels, metric: str, n_bootstrap: int, seed: int, batch_size: int):
    import numpy as np

    n_seeds, _, _, _, _ = values.shape
    label_indices = [np.flatnonzero(labels == label) for label in sorted(np.unique(labels))]
    point = point_estimates(values, metric)
    estimate_names = list(point.keys())
    boot = {name: np.empty(n_bootstrap, dtype=float) for name in estimate_names}
    rng = np.random.default_rng(seed)

    for start in range(0, n_bootstrap, batch_size):
        stop = min(start + batch_size, n_bootstrap)
        for boot_idx in range(start, stop):
            sampled_seed_idx = rng.integers(0, n_seeds, size=n_seeds)
            total = np.zeros((len(TEST_INPUTS), 2, 2), dtype=np.int64)
            for seed_idx in sampled_seed_idx:
                sampled_subject_idx = []
                for idx in label_indices:
                    sampled_subject_idx.extend(
                        rng.choice(idx, size=len(idx), replace=True).tolist()
                    )
                total += values[seed_idx, np.array(sampled_subject_idx, dtype=int)].sum(axis=0)
            estimates = {}
            for input_idx, test_input in enumerate(TEST_INPUTS):
                estimate = ESTIMATE_BY_TEST_INPUT[test_input]
                estimates[estimate] = metric_from_confusion(metric, total[input_idx])
            baseline = estimates["baseline"]
            for test_input in EDITED_INPUTS:
                estimate = ESTIMATE_BY_TEST_INPUT[test_input]
                estimates[f"drop_{estimate}"] = baseline - estimates[estimate]
                estimates[f"retention_{estimate}"] = estimates[estimate] / max(baseline, 1e-12)
            for name in estimate_names:
                boot[name][boot_idx] = estimates[name]
    return point, boot


def write_markdown(path: Path, rows) -> None:
    display = rows[
        [
            "model",
            "task",
            "metric",
            "estimate",
            "point",
            "ci_lower",
            "ci_upper",
            "n_seeds",
            "n_subjects",
        ]
    ].copy()
    for col in ["point", "ci_lower", "ci_upper"]:
        display[col] = display[col].map(lambda value: f"{value:.3f}")
    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in display.columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    import pandas as pd

    args = parse_args()
    frames = []
    for item in args.input:
        model, seed, path = parse_input(item)
        frames.append(load_prediction_input(model, seed, path))
    data = pd.concat(frames, ignore_index=True)

    rows = []
    for (model, task), group in data.groupby(["model", "task"], sort=False):
        seeds, subjects, labels, values = build_arrays(group)
        for metric in METRICS:
            point, boot = bootstrap(
                values,
                labels,
                metric=metric,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed + stable_seed_offset(model, task, metric),
                batch_size=args.bootstrap_batch_size,
            )
            for estimate, point_value in point.items():
                lower, upper = ci_bounds(boot[estimate], args.ci)
                sign_tests = bootstrap_sign_p_values(boot[estimate])
                rows.append(
                    {
                        "model": model,
                        "task": task,
                        "metric": metric,
                        "estimate": estimate,
                        "point": point_value,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "ci": float(args.ci),
                        "n_seeds": int(len(seeds)),
                        "n_subjects": int(len(subjects)),
                        "n_bootstrap": int(args.n_bootstrap),
                        "bootstrap_unit_1": "seed",
                        "bootstrap_unit_2": "eval_subject_stratified_by_label",
                        "p_one_sided_positive": sign_tests["p_one_sided_positive"],
                        "p_two_sided_zero": sign_tests["p_two_sided_zero"],
                        "n_bootstrap_nonpositive": sign_tests["n_nonpositive"],
                        "n_bootstrap_nonnegative": sign_tests["n_nonnegative"],
                        "n_bootstrap_valid": sign_tests["n_bootstrap_valid"],
                    }
                )

    out = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    write_markdown(args.output_md, out)

    focus = out[
        (out["metric"] == "balanced_accuracy")
        & out["estimate"].isin(
            ["baseline", "sham", "aperiodic", "flattened", "drop_sham", "drop_aperiodic", "drop_flattened"]
        )
    ]
    print(f"Wrote: {args.output_csv}")
    print(f"Wrote: {args.output_md}")
    print(
        focus[
            ["model", "task", "estimate", "point", "ci_lower", "ci_upper", "n_seeds", "n_subjects"]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
