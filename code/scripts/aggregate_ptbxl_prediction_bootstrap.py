#!/usr/bin/env python
from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_MAP = {
    "raw_eeg": "baseline",
    "phase_sham": "sham",
    "phase_aperiodic": "aperiodic",
    "phase_flattened": "flattened",
}
INPUT_ORDER = ["baseline", "sham", "aperiodic", "flattened"]
METRICS = ["balanced_accuracy", "macro_f1", "accuracy"]
ESTIMATES = [
    "baseline",
    "sham",
    "aperiodic",
    "flattened",
    "retention_sham",
    "retention_aperiodic",
    "retention_flattened",
    "drop_sham",
    "drop_aperiodic",
    "drop_flattened",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate PTB-XL intervention prediction CSVs with pooled-confusion patient bootstrap."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Model input in the form model_name=/path/to/predictions.csv",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-batch-size", type=int, default=1000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--unit-column",
        default="subject",
        help="Bootstrap unit column. Use pair_id for age/sex-matched PTB-XL controls.",
    )
    return parser.parse_args()


def stable_seed_offset(*parts) -> int:
    text = "::".join(str(part) for part in parts)
    return zlib.crc32(text.encode("utf-8")) % 1_000_000


def parse_model_input(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"Expected model=path, got {text}")
    model, path = text.split("=", 1)
    return model.strip(), Path(path)


def ci_bounds(values: np.ndarray, ci: float):
    alpha = 1.0 - ci
    return float(np.quantile(values, alpha / 2.0)), float(np.quantile(values, 1.0 - alpha / 2.0))


def sign_p_values(values: np.ndarray):
    values = values[np.isfinite(values)]
    n = int(len(values))
    if n == 0:
        return np.nan, np.nan, 0, 0, 0
    n_nonpositive = int(np.count_nonzero(values <= 0.0))
    n_nonnegative = int(np.count_nonzero(values >= 0.0))
    p_one = n_nonpositive / n
    p_two = min(1.0, 2.0 * min(n_nonpositive / n, n_nonnegative / n))
    return float(p_one), float(p_two), n_nonpositive, n_nonnegative, n


def metrics_from_counts(counts: np.ndarray):
    # counts layout: tn, fp, fn, tp
    tn = counts[..., 0].astype(float)
    fp = counts[..., 1].astype(float)
    fn = counts[..., 2].astype(float)
    tp = counts[..., 3].astype(float)
    total = np.maximum(tn + fp + fn + tp, 1.0)
    accuracy = (tn + tp) / total
    recall0 = tn / np.maximum(tn + fp, 1e-12)
    recall1 = tp / np.maximum(tp + fn, 1e-12)
    balanced_accuracy = 0.5 * (recall0 + recall1)
    f1_0 = 2.0 * tn / np.maximum(2.0 * tn + fn + fp, 1e-12)
    f1_1 = 2.0 * tp / np.maximum(2.0 * tp + fp + fn, 1e-12)
    macro_f1 = 0.5 * (f1_0 + f1_1)
    return {
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
    }


def estimate_from_metric(metric_values: dict[str, np.ndarray]):
    baseline = metric_values["baseline"]
    out = {
        "baseline": baseline,
        "sham": metric_values["sham"],
        "aperiodic": metric_values["aperiodic"],
        "flattened": metric_values["flattened"],
    }
    out["retention_sham"] = out["sham"] / np.maximum(baseline, 1e-12)
    out["retention_aperiodic"] = out["aperiodic"] / np.maximum(baseline, 1e-12)
    out["retention_flattened"] = out["flattened"] / np.maximum(baseline, 1e-12)
    out["drop_sham"] = baseline - out["sham"]
    out["drop_aperiodic"] = baseline - out["aperiodic"]
    out["drop_flattened"] = baseline - out["flattened"]
    return out


def prepare_count_cube(group: pd.DataFrame, unit_column: str):
    seeds = np.array(sorted(group["seed"].unique()))
    if unit_column not in group.columns:
        raise ValueError(f"Missing bootstrap unit column: {unit_column}")
    units = np.array(sorted(group[unit_column].astype(str).unique()))
    seed_index = {seed: i for i, seed in enumerate(seeds)}
    unit_index = {unit: i for i, unit in enumerate(units)}
    input_index = {name: i for i, name in enumerate(INPUT_ORDER)}
    counts = np.zeros((len(seeds), len(units), len(INPUT_ORDER), 4), dtype=np.int64)

    for row in group.itertuples(index=False):
        si = seed_index[row.seed]
        unit = unit_index[str(getattr(row, unit_column))]
        ii = input_index[row.canonical_input]
        y_true = int(row.y_true)
        y_pred = int(row.y_pred)
        if y_true == 0 and y_pred == 0:
            counts[si, unit, ii, 0] += 1
        elif y_true == 0 and y_pred == 1:
            counts[si, unit, ii, 1] += 1
        elif y_true == 1 and y_pred == 0:
            counts[si, unit, ii, 2] += 1
        elif y_true == 1 and y_pred == 1:
            counts[si, unit, ii, 3] += 1
        else:
            raise ValueError(f"Unexpected labels y_true={y_true}, y_pred={y_pred}")
    return counts, len(seeds), len(units)


def bootstrap_model(group: pd.DataFrame, n_bootstrap: int, seed: int, batch_size: int, unit_column: str):
    counts, n_seeds, n_units = prepare_count_cube(group, unit_column)
    point_counts = counts.sum(axis=(0, 1))
    point_metrics = metrics_from_counts(point_counts)
    point = {
        metric: estimate_from_metric({inp: point_metrics[metric][idx] for idx, inp in enumerate(INPUT_ORDER)})
        for metric in METRICS
    }

    rng = np.random.default_rng(seed)
    boot = {metric: {estimate: np.empty(n_bootstrap, dtype=float) for estimate in ESTIMATES} for metric in METRICS}
    for start in range(0, n_bootstrap, batch_size):
        stop = min(start + batch_size, n_bootstrap)
        current = stop - start
        seed_idx = rng.integers(0, n_seeds, size=(current, n_seeds))
        unit_idx = rng.integers(0, n_units, size=(current, n_units))
        sampled = counts[seed_idx[:, :, None], unit_idx[:, None, :], :, :]
        pooled = sampled.sum(axis=(1, 2))
        pooled_metrics = metrics_from_counts(pooled)
        for metric in METRICS:
            metric_values = {inp: pooled_metrics[metric][:, idx] for idx, inp in enumerate(INPUT_ORDER)}
            estimates = estimate_from_metric(metric_values)
            for estimate, values in estimates.items():
                boot[metric][estimate][start:stop] = values
    return point, boot, n_seeds, n_units


def write_markdown(path: Path, rows: pd.DataFrame):
    display = rows[
        ["model", "task", "metric", "estimate", "point", "ci_lower", "ci_upper", "n_seeds", "n_subjects"]
    ].copy()
    for col in ["point", "ci_lower", "ci_upper"]:
        display[col] = display[col].map(lambda value: f"{value:.3f}")
    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in display.columns) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    frames = []
    args = parse_args()
    for item in args.input:
        model, path = parse_model_input(item)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame["model"] = model
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data["canonical_input"] = data["test_input"].map(INPUT_MAP)
    unknown = sorted(data.loc[data["canonical_input"].isna(), "test_input"].unique())
    if unknown:
        raise ValueError(f"Unknown test_input values: {unknown}")
    data["subject"] = data["subject"].astype(str)

    rows = []
    for (model, task), group in data.groupby(["model", "task"], sort=False):
        point, boot, n_seeds, n_units = bootstrap_model(
            group,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + stable_seed_offset(model, task),
            batch_size=args.bootstrap_batch_size,
            unit_column=args.unit_column,
        )
        for metric in METRICS:
            for estimate in ESTIMATES:
                values = boot[metric][estimate]
                lower, upper = ci_bounds(values, args.ci)
                p_one, p_two, n_nonpos, n_nonneg, n_valid = sign_p_values(values)
                rows.append(
                    {
                        "model": model,
                        "task": task,
                        "metric": metric,
                        "estimate": estimate,
                        "point": float(point[metric][estimate]),
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "ci": float(args.ci),
                        "n_seeds": n_seeds,
                        "n_subjects": n_units,
                        "n_bootstrap": int(args.n_bootstrap),
                        "bootstrap_unit_1": "seed",
                        "bootstrap_unit_2": args.unit_column,
                        "p_one_sided_positive": p_one,
                        "p_two_sided_zero": p_two,
                        "n_bootstrap_nonpositive": n_nonpos,
                        "n_bootstrap_nonnegative": n_nonneg,
                        "n_bootstrap_valid": n_valid,
                    }
                )

    out = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    write_markdown(args.output_md, out)
    focus = out[
        (out["metric"] == "balanced_accuracy")
        & out["estimate"].isin(["baseline", "sham", "aperiodic", "flattened", "drop_sham", "drop_flattened"])
    ]
    print(f"Wrote: {args.output_csv}")
    print(f"Wrote: {args.output_md}")
    print(focus[["model", "task", "estimate", "point", "ci_lower", "ci_upper", "n_seeds", "n_subjects"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
