#!/usr/bin/env python
from __future__ import annotations

import argparse
import zlib
from pathlib import Path


DEFAULT_INPUTS = {
    "linear_ridge": "results/sleep_edf_subset/interventions_specparam_flattening/intervention_fold_metrics.csv",
    "deep_mlp": "results/sleep_edf_subset/deep_mlp_interventions_specparam/deep_mlp_intervention_fold_metrics.csv",
    "raw_cnn": "results/sleep_edf_subset/raw_cnn_interventions/raw_cnn_intervention_fold_metrics.csv",
    "braindecode_eegnet": "results/sleep_edf_subset/braindecode_eegnet_interventions/braindecode_eegnet_intervention_fold_metrics.csv",
}

MODEL_INPUTS = {
    "linear_ridge": ["full_log_psd", "aperiodic_spectrum", "flattened_log_psd"],
    "deep_mlp": ["full_log_psd", "aperiodic_spectrum", "flattened_log_psd"],
    "raw_cnn": ["raw_eeg", "phase_aperiodic", "phase_flattened"],
    "braindecode_eegnet": ["raw_eeg", "phase_aperiodic", "phase_flattened"],
}

METRICS = ["balanced_accuracy", "macro_f1", "accuracy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap intervention metrics from paired subject-fold metrics."
    )
    parser.add_argument(
        "--linear-fold-csv",
        type=Path,
        default=Path(DEFAULT_INPUTS["linear_ridge"]),
    )
    parser.add_argument(
        "--deep-fold-csv",
        type=Path,
        default=Path(DEFAULT_INPUTS["deep_mlp"]),
    )
    parser.add_argument(
        "--raw-cnn-fold-csv",
        type=Path,
        default=Path(DEFAULT_INPUTS["raw_cnn"]),
    )
    parser.add_argument(
        "--braindecode-eegnet-fold-csv",
        type=Path,
        default=Path(DEFAULT_INPUTS["braindecode_eegnet"]),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("reports/tables/intervention_bootstrap_ci.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("reports/tables/intervention_bootstrap_ci.md"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ci", type=float, default=0.95)
    return parser.parse_args()


def bootstrap_values(values_by_fold, n_bootstrap: int, seed: int):
    import numpy as np

    rng = np.random.default_rng(seed)
    values_by_fold = np.asarray(values_by_fold, dtype=float)
    n_folds = values_by_fold.shape[0]
    boot = np.empty(n_bootstrap, dtype=float)
    for idx in range(n_bootstrap):
        sample = rng.integers(0, n_folds, size=n_folds)
        boot[idx] = values_by_fold[sample].mean()
    return boot


def ci_bounds(values, ci: float):
    import numpy as np

    alpha = 1.0 - ci
    return (
        float(np.quantile(values, alpha / 2.0)),
        float(np.quantile(values, 1.0 - alpha / 2.0)),
    )


def stable_seed_offset(*parts) -> int:
    text = "::".join(str(part) for part in parts)
    return zlib.crc32(text.encode("utf-8")) % 1_000_000


def paired_bootstrap_task(task_df, metric: str, test_inputs: list[str], n_bootstrap: int, seed: int):
    import numpy as np

    pivot = task_df.pivot(index="fold", columns="test_input", values=metric)
    pivot = pivot[test_inputs].dropna()
    if len(pivot) < 2:
        raise ValueError("Need at least two complete folds for bootstrap.")

    rng = np.random.default_rng(seed)
    values = pivot.to_numpy(dtype=float)
    n_folds = values.shape[0]
    baseline_input, aperiodic_input, flattened_input = test_inputs
    columns = {name: idx for idx, name in enumerate(test_inputs)}

    out = {
        baseline_input: np.empty(n_bootstrap, dtype=float),
        aperiodic_input: np.empty(n_bootstrap, dtype=float),
        flattened_input: np.empty(n_bootstrap, dtype=float),
        "retention_aperiodic": np.empty(n_bootstrap, dtype=float),
        "retention_flattened": np.empty(n_bootstrap, dtype=float),
        "drop_aperiodic": np.empty(n_bootstrap, dtype=float),
        "drop_flattened": np.empty(n_bootstrap, dtype=float),
    }

    for boot_idx in range(n_bootstrap):
        sample = rng.integers(0, n_folds, size=n_folds)
        means = values[sample].mean(axis=0)
        full = means[columns[baseline_input]]
        ap = means[columns[aperiodic_input]]
        flat = means[columns[flattened_input]]

        out[baseline_input][boot_idx] = full
        out[aperiodic_input][boot_idx] = ap
        out[flattened_input][boot_idx] = flat
        out["retention_aperiodic"][boot_idx] = ap / max(full, 1e-12)
        out["retention_flattened"][boot_idx] = flat / max(full, 1e-12)
        out["drop_aperiodic"][boot_idx] = full - ap
        out["drop_flattened"][boot_idx] = full - flat

    point = {
        baseline_input: float(pivot[baseline_input].mean()),
        aperiodic_input: float(pivot[aperiodic_input].mean()),
        flattened_input: float(pivot[flattened_input].mean()),
    }
    point["retention_aperiodic"] = point[aperiodic_input] / max(point[baseline_input], 1e-12)
    point["retention_flattened"] = point[flattened_input] / max(point[baseline_input], 1e-12)
    point["drop_aperiodic"] = point[baseline_input] - point[aperiodic_input]
    point["drop_flattened"] = point[baseline_input] - point[flattened_input]

    return point, out, int(len(pivot))


def load_model_rows(
    model_name: str,
    path: Path,
    test_inputs: list[str],
    n_bootstrap: int,
    seed: int,
    ci: float,
):
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(path)
    if len(test_inputs) != 3:
        raise ValueError(f"Expected three test inputs for {model_name}, got {test_inputs}")

    df = pd.read_csv(path)
    rows = []
    for task, task_df in df.groupby("task", sort=False):
        for metric in METRICS:
            point, boot, n_folds = paired_bootstrap_task(
                task_df,
                metric=metric,
                test_inputs=test_inputs,
                n_bootstrap=n_bootstrap,
                seed=seed + stable_seed_offset(model_name, task, metric),
            )
            for estimate, boot_values in boot.items():
                lower, upper = ci_bounds(boot_values, ci)
                rows.append(
                    {
                        "model": model_name,
                        "task": task,
                        "metric": metric,
                        "estimate": estimate,
                        "baseline_input": test_inputs[0],
                        "aperiodic_input": test_inputs[1],
                        "flattened_input": test_inputs[2],
                        "point": point[estimate],
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "ci": ci,
                        "n_folds": n_folds,
                        "n_bootstrap": n_bootstrap,
                    }
                )
    return rows


def write_markdown(path: Path, rows) -> None:
    display_cols = [
        "model",
        "task",
        "metric",
        "estimate",
        "point",
        "ci_lower",
        "ci_upper",
    ]
    rounded = rows[display_cols].copy()
    for col in ["point", "ci_lower", "ci_upper"]:
        rounded[col] = rounded[col].map(lambda val: f"{val:.3f}")

    lines = [
        "| " + " | ".join(display_cols) + " |",
        "| " + " | ".join(["---"] * len(display_cols)) + " |",
    ]
    for _, row in rounded.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in display_cols) + " |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    import pandas as pd

    args = parse_args()
    all_rows = []
    all_rows.extend(
        load_model_rows(
            "linear_ridge",
            args.linear_fold_csv,
            test_inputs=MODEL_INPUTS["linear_ridge"],
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            ci=args.ci,
        )
    )
    all_rows.extend(
        load_model_rows(
            "deep_mlp",
            args.deep_fold_csv,
            test_inputs=MODEL_INPUTS["deep_mlp"],
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + 10_000,
            ci=args.ci,
        )
    )
    all_rows.extend(
        load_model_rows(
            "raw_cnn",
            args.raw_cnn_fold_csv,
            test_inputs=MODEL_INPUTS["raw_cnn"],
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + 20_000,
            ci=args.ci,
        )
    )
    all_rows.extend(
        load_model_rows(
            "braindecode_eegnet",
            args.braindecode_eegnet_fold_csv,
            test_inputs=MODEL_INPUTS["braindecode_eegnet"],
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + 30_000,
            ci=args.ci,
        )
    )

    out = pd.DataFrame(all_rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    write_markdown(args.output_md, out)

    focus = out[
        (out["metric"] == "balanced_accuracy")
        & out["estimate"].isin(
            [
                "full_log_psd",
                "aperiodic_spectrum",
                "flattened_log_psd",
                "raw_eeg",
                "phase_aperiodic",
                "phase_flattened",
                "retention_aperiodic",
                "drop_flattened",
            ]
        )
    ].copy()
    print(f"Wrote: {args.output_csv}")
    print(f"Wrote: {args.output_md}")
    print(
        focus[
            ["model", "task", "estimate", "point", "ci_lower", "ci_upper", "n_folds"]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
