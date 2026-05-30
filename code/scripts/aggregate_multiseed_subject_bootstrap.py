#!/usr/bin/env python
from __future__ import annotations

import argparse
import zlib
from pathlib import Path


METRICS = ["balanced_accuracy", "macro_f1", "accuracy"]
CANONICAL_INPUTS = {
    "full_log_psd": "baseline",
    "raw_eeg": "baseline",
    "phase_sham": "sham",
    "aperiodic_spectrum": "aperiodic",
    "phase_aperiodic": "aperiodic",
    "flattened_log_psd": "flattened",
    "phase_flattened": "flattened",
}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate multiseed neural subject metrics with a two-level "
            "seed/subject hierarchical bootstrap."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Model input in the form model_name=/path/to/subject_metrics.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/reports/tables/full_sleep_edf_multiseed_subject_bootstrap.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds/reports/tables/full_sleep_edf_multiseed_subject_bootstrap.md"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument(
        "--bootstrap-batch-size",
        type=int,
        default=2000,
        help="Number of bootstrap replicates to vectorize at once.",
    )
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stable_seed_offset(*parts) -> int:
    text = "::".join(str(part) for part in parts)
    return zlib.crc32(text.encode("utf-8")) % 1_000_000


def parse_model_input(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"Expected model=path input, got: {text}")
    model, path = text.split("=", 1)
    model = model.strip()
    if not model:
        raise ValueError(f"Missing model name in input: {text}")
    return model, Path(path)


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


def estimate_values(sample, metric: str) -> dict[str, float]:
    pivot = sample.pivot_table(
        index=["seed", "subject"],
        columns="canonical_input",
        values=metric,
        aggfunc="mean",
    )
    pivot = pivot.dropna(subset=["baseline", "aperiodic", "flattened"])
    if pivot.empty:
        raise ValueError("No complete seed/subject rows after pivot.")

    baseline = float(pivot["baseline"].mean())
    aperiodic = float(pivot["aperiodic"].mean())
    flattened = float(pivot["flattened"].mean())
    return {
        "baseline": baseline,
        "aperiodic": aperiodic,
        "flattened": flattened,
        "retention_aperiodic": aperiodic / max(baseline, 1e-12),
        "retention_flattened": flattened / max(baseline, 1e-12),
        "drop_aperiodic": baseline - aperiodic,
        "drop_flattened": baseline - flattened,
    }


def estimates_from_means(means) -> dict[str, float]:
    import numpy as np

    baseline = float(means[0])
    sham = float(means[1])
    aperiodic = float(means[2])
    flattened = float(means[3])
    out = {
        "baseline": baseline,
    }
    if np.isfinite(sham):
        out["sham"] = sham
        out["retention_sham"] = sham / max(baseline, 1e-12)
        out["drop_sham"] = baseline - sham
    if np.isfinite(aperiodic):
        out["aperiodic"] = aperiodic
        out["retention_aperiodic"] = aperiodic / max(baseline, 1e-12)
        out["drop_aperiodic"] = baseline - aperiodic
    if np.isfinite(flattened):
        out["flattened"] = flattened
        out["retention_flattened"] = flattened / max(baseline, 1e-12)
        out["drop_flattened"] = baseline - flattened
    return out


def prepare_metric_cube(group, metric: str):
    import numpy as np
    import pandas as pd

    seeds = np.array(sorted(group["seed"].unique()))
    subjects = np.array(sorted(group["subject"].unique()))
    full_index = pd.MultiIndex.from_product(
        [seeds, subjects], names=["seed", "subject"]
    )
    pivot = group.pivot_table(
        index=["seed", "subject"],
        columns="canonical_input",
        values=metric,
        aggfunc="mean",
    )
    pivot = pivot.reindex(
        index=full_index, columns=["baseline", "sham", "aperiodic", "flattened"]
    )
    values = pivot.to_numpy(dtype=float).reshape(len(seeds), len(subjects), 4).copy()

    # Keep the bootstrap unit identical across the three inputs: a seed/subject
    # pair contributes only when all intervention conditions are available.
    required_complete = np.isfinite(values[:, :, [0, 2, 3]]).all(axis=2)
    values[~required_complete] = np.nan
    if not np.isfinite(values).any():
        raise ValueError("No complete seed/subject rows after pivot.")
    return values, int(len(seeds)), int(len(subjects))


def hierarchical_bootstrap(
    group,
    metric: str,
    n_bootstrap: int,
    seed: int,
    batch_size: int,
):
    import numpy as np

    values, n_seeds, n_subjects = prepare_metric_cube(group, metric)
    point = estimates_from_means(np.nanmean(values, axis=(0, 1)))
    rng = np.random.default_rng(seed)
    boot = {estimate: np.empty(n_bootstrap, dtype=float) for estimate in ESTIMATES}

    for start in range(0, n_bootstrap, batch_size):
        stop = min(start + batch_size, n_bootstrap)
        current = stop - start
        seed_idx = rng.integers(0, n_seeds, size=(current, n_seeds))
        subject_idx = rng.integers(0, n_subjects, size=(current, n_subjects))
        sampled = values[seed_idx[:, :, None], subject_idx[:, None, :], :]
        means = np.nanmean(sampled, axis=(1, 2))

        baseline = means[:, 0]
        sham = means[:, 1]
        aperiodic = means[:, 2]
        flattened = means[:, 3]
        boot["baseline"][start:stop] = baseline
        boot["sham"][start:stop] = sham
        boot["aperiodic"][start:stop] = aperiodic
        boot["flattened"][start:stop] = flattened
        boot["retention_sham"][start:stop] = sham / np.maximum(
            baseline, 1e-12
        )
        boot["retention_aperiodic"][start:stop] = aperiodic / np.maximum(
            baseline, 1e-12
        )
        boot["retention_flattened"][start:stop] = flattened / np.maximum(
            baseline, 1e-12
        )
        boot["drop_sham"][start:stop] = baseline - sham
        boot["drop_aperiodic"][start:stop] = baseline - aperiodic
        boot["drop_flattened"][start:stop] = baseline - flattened

    return point, boot, n_seeds, n_subjects


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
        model, path = parse_model_input(item)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame["model"] = model
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data["canonical_input"] = data["test_input"].map(CANONICAL_INPUTS)
    unknown = sorted(data.loc[data["canonical_input"].isna(), "test_input"].unique())
    if unknown:
        raise ValueError(f"Unknown test_input values: {unknown}")
    if "seed" not in data.columns:
        raise ValueError("Subject metric inputs must include a seed column.")

    rows = []
    for (model, task), group in data.groupby(["model", "task"], sort=False):
        for metric in METRICS:
            point, boot, n_seeds, n_subjects = hierarchical_bootstrap(
                group,
                metric=metric,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed + stable_seed_offset(model, task, metric),
                batch_size=args.bootstrap_batch_size,
            )
            for estimate in ESTIMATES:
                if estimate not in point:
                    continue
                values = boot[estimate]
                values = values[~pd.isna(values)]
                if len(values) == 0:
                    continue
                lower, upper = ci_bounds(values, args.ci)
                sign_tests = bootstrap_sign_p_values(values)
                rows.append(
                    {
                        "model": model,
                        "task": task,
                        "metric": metric,
                        "estimate": estimate,
                        "point": point[estimate],
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "ci": float(args.ci),
                        "n_seeds": n_seeds,
                        "n_subjects": n_subjects,
                        "n_bootstrap": int(args.n_bootstrap),
                        "bootstrap_unit_1": "seed",
                        "bootstrap_unit_2": "subject",
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
            ["baseline", "sham", "aperiodic", "flattened", "drop_sham", "drop_flattened"]
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
