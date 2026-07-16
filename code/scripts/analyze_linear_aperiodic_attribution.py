#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute linear aperiodic attribution ratio for ridge spectral models."
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=Path("results/sleep_edf_subset/psd_index.csv"),
    )
    parser.add_argument(
        "--decomp-npz",
        type=Path,
        default=Path("results/sleep_edf_subset/specparam/specparam_fixed.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/sleep_edf_subset/linear_attribution"),
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    return parser.parse_args()


def make_task_labels(stages):
    import numpy as np

    stages = np.asarray(stages)
    return {
        "wake_vs_sleep": np.where(stages == "W", "W", "Sleep"),
        "n2_vs_n3": np.where(np.isin(stages, ["N2", "N3"]), stages, None),
        "five_stage": stages,
    }


def encode_labels(labels):
    import numpy as np
    from sklearn.preprocessing import LabelEncoder

    labels = np.asarray(labels)
    mask = labels != None  # noqa: E711
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels[mask])
    return mask, y, encoder.classes_.tolist()


def make_group_folds(groups, n_splits: int):
    import numpy as np
    from sklearn.model_selection import GroupKFold

    unique_groups = np.unique(groups)
    n_splits = min(n_splits, len(unique_groups))
    splitter = GroupKFold(n_splits=n_splits)
    placeholder_y = np.zeros(len(groups), dtype=int)
    return list(splitter.split(placeholder_y, groups=groups))


def aperiodic_projection_matrix(freqs):
    import numpy as np

    basis = np.column_stack([np.ones_like(freqs), -np.log(freqs)])
    return basis @ np.linalg.pinv(basis)


def coefficient_aar(coef_matrix, projection):
    import numpy as np

    projected = coef_matrix @ projection.T
    total_energy = float(np.sum(coef_matrix**2))
    aperiodic_energy = float(np.sum(projected**2))
    if total_energy <= 0:
        return 0.0, 0.0, 0.0
    return aperiodic_energy / total_energy, aperiodic_energy, total_energy


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows):
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(rows)
    summaries = []
    for key, group in df.groupby(["task", "class_label"], sort=False):
        task, class_label = key
        vals = group["aar"].to_numpy(dtype=float)
        summaries.append(
            {
                "task": task,
                "class_label": class_label,
                "n": int(len(group)),
                "aar_mean": float(vals.mean()),
                "aar_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "aar_median": float(np.median(vals)),
                "aar_min": float(vals.min()),
                "aar_max": float(vals.max()),
            }
        )

    for task, group in df.groupby("task", sort=False):
        vals = group["aar"].to_numpy(dtype=float)
        summaries.append(
            {
                "task": task,
                "class_label": "__overall__",
                "n": int(len(group)),
                "aar_mean": float(vals.mean()),
                "aar_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "aar_median": float(np.median(vals)),
                "aar_min": float(vals.min()),
                "aar_max": float(vals.max()),
            }
        )
    return summaries


def plot_summary(summary_rows, output_png: Path, output_pdf: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(summary_rows)
    overall = df[df["class_label"] == "__overall__"].copy()
    labels = {
        "wake_vs_sleep": "Wake vs Sleep",
        "n2_vs_n3": "N2 vs N3",
        "five_stage": "Five-stage",
    }
    overall["label"] = overall["task"].map(labels)
    order = ["wake_vs_sleep", "n2_vs_n3", "five_stage"]
    overall["order"] = overall["task"].map({task: idx for idx, task in enumerate(order)})
    overall = overall.sort_values("order")

    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.bar(
        overall["label"],
        overall["aar_mean"],
        yerr=overall["aar_std"],
        color="#4c7c9f",
        edgecolor="#222222",
        linewidth=0.7,
        capsize=3,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Aperiodic Attribution Ratio")
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import RidgeClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    args = parse_args()
    index = pd.read_csv(args.index_csv)
    decomp = np.load(args.decomp_npz)

    log_psd = decomp["log_psd"]
    freqs = decomp["freqs"]
    channels = [str(ch) for ch in decomp["channels"].tolist()]
    features = log_psd.reshape(log_psd.shape[0], -1)
    projection = aperiodic_projection_matrix(freqs)

    task_labels = make_task_labels(index["stage"].to_numpy())
    groups_all = index["subject"].to_numpy()

    rows = []
    coef_rows = []
    for task_name, labels in task_labels.items():
        mask, y, classes = encode_labels(labels)
        task_groups = groups_all[mask]
        task_features = features[mask]
        folds = make_group_folds(task_groups, args.n_splits)

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            clf = make_pipeline(
                StandardScaler(),
                RidgeClassifier(
                    alpha=args.ridge_alpha,
                    class_weight="balanced",
                    solver="lsqr",
                ),
            )
            clf.fit(task_features[train_idx], y[train_idx])
            scaler = clf.named_steps["standardscaler"]
            ridge = clf.named_steps["ridgeclassifier"]

            coef = ridge.coef_
            if coef.ndim == 1:
                coef = coef[None, :]

            # Convert coefficients from standardized feature units back to
            # original log-PSD units. This is the local linear sensitivity.
            original_units = coef / np.maximum(scaler.scale_[None, :], 1e-12)
            coef_tensor = original_units.reshape(coef.shape[0], len(channels), len(freqs))

            if len(classes) == 2 and coef_tensor.shape[0] == 1:
                class_items = [(classes[1], coef_tensor[0]), (classes[0], -coef_tensor[0])]
            else:
                class_items = [
                    (classes[class_idx], coef_tensor[class_idx])
                    for class_idx in range(coef_tensor.shape[0])
                ]

            for class_label, class_coef in class_items:
                aar, ap_energy, total_energy = coefficient_aar(class_coef, projection)
                rows.append(
                    {
                        "task": task_name,
                        "fold": fold_idx,
                        "class_label": class_label,
                        "aar": aar,
                        "aperiodic_energy": ap_energy,
                        "total_energy": total_energy,
                        "train_subjects": ",".join(sorted(np.unique(task_groups[train_idx]))),
                        "test_subjects": ",".join(sorted(np.unique(task_groups[test_idx]))),
                    }
                )

                projected = class_coef @ projection.T
                residual = class_coef - projected
                for ch_idx, channel in enumerate(channels):
                    coef_rows.append(
                        {
                            "task": task_name,
                            "fold": fold_idx,
                            "class_label": class_label,
                            "channel": channel,
                            "aperiodic_energy": float(np.sum(projected[ch_idx] ** 2)),
                            "residual_energy": float(np.sum(residual[ch_idx] ** 2)),
                            "total_energy": float(np.sum(class_coef[ch_idx] ** 2)),
                        }
                    )

    summary_rows = summarize_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "linear_aar_by_fold.csv", rows)
    write_csv(args.output_dir / "linear_aar_by_channel.csv", coef_rows)
    write_csv(args.output_dir / "linear_aar_summary.csv", summary_rows)

    plot_summary(
        summary_rows,
        args.output_dir / "linear_aar_summary.png",
        args.output_dir / "linear_aar_summary.pdf",
    )

    print(f"Wrote outputs to: {args.output_dir}")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

