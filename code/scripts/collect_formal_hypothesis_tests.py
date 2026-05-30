#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path


Q_DEFAULT = 0.05


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collect bootstrap p-values from the primary intervention summaries "
            "and apply Benjamini-Hochberg FDR correction."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/mnt/data/aperiodic_confounds"),
    )
    parser.add_argument("--q", type=float, default=Q_DEFAULT)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("reports/tables/formal_hypothesis_tests"),
    )
    return parser.parse_args()


def read_table(project_root: Path, rel: str):
    import pandas as pd

    path = project_root / rel
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {
        "model",
        "task",
        "metric",
        "estimate",
        "point",
        "ci_lower",
        "ci_upper",
        "n_subjects",
        "n_bootstrap",
        "p_one_sided_positive",
        "p_two_sided_zero",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame["source_file"] = rel
    return frame


def bh_adjust(p_values):
    import numpy as np

    p = np.asarray(p_values, dtype=float)
    m = len(p)
    adjusted = np.full(m, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(p))
    if len(finite) == 0:
        return adjusted

    order = finite[np.argsort(p[finite])]
    ranked = p[order]
    raw_adj = ranked * len(finite) / np.arange(1, len(finite) + 1)
    monotone = np.minimum.accumulate(raw_adj[::-1])[::-1]
    adjusted[order] = np.minimum(monotone, 1.0)
    return adjusted


def p_display(value, n_bootstrap):
    if value is None or not math.isfinite(float(value)):
        return ""
    value = float(value)
    try:
        n_bootstrap = int(n_bootstrap)
    except (TypeError, ValueError):
        n_bootstrap = 10000
    if value == 0.0 and n_bootstrap > 0:
        return f"<{1.0 / n_bootstrap:.4f}"
    if value < 0.0001:
        return "<0.0001"
    return f"{value:.4f}"


def fmt_ci(row):
    return f"{float(row['effect']):.3f} [{float(row['ci_lower']):.3f}, {float(row['ci_upper']):.3f}]"


def add_family_results(rows, q: float):
    import pandas as pd

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["p_fdr"] = math.nan
    out["significant_fdr"] = False
    for family, idx in out.groupby("family").groups.items():
        adjusted = bh_adjust(out.loc[idx, "p_raw"].to_numpy(dtype=float))
        out.loc[idx, "p_fdr"] = adjusted
        out.loc[idx, "significant_fdr"] = adjusted <= q

    out["effect_95ci"] = out.apply(fmt_ci, axis=1)
    out["p_raw_display"] = out.apply(
        lambda row: p_display(row["p_raw"], row["n_bootstrap"]), axis=1
    )
    out["p_fdr_display"] = out.apply(
        lambda row: p_display(row["p_fdr"], row["n_bootstrap"]), axis=1
    )
    out["significance"] = out["significant_fdr"].map(lambda value: "*" if bool(value) else "ns")
    return out


def pick_row(frame, *, model, task, estimate, metric="balanced_accuracy"):
    rows = frame[
        (frame["model"] == model)
        & (frame["task"] == task)
        & (frame["metric"] == metric)
        & (frame["estimate"] == estimate)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one row for model={model}, task={task}, metric={metric}, "
            f"estimate={estimate}; found {len(rows)}"
        )
    return rows.iloc[0]


def append_test(rows, frame, *, family, domain, task, model, display_model, estimate, hypothesis, p_column, note=""):
    row = pick_row(frame, model=model, task=task, estimate=estimate)
    rows.append(
        {
            "family": family,
            "domain": domain,
            "task": task,
            "model": display_model,
            "metric": row["metric"],
            "estimate": estimate,
            "effect": float(row["point"]),
            "ci_lower": float(row["ci_lower"]),
            "ci_upper": float(row["ci_upper"]),
            "p_raw": float(row[p_column]),
            "hypothesis": hypothesis,
            "n_subjects": int(row["n_subjects"]),
            "n_seeds": row.get("n_seeds", ""),
            "n_bootstrap": int(row["n_bootstrap"]),
            "source_file": row["source_file"],
            "note": note,
        }
    )


def collect_primary_and_sham(project_root: Path):
    tables = {
        "sleep_full": read_table(project_root, "reports/tables/full_sleep_edf_multiseed_subject_bootstrap.csv"),
        "sleep_reviewer": read_table(project_root, "reports/tables/sleep_edf_reviewer_resistance_bootstrap.csv"),
        "tuab_neural": read_table(project_root, "reports/tables/tuab_full_multiseed_neural_subject_bootstrap.csv"),
        "tuab_matched": read_table(project_root, "reports/tables/tuab_full_age_matched_multiseed_neural_subject_bootstrap.csv"),
        "tuab_fm": read_table(project_root, "reports/tables/tuab_full_foundation_multiseed_subject_bootstrap.csv"),
        "physionet": read_table(project_root, "reports/tables/physionet_mi_multiseed_neural_subject_bootstrap.csv"),
    }

    rows = []
    primary_hypothesis = "H0: flattening drop <= 0; H1: flattening drop > 0"
    sham_hypothesis = "H0: sham drop = 0; H1: sham drop != 0"

    sleep_tasks = ["wake_vs_sleep", "five_stage", "n2_vs_n3"]
    for task in sleep_tasks:
        append_test(
            rows,
            tables["sleep_full"],
            family="primary_flattening",
            domain="Sleep-EDF",
            task=task,
            model="deep_mlp",
            display_model="MLP",
            estimate="drop_flattened",
            hypothesis=primary_hypothesis,
            p_column="p_one_sided_positive",
        )
        for model, display in [
            ("raw_cnn_sham", "CNN"),
            ("braindecode_eegnet", "EEGNet"),
            ("braindecode_shallow_fbcsp", "ShallowFBCSPNet"),
            ("braindecode_deep4", "Deep4Net"),
        ]:
            append_test(
                rows,
                tables["sleep_reviewer"],
                family="primary_flattening",
                domain="Sleep-EDF",
                task=task,
                model=model,
                display_model=display,
                estimate="drop_flattened",
                hypothesis=primary_hypothesis,
                p_column="p_one_sided_positive",
            )
            append_test(
                rows,
                tables["sleep_reviewer"],
                family="sham_controls",
                domain="Sleep-EDF",
                task=task,
                model=model,
                display_model=display,
                estimate="drop_sham",
                hypothesis=sham_hypothesis,
                p_column="p_two_sided_zero",
            )

    for model, display in [
        ("eegnet", "EEGNet"),
        ("shallow_fbcsp", "ShallowFBCSPNet"),
        ("deep4", "Deep4Net"),
    ]:
        append_test(
            rows,
            tables["tuab_neural"],
            family="primary_flattening",
            domain="TUAB full",
            task="tuab_normal_vs_abnormal",
            model=model,
            display_model=display,
            estimate="drop_flattened",
            hypothesis=primary_hypothesis,
            p_column="p_one_sided_positive",
        )
        append_test(
            rows,
            tables["tuab_neural"],
            family="sham_controls",
            domain="TUAB full",
            task="tuab_normal_vs_abnormal",
            model=model,
            display_model=display,
            estimate="drop_sham",
            hypothesis=sham_hypothesis,
            p_column="p_two_sided_zero",
        )

        append_test(
            rows,
            tables["tuab_matched"],
            family="primary_flattening",
            domain="TUAB full age/sex-matched",
            task="tuab_normal_vs_abnormal",
            model=model,
            display_model=display,
            estimate="drop_flattened",
            hypothesis=primary_hypothesis,
            p_column="p_one_sided_positive",
        )
        append_test(
            rows,
            tables["tuab_matched"],
            family="sham_controls",
            domain="TUAB full age/sex-matched",
            task="tuab_normal_vs_abnormal",
            model=model,
            display_model=display,
            estimate="drop_sham",
            hypothesis=sham_hypothesis,
            p_column="p_two_sided_zero",
        )

    for model in ["BIOT", "LaBraM", "EEGPT", "CBraMod", "REVE-base", "EEGMamba", "BENDR"]:
        note = ""
        display = "REVE" if model == "REVE-base" else model
        if model == "BENDR":
            note = "BENDR sham collapsed to chance; flattening result is reported as intervention fragility, not clean aperiodic reliance."
        append_test(
            rows,
            tables["tuab_fm"],
            family="primary_flattening",
            domain="TUAB full foundation models",
            task="tuab_normal_vs_abnormal",
            model=model,
            display_model=display,
            estimate="drop_flattened",
            hypothesis=primary_hypothesis,
            p_column="p_one_sided_positive",
            note=note,
        )
        append_test(
            rows,
            tables["tuab_fm"],
            family="sham_controls",
            domain="TUAB full foundation models",
            task="tuab_normal_vs_abnormal",
            model=model,
            display_model=display,
            estimate="drop_sham",
            hypothesis=sham_hypothesis,
            p_column="p_two_sided_zero",
            note=note,
        )

    for model, display in [
        ("eegnet", "EEGNet"),
        ("shallow_fbcsp", "ShallowFBCSPNet"),
        ("deep4", "Deep4Net"),
    ]:
        append_test(
            rows,
            tables["physionet"],
            family="primary_flattening",
            domain="PhysioNet MI",
            task="imagined_left_vs_right_fist",
            model=model,
            display_model=display,
            estimate="drop_flattened",
            hypothesis=primary_hypothesis,
            p_column="p_one_sided_positive",
        )
        append_test(
            rows,
            tables["physionet"],
            family="sham_controls",
            domain="PhysioNet MI",
            task="imagined_left_vs_right_fist",
            model=model,
            display_model=display,
            estimate="drop_sham",
            hypothesis=sham_hypothesis,
            p_column="p_two_sided_zero",
        )

    return rows


def write_markdown(path: Path, frame):
    display_cols = [
        "family",
        "domain",
        "task",
        "model",
        "estimate",
        "effect_95ci",
        "p_raw_display",
        "p_fdr_display",
        "significance",
        "note",
    ]
    lines = [
        "| " + " | ".join(display_cols) + " |",
        "| " + " | ".join(["---"] * len(display_cols)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in display_cols) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    rows = collect_primary_and_sham(args.project_root)
    out = add_family_results(rows, q=args.q)

    output_prefix = args.output_prefix
    if not output_prefix.is_absolute():
        output_prefix = args.project_root / output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(output_prefix.with_suffix(".csv"), index=False)
    write_markdown(output_prefix.with_suffix(".md"), out)

    primary = out[out["family"] == "primary_flattening"].copy()
    sham = out[out["family"] == "sham_controls"].copy()
    primary.to_csv(output_prefix.with_name(output_prefix.name + "_primary.csv"), index=False)
    sham.to_csv(output_prefix.with_name(output_prefix.name + "_sham.csv"), index=False)

    print(f"Wrote: {output_prefix.with_suffix('.csv')}")
    print(f"Wrote: {output_prefix.with_suffix('.md')}")
    print(f"Primary flattening tests: {len(primary)}")
    print(f"Primary FDR-significant at q={args.q}: {int(primary['significant_fdr'].sum())}")
    print(f"Sham control tests: {len(sham)}")
    print(f"Sham FDR-significant at q={args.q}: {int(sham['significant_fdr'].sum())}")
    print(
        primary[
            [
                "domain",
                "task",
                "model",
                "effect_95ci",
                "p_raw_display",
                "p_fdr_display",
                "significance",
                "note",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
