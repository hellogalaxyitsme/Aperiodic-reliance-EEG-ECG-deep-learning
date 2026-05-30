#!/usr/bin/env python3
"""Generate Supplementary Information artifacts for the NMI manuscript.

This script only reads existing result summaries and writes manuscript-side
tables/text. It never reads raw datasets and never modifies any dataset files.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Supplementary Information"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def fmt_float(value: object, digits: int = 3) -> str:
    if value in ("", None):
        return ""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f):
        return ""
    return f"{f:.{digits}f}"


def fmt_ci(row: dict[str, object], digits: int = 3) -> str:
    point = fmt_float(row.get("point"), digits)
    lo = fmt_float(row.get("ci_lower"), digits)
    hi = fmt_float(row.get("ci_upper"), digits)
    if point and lo and hi:
        return f"{point} [{lo}, {hi}]"
    return point


def tex_escape(value: object) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def write_md_table(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w") as f:
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            values = [str(row.get(key, "")).replace("\n", " ") for key in fieldnames]
            f.write("| " + " | ".join(values) + " |\n")


def write_tex_longtable(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
    caption: str,
    label: str,
) -> None:
    alignment = "p{0.10\\linewidth}" * len(fieldnames)
    with path.open("w") as f:
        f.write("\\begin{longtable}{" + alignment + "}\n")
        f.write(f"\\caption{{{tex_escape(caption)}}}\\label{{{label}}}\\\\\n")
        f.write("\\toprule\n")
        f.write(" & ".join(tex_escape(name) for name in fieldnames) + "\\\\\n")
        f.write("\\midrule\n\\endfirsthead\n")
        f.write("\\toprule\n")
        f.write(" & ".join(tex_escape(name) for name in fieldnames) + "\\\\\n")
        f.write("\\midrule\n\\endhead\n")
        for row in rows:
            f.write(" & ".join(tex_escape(row.get(key, "")) for key in fieldnames) + "\\\\\n")
        f.write("\\bottomrule\n\\end{longtable}\n")


def normalize_result_rows() -> list[dict[str, object]]:
    sources = [
        ("Sleep-EDF", "full 78-subject cohort", "reports/tables/full_sleep_edf_multiseed_subject_bootstrap.csv", ""),
        ("Sleep-EDF", "reviewer-resistance controls", "reports/tables/sleep_edf_reviewer_resistance_bootstrap.csv", ""),
        ("TUAB", "full v3.0.1 neural multiseed", "reports/tables/tuab_full_multiseed_neural_subject_bootstrap.csv", ""),
        ("TUAB", "full v3.0.1 age/sex-matched neural", "reports/tables/tuab_full_age_matched_multiseed_neural_subject_bootstrap.csv", ""),
        ("TUAB", "full v3.0.1 foundation-model multiseed", "reports/tables/tuab_full_foundation_multiseed_subject_bootstrap.csv", ""),
        ("PhysioNet MI", "109-subject cohort", "reports/tables/physionet_mi_multiseed_neural_subject_bootstrap.csv", ""),
        ("TUAB", "full v3.0.1 PSD ridge, SpecParam decomposition", "results/tuab_full_v3_0_1/psd_interventions_specparam/tuab_psd_intervention_subject_bootstrap.csv", "ridge"),
        ("TUAB", "full v3.0.1 age/sex-matched PSD ridge", "results/tuab_full_v3_0_1/age_matched/psd_interventions_specparam_tuab_full_age_sex_matched_caliper5/tuab_psd_intervention_subject_bootstrap.csv", "ridge"),
        ("PhysioNet MI", "PSD ridge matrix", "results/physionet_mi/psd_interventions/psd_intervention_subject_bootstrap.csv", "ridge"),
        ("Sleep-EDF", "IRASA ridge matrix", "results/sleep_edf_full/irasa/irasa_ridge_interventions_stage_balanced_5k/irasa_ridge_intervention_subject_bootstrap.csv", "ridge"),
    ]
    rows: list[dict[str, object]] = []
    for domain, cohort, rel, default_model in sources:
        path = ROOT / rel
        for row in read_csv(path):
            train = row.get("train_input", "")
            test = row.get("test_input", "")
            estimate = row.get("estimate", "")
            condition = test if test else estimate
            rows.append(
                {
                    "domain": domain,
                    "cohort": cohort,
                    "task": row.get("task", ""),
                    "model": row.get("model") or default_model,
                    "train_input": train,
                    "test_or_condition": condition,
                    "metric": row.get("metric", ""),
                    "estimate": estimate,
                    "point": fmt_float(row.get("point"), 6),
                    "ci_lower": fmt_float(row.get("ci_lower"), 6),
                    "ci_upper": fmt_float(row.get("ci_upper"), 6),
                    "formatted_95ci": fmt_ci(row, 3),
                    "n_subjects": row.get("n_subjects", row.get("n_eval_subjects", "")),
                    "n_seeds": row.get("n_seeds", ""),
                    "n_bootstrap": row.get("n_bootstrap", ""),
                    "source_file": rel,
                }
            )
    return rows


def specparam_fit_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {
            "dataset": "Sleep-EDF",
            "scope": "full 78-subject cohort",
            "method": "SpecParam fixed aperiodic mode",
            "frequency_range_hz": "1-45",
            "n_epochs_or_trials": "195469",
            "n_channels": "2",
            "n_spectra": "390938",
            "ok_fraction": "1.000",
            "mean_r2": "0.910",
            "median_r2": "0.968",
            "p10_r2": "",
            "mean_mae": "",
            "median_mae": "",
            "mean_n_peaks": "5.457",
            "provenance": "Logged Sleep-EDF full-cohort SpecParam summary in experiments.md/summary.md",
        }
    ]

    tuab = read_json(ROOT / "results/tuab_full_v3_0_1/preprocess_20s_100hz/specparam/specparam_fixed_20s.summary.json")
    if tuab:
        rows.append(
            {
                "dataset": "TUAB",
                "scope": "full v3.0.1 corpus, 20-s epochs",
                "method": "SpecParam fixed aperiodic mode",
                "frequency_range_hz": f"{tuab.get('settings', {}).get('freq_min', '')}-{tuab.get('settings', {}).get('freq_max', '')}",
                "n_epochs_or_trials": str(tuab.get("shape", [""])[0]),
                "n_channels": str(tuab.get("shape", ["", ""])[1]),
                "n_spectra": str(int(tuab.get("shape", [0, 0])[0]) * int(tuab.get("shape", [0, 0])[1])),
                "ok_fraction": fmt_float(tuab.get("ok_fraction"), 3),
                "mean_r2": fmt_float(tuab.get("mean_r_squared"), 3),
                "median_r2": fmt_float(tuab.get("median_r_squared"), 3),
                "p10_r2": fmt_float(tuab.get("p10_r_squared"), 3),
                "mean_mae": fmt_float(tuab.get("mean_error_mae"), 3),
                "median_mae": fmt_float(tuab.get("median_error_mae"), 3),
                "mean_n_peaks": fmt_float(tuab.get("mean_n_peaks"), 3),
                "provenance": "results/tuab_full_v3_0_1/preprocess_20s_100hz/specparam/specparam_fixed_20s.summary.json",
            }
        )

    mi = read_json(ROOT / "results/physionet_mi/specparam/imagined_fists_specparam_fixed.summary.json")
    if mi:
        rows.append(
            {
                "dataset": "PhysioNet MI",
                "scope": "imagined left-vs-right fist, cue-locked trials",
                "method": "SpecParam fixed aperiodic mode on multitaper PSD",
                "frequency_range_hz": f"{mi.get('settings', {}).get('freq_min', '')}-{mi.get('settings', {}).get('freq_max', '')}",
                "n_epochs_or_trials": str(mi.get("shape", [""])[0]),
                "n_channels": str(mi.get("shape", ["", ""])[1]),
                "n_spectra": str(int(mi.get("shape", [0, 0])[0]) * int(mi.get("shape", [0, 0])[1])),
                "ok_fraction": fmt_float(mi.get("ok_fraction"), 3),
                "mean_r2": fmt_float(mi.get("mean_r_squared"), 3),
                "median_r2": fmt_float(mi.get("median_r_squared"), 3),
                "p10_r2": fmt_float(mi.get("p10_r_squared"), 3),
                "mean_mae": fmt_float(mi.get("mean_error_mae"), 3),
                "median_mae": fmt_float(mi.get("median_error_mae"), 3),
                "mean_n_peaks": fmt_float(mi.get("mean_n_peaks"), 3),
                "provenance": "results/physionet_mi/specparam/imagined_fists_specparam_fixed.summary.json",
            }
        )
    return rows


def irasa_rows() -> list[dict[str, object]]:
    rel = "results/sleep_edf_full/irasa/irasa_ridge_interventions_stage_balanced_5k/irasa_ridge_intervention_subject_bootstrap.csv"
    rows = []
    for row in read_csv(ROOT / rel):
        rows.append(
            {
                "task": row.get("task", ""),
                "train_input": row.get("train_input", ""),
                "test_input": row.get("test_input", ""),
                "metric": row.get("metric", ""),
                "estimate": row.get("estimate", ""),
                "point": fmt_float(row.get("point"), 6),
                "ci_lower": fmt_float(row.get("ci_lower"), 6),
                "ci_upper": fmt_float(row.get("ci_upper"), 6),
                "formatted_95ci": fmt_ci(row, 3),
                "n_subjects": row.get("n_subjects", ""),
                "n_bootstrap": row.get("n_bootstrap", ""),
            }
        )
    return rows


def to_float(value: object) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or f < 0 or f > 120:
        return None
    return f


def demographic_rows() -> list[dict[str, object]]:
    unmatched = read_csv(ROOT / "results/tuab_full_v3_0_1/age_matched/tuab_full_age_sex_matched_caliper5_header_metadata_subjects.csv")
    matched = read_csv(ROOT / "results/tuab_full_v3_0_1/age_matched/tuab_full_age_sex_matched_caliper5_subjects.csv")
    pairs = read_csv(ROOT / "results/tuab_full_v3_0_1/age_matched/tuab_full_age_sex_matched_caliper5_pairs.csv")
    summary = read_json(ROOT / "results/tuab_full_v3_0_1/age_matched/tuab_full_age_sex_matched_caliper5_summary.json")

    def summarize_subjects(rows: list[dict[str, str]], cohort: str, age_key: str) -> list[dict[str, object]]:
        grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[(row.get("official_split", ""), row.get("label", ""))].append(row)
        out = []
        for (split, label), group in sorted(grouped.items()):
            ages = [age for age in (to_float(r.get(age_key)) for r in group) if age is not None]
            sexes = defaultdict(int)
            for r in group:
                sexes[r.get("sex", "unknown") or "unknown"] += 1
            out.append(
                {
                    "cohort": cohort,
                    "official_split": split,
                    "label": label,
                    "n_subjects": len(group),
                    "n_valid_age": len(ages),
                    "mean_age": fmt_float(mean(ages) if ages else "", 2),
                    "median_age": fmt_float(median(ages) if ages else "", 2),
                    "min_age": fmt_float(min(ages) if ages else "", 2),
                    "max_age": fmt_float(max(ages) if ages else "", 2),
                    "n_female": sexes.get("female", 0),
                    "n_male": sexes.get("male", 0),
                    "n_unknown_sex": sum(v for k, v in sexes.items() if k not in ("female", "male")),
                    "matching_detail": "",
                }
            )
        return out

    rows = summarize_subjects(unmatched, "full TUAB v3.0.1 corpus with usable header metadata", "age_years_first_available")
    rows.extend(summarize_subjects(matched, "full TUAB age/sex-matched caliper-5 subset", "age_years"))

    for split in sorted({p.get("official_split", "") for p in pairs}):
        split_pairs = [p for p in pairs if p.get("official_split", "") == split]
        diffs = [to_float(p.get("abs_age_diff")) for p in split_pairs]
        diffs = [d for d in diffs if d is not None]
        rows.append(
            {
                "cohort": "matching pairs",
                "official_split": split,
                "label": "abnormal-normal pairs",
                "n_subjects": len(split_pairs) * 2,
                "n_valid_age": len(split_pairs) * 2,
                "mean_age": "",
                "median_age": "",
                "min_age": "",
                "max_age": "",
                "n_female": "",
                "n_male": "",
                "n_unknown_sex": "",
                "matching_detail": (
                    f"{len(split_pairs)} same-sex pairs; mean absolute age difference "
                    f"{fmt_float(mean(diffs) if diffs else '', 2)} years; maximum "
                    f"{fmt_float(max(diffs) if diffs else '', 2)} years"
                ),
            }
        )

    rows.append(
        {
            "cohort": "matching summary",
            "official_split": "all",
            "label": "metadata availability",
            "n_subjects": summary.get("n_input_subject_rows", ""),
            "n_valid_age": summary.get("n_valid_age_subject_rows", ""),
            "mean_age": "",
            "median_age": "",
            "min_age": "",
            "max_age": "",
            "n_female": "",
            "n_male": "",
            "n_unknown_sex": summary.get("n_skipped_subject_rows", ""),
            "matching_detail": (
                f"caliper={summary.get('caliper_years', '')} years; same_sex={summary.get('same_sex', '')}; "
                f"matched_pairs={summary.get('n_pairs_total', '')}; matched_subject_rows={summary.get('n_subject_rows', '')}"
            ),
        }
    )
    return rows


def physionet_matrix_rows() -> list[dict[str, object]]:
    rel = "results/physionet_mi/psd_interventions/psd_intervention_subject_bootstrap.csv"
    rows = []
    for row in read_csv(ROOT / rel):
        if row.get("metric") != "balanced_accuracy":
            continue
        rows.append(
            {
                "task": row.get("task", ""),
                "train_input": row.get("train_input", ""),
                "test_input": row.get("test_input", ""),
                "estimate": row.get("estimate", ""),
                "balanced_accuracy": fmt_float(row.get("point"), 6),
                "ci_lower": fmt_float(row.get("ci_lower"), 6),
                "ci_upper": fmt_float(row.get("ci_upper"), 6),
                "formatted_95ci": fmt_ci(row, 3),
                "n_subjects": row.get("n_subjects", ""),
                "n_bootstrap": row.get("n_bootstrap", ""),
            }
        )
    return rows


def simulation_rows() -> list[dict[str, object]]:
    rel = "reports/tables/aperiodic_simulation_validation/simulation_validation_metrics.csv"
    rows = []
    for row in read_csv(ROOT / rel):
        rows.append(
            {
                "scenario": row.get("scenario", ""),
                "train_input": row.get("train_input", ""),
                "test_input": row.get("test_input", ""),
                "balanced_accuracy_mean": fmt_float(row.get("balanced_accuracy_mean"), 3),
                "balanced_accuracy_std": fmt_float(row.get("balanced_accuracy_std"), 3),
                "n_folds": row.get("n_folds", ""),
            }
        )
    return rows


def foundation_model_config_rows() -> list[dict[str, object]]:
    return [
        {
            "model": "BIOT",
            "checkpoint_or_source": "PREST-16 pretrained encoder",
            "input_window": "10 s at 200 Hz",
            "channels_or_montage": "16 TCP-style bipolar channels",
            "model_specific_preprocessing": "Fine-tuned binary head; BIOT-style window normalization",
            "audit_note": "Raw, sham, aperiodic-shaped and flattened inputs evaluated over three seeds",
        },
        {
            "model": "LaBraM",
            "checkpoint_or_source": "labram-base.pth",
            "input_window": "10 s at 200 Hz",
            "channels_or_montage": "23 referential TUAB channels",
            "model_specific_preprocessing": "0.1-75 Hz bandpass; 50 Hz notch; shared LaBraM-style cache",
            "audit_note": "Raw, sham, aperiodic-shaped and flattened inputs evaluated over three seeds",
        },
        {
            "model": "EEGPT",
            "checkpoint_or_source": "Braindecode EEGPT pretrained checkpoint",
            "input_window": "10 s at 200 Hz",
            "channels_or_montage": "23 referential TUAB channels",
            "model_specific_preprocessing": "0.1-75 Hz bandpass; 50 Hz notch; fine-tuned downstream projection/head",
            "audit_note": "Raw, sham, aperiodic-shaped and flattened inputs evaluated over three seeds",
        },
        {
            "model": "CBraMod",
            "checkpoint_or_source": "cbramod-pretrained",
            "input_window": "10 s at 200 Hz",
            "channels_or_montage": "16 longitudinal bipolar channels derived from the 23-channel cache",
            "model_specific_preprocessing": "Inputs divided by 100 before the encoder",
            "audit_note": "Raw, sham, aperiodic-shaped and flattened inputs evaluated over three seeds",
        },
        {
            "model": "REVE",
            "checkpoint_or_source": "brain-bzh/reve-base",
            "input_window": "10 s at 200 Hz",
            "channels_or_montage": "21 TUAB channels recognized by the REVE position bank",
            "model_specific_preprocessing": "Z-scored inputs clipped to +/-15",
            "audit_note": "Raw, sham, aperiodic-shaped and flattened inputs evaluated over three seeds",
        },
        {
            "model": "EEGMamba",
            "checkpoint_or_source": "weighting666/EEGMamba",
            "input_window": "10 s at 200 Hz",
            "channels_or_montage": "16 longitudinal bipolar channels derived from the 23-channel cache",
            "model_specific_preprocessing": "Inputs divided by 100; run in the isolated SSM-compatible environment",
            "audit_note": "Raw, sham, aperiodic-shaped and flattened inputs evaluated over three seeds",
        },
        {
            "model": "BENDR",
            "checkpoint_or_source": "Braindecode BENDR checkpoint",
            "input_window": "10 s at 200 Hz",
            "channels_or_montage": "23 referential TUAB channels",
            "model_specific_preprocessing": "0.1-75 Hz bandpass; 50 Hz notch; encoder-only mode; +/-500 uV clip followed by 1e-6 volt scaling",
            "audit_note": "Reported separately as intervention fragility because the sham condition collapsed to chance",
        },
    ]


def preprocessing_rows() -> list[dict[str, object]]:
    return [
        {
            "dataset": "Sleep-EDF Expanded Sleep Cassette",
            "analysis_scope": "78 subjects; wake-vs-sleep, five-stage staging and N2-vs-N3 tasks",
            "epoch_or_trial_definition": "30 s hypnogram-aligned epochs; wake trimmed to retain wake within 30 min of sleep",
            "channels_or_reference": "Fpz-Cz and Pz-Oz EEG derivations",
            "filtering_and_resampling": "EDF volt units converted to microvolts; zero-phase fourth-order Butterworth 0.5-45 Hz bandpass; resampled to 100 Hz; per-epoch channel mean removed",
            "exclusion_or_rejection": "Epochs extending outside the PSG recording were discarded; no additional amplitude-threshold artifact rejection",
            "spectral_estimation": "SpecParam fixed-mode fits over 1-45 Hz for aperiodic and flattened spectra",
        },
        {
            "dataset": "TUAB v3.0.1",
            "analysis_scope": "Full official train/evaluation split; 2,993 EDF files; 204,122 raw-neural epochs; 253 evaluation subjects",
            "epoch_or_trial_definition": "Non-overlapping 20 s windows with 20 s stride",
            "channels_or_reference": "21 referential channels: FP1, FP2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, A1, A2, FZ, CZ, PZ",
            "filtering_and_resampling": "Signals converted to microvolts; zero-phase fourth-order Butterworth 1-45 Hz bandpass; resampled to 100 Hz; per-window channel mean removed",
            "exclusion_or_rejection": "Windows with incomplete samples or missing required channels were excluded; no additional amplitude-threshold artifact rejection",
            "spectral_estimation": "Multitaper PSDs over 1-45 Hz with 2 Hz bandwidth; SpecParam fixed-mode fits over 1-45 Hz with up to six peaks",
        },
        {
            "dataset": "PhysioNet EEG Motor Movement/Imagery",
            "analysis_scope": "109 subjects; imagined left-vs-right fist task",
            "epoch_or_trial_definition": "Cue-locked T1/T2 trials from 0.5-4.0 s after cue; rest annotations excluded",
            "channels_or_reference": "All 64 EEG channels from the PhysioNet MI recordings",
            "filtering_and_resampling": "EDF volt units converted to microvolts; zero-phase fourth-order Butterworth 1-45 Hz bandpass; resampled to 160 Hz; per-trial channel mean removed",
            "exclusion_or_rejection": "Trials extending beyond the EDF recording boundary were discarded; no additional amplitude-threshold artifact rejection",
            "spectral_estimation": "Multitaper PSDs; SpecParam fixed-mode fits over 2-45 Hz because short trials give coarser low-frequency resolution",
        },
    ]


def write_supplementary_tex() -> None:
    text = r"""\section*{Supplementary Information}

This Supplementary Information accompanies the manuscript ``Aperiodic spectral structure is a task-dependent confound in EEG deep learning''. All tables below were generated from the result CSV and JSON artifacts in the repository. Subject-level confidence intervals use the bootstrap summaries reported in the corresponding result files.

\subsection*{Supplementary Table 1. Complete numerical results}

Supplementary Table 1 reports all available domain $\times$ task $\times$ model $\times$ condition combinations with confidence intervals. It includes PSD ridge baselines, raw neural interventions, reviewer-resistance controls, full-TUAB foundation-model audits, full-TUAB age/sex-matched analyses and IRASA-derived ridge interventions.

\input{supplementary_table_1_complete_results.tex}

\subsection*{Supplementary Table 2. SpecParam fit quality by dataset}

Supplementary Table 2 reports decomposition quality for the SpecParam fixed-mode fits used to construct aperiodic-shaped and flattened spectra. For PhysioNet MI, fit quality is reported explicitly because short cue-locked trials have coarser Fourier resolution than the 20--30 s epochs used in TUAB and Sleep-EDF.

\input{supplementary_table_2_specparam_fit_quality.tex}

\subsection*{Supplementary Table 3. IRASA downstream ridge intervention}

Supplementary Table 3 repeats the downstream ridge intervention using IRASA-derived aperiodic and flattened spectra for Sleep-EDF. The goal is to test whether the main PSD-ridge conclusion depends on the SpecParam decomposition.

\input{supplementary_table_3_irasa_ridge_intervention.tex}

\subsection*{Supplementary Table 4. TUAB age metadata and matching}

Supplementary Table 4 reports demographic metadata availability and the same-sex, $\pm$5-year age-matching control used for the full-TUAB confound analysis. The age-matched analysis is not a replacement for the main TUAB audit; it is a control that asks whether aperiodic reliance survives after reducing the known age imbalance between abnormal and normal recordings.

\input{supplementary_table_4_tuab_age_matching.tex}

\subsection*{Supplementary Table 5. PhysioNet MI PSD intervention matrix}

Supplementary Table 5 reports the complete train-by-test PSD ridge intervention matrix for PhysioNet Motor Movement/Imagery. This matrix separates aperiodic sufficiency, flattened residual performance and train/test mismatch effects in the short-trial motor-imagery domain.

\input{supplementary_table_5_physionet_mi_psd_matrix.tex}

\subsection*{Supplementary Table 6. Foundation-model audit configurations}

Supplementary Table 6 lists the checkpoint, input channel and preprocessing configuration used for each full-TUAB foundation-model audit. All models were evaluated on the same normal-versus-abnormal target under raw, sham, aperiodic-shaped and flattened conditions. BENDR is listed for completeness, but its intervention result is interpreted as fragility because the sham condition collapsed to chance.

\input{supplementary_table_6_foundation_model_config.tex}

\subsection*{Supplementary Table 7. Dataset preprocessing details}

Supplementary Table 7 provides the concrete preprocessing choices used for each dataset, including epoch or trial definitions, channel handling, filtering, resampling and exclusion rules.

\input{supplementary_table_7_preprocessing_details.tex}

\subsection*{Supplementary Note 1. Mathematical framework}

\paragraph{Theorem 1: aperiodic slope changes induce apparent band-power changes.}
Assume a pure aperiodic power spectrum
\[
S(f)=A f^{-\chi}, \qquad f \in [f_{\min}, f_{\max}].
\]
For any frequency band $I=[u,v]$ with $u>0$, the band power is
\[
B_I(A,\chi)=A\int_u^v f^{-\chi}\,df.
\]
If $\chi\neq1$,
\[
B_I(A,\chi)=A\frac{v^{1-\chi}-u^{1-\chi}}{1-\chi},
\]
and if $\chi=1$,
\[
B_I(A,1)=A\log(v/u).
\]
Thus $\partial B_I/\partial A>0$ and $\partial B_I/\partial \chi$ is generally non-zero. Absolute band-power differences can therefore arise from broadband offset or exponent changes even when no periodic oscillatory peak changes. A reported change in theta, alpha or beta power is not automatically an oscillatory effect unless the aperiodic component has been accounted for.

\paragraph{Theorem 2: band ratios can act as aperiodic-exponent detectors.}
For two bands $I=[u,v]$ and $J=[r,s]$ under a pure aperiodic spectrum,
\[
\mathrm{BR}_{I,J}(\chi)=
\frac{\int_u^v f^{-\chi}\,df}{\int_r^s f^{-\chi}\,df}.
\]
The scale parameter $A$ cancels, but the exponent $\chi$ remains. Ratios such as theta/beta or theta/alpha can therefore increase when the aperiodic exponent steepens, even without a theta peak increase or beta peak decrease. Ratio-based EEG biomarkers should therefore be interpreted jointly with aperiodic exponent and periodic peak estimates.

\paragraph{Theorem 3: local classifier sensitivity decomposes into aperiodic and periodic components.}
Let a differentiable classifier produce a class logit $h_\theta(y)$ for vectorized log-spectrum $y$. Around $y$,
\[
h_\theta(y+\delta)-h_\theta(y)=\langle \nabla_y h_\theta(y),\delta\rangle+O(\|\delta\|^2).
\]
Define an aperiodic tangent space $\mathcal{A}_i$ spanned by derivatives of the aperiodic model with respect to its parameters. In fixed mode this space is spanned by $\{1,-\log f\}$ across channel-frequency bins; in knee mode it also includes the knee derivative. Let $P_\mathcal{A}$ be the orthogonal projection onto this tangent space and $P_\mathcal{P}=I-P_\mathcal{A}$. The first-order logit gradient decomposes as
\[
\nabla_y h_\theta = P_\mathcal{A}\nabla_y h_\theta + P_\mathcal{P}\nabla_y h_\theta.
\]
The aperiodic attribution ratio can be written
\[
\mathrm{AAR}_i=
\frac{\|P_\mathcal{A}\nabla_y h_\theta(y_i)\|_2^2}
{\|\nabla_y h_\theta(y_i)\|_2^2},
\]
with $\mathrm{OAR}_i=1-\mathrm{AAR}_i$. This formalizes the distinction between sensitivity aligned with broadband aperiodic structure and sensitivity aligned with residual periodic structure.

\subsection*{Supplementary Note 2. Simulation validation protocol and results}

Synthetic validation is necessary because real EEG does not provide ground-truth labels for aperiodic and periodic spectral generators. We used four simulation families: pure aperiodic classification, pure periodic classification, mixed aperiodic-plus-periodic classification and confound-shift classification. Simulated spectra were evaluated through the same full-spectrum, aperiodic-spectrum and flattened-spectrum ridge intervention matrix used for the empirical PSD analyses.

In the pure aperiodic scenario, full-spectrum and aperiodic-spectrum classifiers performed near ceiling, while flattened spectra collapsed to chance. In the pure periodic scenario, flattened/residual inputs preserved performance and aperiodic-only inputs did not dominate. In the mixed scenario, performance varied monotonically with the injected mixture of broadband and periodic class evidence. In the train-confounded/test-unconfounded scenario, models trained on shortcut aperiodic structure failed when the shortcut was removed at test time. These patterns demonstrate that the audit has the intended sensitivity and specificity: it detects aperiodic reliance when the ground-truth signal is aperiodic, preserves periodic evidence when present and exposes shortcut dependence under distribution shift.

\input{supplementary_note_2_simulation_results.tex}

\subsection*{Supplementary Note 3. Proposed aperiodic reporting checklist}

We propose the following checklist for EEG machine-learning studies that interpret performance or learned features in terms of oscillatory band power.

\begin{enumerate}
\item Report the PSD estimator, frequency range, epoch/window length, tapering or Welch parameters and any frequency exclusion rules.
\item Report the aperiodic decomposition method, model mode, parameter settings and fit-quality summary by dataset and class.
\item Report periodic-adjusted band power or residual spectra whenever making oscillatory claims.
\item Include aperiodic-only and flattened/residual controls for PSD-based models.
\item For raw EEG models, include phase-preserving Fourier interventions and a sham reconstruction control.
\item Aggregate uncertainty at the subject level and report the bootstrap or mixed-effects unit explicitly.
\item Keep official dataset train/test boundaries intact and prevent subject leakage across splits.
\item Report demographic and acquisition confounds that can influence aperiodic structure, including age, sex, site and montage when available.
\item When conclusions depend on spectral decomposition, include a second decomposition method or sensitivity analysis, such as SpecParam--IRASA agreement.
\item Validate the audit pipeline on simulations with known aperiodic and periodic ground truth.
\item Distinguish physiological aperiodic biomarkers from oscillatory mechanisms in the interpretation.
\item Release subject-selection rules, random seeds, model checkpoints or prediction files sufficient to reproduce each reported intervention table.
\end{enumerate}
"""
    (OUT / "supplementary_information.tex").write_text(text)


def write_readme() -> None:
    readme = """# Supplementary Information

This folder contains the generated Supplementary Information package for the NMI manuscript draft.

Generated files:

- `supplementary_information.tex`: main SI text with Supplementary Tables 1-7 and Supplementary Notes 1-3.
- `supplementary_table_1_complete_results.*`: normalized table of all available domain x task x model x condition results with confidence intervals.
- `supplementary_table_2_specparam_fit_quality.*`: SpecParam fit-quality summary by dataset, including full-TUAB fixed-mode fits.
- `supplementary_table_3_irasa_ridge_intervention.*`: Sleep-EDF IRASA downstream ridge intervention results.
- `supplementary_table_4_tuab_age_matching.*`: Full-TUAB demographic metadata and age/sex-matching details.
- `supplementary_table_5_physionet_mi_psd_matrix.*`: PhysioNet MI PSD ridge train-by-test intervention matrix.
- `supplementary_table_6_foundation_model_config.*`: Full-TUAB foundation-model checkpoint and preprocessing configurations.
- `supplementary_table_7_preprocessing_details.*`: Dataset-level preprocessing, channel, filtering and exclusion details.
- `supplementary_note_2_simulation_results.*`: synthetic validation results used in Supplementary Note 2.

Regenerate with:

```bash
python3 code/scripts/generate_nmi_supplementary_information.py
```

The generator reads only result CSV/JSON artifacts and does not touch raw datasets.
"""
    (OUT / "README.md").write_text(readme)


def main() -> None:
    OUT.mkdir(exist_ok=True)

    table1 = normalize_result_rows()
    table1_fields = [
        "domain",
        "cohort",
        "task",
        "model",
        "train_input",
        "test_or_condition",
        "metric",
        "estimate",
        "point",
        "ci_lower",
        "ci_upper",
        "formatted_95ci",
        "n_subjects",
        "n_seeds",
        "n_bootstrap",
        "source_file",
    ]
    write_csv(OUT / "supplementary_table_1_complete_results.csv", table1, table1_fields)
    write_md_table(OUT / "supplementary_table_1_complete_results.md", table1, table1_fields)
    write_tex_longtable(
        OUT / "supplementary_table_1_complete_results.tex",
        table1,
        ["domain", "cohort", "task", "model", "test_or_condition", "metric", "estimate", "formatted_95ci"],
        "Complete numerical results for all available domain, task, model and condition combinations.",
        "tab:supp_complete_results",
    )

    table2 = specparam_fit_rows()
    table2_fields = [
        "dataset",
        "scope",
        "method",
        "frequency_range_hz",
        "n_epochs_or_trials",
        "n_channels",
        "n_spectra",
        "ok_fraction",
        "mean_r2",
        "median_r2",
        "p10_r2",
        "mean_mae",
        "median_mae",
        "mean_n_peaks",
        "provenance",
    ]
    write_csv(OUT / "supplementary_table_2_specparam_fit_quality.csv", table2, table2_fields)
    write_md_table(OUT / "supplementary_table_2_specparam_fit_quality.md", table2, table2_fields)
    write_tex_longtable(
        OUT / "supplementary_table_2_specparam_fit_quality.tex",
        table2,
        ["dataset", "scope", "frequency_range_hz", "n_spectra", "ok_fraction", "mean_r2", "median_r2", "mean_mae", "mean_n_peaks"],
        "SpecParam fit quality summary by dataset.",
        "tab:supp_specparam_fit_quality",
    )

    table3 = irasa_rows()
    table3_fields = ["task", "train_input", "test_input", "metric", "estimate", "point", "ci_lower", "ci_upper", "formatted_95ci", "n_subjects", "n_bootstrap"]
    write_csv(OUT / "supplementary_table_3_irasa_ridge_intervention.csv", table3, table3_fields)
    write_md_table(OUT / "supplementary_table_3_irasa_ridge_intervention.md", table3, table3_fields)
    write_tex_longtable(
        OUT / "supplementary_table_3_irasa_ridge_intervention.tex",
        table3,
        ["task", "train_input", "test_input", "metric", "estimate", "formatted_95ci"],
        "IRASA downstream ridge intervention results.",
        "tab:supp_irasa_ridge",
    )

    table4 = demographic_rows()
    table4_fields = [
        "cohort",
        "official_split",
        "label",
        "n_subjects",
        "n_valid_age",
        "mean_age",
        "median_age",
        "min_age",
        "max_age",
        "n_female",
        "n_male",
        "n_unknown_sex",
        "matching_detail",
    ]
    write_csv(OUT / "supplementary_table_4_tuab_age_matching.csv", table4, table4_fields)
    write_md_table(OUT / "supplementary_table_4_tuab_age_matching.md", table4, table4_fields)
    write_tex_longtable(
        OUT / "supplementary_table_4_tuab_age_matching.tex",
        table4,
        ["cohort", "official_split", "label", "n_subjects", "n_valid_age", "mean_age", "n_female", "n_male", "matching_detail"],
        "TUAB age metadata and age/sex matching details.",
        "tab:supp_tuab_age_matching",
    )

    table5 = physionet_matrix_rows()
    table5_fields = ["task", "train_input", "test_input", "estimate", "balanced_accuracy", "ci_lower", "ci_upper", "formatted_95ci", "n_subjects", "n_bootstrap"]
    write_csv(OUT / "supplementary_table_5_physionet_mi_psd_matrix.csv", table5, table5_fields)
    write_md_table(OUT / "supplementary_table_5_physionet_mi_psd_matrix.md", table5, table5_fields)
    write_tex_longtable(
        OUT / "supplementary_table_5_physionet_mi_psd_matrix.tex",
        table5,
        ["task", "train_input", "test_input", "estimate", "formatted_95ci"],
        "PhysioNet MI PSD intervention matrix.",
        "tab:supp_physionet_mi_psd_matrix",
    )

    table6 = foundation_model_config_rows()
    table6_fields = [
        "model",
        "checkpoint_or_source",
        "input_window",
        "channels_or_montage",
        "model_specific_preprocessing",
        "audit_note",
    ]
    write_csv(OUT / "supplementary_table_6_foundation_model_config.csv", table6, table6_fields)
    write_md_table(OUT / "supplementary_table_6_foundation_model_config.md", table6, table6_fields)
    write_tex_longtable(
        OUT / "supplementary_table_6_foundation_model_config.tex",
        table6,
        table6_fields,
        "Foundation-model audit configurations for full-TUAB normal-versus-abnormal classification.",
        "tab:fm_config",
    )

    table7 = preprocessing_rows()
    table7_fields = [
        "dataset",
        "analysis_scope",
        "epoch_or_trial_definition",
        "channels_or_reference",
        "filtering_and_resampling",
        "exclusion_or_rejection",
        "spectral_estimation",
    ]
    write_csv(OUT / "supplementary_table_7_preprocessing_details.csv", table7, table7_fields)
    write_md_table(OUT / "supplementary_table_7_preprocessing_details.md", table7, table7_fields)
    write_tex_longtable(
        OUT / "supplementary_table_7_preprocessing_details.tex",
        table7,
        table7_fields,
        "Dataset preprocessing details.",
        "tab:supp_preprocessing_details",
    )

    sim = simulation_rows()
    sim_fields = ["scenario", "train_input", "test_input", "balanced_accuracy_mean", "balanced_accuracy_std", "n_folds"]
    write_csv(OUT / "supplementary_note_2_simulation_results.csv", sim, sim_fields)
    write_md_table(OUT / "supplementary_note_2_simulation_results.md", sim, sim_fields)
    write_tex_longtable(
        OUT / "supplementary_note_2_simulation_results.tex",
        sim,
        sim_fields,
        "Simulation validation results.",
        "tab:supp_simulation_validation",
    )

    write_supplementary_tex()
    write_readme()
    print(f"Wrote Supplementary Information package to {OUT}")
    print(f"Supplementary Table 1 rows: {len(table1)}")
    print(f"Supplementary Table 2 rows: {len(table2)}")
    print(f"Supplementary Table 3 rows: {len(table3)}")
    print(f"Supplementary Table 4 rows: {len(table4)}")
    print(f"Supplementary Table 5 rows: {len(table5)}")
    print(f"Supplementary Table 6 rows: {len(table6)}")
    print(f"Supplementary Table 7 rows: {len(table7)}")
    print(f"Simulation result rows: {len(sim)}")


if __name__ == "__main__":
    main()
