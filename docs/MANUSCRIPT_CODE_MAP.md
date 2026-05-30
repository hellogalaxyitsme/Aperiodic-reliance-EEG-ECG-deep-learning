# Manuscript-to-Code Map

This map links the main experimental claims to the scripts and aggregate tables
included in the repository.

## Spectral Audit Framework

- Core utilities: `code/src/aperiodic_eeg/spectral.py`
- PSD intervention protocol: `code/scripts/run_sleep_edf_aperiodic_baselines.py`,
  `code/scripts/run_tuab_psd_interventions.py`,
  `code/scripts/run_physionet_mi_psd_interventions.py`,
  `code/scripts/run_ptbxl_psd_interventions.py`
- Raw Fourier interventions: model-specific `run_*_intervention*.py` scripts.

## Sleep-EDF

- Preprocessing: `code/scripts/make_sleep_edf_epochs.py`,
  `code/scripts/extract_sleep_edf_raw_epochs.py`, `code/scripts/extract_sleep_edf_psd.py`
- SpecParam/IRASA: `code/scripts/fit_sleep_edf_specparam.py`,
  `code/scripts/fit_sleep_edf_irasa.py`, `code/scripts/compare_specparam_irasa.py`
- Neural audits: `code/scripts/run_sleep_edf_braindecode_eegnet_intervention.py`,
  `code/scripts/run_sleep_edf_raw_cnn_intervention.py`,
  `code/scripts/run_sleep_edf_deep_intervention_mlp.py`,
  `code/scripts/launch_sleep_edf_full_multiseed_neural.sh`
- Aggregate tables: `results/tables/full_sleep_edf_multiseed_subject_bootstrap.*`,
  `results/tables/sleep_edf_reviewer_resistance_bootstrap.*`

## TUAB Full Corpus

- Manifest/header metadata: `code/scripts/make_tuab_full_manifest.py`,
  `code/scripts/extract_tuab_header_metadata.py`
- Preprocessing: `code/scripts/extract_tuab_raw_epochs.py`,
  `code/scripts/extract_tuab_psd.py`, `code/scripts/run_tuab_full_preprocessing.sh`
- Neural audits: `code/scripts/launch_tuab_full_multiseed_neural.sh`
- Age/sex matching: `code/scripts/make_tuab_age_matched_subset.py`,
  `code/scripts/launch_tuab_full_age_matched_multiseed_neural.sh`,
  `code/scripts/launch_tuab_full_age_matched_psd.sh`
- Temporal acquisition-proxy audit:
  `code/scripts/run_tuab_site_temporal_psd_audit.py`
- Aggregate tables:
  `results/tables/tuab_full_multiseed_neural_subject_bootstrap.*`,
  `results/tables/tuab_full_age_matched_multiseed_neural_subject_bootstrap.*`

## TUAB Foundation Models

- Sequential multi-seed launcher:
  `code/scripts/launch_tuab_full_foundation_multiseed_sequential.sh`
- Model scripts:
  `code/scripts/run_tuab_biot_intervention.py`,
  `code/scripts/run_tuab_labram_intervention.py`,
  `code/scripts/run_tuab_eegpt_intervention.py`,
  `code/scripts/run_tuab_cbramod_intervention.py`,
  `code/scripts/run_tuab_reve_intervention.py`,
  `code/scripts/run_tuab_eegmamba_intervention.py`,
  `code/scripts/run_tuab_bendr_intervention.py`
- Aggregation:
  `code/scripts/aggregate_foundation_multiseed_predictions.py`
- Aggregate table:
  `results/tables/tuab_full_foundation_multiseed_subject_bootstrap.*`

## PhysioNet Motor Imagery

- Trial preparation: `code/scripts/make_physionet_mi_trials.py`
- PSD/raw extraction: `code/scripts/extract_physionet_mi_psd.py`,
  `code/scripts/extract_physionet_mi_raw_trials.py`
- Neural audits: `code/scripts/launch_physionet_mi_multiseed_neural.sh`
- Aggregate table:
  `results/tables/physionet_mi_multiseed_neural_subject_bootstrap.*`

## PTB-XL ECG

- Preparation: `code/scripts/prepare_ptbxl_1f_demo.py`
- PSD audit: `code/scripts/run_ptbxl_psd_interventions.py`
- Neural architecture audit: `code/scripts/run_ptbxl_raw_cnn_interventions.py`,
  `code/scripts/launch_ptbxl_ecg_architecture_audit.sh`
- Age/sex controls: `code/scripts/run_ptbxl_age_sex_matched_psd.py`,
  `code/scripts/launch_ptbxl_age_sex_matched_neural.sh`
- Aggregation:
  `code/scripts/aggregate_ptbxl_prediction_bootstrap.py`
- Aggregate tables:
  `results/tables/ptbxl_ecg_architectures_prediction_bootstrap.*`,
  `results/tables/ptbxl_matched_ecg_architectures_prediction_bootstrap.*`

## Formal Hypothesis Tests and Figures

- FDR tests: `code/scripts/collect_formal_hypothesis_tests.py`,
  `code/scripts/run_formal_hypothesis_tests_from_saved_outputs.sh`
- Main/Extended figures:
  `code/scripts/generate_nmi_journal_images.py`,
  `code/scripts/generate_nmi_extended_data_figures.py`,
  `code/scripts/generate_nmi_update_figures_matplotlib.py`
- Supplementary tables:
  `code/scripts/generate_nmi_supplementary_information.py`,
  `code/scripts/generate_nmi_update_assets.py`

