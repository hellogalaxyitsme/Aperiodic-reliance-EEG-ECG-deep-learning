# Reproducibility Guide

The workflow has four stages: obtain datasets, build preprocessing caches, run
audits and aggregate results. Full neural and foundation-model experiments are
GPU intensive; PSD and aggregation stages can usually run on CPU.

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Foundation models may require additional dependencies or local checkouts. See
[`FOUNDATION_MODEL_CONFIGS.md`](FOUNDATION_MODEL_CONFIGS.md).

## 2. Dataset Preparation

Download public datasets directly from their official sources and place them
under `data/`. TUAB requires authorized access from the TUH EEG Corpus.

```bash
bash code/scripts/download_sleep_edf_all_sleep_cassette.sh
bash code/scripts/download_physionet_mi.sh
bash code/scripts/download_ptbxl_records100.sh
bash code/scripts/download_tuab_full_resumable.sh
```

Expected dataset locations are listed in
[`DATA_ACCESS.md`](DATA_ACCESS.md).

## 3. Preprocessing

Representative preprocessing entry points:

```bash
bash code/scripts/run_sleep_edf_full_preprocessing.sh
bash code/scripts/run_tuab_full_preprocessing.sh
python code/scripts/prepare_ptbxl_1f_demo.py --help
python code/scripts/make_physionet_mi_trials.py --help
```

The preprocessing choices used in the experiments are summarized in
[`PREPROCESSING.md`](PREPROCESSING.md).

## 4. Main Audits

```bash
bash code/scripts/launch_sleep_edf_full_multiseed_neural.sh
bash code/scripts/launch_sleep_edf_validation_controls.sh
bash code/scripts/launch_tuab_full_multiseed_neural.sh
bash code/scripts/launch_tuab_full_foundation_multiseed_sequential.sh
bash code/scripts/launch_physionet_mi_multiseed_neural.sh
bash code/scripts/launch_ptbxl_ecg_architecture_audit.sh
bash code/scripts/launch_ptbxl_age_sex_matched_neural.sh
python code/scripts/run_tuab_site_temporal_psd_audit.py --help
```

The launch scripts save model outputs under `results/`. If running on another
cluster, set `PROJECT_ROOT`, `PYTHON`, dataset roots and checkpoint paths
explicitly.

## 5. Aggregation and Statistical Testing

```bash
python code/scripts/aggregate_multiseed_subject_bootstrap.py --help
python code/scripts/aggregate_foundation_multiseed_predictions.py --help
python code/scripts/aggregate_ptbxl_prediction_bootstrap.py --help
bash code/scripts/run_formal_hypothesis_tests_from_saved_outputs.sh
```

The repository includes aggregate outputs under `results/tables/` so that the
reported numerical summaries and hypothesis-test tables can be inspected
without rerunning the full GPU workloads.
