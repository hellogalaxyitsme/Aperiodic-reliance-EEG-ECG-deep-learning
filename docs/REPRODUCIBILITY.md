# Reproducibility Notes

The original full experiments were run on an H200 server because the full TUAB
and foundation-model audits require GPU acceleration and large local datasets.
The commands below describe the intended order of execution. Paths can be
changed through the command-line arguments in each script.

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Foundation-model audits may require additional model-specific dependencies and
downloaded checkpoints. See `supplementary/supplementary_table_6_*`.

## 2. Dataset Preparation

Run the relevant download scripts after obtaining dataset access:

```bash
bash code/scripts/download_sleep_edf_all_sleep_cassette.sh
bash code/scripts/download_physionet_mi.sh
bash code/scripts/download_ptbxl_records100.sh
bash code/scripts/download_tuab_full_resumable.sh
```

TUAB requires authorized credentials and is not public-downloadable from this
repository.

## 3. Preprocessing

Representative preprocessing entry points:

```bash
bash code/scripts/run_sleep_edf_full_preprocessing.sh
bash code/scripts/run_tuab_full_preprocessing.sh
python code/scripts/prepare_ptbxl_1f_demo.py --help
python code/scripts/make_physionet_mi_trials.py --help
```

## 4. Main Audits

Run dataset/model families:

```bash
bash code/scripts/launch_sleep_edf_full_multiseed_neural.sh
bash code/scripts/launch_tuab_full_multiseed_neural.sh
bash code/scripts/launch_tuab_full_foundation_multiseed_sequential.sh
bash code/scripts/launch_physionet_mi_multiseed_neural.sh
bash code/scripts/launch_ptbxl_ecg_architecture_audit.sh
bash code/scripts/launch_ptbxl_age_sex_matched_neural.sh
python code/scripts/run_tuab_site_temporal_psd_audit.py --help
```

The launch scripts record the exact H200-style commands used during the project;
reviewers can adapt paths and resource settings for their own environment.

## 5. Aggregation, Hypothesis Tests and Figures

After model outputs are available:

```bash
python code/scripts/aggregate_multiseed_subject_bootstrap.py --help
python code/scripts/aggregate_foundation_multiseed_predictions.py --help
python code/scripts/aggregate_ptbxl_prediction_bootstrap.py --help
bash code/scripts/run_formal_hypothesis_tests_from_saved_outputs.sh
python code/scripts/generate_nmi_journal_images.py
python code/scripts/generate_nmi_extended_data_figures.py
python code/scripts/generate_nmi_update_assets.py
python code/scripts/generate_nmi_update_figures_matplotlib.py
```

The submitted aggregate outputs are already present under `results/tables/`,
`figures/` and `supplementary/`.
