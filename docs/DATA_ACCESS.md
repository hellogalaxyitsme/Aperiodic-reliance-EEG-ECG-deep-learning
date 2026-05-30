# Data Access

Raw recordings are not distributed in this repository.

## Sleep-EDF

Dataset: Sleep-EDF Expanded Sleep Cassette, available through PhysioNet.

Expected local location in the original runs:

```text
data/sleep-edf/sleep-cassette/
```

Relevant scripts:

```text
code/scripts/download_sleep_edf_all_sleep_cassette.sh
code/scripts/make_sleep_edf_epochs.py
code/scripts/extract_sleep_edf_raw_epochs.py
code/scripts/extract_sleep_edf_psd.py
code/scripts/fit_sleep_edf_specparam.py
```

## TUAB

Dataset: TUH Abnormal EEG Corpus v3.0.1. Access requires the Temple University
Hospital EEG Corpus data-use agreement. The repository includes code and
aggregate outputs, but not EDF files, per-EDF restricted manifests or raw caches.

Expected local location in the original runs:

```text
data/tuh_eeg_abnormal/v3.0.1/edf/
```

Relevant scripts:

```text
code/scripts/download_tuab_full_resumable.sh
code/scripts/make_tuab_full_manifest.py
code/scripts/extract_tuab_header_metadata.py
code/scripts/extract_tuab_raw_epochs.py
code/scripts/extract_tuab_psd.py
code/scripts/fit_tuab_specparam_full.py
code/scripts/run_tuab_psd_interventions.py
code/scripts/run_tuab_site_temporal_psd_audit.py
```

## PhysioNet EEG Motor Movement/Imagery

Dataset: EEG Motor Movement/Imagery Dataset, available through PhysioNet.

Relevant scripts:

```text
code/scripts/download_physionet_mi.sh
code/scripts/make_physionet_mi_trials.py
code/scripts/extract_physionet_mi_raw_trials.py
code/scripts/extract_physionet_mi_psd.py
code/scripts/fit_physionet_mi_specparam.py
```

## PTB-XL ECG

Dataset: PTB-XL v1.0.3 records100, available through PhysioNet.

Relevant scripts:

```text
code/scripts/download_ptbxl_records100.sh
code/scripts/download_ptbxl_records100_s3.sh
code/scripts/prepare_ptbxl_1f_demo.py
code/scripts/run_ptbxl_psd_interventions.py
code/scripts/run_ptbxl_raw_cnn_interventions.py
code/scripts/run_ptbxl_age_sex_matched_psd.py
```
