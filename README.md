# Phase-preserving spectral interventions reveal task-dependent aperiodic reliance in EEG and ECG deep learning

This repository contains the code and schema for analysing 
whether physiological time-series models rely on broadband aperiodic
(`1/f`-like) spectral structure. The experiments cover EEG sleep staging,
clinical EEG abnormality detection, EEG motor imagery, and ECG abnormality
detection.

Raw recordings, restricted datasets, downloaded pretrained checkpoints, and
large intermediate caches are not distributed here. 

## Repository Layout

```text
code/src/                    Reusable Python utilities
code/scripts/                Download, preprocessing, intervention, model and aggregation scripts
code/configs/                Small configuration files
code/revision/               Analyses added during peer review (docs/REVISION_ANALYSES.md)
docs/                        Dataset, preprocessing and reproducibility notes
results/tables/              Aggregate result tables and statistical-test outputs
results/revision/            Aggregate outputs of the revision analyses
reports/tables               Symlink to results/tables for script compatibility
```

## Included Analyses

- Spectral decomposition and intervention analyses for full, sham,
  aperiodic-shaped and flattened representations.
- Sleep-EDF EEG analyses across wake-versus-sleep, five-stage sleep staging and
  N2-versus-N3 classification.
- TUAB v3.0.1 normal-versus-abnormal EEG analyses, including full-corpus neural
  models, age/sex-matched controls, temporal acquisition-proxy controls and
  seven pretrained foundation models.
- PhysioNet EEG Motor Movement/Imagery analyses.
- PTB-XL ECG normal-versus-abnormal analyses with unmatched and age/sex-matched
  controls.
- Subject-level and hierarchical bootstrap aggregation, including formal
  bootstrap p-values and Benjamini-Hochberg FDR correction.
- Revision analyses: envelope-estimator sensitivity, ECG envelope estimators
  and a median-beat morphology baseline, time-domain waveform changes,
  training-time flattening augmentation, and the corrected BENDR and pooled
  balanced-accuracy evaluations (see [`docs/REVISION_ANALYSES.md`](docs/REVISION_ANALYSES.md)).

## Not Included

- Raw Sleep-EDF, TUAB, PhysioNet MI or PTB-XL recordings.
- TUAB restricted EDF files, restricted file manifests or raw subject caches.
- Downloaded third-party pretrained model checkpoints.
- Large intermediate arrays, window-level caches and per-record prediction
  files.

Dataset access and expected directory conventions are described in
[`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md).

## Environment

The core scripts target Python 3.10 or newer on Linux. GPU acceleration is
recommended for neural and foundation-model runs.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Some foundation models require their original repositories or isolated
dependency environments. See
[`docs/FOUNDATION_MODEL_CONFIGS.md`](docs/FOUNDATION_MODEL_CONFIGS.md).

## Quick Checks

```bash
python -m compileall code/src code/scripts
python code/scripts/collect_formal_hypothesis_tests.py --help
python code/scripts/run_aperiodic_simulation_validation.py --help
python code/scripts/run_tuab_temporal_psd_analysis.py --help
```

## Reproducibility

Start with [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the
recommended execution order. Aggregate output files are indexed in
[`docs/RESULTS_INDEX.md`](docs/RESULTS_INDEX.md), and preprocessing details are
summarized in [`docs/PREPROCESSING.md`](docs/PREPROCESSING.md).
