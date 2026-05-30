# EEG-ECG Aperiodic Spectral Audit

Code and aggregate outputs for the Nature Machine Intelligence submission:

**A spectral audit reveals task-dependent aperiodic reliance in EEG deep learning**

This repository contains the analysis code used to audit whether EEG and ECG
machine-learning models rely on broadband aperiodic, 1/f-like spectral
structure. The package is organized for reviewer inspection: raw datasets,
intermediate caches, pretrained checkpoints and subject-level restricted TUAB
files are not included.

## Repository Layout

```text
code/src/                    Reusable Python utilities
code/scripts/                Dataset preparation, interventions, models, tables and figures
code/configs/                Small configuration files
results/tables/              Aggregate result tables used in the manuscript
figures/                     Main and Extended Data figure PDFs/PNGs
supplementary/               Supplementary tables and notes
paper/                       Current LaTeX manuscript, bibliography and Nature template files
docs/                        Experiment log and project notes
run_logs/                    Selected run logs for the newest ECG/TUAB audit blocks
```

For compatibility with the original project scripts, the repository also keeps
lightweight symlinks named `reports/tables`, `Journal Images` and
`Supplementary Information` pointing to the cleaner reviewer-facing folders.

## What Is Included

- Sleep-EDF, TUAB, PhysioNet MI and PTB-XL preprocessing scripts.
- PSD aperiodic decomposition and intervention pipelines.
- Raw-signal Fourier intervention pipelines.
- Neural model audits for EEGNet, ShallowFBCSPNet, Deep4Net, CNN and MLP.
- TUAB foundation-model audit scripts for BIOT, LaBraM, EEGPT, CBraMod, REVE,
  EEGMamba and BENDR.
- PTB-XL ECG audit scripts for ResNet1D-Wang, Inception1D and XResNet1D101.
- Bootstrap aggregation and BH-FDR hypothesis-testing scripts.
- Figure and supplementary table generation scripts.
- Aggregate numerical tables corresponding to the submitted manuscript.

## What Is Not Included

- Raw Sleep-EDF, PhysioNet MI, PTB-XL or TUAB recordings.
- TUAB restricted EDF files or subject/file-level raw caches.
- Model checkpoint weights downloaded from third-party repositories.
- Large intermediate NumPy caches and per-record prediction files.

See [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md) for dataset access and expected
local directory conventions.

## Environment

The core analysis was run with Python 3.10+ on Linux/H200. A minimal environment
can be created with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Some foundation models require their original repositories/checkpoints and, in
the case of EEGMamba, an isolated environment. The exact configurations are
summarized in `supplementary/supplementary_table_6_foundation_model_config.*`
and in the model-specific scripts under `scripts/`.

## Quick Verification

These commands check the importable package, regenerate manuscript summary
tables from saved aggregate outputs, and regenerate the two newest Extended Data
figures:

```bash
python -m compileall code/src code/scripts
python code/scripts/collect_formal_hypothesis_tests.py --help
python code/scripts/generate_nmi_update_assets.py
python code/scripts/generate_nmi_update_figures_matplotlib.py
```

The complete end-to-end experiments require downloading the datasets and, for
TUAB, approved access from the Temple University Hospital EEG Corpus.

## Reproducibility Map

For a paper-section-to-code mapping, see
[docs/MANUSCRIPT_CODE_MAP.md](docs/MANUSCRIPT_CODE_MAP.md). For the practical
run order, see [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).
