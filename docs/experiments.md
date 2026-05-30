# Experiments Log

This file is the living experiment ledger for the project:

```text
The Aperiodic Confound in EEG Deep Learning
```

All future runs, launch statuses, intermediate checks, completed results and
interpretations should be appended to this file. Do not delete old entries. If a
run is superseded, add a new note explaining why. `current_stage.md` should only
summarize where the project is and point here for experiment/run/result details.

## Safety And Execution Rules

- H200 is shared institutional storage.
- Do not delete datasets or shared cluster files.
- Do not use `rsync --delete`, `rm -rf`, dataset moves, or cleanup commands on
  cluster storage.
- Code is written locally on the Mac and synced additively to H200.
- Datasets stay on H200 only.

Local project:

```text
/Users/jass/Desktop/Aperiodic Idea
```

Remote project:

```text
/mnt/data/aperiodic_confounds
```

Safe sync command:

```bash
rsync -av --exclude .DS_Store --exclude __pycache__ --exclude '*.pyc' ./ h200:/mnt/data/aperiodic_confounds/
```

## Environment

H200 access:

```bash
ssh h200
```

Verified remote host:

```text
hostname: neurodx-3
user: vinay
GPU: NVIDIA H200
GPU memory: 143771 MiB VRAM
```

Python environments:

```text
/mnt/data/aperiodic_confounds/.venv
/mnt/data/.venvs/ml/bin/python3
/mnt/data/aperiodic_confounds/.python_deps
```

Notes:

- Project venv is used for MNE, sklearn, specparam, pandas, matplotlib reports.
- Shared ML venv is used for CUDA/PyTorch.
- Braindecode was installed into the project-local `.python_deps` folder and is
  used via `PYTHONPATH=/mnt/data/aperiodic_confounds/.python_deps`.

## Dataset

Dataset selected for MVP:

```text
Sleep-EDF Expanded Sleep-Cassette subset
```

Reason:

- Publicly available.
- Small enough for fast iteration.
- Cleaner and more standardized than TDBRAIN for the first proof of concept.
- More manageable than TUH for MVP development.

Remote dataset path:

```text
/mnt/data/aperiodic_confounds/data/sleep-edf/sleep-cassette
```

Subset:

```text
10 subjects / 20 recordings
SC4001/SC4002 through SC4091/SC4092
40 EDF files
about 963 MB
```

Verification:

```text
20 PSG/Hypnogram pairs found
```

## Run Ledger

### 1. H200 Access And Project Setup

Status: complete

Actions:

- Verified SSH alias `h200`.
- Verified GPU and storage.
- Created remote project directory:

```text
/mnt/data/aperiodic_confounds
```

- Added `instructions.md` with cluster access and no-deletion rules.
- Created local `code/` repo structure and synced it to H200.

### 2. Sleep-EDF Download And Verification

Status: complete

Scripts:

```text
code/scripts/download_sleep_edf_subset.sh
code/scripts/verify_sleep_edf.py
```

Outputs:

```text
/mnt/data/aperiodic_confounds/data/sleep-edf/sleep-cassette
```

Result:

```text
20 PSG/Hypnogram pairs verified
```

### 3. Epoch Manifest

Status: complete

Script:

```text
code/scripts/make_sleep_edf_epochs.py
```

Output:

```text
/mnt/data/aperiodic_confounds/results/sleep_edf_subset/epochs.csv
```

Settings:

```text
epoch length: 30 seconds
wake trim: 30 minutes around sleep
```

Result:

```text
total epochs: 21039
W: 3656
N1: 1689
N2: 8927
N3: 3074
REM: 3693
```

### 4. Welch PSD Extraction

Status: complete

Script:

```text
code/scripts/extract_sleep_edf_psd.py
```

Outputs:

```text
/mnt/data/aperiodic_confounds/results/sleep_edf_subset/psd_welch_fpz_pz.npz
/mnt/data/aperiodic_confounds/results/sleep_edf_subset/psd_index.csv
```

Settings:

```text
channels: Fpz-Cz, Pz-Oz
frequency range: 1-45 Hz
Welch window: 4 seconds
overlap: 0.5
```

Result:

```text
PSD shape: (21039, 2, 177)
```

### 5. Fixed 1/f Baselines

Status: complete

Script:

```text
code/scripts/run_sleep_edf_aperiodic_baselines.py
```

Output:

```text
/mnt/data/aperiodic_confounds/results/sleep_edf_subset/baselines
```

Model:

```text
Ridge classifier
5 subject-held-out folds
```

Fixed 1/f fit quality:

```text
mean R2: 0.867
median R2: 0.914
```

Balanced accuracy:

| Task | Full PSD | Aperiodic | Residual |
| --- | ---: | ---: | ---: |
| Wake vs Sleep | 0.940 | 0.891 | 0.917 |
| N2 vs N3 | 0.895 | 0.789 | 0.771 |
| Five-stage | 0.709 | 0.531 | 0.638 |

Interpretation:

- A simple fixed 1/f decomposition already showed that aperiodic structure
  carries substantial classification signal.
- Five-stage classification retained meaningful performance from aperiodic
  features alone.

### 6. Specparam Decomposition

Status: complete

Script:

```text
code/scripts/fit_sleep_edf_specparam.py
```

Outputs:

```text
/mnt/data/aperiodic_confounds/results/sleep_edf_subset/specparam/specparam_fixed.npz
/mnt/data/aperiodic_confounds/results/sleep_edf_subset/specparam/specparam_fixed.summary.json
```

Settings:

```text
mode: fixed
frequency range: 1-45 Hz
max_n_peaks: 6
min_peak_height: 0.1
peak_threshold: 2.0
peak_width_limits: 0.5-8 Hz
```

Result:

```text
spectra fit: 42078
ok fraction: 1.0
mean R2: 0.9475
median R2: 0.9782
mean n_peaks: 5.615
```

Interpretation:

- Specparam gave better fit quality than the simple fixed 1/f regression.
- The peak cap was often reached, so sensitivity checks were needed.

### 7. Specparam Ridge Baselines

Status: complete

Script:

```text
code/scripts/run_sleep_edf_aperiodic_baselines.py --decomposition precomputed
```

Output:

```text
/mnt/data/aperiodic_confounds/results/sleep_edf_subset/baselines_specparam
```

Balanced accuracy:

| Task | Full PSD | Aperiodic | Residual | Aperiodic retention |
| --- | ---: | ---: | ---: | ---: |
| Wake vs Sleep | 0.940 | 0.894 | 0.917 | 95.2% |
| N2 vs N3 | 0.895 | 0.858 | 0.768 | 95.8% |
| Five-stage | 0.710 | 0.529 | 0.636 | 74.6% |

Interpretation:

- Aperiodic-only performance was very high for binary tasks.
- Full five-stage staging still retained about three quarters of full-spectrum
  performance from aperiodic structure alone.

### 8. Specparam Diagnostics

Status: complete

Script:

```text
code/scripts/analyze_specparam_diagnostics.py
```

Output:

```text
/mnt/data/aperiodic_confounds/results/sleep_edf_subset/specparam/diagnostics
```

Result:

```text
fit_ok_fraction: 1.0
mean R2: 0.948
low_r2_fraction: 0.107
mean_n_peaks: 5.62
peak_cap_fraction: 0.670
bad_exponent_fraction: 0.0035
```

Interpretation:

- Most fits were strong.
- Wake, especially Pz-Oz, had lower fit quality.
- High peak-cap fraction justified peak-constraint sensitivity analysis.

### 9. Matched Controls

Status: complete

Script:

```text
code/scripts/run_sleep_edf_matched_baselines.py
```

Outputs:

```text
/mnt/data/aperiodic_confounds/results/sleep_edf_subset/matched_specparam
/mnt/data/aperiodic_confounds/results/sleep_edf_subset/matched_specparam_q3
/mnt/data/aperiodic_confounds/results/sleep_edf_subset/matched_specparam_q2
```

Matching variables:

```text
mean aperiodic offset
mean aperiodic exponent
```

Strict q4 matching:

| Task | n | Full PSD | Aperiodic | Residual |
| --- | ---: | ---: | ---: | ---: |
| Wake vs Sleep | 4788 | 0.827 | 0.783 | 0.770 |
| N2 vs N3 | 2668 | 0.756 | 0.634 | 0.711 |
| Five-stage | 210 | too small | too small | too small |

Looser q2 matching:

| Task | n | Full PSD | Aperiodic | Residual |
| --- | ---: | ---: | ---: | ---: |
| Wake vs Sleep | 7312 | 0.894 | 0.827 | 0.871 |
| N2 vs N3 | 4908 | 0.801 | 0.758 | 0.705 |
| Five-stage | 365 | too small | too small | too small |

Interpretation:

- Matching reduces sample size and performance, as expected.
- Aperiodic signal remains nontrivial even after coarse matching.
- Five-stage matched-control results are too underpowered to emphasize.

### 10. Specparam Sensitivity Grid

Status: complete

Scripts:

```text
code/scripts/run_specparam_sensitivity_grid.sh
code/scripts/summarize_specparam_sensitivity.py
```

Output:

```text
/mnt/data/aperiodic_confounds/results/sleep_edf_subset/specparam_sensitivity
```

Settings tested:

```text
main_p6_h010
conservative_p4_h015
moderate_p8_h010
stricter_p6_h020
```

Balanced accuracy ranges across settings:

| Task | Aperiodic BA range | Residual BA range |
| --- | ---: | ---: |
| Wake vs Sleep | 0.894-0.896 | 0.916-0.917 |
| N2 vs N3 | 0.851-0.858 | 0.768-0.772 |
| Five-stage | 0.527-0.530 | 0.636-0.637 |

Interpretation:

- The classification story is stable across reasonable specparam settings.
- The aperiodic effect is not an artifact of a single peak-constraint choice.

### 11. Linear Intervention Evaluation

Status: complete

Script:

```text
code/scripts/run_sleep_edf_intervention_eval.py
```

Output:

```text
/mnt/data/aperiodic_confounds/results/sleep_edf_subset/interventions_specparam_flattening
```

Experiment:

```text
Train ridge classifier on full log PSD.
Evaluate the same trained model on:
1. full_log_psd
2. aperiodic_spectrum
3. flattened_log_psd
```

Balanced accuracy:

| Task | Full PSD | Aperiodic | Flattened | Flattening drop |
| --- | ---: | ---: | ---: | ---: |
| Wake vs Sleep | 0.940 | 0.798 | 0.500 | 0.440 |
| N2 vs N3 | 0.895 | 0.814 | 0.500 | 0.395 |
| Five-stage | 0.710 | 0.499 | 0.200 | 0.510 |

Interpretation:

- This became the first central causal-style result.
- Aperiodic-only inputs preserved substantial performance.
- Flattening the aperiodic component collapsed performance to chance or near
  chance.

### 12. Paper-Style Artifact For Linear Intervention

Status: complete

Script:

```text
code/scripts/plot_sleep_edf_intervention_results.py
```

Outputs:

```text
/mnt/data/aperiodic_confounds/reports/figures/sleep_edf_intervention_performance.png
/mnt/data/aperiodic_confounds/reports/figures/sleep_edf_intervention_performance.pdf
/mnt/data/aperiodic_confounds/reports/tables/sleep_edf_intervention_summary.csv
/mnt/data/aperiodic_confounds/reports/tables/sleep_edf_intervention_summary.md
```

### 13. Deep MLP PSD Intervention

Status: complete

Script:

```text
code/scripts/run_sleep_edf_deep_intervention_mlp.py
```

Output:

```text
/mnt/data/aperiodic_confounds/results/sleep_edf_subset/deep_mlp_interventions_specparam
```

Environment:

```text
/mnt/data/.venvs/ml/bin/python3
GPU: NVIDIA H200
```

Experiment:

```text
Train MLP on full log PSD.
Evaluate on full, aperiodic-only, and flattened log PSD inputs.
```

Balanced accuracy:

| Task | Full PSD | Aperiodic | Flattened | Aperiodic retention |
| --- | ---: | ---: | ---: | ---: |
| Wake vs Sleep | 0.946 | 0.812 | 0.500 | 0.858 |
| N2 vs N3 | 0.903 | 0.842 | 0.501 | 0.933 |
| Five-stage | 0.736 | 0.576 | 0.200 | 0.782 |

Interpretation:

- The same pattern held in a neural model trained on PSD features.
- This showed that aperiodic reliance was not limited to a linear classifier.

### 14. Linear Aperiodic Attribution

Status: complete

Script:

```text
code/scripts/analyze_linear_aperiodic_attribution.py
```

Output:

```text
/mnt/data/aperiodic_confounds/results/sleep_edf_subset/linear_attribution
```

Raw coefficient-energy AAR:

| Task | AAR |
| --- | ---: |
| Wake vs Sleep | 0.016 |
| N2 vs N3 | 0.032 |
| Five-stage | 0.018 |

Interpretation:

- Full-spectrum ridge weights are not simply a smooth `[1, -log(f)]` template.
- This does not contradict intervention results because model behavior depends
  on data-conditioned inputs, not coefficient geometry alone.

### 15. Data-Conditioned Logit Contributions

Status: complete

Script:

```text
code/scripts/analyze_linear_logit_contributions.py
```

Output:

```text
/mnt/data/aperiodic_confounds/results/sleep_edf_subset/linear_logit_contributions
```

Absolute margin contribution fractions:

| Task | Aperiodic contribution | Flattened contribution |
| --- | ---: | ---: |
| Wake vs Sleep | 69.6% | 30.4% |
| N2 vs N3 | 56.8% | 43.2% |
| Five-stage | 58.8% | 41.2% |

Interpretation:

- This attribution-style analysis aligned better with the intervention results.
- Aperiodic variation contributes materially to held-out decision margins.

### 16. Bootstrap Confidence Intervals

Status: complete

Script:

```text
code/scripts/bootstrap_intervention_metrics.py
```

Outputs:

```text
/mnt/data/aperiodic_confounds/reports/tables/intervention_bootstrap_ci.csv
/mnt/data/aperiodic_confounds/reports/tables/intervention_bootstrap_ci.md
```

Method:

```text
paired fold-bootstrap over 5 subject-held-out folds
10000 bootstrap samples
95% confidence intervals
```

Initial models included:

```text
linear_ridge
deep_mlp
```

Later extended to:

```text
raw_cnn
braindecode_eegnet
```

Note:

- These are first-pass fold-level CIs, not subject-level or epoch-level
  hierarchical uncertainty estimates.

### 17. Raw Epoch Cache

Status: complete

Script:

```text
code/scripts/extract_sleep_edf_raw_epochs.py
```

Outputs:

```text
/mnt/data/aperiodic_confounds/results/sleep_edf_subset/raw_epochs_fpz_pz_100hz.npz
/mnt/data/aperiodic_confounds/results/sleep_edf_subset/raw_epochs_index.csv
```

Settings:

```text
channels: Fpz-Cz, Pz-Oz
target sampling rate: 100 Hz
bandpass: 0.5-45 Hz
epoch length: 30 seconds
```

Result:

```text
raw epoch shape: (21039, 2, 3000)
cache size: about 482 MB
```

### 18. Custom Raw CNN Baseline

Status: complete

Script:

```text
code/scripts/run_sleep_edf_raw_cnn.py
```

Output:

```text
/mnt/data/aperiodic_confounds/results/sleep_edf_subset/raw_cnn
```

Balanced accuracy:

| Task | Raw EEG BA | Macro F1 | Accuracy |
| --- | ---: | ---: | ---: |
| Wake vs Sleep | 0.956 +/- 0.016 | 0.940 | 0.965 |
| N2 vs N3 | 0.935 +/- 0.031 | 0.919 | 0.937 |
| Five-stage | 0.772 +/- 0.024 | 0.745 | 0.784 |

Interpretation:

- A raw EEG neural model performs strongly.
- This provided a bridge from PSD-feature experiments to raw deep EEG models.

### 19. Custom Raw CNN Aperiodic Intervention

Status: complete

Script:

```text
code/scripts/run_sleep_edf_raw_cnn_intervention.py
```

Output:

```text
/mnt/data/aperiodic_confounds/results/sleep_edf_subset/raw_cnn_interventions
```

Experiment:

```text
Train custom raw CNN on original raw EEG.
Evaluate the same fold models on:
1. raw_eeg
2. phase_aperiodic
3. phase_flattened
```

Intervention:

```text
phase-preserving FFT amplitude edits
1-45 Hz
per-channel RMS matched to original epoch
```

Balanced accuracy:

| Task | Raw EEG | Phase aperiodic | Phase flattened |
| --- | ---: | ---: | ---: |
| Wake vs Sleep | 0.957 +/- 0.018 | 0.893 +/- 0.040 | 0.538 +/- 0.062 |
| N2 vs N3 | 0.936 +/- 0.026 | 0.902 +/- 0.073 | 0.935 +/- 0.017 |
| Five-stage | 0.775 +/- 0.049 | 0.666 +/- 0.049 | 0.346 +/- 0.031 |

Interpretation:

- Wake-vs-Sleep and five-stage classification were strongly harmed by
  flattening the aperiodic envelope.
- N2-vs-N3 survived flattening, suggesting the custom raw CNN could use
  phase, morphology, residual oscillatory structure, or temporal slow-wave
  shape for that contrast.

### 20. Combined Intervention Report

Status: complete

Script:

```text
code/scripts/plot_combined_intervention_results.py
```

Outputs:

```text
/mnt/data/aperiodic_confounds/reports/figures/combined_intervention_performance.png
/mnt/data/aperiodic_confounds/reports/figures/combined_intervention_performance.pdf
/mnt/data/aperiodic_confounds/reports/tables/combined_intervention_summary.csv
/mnt/data/aperiodic_confounds/reports/tables/combined_intervention_summary.md
```

Initial combined models:

```text
linear_ridge
deep_mlp
raw_cnn
```

Later updated to include:

```text
braindecode_eegnet
```

### 21. Braindecode EEGNet Intervention

Status: complete

Script:

```text
code/scripts/run_sleep_edf_braindecode_eegnet_intervention.py
```

Output:

```text
/mnt/data/aperiodic_confounds/results/sleep_edf_subset/braindecode_eegnet_interventions
```

Model:

```text
braindecode.models.EEGNet
braindecode version: 1.5.1
F1: 8
D: 2
F2: 16
kernel_length: 64
depthwise_kernel_length: 16
dropout: 0.25
epochs: 80
batch size: 256
optimizer: AdamW
```

Environment:

```text
PyTorch: 2.11.0+cu130
GPU: NVIDIA H200
Braindecode deps: /mnt/data/aperiodic_confounds/.python_deps
```

Experiment:

```text
Train Braindecode EEGNet on original raw EEG.
Evaluate same fold models on:
1. raw_eeg
2. phase_aperiodic
3. phase_flattened
```

Balanced accuracy with 95% fold-bootstrap CI:

| Task | Original | Aperiodic | Flattened | Flattening drop |
| --- | ---: | ---: | ---: | ---: |
| Wake vs Sleep | 0.957 [0.945, 0.967] | 0.894 [0.862, 0.927] | 0.576 [0.538, 0.625] | 0.381 [0.325, 0.429] |
| N2 vs N3 | 0.927 [0.896, 0.946] | 0.886 [0.804, 0.937] | 0.893 [0.807, 0.942] | 0.033 [-0.002, 0.091] |
| Five-stage | 0.752 [0.729, 0.775] | 0.623 [0.609, 0.637] | 0.468 [0.443, 0.493] | 0.285 [0.268, 0.301] |

Interpretation:

- EEGNet confirms the main result in a canonical EEG deep-learning architecture.
- Aperiodic-only test inputs retain high performance.
- Flattening the aperiodic envelope causes a clear performance drop for
  Wake-vs-Sleep and five-stage staging.
- N2-vs-N3 remains much less affected; the flattening-drop CI crosses zero.
  This suggests additional non-aperiodic features contribute to this
  distinction.

## Current Best Combined Results

Balanced accuracy with 95% fold-bootstrap CI:

| Model | Task | Original BA | Aperiodic BA | Flattened BA | Aperiodic retention | Flattening drop |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Linear ridge PSD | Wake vs Sleep | 0.940 [0.923, 0.952] | 0.798 [0.757, 0.836] | 0.500 [0.500, 0.500] | 0.849 [0.818, 0.886] | 0.440 [0.423, 0.452] |
| Linear ridge PSD | N2 vs N3 | 0.895 [0.858, 0.926] | 0.814 [0.733, 0.890] | 0.500 [0.500, 0.500] | 0.909 [0.841, 0.978] | 0.395 [0.358, 0.426] |
| Linear ridge PSD | Five-stage | 0.710 [0.679, 0.734] | 0.499 [0.480, 0.515] | 0.200 [0.200, 0.200] | 0.703 [0.686, 0.726] | 0.510 [0.479, 0.534] |
| Deep MLP PSD | Wake vs Sleep | 0.946 [0.939, 0.955] | 0.812 [0.762, 0.856] | 0.500 [0.500, 0.500] | 0.858 [0.807, 0.900] | 0.446 [0.439, 0.455] |
| Deep MLP PSD | N2 vs N3 | 0.903 [0.872, 0.929] | 0.842 [0.785, 0.883] | 0.501 [0.500, 0.502] | 0.933 [0.902, 0.956] | 0.402 [0.370, 0.429] |
| Deep MLP PSD | Five-stage | 0.736 [0.705, 0.769] | 0.576 [0.527, 0.628] | 0.200 [0.200, 0.200] | 0.782 [0.712, 0.859] | 0.536 [0.505, 0.569] |
| Raw EEG CNN | Wake vs Sleep | 0.957 [0.944, 0.971] | 0.893 [0.865, 0.927] | 0.538 [0.502, 0.594] | 0.933 [0.906, 0.959] | 0.420 [0.363, 0.458] |
| Raw EEG CNN | N2 vs N3 | 0.936 [0.914, 0.955] | 0.902 [0.835, 0.945] | 0.935 [0.922, 0.948] | 0.963 [0.914, 0.995] | 0.001 [-0.010, 0.009] |
| Raw EEG CNN | Five-stage | 0.775 [0.738, 0.812] | 0.666 [0.639, 0.710] | 0.346 [0.321, 0.370] | 0.859 [0.798, 0.916] | 0.429 [0.410, 0.452] |
| Braindecode EEGNet | Wake vs Sleep | 0.957 [0.945, 0.967] | 0.894 [0.862, 0.927] | 0.576 [0.538, 0.625] | 0.934 [0.907, 0.967] | 0.381 [0.325, 0.429] |
| Braindecode EEGNet | N2 vs N3 | 0.927 [0.896, 0.946] | 0.886 [0.804, 0.937] | 0.893 [0.807, 0.942] | 0.956 [0.897, 0.994] | 0.033 [-0.002, 0.091] |
| Braindecode EEGNet | Five-stage | 0.752 [0.729, 0.775] | 0.623 [0.609, 0.637] | 0.468 [0.443, 0.493] | 0.828 [0.801, 0.853] | 0.285 [0.268, 0.301] |

## Future Run Template

Append new entries below using this template.

```markdown
### YYYY-MM-DD - Short Run Name

Status:

Purpose:

Script/command:

Remote output path:

Inputs:

Settings:

Result:

Interpretation:

Follow-up:
```

### 2026-05-23 - Reviewer-Resistance Control Package Implementation

Status: implemented and smoke-tested

Purpose:

Address the strongest reviewer threats raised after the original 10-subject
pilot: raw-intervention artifacts, PSD flattening interpretation, missing
standard EEG-DL architectures, missing IRASA agreement, and missing simulation
validation.

Implemented:

```text
code/scripts/analyze_raw_intervention_diagnostics.py
code/scripts/run_sleep_edf_psd_train_input_controls.py
code/scripts/run_aperiodic_simulation_validation.py
code/scripts/fit_sleep_edf_irasa.py
code/scripts/compare_specparam_irasa.py
code/scripts/launch_sleep_edf_reviewer_resistance_controls.sh
```

Patched:

```text
code/scripts/run_sleep_edf_raw_cnn_intervention.py
code/scripts/run_sleep_edf_braindecode_eegnet_intervention.py
code/scripts/aggregate_multiseed_subject_bootstrap.py
```

Key changes:

```text
1. Added phase_sham to raw CNN and Braindecode raw-intervention evaluations.
2. Generalized the Braindecode runner beyond EEGNet:
   eegnet, shallow_fbcsp, deep4, usleep, eegconformer.
3. Added PSD train-input controls:
   train directly on full_log_psd, aperiodic_spectrum, flattened_log_psd.
4. Added raw-intervention time-domain diagnostics:
   RMS, std, peak-to-peak, skew, kurtosis, line length, zero-crossing rate,
   RMSE/correlation vs raw.
5. Added IRASA fitting and SpecParam-vs-IRASA agreement tables.
6. Added synthetic simulation validation with known aperiodic/oscillatory
   ground truth scenarios.
7. Extended hierarchical bootstrap aggregation to include sham estimates:
   sham, retention_sham, drop_sham.
```

Remote smoke tests:

```text
/mnt/data/aperiodic_confounds/results/smoke/reviewer_controls/simulation
/mnt/data/aperiodic_confounds/results/smoke/reviewer_controls/raw_diagnostics
/mnt/data/aperiodic_confounds/results/smoke/reviewer_controls/irasa
/mnt/data/aperiodic_confounds/results/smoke/reviewer_controls/irasa_agreement
/mnt/data/aperiodic_confounds/results/smoke/reviewer_controls/psd_train_controls
/mnt/data/aperiodic_confounds/results/smoke/reviewer_controls/braindecode_shallow
/mnt/data/aperiodic_confounds/results/smoke/reviewer_controls/bootstrap_sham.csv
```

Smoke-test result:

All new scripts passed Python/shell syntax checks on H200. Small capped runs
completed for simulation validation, raw intervention diagnostics, IRASA
decomposition/agreement, PSD train-input controls, ShallowFBCSPNet intervention,
all Braindecode architecture constructors, and sham-aware hierarchical
bootstrap aggregation.

Full run command:

```bash
cd /mnt/data/aperiodic_confounds
nohup bash code/scripts/launch_sleep_edf_reviewer_resistance_controls.sh \
  > logs/reviewer_resistance_controls_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Safety:

The launcher writes additive outputs under
`/mnt/data/aperiodic_confounds/results/sleep_edf_full/reviewer_resistance_controls`
and report tables under `/mnt/data/aperiodic_confounds/reports/tables`. It does
not delete or overwrite datasets.

Core staged run:

```text
status: complete
launched: 2026-05-23
completed: 2026-05-24 02:32
launcher pid: 2556244
wrapper pid: 2556241
log: /mnt/data/aperiodic_confounds/logs/reviewer_resistance_controls_core_20260523_214141.log
environment:
  RUN_IRASA=0
  BRAIND_ARCHS="eegnet shallow_fbcsp deep4"
```

Core staged scope:

```text
included:
  raw intervention distribution diagnostics
  simulation validation
  PSD train-input controls, seeds 42 43 44
  custom raw CNN with sham control, seeds 42 43 44
  Braindecode EEGNet with sham control, seeds 42 43 44
  Braindecode ShallowFBCSPNet with sham control, seeds 42 43 44
  Braindecode Deep4Net with sham control, seeds 42 43 44
  sham-aware hierarchical bootstrap aggregation
excluded for later staged runs:
  full IRASA
  USleep
  EEGConformer
```

Initial status:

```text
raw diagnostics: complete
simulation validation: complete
PSD train-input controls: complete
custom raw CNN with sham: complete
Braindecode EEGNet/ShallowFBCSPNet/Deep4Net with sham: complete
sham-aware hierarchical bootstrap: complete
GPU/process status at completion check: idle, no launcher process
```

Core staged outputs:

```text
/mnt/data/aperiodic_confounds/reports/tables/sleep_edf_reviewer_resistance_bootstrap.csv
/mnt/data/aperiodic_confounds/reports/tables/sleep_edf_reviewer_resistance_bootstrap.md
/mnt/data/aperiodic_confounds/reports/tables/raw_intervention_diagnostics/raw_intervention_distribution_diagnostics.csv
/mnt/data/aperiodic_confounds/reports/tables/raw_intervention_diagnostics/raw_intervention_distribution_diagnostics.md
/mnt/data/aperiodic_confounds/results/simulations/aperiodic_validation/simulation_validation_metrics.csv
/mnt/data/aperiodic_confounds/results/simulations/aperiodic_validation/simulation_validation_metrics.md
```

Core staged balanced-accuracy highlights:

```text
Sham control:
  Raw CNN, EEGNet, ShallowFBCSPNet, and Deep4Net all had sham performance
  indistinguishable from raw performance. Drop_sham was 0.000 or numerically
  negligible across tasks. This directly addresses the concern that the FFT
  surgery pipeline itself causes the raw-model performance drop.

PSD train-input controls:
  Wake vs Sleep:
    full 0.929 [0.918, 0.939]
    aperiodic-trained 0.893 [0.876, 0.908]
    flattened-trained 0.913 [0.900, 0.925]
  N2 vs N3:
    full 0.850 [0.824, 0.874]
    aperiodic-trained 0.759 [0.727, 0.790]
    flattened-trained 0.745 [0.720, 0.768]
  Five-stage:
    full 0.694 [0.674, 0.712]
    aperiodic-trained 0.572 [0.549, 0.595]
    flattened-trained 0.633 [0.614, 0.652]

Raw CNN with sham:
  Wake vs Sleep:
    raw 0.940 [0.932, 0.948]
    sham 0.940 [0.932, 0.948]
    aperiodic 0.871 [0.853, 0.889]
    flattened 0.511 [0.505, 0.520]
    drop_flattened 0.429 [0.417, 0.439]
  N2 vs N3:
    raw 0.877 [0.852, 0.901]
    sham 0.877 [0.852, 0.901]
    flattened 0.875 [0.858, 0.891]
    drop_flattened 0.002 [-0.021, 0.023]
  Five-stage:
    raw 0.736 [0.718, 0.753]
    sham 0.736 [0.718, 0.753]
    aperiodic 0.648 [0.628, 0.666]
    flattened 0.281 [0.259, 0.303]
    drop_flattened 0.455 [0.427, 0.484]

Braindecode architecture pattern:
  EEGNet, ShallowFBCSPNet, and Deep4Net all replicate the central raw-model
  pattern: large flattening drops for Wake-vs-Sleep and Five-stage, near-zero
  flattening drops for N2-vs-N3, and no sham disruption.
```

### 2026-05-23 - Full Sleep-EDF Multiseed Neural Robustness

Status: complete

Purpose:

Scale from the 10-subject Sleep-EDF MVP to all Sleep-Cassette subjects and run
multiple random seeds for all neural models used so far.

Dataset:

```text
Sleep-EDF Expanded Sleep-Cassette
153 PSG/Hypnogram recording pairs
306 EDF files
```

Download:

```text
script: code/scripts/download_sleep_edf_all_sleep_cassette.sh
log: /mnt/data/aperiodic_confounds/logs/download_sleep_edf_all_retry_20260523_164647.log
status: complete
```

Download verification:

```text
EDF files: 306
Recording pairs: 153
Total EDF size: 7244.6 MiB
First pair: SC4001
Last pair: SC4822
Status: OK
```

Preprocessing:

```text
script: code/scripts/run_sleep_edf_full_preprocessing.sh
log: /mnt/data/aperiodic_confounds/logs/preprocess_sleep_edf_full_20260523_164829.log
status: complete
```

Preprocessing outputs:

```text
/mnt/data/aperiodic_confounds/results/sleep_edf_full/epochs.csv
/mnt/data/aperiodic_confounds/results/sleep_edf_full/psd_welch_fpz_pz.npz
/mnt/data/aperiodic_confounds/results/sleep_edf_full/psd_index.csv
/mnt/data/aperiodic_confounds/results/sleep_edf_full/specparam/specparam_fixed.npz
/mnt/data/aperiodic_confounds/results/sleep_edf_full/raw_epochs_fpz_pz_100hz.npz
/mnt/data/aperiodic_confounds/results/sleep_edf_full/raw_epochs_index.csv
```

Preprocessing result:

```text
epochs: 195469
PSD shape: (195469, 2, 177)
raw shape: (195469, 2, 3000)
specparam spectra: 390938
specparam ok_fraction: 1.0
specparam mean R2: 0.9097
specparam median R2: 0.9677
specparam mean_n_peaks: 5.4566
```

Neural multiseed run:

```text
script: code/scripts/launch_sleep_edf_full_multiseed_neural.sh
log: /mnt/data/aperiodic_confounds/logs/full_multiseed_neural_20260523_165523.log
pid at launch: 2405689
seeds: 42 43 44
models: deep_mlp, raw_cnn, braindecode_eegnet
status: complete
```

Aggregation:

```text
script: code/scripts/aggregate_multiseed_subject_bootstrap.py
bootstrap units: seed and subject
bootstrap samples: 10000
output:
/mnt/data/aperiodic_confounds/reports/tables/full_sleep_edf_multiseed_subject_bootstrap.csv
/mnt/data/aperiodic_confounds/reports/tables/full_sleep_edf_multiseed_subject_bootstrap.md
status: complete
```

Aggregation note:

The first bootstrap implementation was correct in intent but too slow because
it rebuilt pandas pivots and concatenated frames inside every bootstrap
iteration. It was stopped after the neural runs had completed, optimized to use
a precomputed seed-by-subject metric cube with batched NumPy resampling, and
rerun only for report generation. No model outputs or datasets were deleted.

Balanced accuracy, subject-level hierarchical bootstrap over seed and subject:

| Model | Task | Baseline | Aperiodic | Flattened | Flattening drop |
| --- | --- | ---: | ---: | ---: | ---: |
| Deep MLP PSD | Wake vs Sleep | 0.929 [0.918, 0.939] | 0.783 [0.757, 0.808] | 0.500 [0.500, 0.500] | 0.429 [0.418, 0.439] |
| Deep MLP PSD | N2 vs N3 | 0.850 [0.824, 0.875] | 0.763 [0.733, 0.791] | 0.478 [0.442, 0.511] | 0.372 [0.327, 0.420] |
| Deep MLP PSD | Five-stage | 0.693 [0.674, 0.711] | 0.509 [0.489, 0.528] | 0.204 [0.201, 0.207] | 0.490 [0.470, 0.508] |
| Custom raw CNN | Wake vs Sleep | 0.941 [0.933, 0.949] | 0.876 [0.857, 0.892] | 0.506 [0.503, 0.510] | 0.435 [0.425, 0.444] |
| Custom raw CNN | N2 vs N3 | 0.881 [0.853, 0.905] | 0.872 [0.851, 0.890] | 0.875 [0.855, 0.892] | 0.006 [-0.013, 0.022] |
| Custom raw CNN | Five-stage | 0.735 [0.715, 0.754] | 0.648 [0.628, 0.667] | 0.267 [0.250, 0.284] | 0.468 [0.447, 0.490] |
| Braindecode EEGNet | Wake vs Sleep | 0.939 [0.929, 0.948] | 0.877 [0.858, 0.896] | 0.503 [0.501, 0.505] | 0.436 [0.427, 0.445] |
| Braindecode EEGNet | N2 vs N3 | 0.886 [0.861, 0.908] | 0.864 [0.845, 0.883] | 0.884 [0.869, 0.898] | 0.002 [-0.017, 0.019] |
| Braindecode EEGNet | Five-stage | 0.710 [0.690, 0.729] | 0.607 [0.587, 0.626] | 0.318 [0.301, 0.335] | 0.392 [0.375, 0.408] |

Interpretation:

The full Sleep-EDF multiseed result supports the project narrative more
strongly than the initial 10-subject MVP. Wake-vs-Sleep and five-stage staging
show large, stable performance drops when the aperiodic envelope is flattened,
across all neural model families. Aperiodic-only inputs remain strongly
predictive, especially for the raw models and EEGNet. N2-vs-N3 remains the
important exception for raw CNN and EEGNet: flattening barely changes balanced
accuracy, suggesting that this task uses non-aperiodic temporal morphology or
phase-preserved information.

## 2026-05-24: PhysioNet EEG Motor Movement/Imagery Second-Domain Setup

Purpose:

Extend the audit beyond Sleep-EDF into a public motor-imagery domain using
PhysioNet EEG Motor Movement/Imagery (`eegmmidb`, Schalk et al.). This tests
whether the aperiodic-confound narrative generalizes to short-window
cue-locked task EEG rather than only sleep staging.

Safety:

All operations were additive. No shared datasets were deleted or modified.
The dataset was downloaded under the project data root:

```text
/mnt/data/aperiodic_confounds/data/physionet-eegmmidb
```

Code added:

```text
code/src/aperiodic_eeg/physionet_mi.py
code/scripts/download_physionet_mi.sh
code/scripts/make_physionet_mi_trials.py
code/scripts/extract_physionet_mi_raw_trials.py
code/scripts/extract_physionet_mi_psd.py
code/scripts/fit_physionet_mi_specparam.py
code/scripts/run_physionet_mi_aperiodic_baselines.py
```

Download:

```text
script: code/scripts/download_physionet_mi.sh
log: /mnt/data/aperiodic_confounds/logs/physionet_mi_download_20260524_133523.log
source: PhysioNet public S3 mirror
status: complete
```

Downloaded data visible on H200:

```text
subjects: 109
EDF files: 1526
event files: 1526
total EDF/event files: 3052
```

Primary task definition:

```text
task: imagined left fist vs imagined right fist
runs: 4, 8, 12
trial window: 0.5-4.0 s after cue onset
sampling rate: 160 Hz
channels: 64 EEG channels
```

Preprocessing outputs:

```text
/mnt/data/aperiodic_confounds/results/physionet_mi/imagined_fists_trials.csv
/mnt/data/aperiodic_confounds/results/physionet_mi/imagined_fists_raw_160hz.npz
/mnt/data/aperiodic_confounds/results/physionet_mi/imagined_fists_raw_index.csv
/mnt/data/aperiodic_confounds/results/physionet_mi/imagined_fists_psd_multitaper.npz
/mnt/data/aperiodic_confounds/results/physionet_mi/imagined_fists_psd_index.csv
/mnt/data/aperiodic_confounds/results/physionet_mi/specparam/imagined_fists_specparam_fixed.npz
/mnt/data/aperiodic_confounds/results/physionet_mi/specparam/imagined_fists_specparam_fixed.summary.json
```

Preprocessing result:

```text
trials: 4917
subjects: 109
condition counts: left_fist=2479, right_fist=2438
raw shape: (4917, 64, 560)
PSD method: multitaper
PSD shape: (4917, 64, 151)
frequency range: 2.00-44.86 Hz
multitaper bandwidth: 4 Hz
```

Specparam short-window fit quality:

```text
spectra fit: 314688
ok_fraction: 1.0
mean R2: 0.9520
median R2: 0.9705
10th percentile R2: 0.9066
mean MAE: 0.0771
mean_n_peaks: 3.5293
```

This is an important result because the main concern for PhysioNet MI was that
2-4 second trial windows might make aperiodic fitting too unreliable. With the
current `0.5-4.0 s` trial window and multitaper PSD, the fit quality is strong
enough to proceed, while still requiring explicit reporting in the paper.

First subject-held-out ridge baselines:

```text
script: code/scripts/run_physionet_mi_aperiodic_baselines.py
output: /mnt/data/aperiodic_confounds/results/physionet_mi/baselines_specparam
folding: 5-fold GroupKFold by subject
classifier: balanced RidgeClassifier
labels: left_fist vs right_fist
status: complete
```

Balanced accuracy:

| Feature set | Mean | Std |
| --- | ---: | ---: |
| full_log_psd | 0.565 | 0.018 |
| aperiodic_spectrum | 0.526 | 0.011 |
| aperiodic_params | 0.530 | 0.015 |
| periodic_residual | 0.532 | 0.021 |

Interpretation:

This first motor-imagery result is intentionally modest and should not be
treated as a final model benchmark. It shows that left-vs-right imagined fist
classification is only weakly above chance with a simple subject-held-out ridge
classifier, which is plausible for cross-subject MI. Unlike Sleep-EDF, the
aperiodic-only features do not dominate the task. Full PSD performs best, while
aperiodic-only and residual-only representations sit closer to chance. This is
useful for the paper: the second domain may show a different dependence profile
rather than simply reproducing the sleep-stage effect.

## 2026-05-24: PhysioNet MI PSD Intervention And Train-On-Representation Controls

Purpose:

Run the first true intervention test on dataset 2. This mirrors the Sleep-EDF
PSD logic but uses PhysioNet MI short cue-locked trials:

```text
train on full PSD -> test original, aperiodic-only, flattened
train on aperiodic-only -> test original, aperiodic-only, flattened
train on flattened -> test original, aperiodic-only, flattened
```

This directly tests whether a model trained normally collapses when the
aperiodic component is removed, and whether residual/flattened MI information
is learnable when trained for it.

Run:

```text
script: code/scripts/run_physionet_mi_psd_interventions.py
log: /mnt/data/aperiodic_confounds/logs/physionet_mi_psd_interventions_20260524_141551.log
input index: /mnt/data/aperiodic_confounds/results/physionet_mi/imagined_fists_psd_index.csv
input decomposition: /mnt/data/aperiodic_confounds/results/physionet_mi/specparam/imagined_fists_specparam_fixed.npz
output: /mnt/data/aperiodic_confounds/results/physionet_mi/psd_interventions
status: complete
```

Evaluation:

```text
task: imagined left fist vs imagined right fist
trials: 4917
subjects: 109
folding: 5-fold subject-held-out GroupKFold
classifier: balanced RidgeClassifier
uncertainty: subject bootstrap, 10000 samples
specparam mean R2: 0.9520
specparam median R2: 0.9705
```

Subject-bootstrap balanced accuracy:

| Train input | Test input | BA [95% CI] |
| --- | --- | ---: |
| Full log-PSD | Full log-PSD | 0.566 [0.550, 0.581] |
| Full log-PSD | Aperiodic spectrum | 0.518 [0.508, 0.530] |
| Full log-PSD | Flattened log-PSD | 0.556 [0.540, 0.572] |
| Aperiodic spectrum | Full log-PSD | 0.545 [0.533, 0.558] |
| Aperiodic spectrum | Aperiodic spectrum | 0.527 [0.516, 0.538] |
| Aperiodic spectrum | Flattened log-PSD | 0.498 [0.492, 0.503] |
| Flattened log-PSD | Full log-PSD | 0.527 [0.515, 0.539] |
| Flattened log-PSD | Aperiodic spectrum | 0.499 [0.492, 0.505] |
| Flattened log-PSD | Flattened log-PSD | 0.531 [0.516, 0.547] |

Paired full-trained intervention drops:

| Intervention | BA drop [95% CI] |
| --- | ---: |
| Test on aperiodic spectrum | 0.047 [0.030, 0.065] |
| Test on flattened log-PSD | 0.009 [-0.003, 0.022] |

Interpretation:

This result is meaningfully different from Sleep-EDF. In Sleep-EDF, flattening
the aperiodic spectral envelope caused large performance collapses for several
tasks. In PhysioNet MI left-vs-right imagery, the full-PSD ridge model does
not collapse under flattening: balanced accuracy stays close to the original
input, and the flattening drop CI includes zero. Aperiodic-only test input is
worse than full PSD, suggesting that aperiodic structure alone is not the
dominant usable signal for this MI contrast.

The train-on-flattened control also matters. Flattened features trained and
tested on flattened features reach 0.531 BA [0.516, 0.547], which is weak but
above chance. This supports the interpretation that residual spectral
information exists in the MI task, but it is modest in this cross-subject ridge
setting.

Paper implication:

PhysioNet MI is currently a contrastive second-domain result, not a duplicate
of Sleep-EDF. The audit says "aperiodic reliance is task/domain dependent":
large in Sleep-EDF arousal/staging, much weaker for left-vs-right motor
imagery PSD ridge decoding. This nuance strengthens the methodological story
because it shows the pipeline can find both dependence and relative robustness.

## 2026-05-24: PhysioNet MI Raw EEGNet And ShallowFBCSPNet Interventions

Purpose:

Test whether the PhysioNet MI contrastive result also holds for raw neural EEG
models. This mirrors the Sleep-EDF reviewer-resistant raw intervention logic:

```text
train on raw cue-locked trials
test on raw_eeg, phase_sham, phase_aperiodic, phase_flattened
```

Run:

```text
script: code/scripts/run_physionet_mi_braindecode_intervention.py
log: /mnt/data/aperiodic_confounds/logs/physionet_mi_raw_braindecode_eegnet_shallow_20260524_143516.log
output: /mnt/data/aperiodic_confounds/results/physionet_mi/raw_braindecode_interventions
models: Braindecode EEGNet, Braindecode ShallowFBCSPNet
folding: 5-fold subject-held-out
seed: 42
epochs: 80 with patience 12
status: complete
```

Inputs:

```text
raw trials: /mnt/data/aperiodic_confounds/results/physionet_mi/imagined_fists_raw_160hz.npz
index: /mnt/data/aperiodic_confounds/results/physionet_mi/imagined_fists_raw_index.csv
decomposition: /mnt/data/aperiodic_confounds/results/physionet_mi/specparam/imagined_fists_specparam_fixed.npz
task: imagined left fist vs imagined right fist
trials: 4917
subjects: 109
channels: 64
trial samples: 560
sampling rate: 160 Hz
```

Subject-bootstrap balanced accuracy:

| Model | Raw | Sham | Aperiodic | Flattened | Flattening drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| EEGNet | 0.743 [0.719, 0.766] | 0.743 [0.719, 0.766] | 0.733 [0.710, 0.757] | 0.736 [0.714, 0.759] | 0.006 [-0.002, 0.014] |
| ShallowFBCSPNet | 0.669 [0.648, 0.691] | 0.669 [0.648, 0.691] | 0.634 [0.617, 0.652] | 0.659 [0.640, 0.680] | 0.009 [-0.001, 0.020] |

Interpretation:

This is an important second-domain result. The sham intervention is exactly
identical to raw for both neural models, so the Fourier reconstruction pipeline
is not causing damage. More importantly, phase-flattening barely changes
performance for either EEGNet or ShallowFBCSPNet, and both flattening-drop CIs
include zero. This agrees with the PhysioNet MI PSD intervention result and is
very different from Sleep-EDF wake-vs-sleep/five-stage staging.

The aperiodic-only intervention reduces performance slightly, especially for
ShallowFBCSPNet, but it does not become the dominant sufficient signal in the
way it often did for Sleep-EDF. In this motor-imagery domain, raw neural models
appear to rely more on residual/task-specific temporal-spectral structure than
on the aperiodic envelope alone.

Paper implication:

The second domain now supports a nuanced claim:

```text
The audit detects strong aperiodic dependence in Sleep-EDF sleep staging, but
finds relative robustness to aperiodic flattening in PhysioNet left-vs-right
motor imagery, across PSD ridge, EEGNet, and ShallowFBCSPNet.
```

This helps avoid overclaiming. The contribution becomes a general audit method
plus evidence that aperiodic reliance is task/domain dependent, not a blanket
statement that all EEG models always use the aperiodic component.

## 2026-05-24: TUAB 200-Subject Subset Preparation

Purpose:

Prepare a bounded TUAB subset for the clinical normal-vs-abnormal phase of the
project while preserving the official TUAB train/eval protocol.

Access:

```text
TUH key: ~/.ssh/id_ed25519_tuh
server: nedc-tuh-eeg@www.isip.piconepress.com
release: data/tuh_eeg/tuh_eeg_abnormal/v3.0.1
H200 access method: ssh -A h200 agent forwarding
private key copied to H200: no
```

Read-only mapping:

```text
listing: results/tuab_subset_200/tuab_v3_0_1_edf_rsync_listing.txt
README: results/tuab_subset_200/readme/AAREADME.txt
full release EDF files parsed: 2993
full release size from rsync listing: 62.7 GB
```

Subset builder:

```text
script: code/scripts/make_tuab_subset_manifest.py
seed: 20260524
subset: random_stratified_200
train normal subjects: 60
train abnormal subjects: 60
eval normal subjects: 40
eval abnormal subjects: 40
train label-conflict subjects excluded: yes
```

Selected subset:

```text
unique subjects: 200
selected EDF files: 247
selected bytes: 5,059,102,966
selected size: 4.71 GiB
```

Download:

```text
destination: /mnt/data/aperiodic_confounds/data/tuab/v3.0.1_random_stratified_200
log: /mnt/data/aperiodic_confounds/logs/tuab_subset_200_download_20260524_162716.log
status: complete
```

Verification:

```text
expected EDF files: 247
actual EDF files: 247
expected bytes: 5059102966
actual bytes: 5059102966
missing files: 0
extra files: 0
size mismatches: 0
```

Header metadata:

```text
script: code/scripts/extract_tuab_header_metadata.py
signal arrays loaded: no
subject rows: 200
sex available: yes
age available from EDF headers: no
```

Sex balance by subject:

| Split | Label | Female | Male |
| --- | --- | ---: | ---: |
| eval | abnormal | 21 | 19 |
| eval | normal | 19 | 21 |
| train | abnormal | 32 | 28 |
| train | normal | 35 | 25 |

Interpretation:

The subset is now ready for the first TUAB implementation pass. It respects the
official split, samples by subject, balances labels within train/eval, excludes
train subjects that appear under both labels, and is fully reproducible from
the saved manifests. Age matching is not yet possible because the EDF headers
do not expose per-subject age/birthday in this subset. We should report this
limitation and revisit age matching if NEDC/TUH can provide per-subject
demographics later.

## TUAB 200-Subject Preprocessing Cache

Date:

```text
2026-05-24
```

Goal:

Prepare the TUAB random stratified 200-subject subset for the second-domain
aperiodic audit without modifying or deleting any shared dataset files.

Code added:

```text
code/src/aperiodic_eeg/tuab.py
code/scripts/audit_tuab_channels.py
code/scripts/make_tuab_epochs.py
code/scripts/extract_tuab_raw_epochs.py
code/scripts/extract_tuab_psd.py
code/scripts/fit_tuab_specparam_qc.py
```

Channel audit:

```text
selected EDF files: 247
standard EEG channels requested: 21
files with all requested EEG channels: 247 / 247
sampling rates observed: 250 Hz in 236 files, 256 Hz in 10 files, 512 Hz in 1 file
```

Standard EEG channel set:

```text
FP1 FP2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 A1 A2 FZ CZ PZ
```

Epoch manifest:

```text
window length: 20 s
stride: 20 s
used files: 247
skipped files: 0
total epochs: 16458
eval abnormal epochs: 3663
eval normal epochs: 2565
train abnormal epochs: 5808
train normal epochs: 4422
```

Raw cache:

```text
remote file: /mnt/data/aperiodic_confounds/results/tuab_subset_200/raw_epochs_20s_100hz.npz
remote index: /mnt/data/aperiodic_confounds/results/tuab_subset_200/raw_epochs_20s_100hz_index.csv
shape: 16458 x 21 x 2000
target sampling rate: 100 Hz
filter: 1-45 Hz
scale: microvolts
remote size: 2.6G
```

PSD cache:

```text
remote file: /mnt/data/aperiodic_confounds/results/tuab_subset_200/psd_20s_multitaper.npz
remote index: /mnt/data/aperiodic_confounds/results/tuab_subset_200/psd_20s_multitaper_index.csv
method: multitaper
bandwidth: 2 Hz
frequency range: 1-45 Hz
shape: 16458 x 21 x 881
remote size: 1.1G
```

Specparam QC:

```text
script: code/scripts/fit_tuab_specparam_qc.py
sample: 1000 epochs, stratified 250 each from eval/abnormal, eval/normal, train/abnormal, train/normal
spectra fit: 21000
ok fraction: 1.000
median R^2: 0.971
mean R^2: 0.951
median MAE: 0.076
mean exponent: 1.552
median exponent: 1.573
```

By-group specparam exponent means:

| Split | Label | Mean exponent | Median R^2 |
| --- | --- | ---: | ---: |
| eval | abnormal | 1.686 | 0.976 |
| eval | normal | 1.494 | 0.968 |
| train | abnormal | 1.662 | 0.978 |
| train | normal | 1.365 | 0.965 |

Interpretation:

The preprocessing cache is ready for first TUAB baselines. The channel audit is
especially important because it shows the subset can be harmonized without
dropping EDFs. The specparam QC suggests that 20-second TUAB windows support
stable fixed-mode aperiodic fits. The abnormal groups show steeper mean
exponents than normal groups in this QC sample, which is exactly the kind of
signal we must audit carefully because it could reflect clinical pathology,
age/confounding, medication/state differences, or a combination of these.

## TUAB First PSD Intervention Baseline

Date:

```text
2026-05-24
```

Goal:

Run the first TUAB normal-vs-abnormal PSD baseline using the official train/eval
split and subject-stratified bootstrap over eval subjects.

Code:

```text
code/scripts/run_tuab_psd_interventions.py
```

Inputs:

```text
PSD: /mnt/data/aperiodic_confounds/results/tuab_subset_200/psd_20s_multitaper.npz
index: /mnt/data/aperiodic_confounds/results/tuab_subset_200/psd_20s_multitaper_index.csv
decomposition: vectorized fixed 1/f log-log fit
classifier: ridge, class_weight=balanced
train subjects: 120
eval subjects: 80
train epochs: 10230
eval epochs: 6228
bootstrap: 10000 stratified eval-subject resamples
```

Outputs:

```text
remote: /mnt/data/aperiodic_confounds/results/tuab_subset_200/psd_interventions_fixed
local: results/tuab_subset_200/psd_interventions_fixed
```

Fixed decomposition summary:

```text
mean fixed-fit R^2: 0.745
median fixed-fit R^2: 0.793
mean exponent: 1.683
median exponent: 1.703
```

Balanced accuracy on official eval:

| Train input | Test input | Balanced accuracy | 95% CI |
| --- | --- | ---: | --- |
| full PSD | full PSD | 0.656 | 0.590-0.722 |
| full PSD | aperiodic only | 0.530 | 0.481-0.568 |
| full PSD | residual PSD | 0.551 | 0.513-0.586 |
| aperiodic only | full PSD | 0.717 | 0.639-0.796 |
| aperiodic only | aperiodic only | 0.722 | 0.645-0.797 |
| aperiodic only | residual PSD | 0.500 | 0.500-0.500 |
| residual PSD | full PSD | 0.486 | 0.456-0.509 |
| residual PSD | aperiodic only | 0.479 | 0.447-0.505 |
| residual PSD | residual PSD | 0.625 | 0.575-0.676 |

Paired intervention drops for a model trained on full PSD:

```text
full -> aperiodic-only drop: 0.126 balanced accuracy, 95% CI 0.063-0.185
full -> residual-PSD drop: 0.105 balanced accuracy, 95% CI 0.052-0.158
```

Interpretation:

This first TUAB result is important. Unlike PhysioNet MI, where residual
oscillatory information carried the task, TUAB normal-vs-abnormal appears to
contain a strong aperiodic component. A model trained directly on the fixed
aperiodic spectrum reaches about 0.72 balanced accuracy, outperforming the
full-PSD ridge baseline. Residual-only PSD is still above chance at about 0.63,
so the task is not purely aperiodic, but the aperiodic component is clearly
not a small nuisance variable here.

Caution:

This run uses a fast fixed 1/f decomposition, not the full specparam
decomposition. The earlier TUAB specparam QC showed high fit quality, so the
next robustness step is to repeat this official-split baseline with full
specparam features. We should also treat the strong aperiodic result as
potentially confounded by age or other clinical-state differences until
demographics are available.

## TUAB Full-Specparam PSD Intervention Baseline

Date:

```text
2026-05-24
```

Goal:

Repeat the TUAB official-split PSD intervention baseline using full specparam
fixed-mode decomposition instead of the fast vectorized fixed 1/f fit.

Code:

```text
code/scripts/fit_tuab_specparam_full.py
code/scripts/run_tuab_psd_interventions.py
```

Full decomposition artifact:

```text
remote npz: /mnt/data/aperiodic_confounds/results/tuab_subset_200/specparam/specparam_fixed_20s.npz
remote summary: /mnt/data/aperiodic_confounds/results/tuab_subset_200/specparam/specparam_fixed_20s.summary.json
local summary: results/tuab_subset_200/specparam/specparam_fixed_20s.summary.json
shape: 16458 x 21 x 881
spectra fit: 345618
ok fraction: 1.000
mean R^2: 0.953
median R^2: 0.972
p10 R^2: 0.920
mean MAE: 0.081
median MAE: 0.076
mean exponent: 1.557
median exponent: 1.572
mean peaks: 4.965
remote artifact size: 3.1G
```

Baseline output:

```text
remote: /mnt/data/aperiodic_confounds/results/tuab_subset_200/psd_interventions_specparam
local: results/tuab_subset_200/psd_interventions_specparam
train subjects: 120
eval subjects: 80
train epochs: 10230
eval epochs: 6228
bootstrap: 10000 stratified eval-subject resamples
```

Balanced accuracy on official eval:

| Train input | Test input | Balanced accuracy | 95% CI |
| --- | --- | ---: | --- |
| full PSD | full PSD | 0.657 | 0.590-0.722 |
| full PSD | aperiodic only | 0.530 | 0.484-0.569 |
| full PSD | residual PSD | 0.570 | 0.531-0.608 |
| aperiodic only | full PSD | 0.694 | 0.609-0.780 |
| aperiodic only | aperiodic only | 0.676 | 0.596-0.756 |
| aperiodic only | residual PSD | 0.501 | 0.500-0.501 |
| residual PSD | full PSD | 0.527 | 0.479-0.578 |
| residual PSD | aperiodic only | 0.475 | 0.423-0.526 |
| residual PSD | residual PSD | 0.631 | 0.578-0.684 |

Paired intervention drops for a model trained on full PSD:

```text
full -> aperiodic-only drop: 0.127 balanced accuracy, 95% CI 0.063-0.186
full -> residual-PSD drop: 0.086 balanced accuracy, 95% CI 0.035-0.135
```

Interpretation:

The full-specparam result confirms the main TUAB story. Aperiodic-only features
are strongly predictive of normal-vs-abnormal status, reaching 0.676 balanced
accuracy on the official eval subset. Residual PSD is also above chance at
0.631, so TUAB contains both broad aperiodic and oscillatory/residual
information. The full-PSD model drops when tested on either isolated
representation, meaning the trained full-spectrum decision boundary combines
both components rather than transferring perfectly to either one alone.

Comparison with the fast fixed 1/f run:

The full-PSD baseline is essentially unchanged, 0.657 vs 0.656. Residual-only is
also very similar, 0.631 vs 0.625. Aperiodic-only is lower with full specparam,
0.676 vs 0.722, but remains stronger than the full-PSD ridge baseline. This
strengthens the conclusion that the TUAB finding is not an artifact of the
simple fixed-fit decomposition.

## TUAB Raw EEGNet Phase-Preserving Intervention

Date:

```text
2026-05-24
```

Goal:

Test whether the TUAB aperiodic dependence survives in a standard raw EEG deep
model, not just in PSD/ridge models.

Code:

```text
code/scripts/run_tuab_braindecode_eegnet_intervention.py
```

Inputs:

```text
raw EEG: /mnt/data/aperiodic_confounds/results/tuab_subset_200/raw_epochs_20s_100hz.npz
index: /mnt/data/aperiodic_confounds/results/tuab_subset_200/raw_epochs_20s_100hz_index.csv
specparam: /mnt/data/aperiodic_confounds/results/tuab_subset_200/specparam/specparam_fixed_20s.npz
raw shape: 16458 x 21 x 2000
train subjects: 120
eval subjects: 80
train epochs: 10230
eval epochs: 6228
```

Model:

```text
architecture: Braindecode EEGNet
braindecode version: 1.5.1
device: NVIDIA H200
seed: 42
batch size: 128
epochs requested: 80
early stopping epoch: 36
best epoch: 24
best validation loss: 0.273
```

Interventions:

```text
raw_eeg: original raw EEG
phase_sham: Fourier phase/amplitude roundtrip control
phase_aperiodic: original phase with aperiodic-shaped amplitude
phase_flattened: original phase with aperiodic shape divided out
band: 1-45 Hz
per-channel RMS matched: yes
bootstrap: 10000 stratified eval-subject resamples
```

Balanced accuracy on official eval:

| Test input | Balanced accuracy | 95% CI | Drop vs raw |
| --- | ---: | --- | ---: |
| raw EEG | 0.770 | 0.691-0.844 | 0.000 |
| sham | 0.770 | 0.690-0.844 | 0.000 |
| aperiodic-shaped | 0.757 | 0.686-0.824 | 0.012 |
| flattened | 0.573 | 0.521-0.627 | 0.196 |

Interpretation:

This is the strongest TUAB result so far. EEGNet trained on raw EEG reaches
0.770 balanced accuracy on the official eval subset. The sham control has
exactly no effect, which means the Fourier edit pipeline itself is not
disrupting the classifier. The aperiodic-shaped signal preserves almost all
performance, while flattening the aperiodic spectral shape causes a large drop
of 0.196 balanced accuracy.

In plain terms, the raw EEGNet appears to rely heavily on the broad aperiodic
spectral structure in TUAB. This directly supports the central paper narrative:
deep EEG models can learn clinically predictive structure from aperiodic
background activity, and this dependence is not limited to PSD/ridge models.

Caution:

This is one seed and one raw architecture. Before making a high-confidence
paper claim, repeat with multiple seeds and add ShallowConvNet/Deep4Net on the
same official TUAB split.

## TUAB Multi-Seed Neural Robustness Run

Date launched:

```text
2026-05-24
```

Goal:

Run the TUAB neural intervention suite with the same classifier family used in
the Sleep-EDF neural experiments, using multiple random seeds and official
TUAB train/eval split.

Code:

```text
code/scripts/run_tuab_deep_mlp_intervention.py
code/scripts/run_tuab_braindecode_eegnet_intervention.py
code/scripts/launch_tuab_multiseed_neural.sh
code/scripts/aggregate_multiseed_subject_bootstrap.py
```

Classifiers:

```text
PSD ridge: already completed in the TUAB PSD intervention baselines
deep_mlp: PSD neural baseline trained on full log-PSD
raw_cnn: custom raw EEG CNN
eegnet: Braindecode EEGNet
shallow_fbcsp: Braindecode ShallowFBCSPNet / ShallowConvNet-style baseline
deep4: Braindecode Deep4Net / DeepConvNet-style baseline
```

Run configuration:

```text
seeds: 42 43 44
raw models: raw_cnn eegnet shallow_fbcsp deep4
epochs requested: 80
official train subjects: 120
official eval subjects: 80
train epochs: 10230
eval epochs: 6228
bootstrap after all runs: 10000 hierarchical seed/subject resamples
```

Remote run:

```text
run root: /mnt/data/aperiodic_confounds/results/tuab_subset_200/multiseed_neural
log: /mnt/data/aperiodic_confounds/logs/tuab_multiseed_neural_20260524_174452.log
final aggregate CSV: /mnt/data/aperiodic_confounds/reports/tables/tuab_multiseed_neural_subject_bootstrap.csv
final aggregate Markdown: /mnt/data/aperiodic_confounds/reports/tables/tuab_multiseed_neural_subject_bootstrap.md
status at logging: running
```

Early progress:

```text
seed 42 deep_mlp: complete
seed 42 raw_cnn: complete
seed 42 eegnet: running at the time this entry was written
```

Notes:

This run writes additive result folders only. It does not delete or modify any
dataset files. Once complete, the aggregate table should be pulled back locally
and summarized here.

Completion:

```text
completed: 2026-05-24
completed model-seed jobs: 15 / 15
local run root: results/tuab_subset_200/multiseed_neural
local aggregate CSV: reports/tables/tuab_multiseed_neural_subject_bootstrap.csv
local aggregate Markdown: reports/tables/tuab_multiseed_neural_subject_bootstrap.md
```

Aggregate balanced accuracy, hierarchical seed/subject bootstrap:

| Model | Baseline | Aperiodic | Flattened | Flattening drop |
| --- | ---: | ---: | ---: | ---: |
| deep MLP | 0.739 | 0.560 | 0.607 | 0.131 |
| raw CNN | 0.782 | 0.717 | 0.614 | 0.168 |
| EEGNet | 0.763 | 0.702 | 0.605 | 0.159 |
| ShallowFBCSP | 0.754 | 0.702 | 0.589 | 0.165 |
| Deep4Net | 0.759 | 0.713 | 0.529 | 0.231 |

95% CIs for flattening drop:

```text
deep MLP: 0.035-0.231
raw CNN: 0.082-0.256
EEGNet: 0.062-0.261
ShallowFBCSP: 0.075-0.258
Deep4Net: 0.140-0.323
```

Raw-model sham control:

```text
raw CNN: drop 0.000, 95% CI 0.000-0.000
EEGNet: drop 0.000, 95% CI 0.000-0.000
ShallowFBCSP: drop 0.000, 95% CI 0.000-0.000
Deep4Net: drop 0.000, 95% CI 0.000-0.000
```

Interpretation:

The multi-seed TUAB neural result is stable and strongly supports the core
claim. Every neural model loses performance when the aperiodic spectral shape
is flattened, and every flattening-drop CI is strictly above zero. The raw
model sham controls are exactly zero, which argues that the drop is not caused
by the Fourier edit procedure itself. Aperiodic-shaped raw inputs retain high
performance across raw models, typically around 0.70 balanced accuracy.

This means the TUAB result is not architecture-specific: the effect appears in
PSD MLP, custom raw CNN, EEGNet, ShallowFBCSPNet, and Deep4Net. Deep4Net shows
the strongest flattening sensitivity. The deep MLP result differs from the
linear PSD result in that its residual/flattened performance exceeds
aperiodic-only performance, suggesting nonlinear PSD models can use residual
spectral structure more effectively than the earlier ridge baseline.

## Sleep-EDF IRASA-vs-SpecParam Agreement

Date:

```text
2026-05-24
```

Goal:

Run the decomposition-method agreement check requested in `project.md`, using
IRASA as an independent aperiodic estimator and comparing it against the
existing Sleep-EDF full-cohort SpecParam decomposition.

Code:

```text
code/scripts/fit_sleep_edf_irasa.py
code/scripts/compare_specparam_irasa.py
```

Implementation details:

```text
Sleep-EDF full raw epochs: 195469
sample: stage-balanced, 1000 epochs each from W/N1/N2/N3/REM
selected epochs: 5000
channels: Fpz-Cz, Pz-Oz
subjects represented: 78
recordings represented: 153
frequency range: 1-45 Hz
frequency step: 0.25 Hz
IRASA hset: 1.10 to 1.90 in 0.05 steps
Welch window: 4 s
seed: 20260524
unit handling: raw cache converted from microvolts back to volts before Welch
```

Outputs:

```text
remote IRASA artifact:
/mnt/data/aperiodic_confounds/results/sleep_edf_full/irasa/irasa_aperiodic_stage_balanced_5k_volts.npz

local sampled index:
results/sleep_edf_full/irasa/irasa_aperiodic_stage_balanced_5k_volts.index.csv

local agreement table:
reports/tables/irasa_specparam_agreement_stage_balanced_5k_volts/irasa_specparam_agreement.csv
```

Agreement results:

| Metric | Mean | Median | p05 | p95 |
| --- | ---: | ---: | ---: | ---: |
| aperiodic MAE, log10 power | 0.190 | 0.140 | 0.013 | 0.553 |
| bias, SpecParam minus IRASA | -0.078 | -0.037 | -0.549 | 0.249 |
| centered shape MAE, log10 power | 0.186 | 0.149 | 0.014 | 0.484 |
| aperiodic shape correlation | 0.936 | 0.966 | 0.812 | 0.988 |

Interpretation:

SpecParam and IRASA agree strongly on the aperiodic spectral shape in the
stage-balanced Sleep-EDF sample. The median shape correlation is 0.966 and the
5th percentile is still 0.812, so the agreement is not driven only by a small
subset of easy epochs. The small negative bias means SpecParam estimates are
slightly lower than IRASA on average after unit correction. This supports the
claim that the Sleep-EDF decomposition conclusions are not specific to
SpecParam alone.

Caution:

This is a stage-balanced 5k-epoch agreement sample, not a full 195k-epoch IRASA
sweep. A full sweep is possible but computationally much heavier. For paper
purposes, this sample is already useful because it covers all stages, both EEG
channels, 78 subjects, and 153 recordings.

## Sleep-EDF IRASA Downstream Ridge Intervention

Date:

```text
2026-05-24
```

Goal:

Run the downstream classification check requested after the IRASA agreement
analysis: use the IRASA-derived full PSD, aperiodic PSD, and flattened PSD in
the same ridge intervention design used for the SpecParam PSD experiments.

Code:

```text
code/scripts/run_sleep_edf_irasa_ridge_intervention.py
```

Run:

```text
cd /mnt/data/aperiodic_confounds
.venv/bin/python code/scripts/run_sleep_edf_irasa_ridge_intervention.py --n-bootstrap 10000
```

Inputs:

```text
IRASA artifact:
/mnt/data/aperiodic_confounds/results/sleep_edf_full/irasa/irasa_aperiodic_stage_balanced_5k_volts.npz

IRASA sampled index:
/mnt/data/aperiodic_confounds/results/sleep_edf_full/irasa/irasa_aperiodic_stage_balanced_5k_volts.index.csv
```

Outputs:

```text
results/sleep_edf_full/irasa/irasa_ridge_interventions_stage_balanced_5k/irasa_ridge_intervention_fold_metrics.csv
results/sleep_edf_full/irasa/irasa_ridge_interventions_stage_balanced_5k/irasa_ridge_intervention_subject_metrics.csv
results/sleep_edf_full/irasa/irasa_ridge_interventions_stage_balanced_5k/irasa_ridge_intervention_summary_metrics.csv
results/sleep_edf_full/irasa/irasa_ridge_interventions_stage_balanced_5k/irasa_ridge_intervention_subject_bootstrap.csv
results/sleep_edf_full/irasa/irasa_ridge_interventions_stage_balanced_5k/irasa_ridge_intervention_subject_bootstrap.md
```

Design:

```text
sample: same stage-balanced 5k IRASA sample
subjects: 78
folding: 5-fold GroupKFold by subject
model: StandardScaler + RidgeClassifier(alpha=1.0, class_weight=balanced)
train input: full IRASA log-PSD
test inputs: full IRASA log-PSD, IRASA aperiodic spectrum, IRASA flattened PSD
uncertainty: subject bootstrap over held-out subjects, 10000 resamples
```

Subject-bootstrap balanced accuracy:

| Task | Full PSD | IRASA aperiodic | IRASA flattened | Aperiodic drop | Flattened drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wake-vs-Sleep | 0.876 [0.859, 0.893] | 0.867 [0.845, 0.888] | 0.500 [0.500, 0.500] | 0.009 [-0.001, 0.019] | 0.376 [0.359, 0.393] |
| N2-vs-N3 | 0.842 [0.813, 0.868] | 0.859 [0.832, 0.884] | 0.417 [0.378, 0.455] | -0.017 [-0.031, -0.005] | 0.425 [0.371, 0.485] |
| Five-stage | 0.615 [0.592, 0.637] | 0.612 [0.589, 0.634] | 0.210 [0.206, 0.215] | 0.002 [-0.010, 0.015] | 0.404 [0.382, 0.428] |

Interpretation:

The downstream IRASA check strongly reproduces the central Sleep-EDF PSD
intervention pattern. A ridge classifier trained on the full IRASA PSD retains
nearly all performance when evaluated on IRASA aperiodic-only spectra. The
aperiodic-only drop is near zero for Wake-vs-Sleep and five-stage staging, and
is slightly negative for N2-vs-N3, meaning the IRASA aperiodic spectrum performs
slightly better than the full IRASA PSD for that task in this sampled check.

By contrast, the IRASA flattened PSD collapses to chance for Wake-vs-Sleep
and close to five-class chance for five-stage staging. N2-vs-N3 drops below
chance-level balanced accuracy in this intervention, which likely reflects
systematic label flipping or fold-specific decision-boundary inversion after
the spectral envelope is removed. The important paper-level point is that the
large flattening collapse is not a SpecParam-only artifact; it also appears
when the aperiodic component is estimated with IRASA.

Caution:

This is the same stage-balanced 5k-epoch sample used for IRASA agreement, not
the full Sleep-EDF epoch set. It is best framed as decomposition-method
validation of the intervention result, not as a replacement for the full
SpecParam Sleep-EDF runs.

## TUAB Age/Sex-Matched PSD Control

Date:

```text
2026-05-24
```

Goal:

Address the TUAB demographic-confounding concern: abnormal TUAB subjects skew
older, and aperiodic exponent is age-sensitive. We therefore built an
age-matched, sex-matched abnormal-vs-normal subset within the official TUAB
train/eval split and reran the PSD intervention pipeline.

Code:

```text
code/scripts/extract_tuab_header_metadata.py
code/scripts/make_tuab_age_matched_subset.py
code/scripts/run_tuab_psd_interventions.py
```

Important implementation detail:

MNE did not expose age from `raw.info["subject_info"]`, but TUAB v3.0.1 stores
age in the EDF fixed patient header as strings like:

```text
aaaaabdo F 01-JAN-0000 aaaaabdo Age:75
```

The metadata extractor now parses this `Age:` field directly from the first
256 bytes of each EDF. Values above 120 are treated as missing sentinels. In
the 200-subject subset, two train subjects had `Age:999` and were excluded
from age matching.

Age availability and imbalance in the original 200-subject subset:

| Split | Label | n with age | n total | Mean age | Median age |
| --- | --- | ---: | ---: | ---: | ---: |
| eval | abnormal | 40 | 40 | 56.83 | 56.0 |
| eval | normal | 40 | 40 | 43.42 | 39.0 |
| train | abnormal | 59 | 60 | 56.07 | 58.0 |
| train | normal | 59 | 60 | 43.22 | 44.0 |

Age/sex matching:

```text
caliper: +/-5 years
matching: same sex, within official split
train pairs: 34
eval pairs: 27
total pairs: 61
total subjects: 122
```

Matched subset demographics:

| Split | Label | n | Mean age | Median age | Female | Male |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train | abnormal | 34 | 50.15 | 51.5 | 15 | 19 |
| train | normal | 34 | 49.29 | 50.5 | 15 | 19 |
| eval | abnormal | 27 | 51.59 | 51.0 | 12 | 15 |
| eval | normal | 27 | 50.74 | 52.0 | 12 | 15 |

Run:

```text
cd /mnt/data/aperiodic_confounds
.venv/bin/python code/scripts/make_tuab_age_matched_subset.py \
  --caliper-years 5 \
  --same-sex \
  --prefix tuab_age_sex_matched_caliper5

.venv/bin/python code/scripts/run_tuab_psd_interventions.py \
  --decomposition precomputed \
  --decomp-npz results/tuab_subset_200/specparam/specparam_fixed_20s.npz \
  --subject-filter-csv results/tuab_subset_200/age_matched/tuab_age_sex_matched_caliper5_subjects.csv \
  --output-dir results/tuab_subset_200/age_matched/psd_interventions_specparam_age_sex_caliper5 \
  --n-bootstrap 10000
```

Outputs:

```text
results/tuab_subset_200/tuab_age_metadata_audit.md
results/tuab_subset_200/age_matched/tuab_age_sex_matched_caliper5_subjects.csv
results/tuab_subset_200/age_matched/tuab_age_sex_matched_caliper5_pairs.csv
results/tuab_subset_200/age_matched/tuab_age_sex_matched_caliper5_summary.md
results/tuab_subset_200/age_matched/psd_interventions_specparam_age_sex_caliper5/tuab_psd_intervention_subject_bootstrap.md
```

Matched PSD intervention result:

Subject-stratified eval-subject bootstrap, balanced accuracy:

| Train input | Test input | Matched BA | 95% CI |
| --- | --- | ---: | ---: |
| full PSD | full PSD | 0.635 | [0.554, 0.713] |
| full PSD | aperiodic | 0.541 | [0.518, 0.572] |
| full PSD | flattened | 0.544 | [0.494, 0.588] |
| aperiodic | full PSD | 0.675 | [0.595, 0.754] |
| aperiodic | aperiodic | 0.682 | [0.598, 0.763] |
| aperiodic | flattened | 0.503 | [0.499, 0.509] |
| flattened | full PSD | 0.556 | [0.503, 0.615] |
| flattened | aperiodic | 0.496 | [0.444, 0.555] |
| flattened | flattened | 0.616 | [0.546, 0.684] |

Key drops when trained on full PSD:

| Test intervention | Drop | 95% CI |
| --- | ---: | ---: |
| aperiodic-only | 0.093 | [0.015, 0.167] |
| flattened PSD | 0.091 | [0.024, 0.155] |

Comparison to unmatched TUAB PSD result:

| Quantity | Unmatched | Age/sex matched |
| --- | ---: | ---: |
| eval subjects | 80 | 54 |
| full->full BA | 0.657 | 0.635 |
| full->aperiodic BA | 0.530 | 0.541 |
| full->flattened BA | 0.570 | 0.544 |
| aperiodic->aperiodic BA | 0.676 | 0.682 |
| flattened->flattened BA | 0.631 | 0.616 |
| full-trained aperiodic-only drop | 0.127 | 0.093 |
| full-trained flattened drop | 0.086 | 0.091 |

Interpretation:

Age/sex matching does not eliminate TUAB classification performance from
aperiodic spectra. A model trained and tested on aperiodic spectra still
achieves 0.682 balanced accuracy on the matched evaluation subjects, almost
identical to the unmatched aperiodic-trained result of 0.676. This argues
against a simple "TUAB aperiodic result is only an age shortcut" explanation.

However, the full-trained intervention view changes in a more nuanced way. When
the full-PSD model is evaluated on aperiodic-only spectra, balanced accuracy is
0.541 after matching, and the drop from full PSD is 0.093. So the full-PSD
ridge model does not transfer cleanly to aperiodic-only inputs in the matched
subset. The more defensible paper claim is therefore: age is an important
confound and must be controlled, but clinically relevant aperiodic information
appears to remain after strict age/sex matching.

Caution:

This matched analysis is currently PSD/ridge only. The neural TUAB models
should be rerun on the same matched subject manifest before making strong
architecture-general claims about age-controlled TUAB results.

## TUAB BIOT Foundation Model Intervention

Date:

```text
2026-05-24
```

Goal:

Add a foundation-model-style TUAB check using the public BIOT GitHub
implementation and pretrained weights. BIOT reports TUAB normal-vs-abnormal as
a primary benchmark, so this tests whether our aperiodic intervention result
also appears in a published benchmark architecture initialized from pretrained
EEG weights.

Source:

```text
BIOT GitHub: https://github.com/ycq091044/BIOT
local remote copy: /mnt/data/aperiodic_confounds/external/BIOT
checkpoint: external/BIOT/pretrained-models/EEG-PREST-16-channels.ckpt
```

Important implementation detail:

The public BIOT checkpoints in the repo are encoder checkpoints. They do not
include a ready-made TUAB binary classifier head. Therefore the correct
implementation is:

```text
initialize BIOT encoder from official PREST-16 checkpoint
attach binary classification head
fine-tune on our TUAB train subset
evaluate official TUAB eval subset under interventions
```

Code:

```text
code/scripts/run_tuab_biot_intervention.py
```

BIOT preprocessing matched from the official TUAB processor:

```text
channels: 16 bipolar TCP-style pairs
sampling rate: 200 Hz
window length: 10 s
samples per window: 2000
normalization: per-window, per-channel 95th percentile absolute amplitude
n_fft: 200
hop_length: 100
```

Cache:

```text
results/tuab_subset_200/biot_10s_200hz_cache.npz
results/tuab_subset_200/biot_10s_200hz_cache.index.csv
results/tuab_subset_200/biot_10s_200hz_cache.summary.json
```

Cache summary:

```text
input EDF files: 247
skipped EDF files: 0
BIOT windows: 33014
window shape: 16 x 2000
```

Run:

```text
cd /mnt/data/aperiodic_confounds
/mnt/data/.venvs/ml/bin/python3 code/scripts/run_tuab_biot_intervention.py \
  --output-dir results/tuab_subset_200/biot_interventions_prest \
  --cache-npz results/tuab_subset_200/biot_10s_200hz_cache.npz \
  --epochs 30 \
  --batch-size 256 \
  --n-bootstrap 10000
```

Training:

```text
train windows: 20528
eval windows: 12486
train subjects: 120
eval subjects: 80
validation subjects from train split: 18
best epoch by validation loss: 0
best validation loss: 0.4140
validation balanced accuracy at epoch 0: 0.7777
early stopping epoch: 6
```

Intervention:

The intervention was applied after BIOT preprocessing/normalization and before
the model forward pass. It used the same phase-preserving FFT amplitude-edit
logic as the other raw-model interventions. For this BIOT-specific 10 s input,
the aperiodic shape was estimated with a fixed log-power linear aperiodic fit
over 1-45 Hz on each preprocessed window/channel.

Outputs:

```text
results/tuab_subset_200/biot_interventions_prest/tuab_biot_intervention_eval_metrics.csv
results/tuab_subset_200/biot_interventions_prest/tuab_biot_intervention_subject_bootstrap.csv
results/tuab_subset_200/biot_interventions_prest/tuab_biot_intervention_subject_bootstrap.md
results/tuab_subset_200/biot_interventions_prest/tuab_biot_intervention_metadata.json
```

Subject-stratified eval-subject bootstrap, balanced accuracy:

| Test input | BA | 95% CI | Drop vs raw | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| raw EEG | 0.673 | [0.594, 0.752] | - | - |
| phase sham | 0.672 | [0.592, 0.749] | 0.001 | [-0.005, 0.006] |
| aperiodic-shaped | 0.619 | [0.559, 0.676] | 0.054 | [-0.007, 0.115] |
| flattened | 0.520 | [0.485, 0.557] | 0.153 | [0.071, 0.232] |

Interpretation:

The BIOT result strengthens the TUAB story. The sham control is essentially
identical to raw performance, so FFT reconstruction itself is not causing the
drop. Flattening the aperiodic envelope produces a clear balanced-accuracy
drop of 0.153 with a CI strictly above zero. The aperiodic-shaped input retains
some above-chance performance but does not retain full raw performance.

This differs from the PSD/ridge pattern, where aperiodic-trained PSD models can
be very strong. BIOT is a raw waveform foundation architecture, so it likely
uses a mixture of aperiodic spectral envelope, residual oscillatory structure,
and waveform morphology. The key paper-level point is that even a pretrained
foundation-style raw model is sensitive to removing the aperiodic envelope on
TUAB, while the sham control rules out procedure-only disruption.

Caution:

This is a fine-tuned BIOT model initialized from official pretrained encoder
weights, not an off-the-shelf frozen TUAB classifier checkpoint. The public
repository did not provide a TUAB classifier checkpoint with classifier-head
weights.

## TUAB LaBraM Foundation-Model Intervention Check

Date:

```text
2026-05-24
```

Purpose:

Run LaBraM as a second foundation-model-style TUAB check, using the public
LaBraM implementation and official `labram-base.pth` checkpoint. This tests
whether the TUAB aperiodic-dependence story also appears in a transformer EEG
foundation architecture, not only in BIOT.

Important implementation detail:

The public LaBraM checkpoint is a pretrained base checkpoint, not a ready-made
TUAB classifier checkpoint for our subset. Therefore the implemented protocol
was:

```text
initialize LaBraM-base from official labram-base checkpoint
attach binary classification head
fine-tune on our TUAB train subset
evaluate official TUAB eval subset under interventions
```

Code:

```text
code/scripts/run_tuab_labram_intervention.py
```

LaBraM preprocessing matched the official TUAB maker:

```text
channels: 23 referential TUAB channels
sampling rate: 200 Hz
window length: 10 s
samples per window: 2000
filtering: 0.1-75 Hz bandpass
notch: 50 Hz
units: microvolts
model input shape: 23 x 10 x 200
checkpoint: external/LaBraM/checkpoints/labram-base.pth
```

Cache:

```text
results/tuab_subset_200/labram_10s_200hz_cache.npz
results/tuab_subset_200/labram_10s_200hz_cache.index.csv
results/tuab_subset_200/labram_10s_200hz_cache.summary.json
```

Cache summary:

```text
input EDF files: 247
skipped EDF files: 0
LaBraM windows: 33014
window shape: 23 x 2000
cache size: 5.3 GB on H200 storage
```

Run:

```text
cd /mnt/data/aperiodic_confounds
/mnt/data/.venvs/ml/bin/python3 code/scripts/run_tuab_labram_intervention.py \
  --output-dir results/tuab_subset_200/labram_interventions_base \
  --cache-npz results/tuab_subset_200/labram_10s_200hz_cache.npz \
  --epochs 30 \
  --batch-size 64 \
  --n-bootstrap 10000
```

Training:

```text
train windows: 20528
eval windows: 12486
train subjects: 120
eval subjects: 80
validation subjects from train split: 18
best epoch by validation loss: 2
best validation loss: 0.5730
validation balanced accuracy at best epoch: 0.7163
early stopping epoch: 8
balanced loss: false, matching the official LaBraM fine-tuning setup more closely
```

Intervention:

The intervention was applied after LaBraM preprocessing and before the model
forward pass. It used the same phase-preserving FFT amplitude-edit logic as the
BIOT run, with a fixed log-power linear aperiodic fit over 1-45 Hz on each
preprocessed 10 s window/channel.

Outputs:

```text
results/tuab_subset_200/labram_interventions_base/tuab_labram_intervention_eval_metrics.csv
results/tuab_subset_200/labram_interventions_base/tuab_labram_intervention_subject_bootstrap.csv
results/tuab_subset_200/labram_interventions_base/tuab_labram_intervention_subject_bootstrap.md
results/tuab_subset_200/labram_interventions_base/tuab_labram_intervention_metadata.json
```

Subject-stratified eval-subject bootstrap, balanced accuracy:

| Test input | BA | 95% CI | Drop vs raw | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| raw EEG | 0.694 | [0.617, 0.770] | - | - |
| phase sham | 0.695 | [0.617, 0.769] | -0.001 | [-0.003, 0.001] |
| aperiodic-shaped | 0.655 | [0.602, 0.708] | 0.039 | [-0.018, 0.096] |
| flattened | 0.627 | [0.562, 0.690] | 0.067 | [0.023, 0.114] |

Interpretation:

LaBraM shows a different pattern from BIOT. The sham control is unchanged,
which again validates that the FFT intervention pipeline itself is not driving
the result. Flattening the aperiodic envelope still causes a statistically
positive drop, but the drop is smaller than BIOT and performance remains well
above chance.

This suggests LaBraM is less dependent on the aperiodic envelope than BIOT on
this TUAB subset. A reasonable interpretation is that the transformer
architecture and LaBraM pretraining preserve more usable information in
residual oscillatory or waveform structure after flattening. This is good for
the paper: it makes the result more nuanced and architecture-specific instead
of claiming all foundation models fail identically.

Caution:

This is a fine-tuned LaBraM-base model initialized from the official public
pretrained checkpoint, not an off-the-shelf frozen TUAB classifier checkpoint.
The run used our 200-subject TUAB subset while preserving the official
train/eval boundary.

## 2026-05-24: PhysioNet MI Multiseed Neural Robustness Run

Purpose:

Resolve the PhysioNet MI neural-model limitation by repeating raw neural
intervention experiments across multiple seeds and adding Deep4Net for symmetry
with Sleep-EDF and TUAB.

Code:

```text
code/scripts/run_physionet_mi_braindecode_intervention.py
code/scripts/launch_physionet_mi_multiseed_neural.sh
code/scripts/aggregate_multiseed_subject_bootstrap.py
```

Run:

```text
cd /mnt/data/aperiodic_confounds
EPOCHS=80 \
BATCH_SIZE=256 \
N_BOOTSTRAP=10000 \
SEEDS="42 43 44" \
RAW_MODELS="eegnet shallow_fbcsp deep4" \
bash code/scripts/launch_physionet_mi_multiseed_neural.sh
```

Setup:

```text
dataset: PhysioNet EEG Motor Movement/Imagery
task: imagined left fist vs imagined right fist
subjects: 109
trials: 4917
trial window: 0.5-4.0 s after cue onset
evaluation: 5-fold subject-held-out GroupKFold
seeds: 42, 43, 44
models: Braindecode EEGNet, ShallowFBCSPNet, Deep4Net
uncertainty: hierarchical seed/subject bootstrap, 10000 samples
```

Outputs:

```text
results/physionet_mi/multiseed_neural
reports/tables/physionet_mi_multiseed_neural_subject_bootstrap.csv
reports/tables/physionet_mi_multiseed_neural_subject_bootstrap.md
```

Hierarchical seed/subject bootstrap, balanced accuracy:

| Model | Raw | Sham | Aperiodic | Flattened | Flattening drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| EEGNet | 0.744 [0.720, 0.767] | 0.744 [0.720, 0.767] | 0.737 [0.714, 0.760] | 0.739 [0.717, 0.761] | 0.005 [-0.002, 0.013] |
| ShallowFBCSPNet | 0.655 [0.634, 0.678] | 0.655 [0.634, 0.678] | 0.617 [0.597, 0.639] | 0.641 [0.618, 0.665] | 0.014 [0.005, 0.024] |
| Deep4Net | 0.682 [0.661, 0.703] | 0.682 [0.661, 0.703] | 0.684 [0.663, 0.705] | 0.692 [0.669, 0.715] | -0.010 [-0.020, 0.001] |

Interpretation:

The PhysioNet MI neural result is now much stronger. It no longer rests on a
single seed or only two architectures. Across EEGNet, ShallowFBCSPNet, and
Deep4Net, the sham condition is identical to raw, so the Fourier intervention
pipeline is not damaging the signal.

The flattening result remains fundamentally different from Sleep-EDF and TUAB.
EEGNet is essentially unchanged after flattening. ShallowFBCSPNet shows a
statistically positive but very small flattening drop of about 0.014 balanced
accuracy. Deep4Net is slightly better after flattening, with the drop estimate
negative and its confidence interval nearly crossing zero.

This strengthens the paper narrative: our audit does not simply force every EEG
model to fail. It detects task/domain differences. Sleep-EDF and TUAB show
strong aperiodic dependence, while PhysioNet MI left-vs-right motor imagery is
largely robust to aperiodic flattening across standard convolutional EEG
architectures.

## 2026-05-24: TUAB Age/Sex-Matched Multiseed Neural Control

Purpose:

Extend the TUAB age/sex-matched control from PSD ridge to raw neural models.
This addresses the reviewer concern that TUAB aperiodic reliance might be only
an age shortcut, especially because abnormal TUAB subjects skew older and
aperiodic exponent is age-sensitive.

Code:

```text
code/scripts/run_tuab_braindecode_eegnet_intervention.py
code/scripts/launch_tuab_age_matched_multiseed_neural.sh
code/scripts/aggregate_multiseed_subject_bootstrap.py
```

Implementation note:

`run_tuab_braindecode_eegnet_intervention.py` now accepts:

```text
--subject-filter-csv
```

The filter keeps only rows whose `official_split`, `label`, and `subject`
match the age/sex-matched subject manifest. This reuses the existing raw epoch
and specparam caches without modifying or deleting any dataset files.

Run:

```text
cd /mnt/data/aperiodic_confounds
EPOCHS=80 \
BATCH_SIZE_RAW=128 \
N_BOOTSTRAP=10000 \
SEEDS="42 43 44" \
RAW_MODELS="eegnet shallow_fbcsp deep4" \
bash code/scripts/launch_tuab_age_matched_multiseed_neural.sh
```

Setup:

```text
subject filter: results/tuab_subset_200/age_matched/tuab_age_sex_matched_caliper5_subjects.csv
train pairs: 34
eval pairs: 27
train subjects: 68
eval subjects: 54
matched raw epochs: 10063
train epochs: 5846
eval epochs: 4217
models: Braindecode EEGNet, ShallowFBCSPNet, Deep4Net
seeds: 42, 43, 44
uncertainty: hierarchical seed/subject bootstrap, 10000 samples
```

Outputs:

```text
results/tuab_subset_200/age_matched/multiseed_neural
reports/tables/tuab_age_matched_multiseed_neural_subject_bootstrap.csv
reports/tables/tuab_age_matched_multiseed_neural_subject_bootstrap.md
```

Hierarchical seed/subject bootstrap, balanced accuracy:

| Model | Raw | Sham | Aperiodic | Flattened | Flattening drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| EEGNet | 0.710 [0.615, 0.799] | 0.710 [0.615, 0.799] | 0.641 [0.539, 0.739] | 0.563 [0.448, 0.674] | 0.147 [0.057, 0.245] |
| ShallowFBCSPNet | 0.719 [0.628, 0.807] | 0.719 [0.628, 0.807] | 0.675 [0.574, 0.772] | 0.558 [0.445, 0.669] | 0.161 [0.076, 0.250] |
| Deep4Net | 0.717 [0.619, 0.807] | 0.717 [0.619, 0.807] | 0.686 [0.581, 0.782] | 0.509 [0.394, 0.625] | 0.209 [0.118, 0.303] |

Interpretation:

The age/sex-matched raw neural result is a major strengthening point. All three
models still show clear aperiodic-flattening drops after matching, and every
flattening-drop CI is above zero. The sham control is unchanged for all models,
so the drop is not due to the FFT reconstruction procedure.

This means the TUAB raw-neural aperiodic dependence cannot be dismissed as only
an age shortcut. Age remains a real confound and should be reported, but the
matched control suggests that clinically relevant aperiodic structure remains
beyond age/sex imbalance.

## TUAB EEGPT Foundation Model Intervention Check

Date:

```text
2026-05-25
```

Goal:

Add EEGPT as a third foundation-model audit on TUAB, using the same raw
intervention protocol as BIOT and LaBraM.

Source:

```text
EEGPT GitHub: https://github.com/BINE022/EEGPT
local remote copy: /mnt/data/aperiodic_confounds/external/EEGPT
checkpoint source: braindecode/eegpt-pretrained
checkpoint file: external/EEGPT/checkpoint/braindecode_eegpt_pretrained_pytorch_model.bin
```

Implementation note:

The official Figshare checkpoint link in the EEGPT README is protected by a
web challenge from the cluster, so this run used the public Braindecode EEGPT
implementation and the public `braindecode/eegpt-pretrained` PyTorch
checkpoint. Compatible pretrained encoder weights were loaded into the TUAB
EEGPT model; the downstream channel projection and classifier head were
fine-tuned on our TUAB train subset.

Code:

```text
code/scripts/run_tuab_eegpt_intervention.py
```

Preprocessing:

```text
cache: results/tuab_subset_200/labram_10s_200hz_cache.npz
channels: 23 referential TUAB channels
sampling rate: 200 Hz
window length: 10 s
samples per window: 2000
filtering: 0.1-75 Hz bandpass, 50 Hz notch
units: microvolts
normalization: no per-window amplitude normalization
```

Run:

```text
cd /mnt/data/aperiodic_confounds
PYTHONPATH=/mnt/data/aperiodic_confounds/.python_deps \
/mnt/data/.venvs/ml/bin/python3 code/scripts/run_tuab_eegpt_intervention.py \
  --output-dir results/tuab_subset_200/eegpt_interventions_braindecode \
  --epochs 30 \
  --batch-size 64 \
  --n-bootstrap 10000 \
  --device cuda
```

Training:

```text
train windows: 20528
eval windows: 12486
train subjects: 120
eval subjects: 80
validation subjects from train split: 18
best epoch by validation loss: 3
best validation loss: 0.3358
early stopping epoch: 9
```

Outputs:

```text
results/tuab_subset_200/eegpt_interventions_braindecode/tuab_eegpt_intervention_eval_metrics.csv
results/tuab_subset_200/eegpt_interventions_braindecode/tuab_eegpt_intervention_subject_bootstrap.csv
results/tuab_subset_200/eegpt_interventions_braindecode/tuab_eegpt_intervention_subject_bootstrap.md
results/tuab_subset_200/eegpt_interventions_braindecode/tuab_eegpt_intervention_metadata.json
results/tuab_subset_200/eegpt_interventions_braindecode/tuab_eegpt_training_log.csv
```

Subject-stratified eval-subject bootstrap, balanced accuracy:

| Test input | BA | 95% CI | Drop vs raw | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| raw EEG | 0.720 | [0.639, 0.795] | - | - |
| phase sham | 0.719 | [0.638, 0.796] | 0.001 | [-0.001, 0.004] |
| aperiodic-shaped | 0.609 | [0.538, 0.673] | 0.111 | [0.058, 0.165] |
| flattened | 0.673 | [0.595, 0.751] | 0.047 | [-0.011, 0.105] |

Interpretation:

EEGPT is the least flattening-sensitive of the three foundation-model checks so
far. Sham remains neutral, so the result is not an artifact of Fourier
reconstruction. The aperiodic-shaped condition produces a clearer performance
drop than flattening, while flattened performance remains high and the
flattening-drop CI includes zero. This suggests EEGPT uses a broader mixture of
residual waveform, oscillatory, and/or spatial-temporal information rather than
depending as strongly on the smooth aperiodic envelope as BIOT. The paper-level
message becomes stronger and more nuanced: aperiodic reliance is present in
foundation models, but it is not identical across architectures.

## TUAB BENDR Foundation Model Intervention Check

Date:

```text
2026-05-25
```

Goal:

Add BENDR as another TUAB foundation-model audit, using the same 23-channel
10 s TUAB cache, raw/sham/aperiodic-shaped/flattened intervention conditions,
and eval-subject bootstrap reporting used for BIOT, LaBraM, and EEGPT.

Code:

```text
code/scripts/run_tuab_bendr_intervention.py
```

Model source:

```text
Braindecode InterpolatedBENDR
pretrained checkpoint: braindecode/braindecode-bendr
checkpoint file on H200:
/home/vinay/.cache/huggingface/hub/models--braindecode--braindecode-bendr/snapshots/191f221cd56de8203899ea9a8d0f43238724f8b6/model.safetensors
```

Implementation notes:

The full BENDR contextualizer was implemented and smoke-tested, but on the
full TUAB training cache it repeatedly produced non-finite loss after a few
epochs. The reportable run therefore uses Braindecode's documented
`encoder_only=True` BENDR downstream mode, which loads the pretrained
convolutional encoder and uses four-chunk temporal pooling for the classifier
head. This is recorded explicitly in the metadata.

BENDR also required an amplitude-stability guard on the TUAB cache:

```text
input cache units: microvolts
clip before model input: +/-500 uV
input scale after clipping: 1e-6
model input units after scaling: volts
learning rate: 5e-5
gradient clipping: 1.0
```

Run:

```text
cd /mnt/data/aperiodic_confounds
PYTHONPATH=/mnt/data/aperiodic_confounds/.python_deps \
/mnt/data/.venvs/ml/bin/python3 code/scripts/run_tuab_bendr_intervention.py \
  --output-dir results/tuab_subset_200/bendr_interventions_braindecode_encoder_only \
  --encoder-only \
  --epochs 30 \
  --batch-size 128 \
  --n-bootstrap 10000 \
  --device cuda
```

Training:

```text
train windows: 20528
eval windows: 12486
train subjects: 120
eval subjects: 80
validation subjects from train split: 18
best epoch by validation loss: 1
best validation loss: 0.6000
early stopping epoch: 7
```

Outputs:

```text
results/tuab_subset_200/bendr_interventions_braindecode_encoder_only/tuab_bendr_intervention_eval_metrics.csv
results/tuab_subset_200/bendr_interventions_braindecode_encoder_only/tuab_bendr_intervention_subject_bootstrap.csv
results/tuab_subset_200/bendr_interventions_braindecode_encoder_only/tuab_bendr_intervention_subject_bootstrap.md
results/tuab_subset_200/bendr_interventions_braindecode_encoder_only/tuab_bendr_intervention_metadata.json
results/tuab_subset_200/bendr_interventions_braindecode_encoder_only/tuab_bendr_training_log.csv
```

Subject-stratified eval-subject bootstrap, balanced accuracy:

| Test input | BA | 95% CI | Drop vs raw | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| raw EEG | 0.507 | [0.495, 0.520] | - | - |
| phase sham | 0.507 | [0.495, 0.519] | 0.001 | [-0.001, 0.003] |
| aperiodic-shaped | 0.508 | [0.496, 0.521] | -0.001 | [-0.004, 0.003] |
| flattened | 0.507 | [0.495, 0.519] | 0.001 | [-0.003, 0.005] |

Interpretation:

BENDR encoder-only did not learn a useful TUAB abnormality classifier under
this subset/preprocessing protocol; raw balanced accuracy is essentially
chance. Consequently, the near-zero flattening drop should not be interpreted
as evidence of aperiodic robustness. The main value of this run is negative
and methodological: not every public foundation checkpoint transfers cleanly
to the TUAB subset under our unified preprocessing, and BENDR should be kept
separate from the stronger BIOT, LaBraM, and EEGPT foundation-model evidence
unless we later implement BENDR's original preprocessing more faithfully.

## TUAB CBraMod Foundation Model Intervention Check

Date:

```text
2026-05-25
```

Goal:

Add CBraMod as another TUAB foundation-model audit, using the same TUAB subject
subset and raw/sham/aperiodic-shaped/flattened intervention protocol as BIOT,
LaBraM, EEGPT, and BENDR.

Code:

```text
code/scripts/run_tuab_cbramod_intervention.py
```

Model source:

```text
Braindecode CBraMod
pretrained checkpoint: braindecode/cbramod-pretrained
checkpoint file on H200:
/home/vinay/.cache/huggingface/hub/models--braindecode--cbramod-pretrained/snapshots/584cdc415913739a05d84bf0c1cb3db397764507/model.safetensors
```

Implementation notes:

CBraMod has an official TUAB preprocessing convention, so the runner does not
feed the 23 referential cache directly. Instead, it derives the official
16-channel longitudinal bipolar TUAB montage from the shared 23-channel cache:

```text
FP1-F7, F7-T3, T3-T5, T5-O1,
FP2-F8, F8-T4, T4-T6, T6-O2,
FP1-F3, F3-C3, C3-P3, P3-O1,
FP2-F4, F4-C4, C4-P4, P4-O2
```

The resulting bipolar signals are divided by 100, matching the official
CBraMod TUAB dataset loader. CBraMod is then initialized from the public
checkpoint, `proj_out` is replaced with identity, and the classifier uses the
official all-patch-representations three-layer head. The intervention is
applied after this CBraMod TUAB preprocessing and before the model forward
pass.

One minor protocol difference remains: our shared cache is filtered with
0.1-75 Hz and 50 Hz notch, whereas the official CBraMod TUAB preprocessing
uses 0.3-75 Hz and 60 Hz notch. We keep the shared cache for consistency with
the other foundation-model audits and record this in metadata.

Run:

```text
cd /mnt/data/aperiodic_confounds
PYTHONPATH=/mnt/data/aperiodic_confounds/.python_deps \
/mnt/data/.venvs/ml/bin/python3 code/scripts/run_tuab_cbramod_intervention.py \
  --output-dir results/tuab_subset_200/cbramod_interventions_braindecode \
  --epochs 30 \
  --batch-size 64 \
  --n-bootstrap 10000 \
  --device cuda
```

Training:

```text
train windows: 20528
eval windows: 12486
train subjects: 120
eval subjects: 80
validation subjects from train split: 18
best epoch by validation loss: 0
best validation loss: 0.4959
early stopping epoch: 6
```

Outputs:

```text
results/tuab_subset_200/cbramod_interventions_braindecode/tuab_cbramod_intervention_eval_metrics.csv
results/tuab_subset_200/cbramod_interventions_braindecode/tuab_cbramod_intervention_subject_bootstrap.csv
results/tuab_subset_200/cbramod_interventions_braindecode/tuab_cbramod_intervention_subject_bootstrap.md
results/tuab_subset_200/cbramod_interventions_braindecode/tuab_cbramod_intervention_metadata.json
results/tuab_subset_200/cbramod_interventions_braindecode/tuab_cbramod_training_log.csv
```

Subject-stratified eval-subject bootstrap, balanced accuracy:

| Test input | BA | 95% CI | Drop vs raw | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| raw EEG | 0.691 | [0.616, 0.761] | - | - |
| phase sham | 0.692 | [0.615, 0.761] | -0.001 | [-0.002, 0.001] |
| aperiodic-shaped | 0.607 | [0.542, 0.668] | 0.084 | [0.044, 0.123] |
| flattened | 0.657 | [0.585, 0.726] | 0.034 | [-0.033, 0.097] |

Interpretation:

CBraMod transfers meaningfully to the TUAB subset, unlike BENDR. The sham
condition is neutral, so the Fourier reconstruction itself is not damaging
performance. The aperiodic-shaped input produces a clear drop, showing that
CBraMod uses clinically useful information in the broadband spectral envelope.
The flattened condition drops only modestly and its confidence interval crosses
zero, suggesting CBraMod retains more useful residual oscillatory/spatiotemporal
information than BIOT or LaBraM, and is closer to EEGPT in being partially
robust to flattening. This is a useful foundation-model row, but the exact
magnitude should be interpreted alongside the CBraMod-specific bipolar montage
and input scaling.

## TUAB EEGMamba Foundation Model Attempt

Date:

```text
2026-05-25
```

Goal:

Add EEGMamba as the next TUAB foundation-model audit, extending the
foundation-model panel with a state-space/Mamba architecture.

Code:

```text
code/scripts/run_tuab_eegmamba_intervention.py
```

Implementation status:

The runner was implemented using the official `wjq-learning/EEGMamba` GitHub
repo and the public Hugging Face checkpoint:

```text
official repo: /mnt/data/aperiodic_confounds/external/EEGMamba
checkpoint source: weighting666/EEGMamba
checkpoint file:
/home/vinay/.cache/huggingface/hub/models--weighting666--EEGMamba/snapshots/0b060d87acd6f23bf1d0b852bf1726064f335f97/pretrained_EEGMamba.pth
```

The input protocol mirrors CBraMod/EEGMamba's official TUAB convention:

```text
derive 16 longitudinal bipolar TUAB channels from the shared 23-channel cache
divide input by 100
set proj_out to Identity
use official all_patch_reps-style three-layer classifier head
apply raw/sham/aperiodic-shaped/flattened interventions after EEGMamba TUAB preprocessing
```

Smoke command:

```text
cd /mnt/data/aperiodic_confounds
PYTHONPATH=/mnt/data/aperiodic_confounds/.python_deps \
/mnt/data/.venvs/ml/bin/python3 code/scripts/run_tuab_eegmamba_intervention.py \
  --output-dir results/tuab_subset_200/eegmamba_interventions_official_smoke \
  --epochs 1 \
  --batch-size 4 \
  --max-train-windows 16 \
  --max-eval-windows 16 \
  --n-bootstrap 10 \
  --device cuda
```

Outcome:

Blocked by missing `mamba_ssm`.

The official EEGMamba model imports `mamba_ssm.models.config_mamba`,
`mamba_ssm.modules.mamba2`, and related Mamba blocks. The package is not
installed on H200. Installation attempts in the available Python 3.12 CUDA
venv failed because pip fell back to building from source and the environment
does not expose an `nvcc` compiler. No compatible prebuilt `mamba-ssm` wheel
was available for the current Torch/CUDA stack.

Interpretation:

EEGMamba is not scientifically failed; it is infrastructure-blocked. The
runner is ready, the official checkpoint is downloaded, and the blocker is
specifically the missing Mamba SSM runtime. To run this model, we need either
a venv/container with compatible prebuilt `mamba-ssm`, or a CUDA development
environment with `nvcc` available so `mamba-ssm` can be built.

## TUAB EEGMamba Foundation Model Run And Sanity Checks

Date:

```text
2026-05-25
```

Goal:

Resolve the EEGMamba runtime blocker, run the TUAB intervention audit, and
stress-test the surprising weak/absent flattening drop.

Environment:

```text
venv: /mnt/data/aperiodic_confounds/.venvs/eegmamba
torch: 2.5.1+cu124
mamba-ssm: 2.2.6.post3
causal-conv1d: 1.6.0
transformers: 4.44.2
```

Implementation notes:

The successful runtime required an isolated environment. Torch 2.6-compatible
`mamba-ssm` wheels imported with a compiled-symbol mismatch, while the
Torch 2.5.1/CUDA 12.4 wheel combination imported and executed correctly after
adding the matching `causal-conv1d` wheel. The EEGMamba wrapper keeps the
shared TUAB cache in `(window, channel, time)` format and reshapes only inside
the model forward pass to EEGMamba's expected `(batch, channel, patch, sample)`
layout.

Primary run:

```text
results/tuab_subset_200/eegmamba_interventions_official
```

Primary result:

| Input | Balanced accuracy | Drop vs raw |
| --- | ---: | ---: |
| raw EEG | 0.632 [0.571, 0.691] | - |
| phase sham | 0.631 [0.569, 0.691] | 0.001 |
| aperiodic-shaped | 0.558 [0.505, 0.609] | 0.075 |
| flattened | 0.650 [0.590, 0.707] | -0.017 [-0.053, 0.018] |

Sanity-check matrix:

```text
results/tuab_subset_200/eegmamba_sanity_checks
results/tuab_subset_200/eegmamba_sanity_checks/eegmamba_sanity_summary.md
```

All sanity checks used validation balanced accuracy for checkpoint selection
and three seeds (`20260525`, `20260526`, `20260527`). The matrix tested:

```text
full fine-tuning, /100 input scaling
full fine-tuning, no input scaling
full fine-tuning, per-window/channel z-score
frozen EEGMamba backbone, /100 input scaling
```

Summary:

| Run | Raw BA range | Flattened BA range | Flattening drop pattern |
| --- | ---: | ---: | --- |
| /100 full fine-tune | 0.647-0.690 | 0.671-0.694 | near zero or negative |
| no-scale full fine-tune | 0.664-0.703 | 0.615-0.646 | positive but seed-sensitive |
| z-score full fine-tune | 0.653-0.678 | 0.624-0.680 | small/inconsistent |
| /100 frozen backbone | 0.609-0.663 | 0.620-0.676 | near zero or negative |

Interpretation:

The sanity checks support the skepticism: EEGMamba does not show the stable
flattening collapse seen for BIOT/LaBraM. Its raw TUAB performance is modest,
sham is appropriately neutral, and aperiodic-shaped inputs consistently reduce
performance. Flattening, however, is unstable: it is near zero or even negative
under the official-style `/100` scaling and frozen-backbone settings, while
no-scaling creates a positive but seed-sensitive drop. This suggests the
EEGMamba row should be treated as a sensitivity/supplementary result unless we
can exactly reproduce the authors' TUAB preprocessing and fine-tuning protocol.

## TUAB REVE-base Foundation Model Run

Date:

```text
2026-05-25
```

Goal:

Add REVE-base as a newer EEG foundation model with flexible electrode-position
encoding and test whether its TUAB performance depends on aperiodic structure.

Code:

```text
code/scripts/run_tuab_reve_intervention.py
```

Model sources:

```text
encoder: brain-bzh/reve-base
position bank: brain-bzh/reve-positions
```

Implementation notes:

REVE-base is gated on Hugging Face. Access was authenticated on H200 through
the user's Hugging Face account and cached outside the project tree. The runner
uses the Hugging Face `AutoModel` interface with `trust_remote_code=True`.
The model expects 200 Hz EEG and a position tensor. The REVE position bank
recognized 21 of our 23 TUAB referential channels; `T1/T2` were dropped rather
than assigning invented coordinates. Inputs were z-scored per window/channel
and clipped to +/-15, matching REVE's usage guidance. The encoder returns
tokens shaped `(batch, channels, patches, 512)`, which were pooled with REVE's
built-in `attention_pooling` method and passed to a small binary classifier
head. Interventions were applied after REVE preprocessing and before model
forward, matching the other raw-foundation-model audits.

Output:

```text
results/tuab_subset_200/reve_base_interventions
```

Result:

| Input | Balanced accuracy | Drop vs raw |
| --- | ---: | ---: |
| raw EEG | 0.755 [0.674, 0.830] | - |
| phase sham | 0.755 [0.675, 0.833] | 0.000 |
| aperiodic-shaped | 0.663 [0.612, 0.716] | 0.092 |
| flattened | 0.697 [0.634, 0.759] | 0.058 [0.003, 0.115] |

Interpretation:

REVE-base is a strong and useful TUAB foundation-model row. It performs better
than BIOT/LaBraM/CBraMod/EEGMamba in our subset audit, has an exactly neutral
sham control, and shows a statistically positive flattening drop. The drop is
smaller than BIOT/LaBraM but clearer than EEGMamba, suggesting REVE uses both
aperiodic spectral structure and residual morphology/oscillatory information.
This strengthens the foundation-model narrative because a newer model designed
for arbitrary electrode configurations still carries measurable aperiodic
reliance.

## TUAB Full-Dataset Preprocessing Launch

Date:

```text
2026-05-26
```

Goal:

Scale the TUAB preprocessing cache from the 200-subject subset to the full
TUAB v3.0.1 EDF manifest while preserving the exact preprocessing choices used
for the 200-subject experiment.

Code:

```text
code/scripts/run_tuab_full_preprocessing.sh
code/scripts/audit_tuab_channels.py
code/scripts/make_tuab_epochs.py
code/scripts/extract_tuab_raw_epochs.py
code/scripts/extract_tuab_psd.py
code/scripts/fit_tuab_specparam_qc.py
```

Remote paths:

```text
data root: /mnt/data/aperiodic_confounds/data/tuab/v3.0.1_random_stratified_200
manifest: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/tuab_v3_0_1_full_edf_files.csv
output: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/preprocess_20s_100hz
log: /mnt/data/aperiodic_confounds/logs/tuab_full_preprocessing_20260526_030925.log
```

Preprocessing settings:

```text
channels: FP1 FP2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 A1 A2 FZ CZ PZ
epoch length: 20 s
stride: 20 s
raw target sampling rate: 100 Hz
raw filter: 1-45 Hz
raw scale: microvolts
PSD: multitaper, 1-45 Hz, 2 Hz bandwidth
specparam QC: fixed mode, 250 epochs per split/label, 16 jobs
```

Completed stages so far:

```text
channel audit: complete
epoch manifest: complete
raw epoch extraction: running
```

Initial full-dataset audit results:

```text
EDF files: 2993
files with all 21 requested channels: 2993 / 2993
sampling rates: 250 Hz in 2786 files, 256 Hz in 189 files, 512 Hz in 18 files
used files: 2993
skipped files: 0
total 20 s epochs: 204122
eval abnormal epochs: 8492
eval normal epochs: 9925
train abnormal epochs: 94123
train normal epochs: 91582
```

Notes:

The launcher uses stage marker files under
`results/tuab_full_v3_0_1/preprocess_20s_100hz/stage_markers`, so completed
preprocessing stages are skipped on rerun. No dataset files are deleted or
modified.

## TUAB Full-Dataset Specparam and PSD Intervention Launch

Date:

```text
2026-05-26
```

Goal:

Run the same precomputed-specparam PSD ridge intervention used in the TUAB
200-subject experiment, now on the full TUAB v3.0.1 preprocessing cache.

Code:

```text
code/scripts/run_tuab_full_specparam_psd_intervention.sh
code/scripts/fit_tuab_specparam_full.py
code/scripts/run_tuab_psd_interventions.py
```

Remote paths:

```text
PSD: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/preprocess_20s_100hz/psd_20s_multitaper.npz
PSD index: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/preprocess_20s_100hz/psd_20s_multitaper_index.csv
specparam output: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/preprocess_20s_100hz/specparam/specparam_fixed_20s.npz
PSD intervention output: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/psd_interventions_specparam
log: /mnt/data/aperiodic_confounds/logs/tuab_full_specparam_psd_20260526_041055.log
```

Settings:

```text
specparam mode: fixed
frequency range: 1-45 Hz
max peaks: 6
peak width limits: 0.5-8.0 Hz
min peak height: 0.1
peak threshold: 2.0
parallel jobs: 16
chunk spectra: 8192
PSD classifier: RidgeClassifier, alpha=1.0, class_weight=balanced
bootstrap: 10000 stratified eval-subject resamples
```

Status:

```text
launched at: 2026-05-26 04:10 IST
completed at: 2026-05-26 05:00 IST
```

Notes:

The launcher writes stage markers under the full TUAB preprocessing directory.
If the full specparam decomposition completes but a later step fails, rerunning
the launcher will skip the completed decomposition and continue from the PSD
intervention stage. No dataset files are deleted or modified.

Full specparam result:

```text
shape: 204122 x 21 x 881
fit success: 1.000
mean R^2: 0.956
median R^2: 0.971
p10 R^2: 0.924
mean MAE: 0.081
median MAE: 0.076
mean exponent: 1.529
median exponent: 1.529
errors: 0
```

PSD intervention result, official TUAB eval split:

| Train input | Test input | Balanced accuracy | 95% CI |
| --- | --- | ---: | ---: |
| full log-PSD | full log-PSD | 0.752 | [0.708, 0.793] |
| full log-PSD | aperiodic spectrum | 0.588 | [0.555, 0.620] |
| full log-PSD | flattened log-PSD | 0.591 | [0.566, 0.617] |
| aperiodic spectrum | full log-PSD | 0.721 | [0.675, 0.763] |
| aperiodic spectrum | aperiodic spectrum | 0.711 | [0.667, 0.754] |
| aperiodic spectrum | flattened log-PSD | 0.500 | [0.500, 0.501] |
| flattened log-PSD | full log-PSD | 0.633 | [0.596, 0.671] |
| flattened log-PSD | aperiodic spectrum | 0.506 | [0.500, 0.513] |
| flattened log-PSD | flattened log-PSD | 0.724 | [0.683, 0.762] |

Primary full-PSD flattening drop:

```text
balanced accuracy drop: 0.160 [0.125, 0.197]
```

Interpretation:

The full TUAB PSD result strongly preserves the 200-subject finding. Full
log-PSD performs well, and flattening the aperiodic envelope causes a clear
balanced-accuracy drop. Aperiodic-only training remains high-performing on the
official eval split, while the aperiodic-trained model collapses to chance when
tested on flattened spectra. Importantly, models trained directly on flattened
spectra perform well on flattened spectra, showing that residual non-aperiodic
information exists in TUAB; the audit is therefore not merely destroying all
signal, but testing whether models trained on ordinary spectra rely on the
aperiodic component.

## TUAB Full-Dataset One-Seed Raw Neural Launch

Date:

```text
2026-05-26
```

Goal:

Start the full-TUAB raw neural intervention scale-up with one seed before
launching multiseed runs. This is intended to verify runtime, GPU use, and
memory behavior while preserving the TUAB-200 raw-neural intervention protocol.

Code:

```text
code/scripts/run_tuab_braindecode_eegnet_intervention.py
code/scripts/launch_tuab_full_single_seed_neural.sh
```

Remote paths:

```text
raw cache: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/preprocess_20s_100hz/raw_epochs_20s_100hz.npz
raw index: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/preprocess_20s_100hz/raw_epochs_20s_100hz_index.csv
specparam: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/preprocess_20s_100hz/specparam/specparam_fixed_20s.npz
run root: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/single_seed_neural
log: /mnt/data/aperiodic_confounds/logs/tuab_full_single_seed_neural_streamed_20260526_053636.log
```

Settings:

```text
seed: 42
models: EEGNet, ShallowFBCSPNet, Deep4Net
epochs requested: 80
batch size: 512
device: cuda
bootstrap: 10000 eval-subject resamples per model
```

Implementation note:

The first full-cache attempt exposed a memory issue inherited from the
TUAB-200-scale runner: it materialized full training and intervention arrays in
RAM. The runner was patched to stream train/eval batches from the full cache,
compute channel standardization in chunks, validate in batches, and generate
phase-preserving interventions only for eval chunks. This does not change the
model architecture, official split, Fourier intervention, RMS matching, or
metrics; it only avoids full-dataset array duplication.

Initial status:

```text
model: EEGNet
train epochs: 185705
eval epochs: 18417
train subjects: 2076
eval subjects: 253
H200 batch size: 512
GPU utilization observed: ~72%
GPU memory observed: ~3.7 GiB
CPU memory stable after patch: ~50 GiB used, no swap pressure
first epochs: epoch 0 train_loss=0.58582 val_loss=0.73581; epoch 1 train_loss=0.51736 val_loss=1.06413; epoch 2 train_loss=0.50341 val_loss=0.64438
```

Completed result:

```text
completed models: EEGNet, ShallowFBCSPNet, Deep4Net
completed at: 2026-05-26 06:31 IST
local run root: results/tuab_full_v3_0_1/single_seed_neural
local report: reports/tables/tuab_full_single_seed_neural_subject_bootstrap.md
```

Epoch-level official eval balanced accuracy:

| Model | Raw | Sham | Aperiodic | Flattened | Flattening drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| EEGNet | 0.781 | 0.781 | 0.557 | 0.662 | 0.118 |
| ShallowFBCSPNet | 0.794 | 0.794 | 0.598 | 0.676 | 0.118 |
| Deep4Net | 0.803 | 0.803 | 0.589 | 0.661 | 0.142 |

Subject-bootstrap balanced accuracy:

| Model | Raw | Sham | Aperiodic | Flattened | Flattening drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| EEGNet | 0.792 [0.753, 0.830] | 0.792 [0.753, 0.830] | 0.495 [0.443, 0.548] | 0.712 [0.661, 0.761] | 0.080 [0.031, 0.131] |
| ShallowFBCSPNet | 0.801 [0.760, 0.839] | 0.801 [0.760, 0.839] | 0.537 [0.485, 0.587] | 0.723 [0.674, 0.771] | 0.077 [0.031, 0.122] |
| Deep4Net | 0.807 [0.767, 0.845] | 0.807 [0.767, 0.845] | 0.534 [0.484, 0.584] | 0.703 [0.652, 0.754] | 0.104 [0.054, 0.153] |

Interpretation:

The one-seed full-TUAB neural run confirms that the full dataset is tractable on
the H200 after streaming optimization, and the qualitative TUAB-200 neural
finding survives at full scale. All three standard EEG-DL architectures reach
strong raw performance, all sham controls are exactly neutral, and all
flattening-drop subject-bootstrap CIs are above zero. The full-dataset
flattening drops are smaller than the TUAB-200 multiseed drops but remain
consistent and positive across architectures. This makes the next required step
a full multiseed run using the same streamed implementation.

## TUAB Full-Dataset Multiseed Raw Neural Launch

Date:

```text
2026-05-26
```

Goal:

Extend the full-TUAB raw neural result from one seed to the same three-seed
robustness convention used for TUAB-200 and PhysioNet MI.

Code:

```text
code/scripts/launch_tuab_full_multiseed_neural.sh
code/scripts/run_tuab_braindecode_eegnet_intervention.py
code/scripts/aggregate_multiseed_subject_bootstrap.py
```

Remote paths:

```text
run root: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/multiseed_neural
single-seed source reused for seed 42: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/single_seed_neural
log: /mnt/data/aperiodic_confounds/logs/tuab_full_multiseed_neural_20260526_073949.log
final report: /mnt/data/aperiodic_confounds/reports/tables/tuab_full_multiseed_neural_subject_bootstrap.md
```

Settings:

```text
seeds: 42, 43, 44
models: EEGNet, ShallowFBCSPNet, Deep4Net
epochs requested: 80
batch size: 512
device: cuda
bootstrap: 10000 hierarchical seed/subject resamples
```

Initial status:

```text
seed 42 EEGNet/ShallowFBCSPNet/Deep4Net: reused from completed single-seed run
seed 43 EEGNet: running
```

Notes:

The launcher copies the completed seed-42 outputs into the multiseed run root
instead of retraining them, then runs only missing seed/model combinations. It
will aggregate all three seeds after the remaining six model runs complete.

Completed result:

```text
completed at: 2026-05-26 09:33 IST
completed jobs: 9/9
seeds: 42, 43, 44
models: EEGNet, ShallowFBCSPNet, Deep4Net
local run root: results/tuab_full_v3_0_1/multiseed_neural
local report: reports/tables/tuab_full_multiseed_neural_subject_bootstrap.md
remote report: /mnt/data/aperiodic_confounds/reports/tables/tuab_full_multiseed_neural_subject_bootstrap.md
```

Subject-bootstrap balanced accuracy, aggregated over seeds:

| Model | Raw | Sham | Aperiodic | Flattened | Flattening drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| EEGNet | 0.804 [0.765, 0.843] | 0.804 [0.765, 0.843] | 0.578 [0.491, 0.658] | 0.707 [0.655, 0.758] | 0.097 [0.050, 0.144] |
| ShallowFBCSPNet | 0.796 [0.755, 0.836] | 0.796 [0.755, 0.836] | 0.575 [0.511, 0.639] | 0.727 [0.676, 0.774] | 0.070 [0.020, 0.119] |
| Deep4Net | 0.816 [0.777, 0.854] | 0.816 [0.777, 0.854] | 0.574 [0.512, 0.635] | 0.687 [0.633, 0.739] | 0.129 [0.075, 0.184] |

Interpretation:

The full-TUAB multiseed result is complete and stable. Raw performance is strong
for all three neural architectures, sham intervention is exactly neutral, and
flattening the aperiodic envelope produces a positive drop with confidence
intervals above zero for every architecture. The largest flattening drop appears
in Deep4Net, followed by EEGNet and ShallowFBCSPNet. Compared with the earlier
TUAB-200 subset, the full-dataset neural drops are smaller, but the main
necessity result is more credible because it now holds across the full official
TUAB scale, three random seeds, and subject-level aggregation.

## TUAB Full-Dataset BIOT Foundation Model Launch

Date:

```text
2026-05-26
```

Goal:

Scale the TUAB-200 BIOT foundation-model audit to the full TUAB dataset, using
the same BIOT preprocessing and intervention protocol as before.

Code:

```text
code/scripts/run_tuab_biot_intervention.py
code/scripts/launch_tuab_full_biot_intervention.sh
```

Remote paths:

```text
data root: /mnt/data/aperiodic_confounds/data/tuab/v3.0.1_random_stratified_200
manifest: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/tuab_v3_0_1_full_edf_files.csv
cache prefix: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/biot_10s_200hz_cache
run root: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/biot_interventions_prest_full
log: /mnt/data/aperiodic_confounds/logs/tuab_full_biot_20260526_122500.log
checkpoint: /mnt/data/aperiodic_confounds/external/BIOT/pretrained-models/EEG-PREST-16-channels.ckpt
```

Settings:

```text
channels: 16 bipolar TCP-style BIOT pairs
sampling rate: 200 Hz
window length: 10 s
normalization: per-window per-channel 95th percentile absolute amplitude
epochs requested: 30
batch size: 256
intervention batch size: 512
bootstrap: 10000 eval-subject resamples
seed: 20260524
device: cuda
```

Implementation note:

The original TUAB-200 BIOT cache was an in-memory compressed NPZ, which is not
appropriate for full TUAB. The BIOT runner was extended with an `npy`
memmap-backed cache mode, streaming training batches, batched validation, and
chunked raw/sham/aperiodic/flattened intervention evaluation. This preserves
the BIOT preprocessing, pretrained encoder initialization, classifier training,
Fourier intervention, RMS matching, and subject-bootstrap metrics while avoiding
full-dataset array duplication in host RAM.

Initial status:

```text
launched on H200 at: 2026-05-26 12:25 IST
process: /mnt/data/.venvs/ml/bin/python3 code/scripts/run_tuab_biot_intervention.py
planned BIOT windows: 409455
cache x file: results/tuab_full_v3_0_1/biot_10s_200hz_cache.x.npy
cache x size: approximately 49 GiB
status at 12:25 IST: writing BIOT cache, 40/2993 usable EDF files complete
```

Resume note:

```text
cache completed at: 2026-05-26 12:45 IST
cache windows: 409455
initial crash: Python 3.14 DataLoader multiprocessing could not pickle a local dataset class
fix: use single-process streaming DataLoader for the memmap cache
resumed log: /mnt/data/aperiodic_confounds/logs/tuab_full_biot_resume_20260526_135649.log
status at 2026-05-26 14:03 IST: fine-tuning, epoch 4 complete
```

Completed result:

```text
completed at: 2026-05-26 14:07 IST
local run root: results/tuab_full_v3_0_1/biot_interventions_prest_full
remote run root: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/biot_interventions_prest_full
train windows: 372510
eval windows: 36945
eval subjects: 253
best epoch by validation loss: 0
early stopping epoch: 6
```

Subject-stratified eval-subject bootstrap, balanced accuracy:

| Test input | BA | 95% CI | Drop vs raw | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| raw EEG | 0.799 | [0.763, 0.834] | - | - |
| phase sham | 0.798 | [0.762, 0.833] | 0.001 | [-0.001, 0.003] |
| aperiodic-shaped | 0.696 | [0.662, 0.729] | 0.103 | [0.077, 0.128] |
| flattened | 0.682 | [0.648, 0.715] | 0.118 | [0.088, 0.146] |

Interpretation:

The full-TUAB BIOT result strongly preserves the foundation-model finding from
the 200-subject subset. Raw performance is much stronger at full scale, the
sham reconstruction remains essentially neutral, and both aperiodic-shaped and
flattened interventions reduce balanced accuracy with CIs clearly above zero.
The flattened drop is slightly smaller than in the TUAB-200 BIOT run but is now
estimated on 253 official eval subjects and 36,945 eval windows, making it much
more robust for the paper. Importantly, flattened performance remains above
chance, so BIOT is not using only the aperiodic envelope; it uses a mixture of
aperiodic structure and residual waveform/oscillatory information.

## TUAB Full-Dataset LaBraM Foundation Model Preparation

Date:

```text
2026-05-26
```

Goal:

Prepare the full-TUAB LaBraM foundation-model audit so it can be launched after
the full-TUAB BIOT run finishes, using the same LaBraM protocol previously used
on the TUAB-200 subset.

Code:

```text
code/scripts/run_tuab_labram_intervention.py
code/scripts/launch_tuab_full_labram_intervention.sh
```

Remote paths:

```text
data root: /mnt/data/aperiodic_confounds/data/tuab/v3.0.1_random_stratified_200
manifest: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/tuab_v3_0_1_full_edf_files.csv
cache prefix: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/labram_10s_200hz_cache
run root: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/labram_interventions_base_full
checkpoint: /mnt/data/aperiodic_confounds/external/LaBraM/checkpoints/labram-base.pth
```

Settings:

```text
channels: 23 referential TUAB channels
preprocessing: 0.1-75 Hz bandpass, 50 Hz notch, resample to 200 Hz, microvolt units
window length: 10 s
model input shape: 23 x 10 x 200
epochs requested: 30
batch size: 64
intervention batch size: 256
bootstrap: 10000 eval-subject resamples
seed: 20260524
device: cuda
```

Implementation note:

The LaBraM runner was extended with a full-scale `npy` memmap cache mode,
streaming train/validation batches, and chunked raw/sham/aperiodic/flattened
intervention evaluation. This mirrors the full-TUAB BIOT memory strategy while
preserving the TUAB-200 LaBraM preprocessing, checkpoint initialization,
fine-tuning objective, phase-preserving Fourier intervention, RMS matching and
subject-bootstrap metrics.

Status:

```text
prepared and synced to H200
not launched yet; waiting for full-TUAB BIOT to finish
```

Launch update:

```text
launched on H200 at: 2026-05-26 14:33 IST
process: /mnt/data/.venvs/ml/bin/python3 code/scripts/run_tuab_labram_intervention.py
log: /mnt/data/aperiodic_confounds/logs/tuab_full_labram_20260526_143356.log
status at 2026-05-26 14:34 IST: cache planning stage, 2900/2993 EDF files planned
planned windows at file 2900: 396674
```

Completed result:

```text
completed at: 2026-05-26 15:37 IST
local run root: results/tuab_full_v3_0_1/labram_interventions_base_full
remote run root: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/labram_interventions_base_full
usable EDF files: 2990/2993
skipped EDF files: 3, all missing EEG T1-REF and EEG T2-REF
train windows: 372138
eval windows: 36945
train subjects: 2075
eval subjects: 253
best epoch by validation loss: 1
early stopping epoch: 7
```

Subject-stratified eval-subject bootstrap, balanced accuracy:

| Test input | BA | 95% CI | Drop vs raw | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| raw EEG | 0.780 | [0.741, 0.818] | - | - |
| phase sham | 0.780 | [0.740, 0.817] | 0.000 | [-0.001, 0.002] |
| aperiodic-shaped | 0.718 | [0.684, 0.750] | 0.062 | [0.040, 0.084] |
| flattened | 0.710 | [0.670, 0.750] | 0.070 | [0.044, 0.095] |

Interpretation:

The full-TUAB LaBraM result confirms the foundation-model pattern at full scale
but with weaker aperiodic dependence than BIOT. Raw LaBraM reaches strong
balanced accuracy, the sham control is neutral, and flattening produces a
positive drop with the confidence interval above zero. The smaller drop and
higher flattened performance suggest that LaBraM preserves more usable
non-aperiodic information than BIOT on TUAB, which supports an
architecture-specific rather than blanket foundation-model interpretation.

## TUAB Full-Dataset EEGPT Foundation Model Launch

Date:

```text
2026-05-26
```

Goal:

Scale the TUAB-200 EEGPT foundation-model audit to the full TUAB dataset, using
the same Braindecode EEGPT implementation and pretrained checkpoint as before.

Code:

```text
code/scripts/run_tuab_eegpt_intervention.py
code/scripts/launch_tuab_full_eegpt_intervention.sh
```

Remote paths:

```text
cache prefix: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/labram_10s_200hz_cache
run root: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/eegpt_interventions_braindecode_full
log: /mnt/data/aperiodic_confounds/logs/tuab_full_eegpt_20260526_163237.log
checkpoint: /mnt/data/aperiodic_confounds/external/EEGPT/checkpoint/braindecode_eegpt_pretrained_pytorch_model.bin
```

Settings:

```text
cache: reused full-TUAB LaBraM-style 23-channel 10s/200Hz memmap cache
preprocessing: 0.1-75 Hz bandpass, 50 Hz notch, resample to 200 Hz, microvolt units
epochs requested: 30
batch size: 64
intervention batch size: 256
bootstrap: 10000 eval-subject resamples
seed: 20260525
device: cuda
```

Implementation note:

The EEGPT runner was extended to read the full-TUAB LaBraM-style cache in
memmap mode, stream train/validation batches, and evaluate raw/sham/aperiodic
and flattened interventions in chunks. This preserves the TUAB-200 EEGPT
protocol while avoiding full-dataset array duplication in host RAM.

Initial status:

```text
launched on H200 at: 2026-05-26 16:32 IST
process: /mnt/data/.venvs/ml/bin/python3 code/scripts/run_tuab_eegpt_intervention.py
status at 2026-05-26 16:33 IST: running, GPU memory approximately 25.4 GiB, GPU utilization 100%
```

Completed result:

```text
completed at: 2026-05-26 23:11 IST
local run root: results/tuab_full_v3_0_1/eegpt_interventions_braindecode_full
remote run root: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/eegpt_interventions_braindecode_full
train windows: 372138
eval windows: 36945
train subjects: 2075
eval subjects: 253
best epoch by validation loss: 6
early stopping epoch: 12
```

Subject-stratified eval-subject bootstrap, balanced accuracy:

| Test input | BA | 95% CI | Drop vs raw | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| raw EEG | 0.796 | [0.757, 0.833] | - | - |
| phase sham | 0.795 | [0.757, 0.832] | 0.001 | [-0.001, 0.002] |
| aperiodic-shaped | 0.665 | [0.635, 0.695] | 0.131 | [0.102, 0.160] |
| flattened | 0.730 | [0.684, 0.774] | 0.067 | [0.034, 0.099] |

Interpretation:

The full-TUAB EEGPT result is stronger than the earlier TUAB-200 EEGPT result.
Raw EEGPT reaches strong balanced accuracy, the sham control remains neutral,
and the flattened drop is now positive with the confidence interval above zero.
The aperiodic-shaped condition causes a larger drop than flattening, suggesting
that EEGPT does not simply depend on a smooth aperiodic envelope; it likely uses
a mixture of aperiodic slope, residual oscillatory/waveform structure and
spatial-temporal information. Together with BIOT and LaBraM, the full-TUAB
foundation-model panel now supports a consistent but architecture-dependent
aperiodic reliance story.

## TUAB Full-Dataset CBraMod Foundation Model Preparation

Date:

```text
2026-05-26
```

Goal:

Prepare the full-TUAB CBraMod foundation-model audit so it can be launched
after the full-TUAB EEGPT run finishes, using the same CBraMod protocol
previously used on the TUAB-200 subset.

Code:

```text
code/scripts/run_tuab_cbramod_intervention.py
code/scripts/launch_tuab_full_cbramod_intervention.sh
```

Remote paths:

```text
cache prefix: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/labram_10s_200hz_cache
run root: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/cbramod_interventions_braindecode_full
checkpoint: /home/vinay/.cache/huggingface/hub/models--braindecode--cbramod-pretrained/snapshots/584cdc415913739a05d84bf0c1cb3db397764507/model.safetensors
```

Settings:

```text
cache: reused full-TUAB LaBraM-style 23-channel 10s/200Hz memmap cache
CBraMod input: derived 16-channel longitudinal bipolar TUAB montage
input scaling: divide by 100.0, matching the official CBraMod TUAB loader
epochs requested: 30
batch size: 64
intervention batch size: 256
bootstrap: 10000 eval-subject resamples
seed: 20260525
device: cuda
```

Implementation note:

The CBraMod runner was extended to read the full-TUAB LaBraM-style cache in
memmap mode and derive the official 16-channel bipolar CBraMod input
batch-by-batch. Training, validation and raw evaluation stream from the
referential cache without materializing a full bipolar copy. The
raw/sham/aperiodic/flattened interventions are computed on chunked CBraMod
bipolar inputs, preserving the TUAB-200 CBraMod protocol while avoiding
full-dataset array duplication.

Status:

```text
prepared and synced to H200
not launched yet; waiting for full-TUAB EEGPT to finish
```

Launch update:

```text
launched on H200 at: 2026-05-27 09:11 IST
process: /mnt/data/.venvs/ml/bin/python3 code/scripts/run_tuab_cbramod_intervention.py
log: /mnt/data/aperiodic_confounds/logs/tuab_full_cbramod_20260527_091124.log
status at 2026-05-27 09:21 IST: fine-tuning, epoch 2 complete
GPU memory: approximately 4.5 GiB
GPU utilization: approximately 85%
```

Initial training:

```text
epoch 0: train_loss=0.35052, val_loss=0.53920, val_bacc=0.8009
epoch 1: train_loss=0.12249, val_loss=0.95905, val_bacc=0.7953
epoch 2: train_loss=0.06058, val_loss=1.51223, val_bacc=0.7871
```

Completed result:

```text
completed at: 2026-05-27 09:33 IST
local run root: results/tuab_full_v3_0_1/cbramod_interventions_braindecode_full
remote run root: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/cbramod_interventions_braindecode_full
train windows: 372138
eval windows: 36945
eval subjects: 253
best epoch by validation loss: 0
early stopping epoch: 6
```

Subject-stratified eval-subject bootstrap, balanced accuracy:

| Test input | BA | 95% CI | Drop vs raw | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| raw EEG | 0.782 | [0.745, 0.818] | - | - |
| phase sham | 0.783 | [0.745, 0.818] | -0.001 | [-0.002, 0.000] |
| aperiodic-shaped | 0.621 | [0.593, 0.649] | 0.161 | [0.134, 0.189] |
| flattened | 0.700 | [0.660, 0.739] | 0.083 | [0.056, 0.110] |

Interpretation:

The full-TUAB CBraMod result is stronger than the TUAB-200 subset result. Raw
performance is high, the sham control is neutral, and both aperiodic-shaped and
flattened interventions show positive drops with confidence intervals above
zero. CBraMod retains substantial above-chance flattened performance, but it is
nevertheless clearly sensitive to removing aperiodic spectral structure at full
TUAB scale.

## TUAB Full-Dataset REVE-base Foundation Model Launch

Date:

```text
2026-05-27
```

Goal:

Scale the TUAB-200 REVE-base foundation-model audit to the full TUAB dataset,
using the same Hugging Face REVE encoder and REVE position-bank protocol as
before.

Code:

```text
code/scripts/run_tuab_reve_intervention.py
code/scripts/launch_tuab_full_reve_intervention.sh
```

Remote paths:

```text
cache prefix: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/labram_10s_200hz_cache
run root: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/reve_base_interventions_full
log: /mnt/data/aperiodic_confounds/logs/tuab_full_reve_20260527_102024.log
pretrained repo: brain-bzh/reve-base
positions repo: brain-bzh/reve-positions
```

Settings:

```text
cache: reused full-TUAB LaBraM-style 23-channel 10s/200Hz memmap cache
REVE input: selected 21 TUAB referential channels recognized by the REVE position bank
normalization: per-window/channel z-score, clipped to +/-15
epochs requested: 30
batch size: 64
intervention batch size: 256
bootstrap: 10000 eval-subject resamples
seed: 20260525
device: cuda
selection metric: validation balanced accuracy
```

Implementation note:

The REVE runner was extended to read the full-TUAB LaBraM-style cache in memmap
mode and derive REVE's 21-channel normalized input batch-by-batch. Training,
validation and raw evaluation stream from the referential cache without
materializing a full REVE-format copy. The raw/sham/aperiodic/flattened
interventions are computed on chunked REVE-format inputs.

Initial status:

```text
launched on H200 at: 2026-05-27 10:20 IST
process: /mnt/data/.venvs/ml/bin/python3 code/scripts/run_tuab_reve_intervention.py
status at 2026-05-27 10:20 IST: REVE weights loaded, GPU memory approximately 15.6 GiB, GPU utilization 99%
```

Completion update:

```text
completed at: 2026-05-27 14:08 IST
local run root: results/tuab_full_v3_0_1/reve_base_interventions_full
remote run root: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/reve_base_interventions_full
train windows: 372138
eval windows: 36945
eval subjects: 253
best epoch by validation balanced accuracy: 6
best validation balanced accuracy: 0.8413
early stopping epoch: 12
```

Training trajectory:

```text
epoch 0: train_loss=0.15121, val_loss=0.70034, val_bacc=0.8212
epoch 1: train_loss=0.03864, val_loss=0.71758, val_bacc=0.8230
epoch 2: train_loss=0.02677, val_loss=0.90604, val_bacc=0.8352
epoch 3: train_loss=0.02157, val_loss=0.78456, val_bacc=0.8374
epoch 4: train_loss=0.01856, val_loss=0.82510, val_bacc=0.8305
epoch 5: train_loss=0.01564, val_loss=1.05936, val_bacc=0.8384
epoch 6: train_loss=0.01458, val_loss=0.81969, val_bacc=0.8413
epoch 7: train_loss=0.01358, val_loss=0.83758, val_bacc=0.8250
epoch 8: train_loss=0.01220, val_loss=0.88434, val_bacc=0.8344
epoch 9: train_loss=0.01189, val_loss=0.92131, val_bacc=0.8295
epoch 10: train_loss=0.01090, val_loss=0.97885, val_bacc=0.8336
epoch 11: train_loss=0.01052, val_loss=0.88435, val_bacc=0.8337
epoch 12: train_loss=0.01017, val_loss=0.87702, val_bacc=0.8284
```

Subject-stratified eval-subject bootstrap, balanced accuracy:

| Test input | BA | 95% CI | Drop vs raw | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| raw EEG | 0.790 | [0.751, 0.827] | - | - |
| phase sham | 0.790 | [0.750, 0.826] | 0.000 | [0.000, 0.000] |
| aperiodic-shaped | 0.687 | [0.652, 0.719] | 0.103 | [0.075, 0.131] |
| flattened | 0.734 | [0.691, 0.775] | 0.056 | [0.025, 0.087] |

Output files pulled locally:

```text
results/tuab_full_v3_0_1/reve_base_interventions_full/tuab_reve_intervention_eval_metrics.csv
results/tuab_full_v3_0_1/reve_base_interventions_full/tuab_reve_intervention_metadata.json
results/tuab_full_v3_0_1/reve_base_interventions_full/tuab_reve_intervention_predictions.csv
results/tuab_full_v3_0_1/reve_base_interventions_full/tuab_reve_intervention_subject_bootstrap.csv
results/tuab_full_v3_0_1/reve_base_interventions_full/tuab_reve_intervention_subject_bootstrap.md
results/tuab_full_v3_0_1/reve_base_interventions_full/tuab_reve_training_log.csv
```

Interpretation:

REVE-base is a positive full-TUAB foundation-model result. Raw performance is
high, the sham control is exactly neutral, aperiodic-shaped input causes a
large drop, and flattened input causes a smaller but clearly positive drop with
the confidence interval above zero. This places REVE closer to LaBraM/EEGPT in
flattening sensitivity than BIOT, while still confirming that the representation
uses aperiodic spectral structure.

## TUAB Full-Dataset BENDR Foundation Model Preparation

Date:

```text
2026-05-27
```

Goal:

Prepare the full-TUAB BENDR foundation-model audit so it can be launched after
the full-TUAB REVE-base run finishes, using the same reportable BENDR protocol
previously used on the TUAB-200 subset.

Code:

```text
code/scripts/run_tuab_bendr_intervention.py
code/scripts/launch_tuab_full_bendr_intervention.sh
```

Remote paths:

```text
cache prefix: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/labram_10s_200hz_cache
run root: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/bendr_interventions_braindecode_encoder_only_full
checkpoint: /home/vinay/.cache/huggingface/hub/models--braindecode--braindecode-bendr/snapshots/191f221cd56de8203899ea9a8d0f43238724f8b6/model.safetensors
```

Settings:

```text
cache: reused full-TUAB LaBraM-style 23-channel 10s/200Hz memmap cache
BENDR mode: encoder-only, matching the TUAB-200 reportable run
input clipping: +/-500 uV
input scale: 1e-6, converting clipped uV to V
epochs requested: 30
batch size: 128
intervention batch size: 256
bootstrap: 10000 eval-subject resamples
seed: 20260525
device: cuda
```

Implementation note:

The BENDR runner was extended to read the full-TUAB LaBraM-style cache in
memmap mode and apply the same BENDR clip/scale transform batch-by-batch.
Training, validation and raw evaluation stream from the referential cache
without materializing a full scaled copy. The raw/sham/aperiodic/flattened
interventions are computed on chunked BENDR-format inputs. The launcher uses
`--encoder-only`, because the full contextualizer was unstable in the TUAB-200
audit and the encoder-only run was the recorded reportable BENDR protocol.

Status:

```text
prepared and synced to H200
not launched yet; waiting for full-TUAB REVE-base to finish
```

Status update:

```text
REVE-base finished on 2026-05-27 at 14:08 IST.
BENDR remains prepared but was not launched immediately because the user
requested launching full-TUAB EEGMamba first.
```

Readiness check:

```text
checked at: 2026-05-27 14:33 IST
local runner exists: code/scripts/run_tuab_bendr_intervention.py
local full launcher exists: code/scripts/launch_tuab_full_bendr_intervention.sh
remote runner exists: /mnt/data/aperiodic_confounds/code/scripts/run_tuab_bendr_intervention.py
remote full launcher exists: /mnt/data/aperiodic_confounds/code/scripts/launch_tuab_full_bendr_intervention.sh
remote syntax checks: pass
BENDR checkpoint: present
Braindecode InterpolatedBENDR import: pass
full-TUAB memmap cache: present
full-TUAB BENDR output directory/files: not present
conclusion: full-TUAB BENDR code exists and is prepared, but the full-TUAB BENDR run has not been launched.
```

Launch update:

```text
launched on H200 at: 2026-05-27 15:22 IST
process: /mnt/data/.venvs/ml/bin/python3 code/scripts/run_tuab_bendr_intervention.py
pid: 2982363
log: /mnt/data/aperiodic_confounds/logs/tuab_full_bendr_20260527_152220.log
run root: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/bendr_interventions_braindecode_encoder_only_full
status at 2026-05-27 15:23 IST: fine-tuning, epoch 0 complete
GPU memory: approximately 3.5 GiB
GPU utilization: approximately 37%
```

Initial training:

```text
epoch 0: train_loss=0.68457, val_loss=0.68466, val_bacc=0.4996
```

Status check:

```text
checked at: 2026-05-27 15:28 IST
pid: 2982363
process: alive
GPU memory: approximately 3.5 GiB
GPU utilization: approximately 11%
output files: none yet
stage: fine-tuning
```

Latest training:

```text
epoch 0: train_loss=0.68457, val_loss=0.68466, val_bacc=0.4996
epoch 1: train_loss=0.63901, val_loss=0.57608, val_bacc=0.6910
epoch 2: train_loss=0.53380, val_loss=0.52817, val_bacc=0.7279
epoch 3: train_loss=0.51328, val_loss=0.56974, val_bacc=0.7302
epoch 4: train_loss=0.50263, val_loss=0.52867, val_bacc=0.7338
epoch 5: train_loss=0.49357, val_loss=0.53543, val_bacc=0.7377
epoch 6: train_loss=0.48736, val_loss=0.57682, val_bacc=0.7376
epoch 7: train_loss=0.48153, val_loss=0.53983, val_bacc=0.7388
best checkpoint so far: epoch 7 by validation balanced accuracy
```

## TUAB Full-Dataset EEGMamba Foundation Model Launch

Date:

```text
2026-05-27
```

Goal:

Scale the TUAB-200 EEGMamba audit to the full TUAB dataset using the prepared
full-scale memmap runner and the isolated SSM-capable EEGMamba environment.

Code:

```text
code/scripts/run_tuab_eegmamba_intervention.py
code/scripts/launch_tuab_full_eegmamba_intervention.sh
```

Remote paths:

```text
cache prefix: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/labram_10s_200hz_cache
run root: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/eegmamba_interventions_official_full
log: /mnt/data/aperiodic_confounds/logs/tuab_full_eegmamba_20260527_142325.log
official repo: /mnt/data/aperiodic_confounds/external/EEGMamba
checkpoint: /home/vinay/.cache/huggingface/hub/models--weighting666--EEGMamba/snapshots/0b060d87acd6f23bf1d0b852bf1726064f335f97/pretrained_EEGMamba.pth
```

Environment:

```text
python: /mnt/data/aperiodic_confounds/.venvs/eegmamba/bin/python
torch: 2.5.1+cu124
mamba-ssm: 2.2.6.post3
causal-conv1d: 1.6.0
transformers: 4.44.2
```

Settings:

```text
cache: reused full-TUAB LaBraM-style 23-channel 10s/200Hz memmap cache
observed cache shape: 409083 x 23 x 2000
EEGMamba input: 16 longitudinal bipolar channels derived from the 23-channel cache
input normalization: divisor
input divisor: 100.0
epochs requested: 30
batch size: 64
intervention batch size: 256
bootstrap: 10000 eval-subject resamples
seed: 20260525
device: cuda
selection metric: validation balanced accuracy
```

Implementation note:

The EEGMamba runner was extended to read the full-TUAB LaBraM-style cache in
memmap mode and derive EEGMamba's 16-channel bipolar input batch-by-batch.
Training, validation and raw evaluation stream from the referential cache
without materializing a full bipolar EEGMamba-format copy. The
raw/sham/aperiodic/flattened interventions are computed on chunked EEGMamba
bipolar inputs.

Launch status:

```text
launched on H200 at: 2026-05-27 14:23 IST
process: /mnt/data/aperiodic_confounds/.venvs/eegmamba/bin/python code/scripts/run_tuab_eegmamba_intervention.py
pid: 2949652
status at 2026-05-27 14:28 IST: fine-tuning, epoch 0 complete
status at 2026-05-27 14:30 IST: fine-tuning, epoch 1 complete
GPU memory: approximately 3.6 GiB
GPU utilization: approximately 67%
```

Initial training:

```text
epoch 0: train_loss=0.42310, val_loss=0.44917, val_bacc=0.7873
epoch 1: train_loss=0.30124, val_loss=0.50869, val_bacc=0.7920
```

Status check:

```text
checked at: 2026-05-27 14:37 IST
pid: 2949652
process: alive
GPU memory: approximately 3.6 GiB
GPU utilization: approximately 68%
output files: none yet
stage: fine-tuning
```

Latest training:

```text
epoch 0: train_loss=0.42310, val_loss=0.44917, val_bacc=0.7873
epoch 1: train_loss=0.30124, val_loss=0.50869, val_bacc=0.7920
epoch 2: train_loss=0.16015, val_loss=0.70029, val_bacc=0.7922
epoch 3: train_loss=0.06453, val_loss=0.99316, val_bacc=0.8033
best checkpoint so far: epoch 3 by validation balanced accuracy
```

Status check:

```text
checked at: 2026-05-27 14:43 IST
pid: 2949652
process: alive
GPU memory: approximately 3.6 GiB
GPU utilization: approximately 69%
output files: none yet
stage: fine-tuning
```

Latest training:

```text
epoch 0: train_loss=0.42310, val_loss=0.44917, val_bacc=0.7873
epoch 1: train_loss=0.30124, val_loss=0.50869, val_bacc=0.7920
epoch 2: train_loss=0.16015, val_loss=0.70029, val_bacc=0.7922
epoch 3: train_loss=0.06453, val_loss=0.99316, val_bacc=0.8033
epoch 4: train_loss=0.02940, val_loss=1.35552, val_bacc=0.8056
epoch 5: train_loss=0.01868, val_loss=1.68599, val_bacc=0.8097
best checkpoint so far: epoch 5 by validation balanced accuracy
```

Status check:

```text
checked at: 2026-05-27 14:51 IST
pid: 2949652
process: alive
GPU memory: approximately 3.6 GiB
GPU utilization: approximately 71%
output files: none yet
stage: fine-tuning
```

Latest training:

```text
epoch 0: train_loss=0.42310, val_loss=0.44917, val_bacc=0.7873
epoch 1: train_loss=0.30124, val_loss=0.50869, val_bacc=0.7920
epoch 2: train_loss=0.16015, val_loss=0.70029, val_bacc=0.7922
epoch 3: train_loss=0.06453, val_loss=0.99316, val_bacc=0.8033
epoch 4: train_loss=0.02940, val_loss=1.35552, val_bacc=0.8056
epoch 5: train_loss=0.01868, val_loss=1.68599, val_bacc=0.8097
epoch 6: train_loss=0.01550, val_loss=1.81074, val_bacc=0.8012
epoch 7: train_loss=0.01312, val_loss=2.03013, val_bacc=0.8047
epoch 8: train_loss=0.01194, val_loss=1.86933, val_bacc=0.8140
best checkpoint so far: epoch 8 by validation balanced accuracy
```

Status check:

```text
checked at: 2026-05-27 15:04 IST
pid: 2949652
process: alive
GPU memory: approximately 3.6 GiB
GPU utilization: approximately 61%
output files: none yet
stage: fine-tuning
```

Latest training:

```text
epoch 0: train_loss=0.42310, val_loss=0.44917, val_bacc=0.7873
epoch 1: train_loss=0.30124, val_loss=0.50869, val_bacc=0.7920
epoch 2: train_loss=0.16015, val_loss=0.70029, val_bacc=0.7922
epoch 3: train_loss=0.06453, val_loss=0.99316, val_bacc=0.8033
epoch 4: train_loss=0.02940, val_loss=1.35552, val_bacc=0.8056
epoch 5: train_loss=0.01868, val_loss=1.68599, val_bacc=0.8097
epoch 6: train_loss=0.01550, val_loss=1.81074, val_bacc=0.8012
epoch 7: train_loss=0.01312, val_loss=2.03013, val_bacc=0.8047
epoch 8: train_loss=0.01194, val_loss=1.86933, val_bacc=0.8140
epoch 9: train_loss=0.01124, val_loss=2.06075, val_bacc=0.8084
epoch 10: train_loss=0.01022, val_loss=2.37487, val_bacc=0.7973
epoch 11: train_loss=0.00909, val_loss=2.18129, val_bacc=0.8136
epoch 12: train_loss=0.00873, val_loss=2.31018, val_bacc=0.8020
epoch 13: train_loss=0.00812, val_loss=2.37122, val_bacc=0.8010
best checkpoint so far: epoch 8 by validation balanced accuracy
early-stop note: patience is 6; if epoch 14 does not improve over epoch 8, training should stop after epoch 14.
```

Completion update:

```text
checked/completed at: 2026-05-27 15:19 IST
remote completion time from log/files: 2026-05-27 15:08 IST
local run root: results/tuab_full_v3_0_1/eegmamba_interventions_official_full
remote run root: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/eegmamba_interventions_official_full
train windows: 372138
eval windows: 36945
eval subjects: 253
best epoch by validation balanced accuracy: 8
best validation balanced accuracy: 0.8140
early stopping epoch: 14
GPU after completion: idle
```

Training trajectory:

```text
epoch 0: train_loss=0.42310, val_loss=0.44917, val_bacc=0.7873
epoch 1: train_loss=0.30124, val_loss=0.50869, val_bacc=0.7920
epoch 2: train_loss=0.16015, val_loss=0.70029, val_bacc=0.7922
epoch 3: train_loss=0.06453, val_loss=0.99316, val_bacc=0.8033
epoch 4: train_loss=0.02940, val_loss=1.35552, val_bacc=0.8056
epoch 5: train_loss=0.01868, val_loss=1.68599, val_bacc=0.8097
epoch 6: train_loss=0.01550, val_loss=1.81074, val_bacc=0.8012
epoch 7: train_loss=0.01312, val_loss=2.03013, val_bacc=0.8047
epoch 8: train_loss=0.01194, val_loss=1.86933, val_bacc=0.8140
epoch 9: train_loss=0.01124, val_loss=2.06075, val_bacc=0.8084
epoch 10: train_loss=0.01022, val_loss=2.37487, val_bacc=0.7973
epoch 11: train_loss=0.00909, val_loss=2.18129, val_bacc=0.8136
epoch 12: train_loss=0.00873, val_loss=2.31018, val_bacc=0.8020
epoch 13: train_loss=0.00812, val_loss=2.37122, val_bacc=0.8010
epoch 14: train_loss=0.00785, val_loss=2.36979, val_bacc=0.8031
```

Subject-stratified eval-subject bootstrap, balanced accuracy:

| Test input | BA | 95% CI | Drop vs raw | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| raw EEG | 0.773 | [0.736, 0.809] | - | - |
| phase sham | 0.773 | [0.736, 0.809] | 0.000 | [-0.001, 0.001] |
| aperiodic-shaped | 0.678 | [0.643, 0.713] | 0.095 | [0.068, 0.122] |
| flattened | 0.708 | [0.668, 0.746] | 0.066 | [0.040, 0.091] |

Output files pulled locally:

```text
results/tuab_full_v3_0_1/eegmamba_interventions_official_full/tuab_eegmamba_intervention_eval_metrics.csv
results/tuab_full_v3_0_1/eegmamba_interventions_official_full/tuab_eegmamba_intervention_metadata.json
results/tuab_full_v3_0_1/eegmamba_interventions_official_full/tuab_eegmamba_intervention_predictions.csv
results/tuab_full_v3_0_1/eegmamba_interventions_official_full/tuab_eegmamba_intervention_subject_bootstrap.csv
results/tuab_full_v3_0_1/eegmamba_interventions_official_full/tuab_eegmamba_intervention_subject_bootstrap.md
results/tuab_full_v3_0_1/eegmamba_interventions_official_full/tuab_eegmamba_training_log.csv
```

Interpretation:

Full-TUAB EEGMamba is no longer only a cautionary/unstable subset result. At
full scale, raw performance is competitive, sham is neutral, and both
aperiodic-shaped and flattened interventions show positive drops with
confidence intervals above zero. The flattening drop is smaller than BIOT and
closer to REVE/EEGPT/LaBraM-scale sensitivity, suggesting EEGMamba retains
substantial non-aperiodic information while still using aperiodic spectral
structure.

## 2026-05-27 - Full TUAB BENDR Encoder-Only Intervention Run

Status check:

```text
checked at: 2026-05-27 16:00 IST
remote completion time from log/files: 2026-05-27 15:30 IST
remote log: /mnt/data/aperiodic_confounds/logs/tuab_full_bendr_20260527_152220.log
remote run root: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/bendr_interventions_braindecode_encoder_only_full
local run root: results/tuab_full_v3_0_1/bendr_interventions_braindecode_encoder_only_full
GPU after completion: idle
```

Training:

```text
epoch 0: train_loss=0.68457, val_loss=0.68466, val_bacc=0.4996
epoch 1: train_loss=0.63901, val_loss=0.57608, val_bacc=0.6910
epoch 2: train_loss=0.53380, val_loss=0.52817, val_bacc=0.7279
epoch 3: train_loss=0.51328, val_loss=0.56974, val_bacc=0.7302
epoch 4: train_loss=0.50263, val_loss=0.52867, val_bacc=0.7338
epoch 5: train_loss=0.49357, val_loss=0.53543, val_bacc=0.7377
epoch 6: train_loss=0.48736, val_loss=0.57682, val_bacc=0.7376
epoch 7: train_loss=0.48153, val_loss=0.53983, val_bacc=0.7388
epoch 8: train_loss=0.47343, val_loss=0.53381, val_bacc=0.7440
early stopping at epoch 8
```

Subject-stratified eval-subject bootstrap, balanced accuracy:

| Test input | BA | 95% CI | Drop vs raw | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| raw EEG | 0.744 | [0.702, 0.786] | - | - |
| phase sham | 0.500 | [0.500, 0.500] | 0.244 | [0.202, 0.286] |
| aperiodic-shaped | 0.500 | [0.500, 0.500] | 0.244 | [0.201, 0.287] |
| flattened | 0.500 | [0.500, 0.500] | 0.244 | [0.202, 0.286] |

Output files pulled locally:

```text
results/tuab_full_v3_0_1/bendr_interventions_braindecode_encoder_only_full/tuab_bendr_intervention_eval_metrics.csv
results/tuab_full_v3_0_1/bendr_interventions_braindecode_encoder_only_full/tuab_bendr_intervention_metadata.json
results/tuab_full_v3_0_1/bendr_interventions_braindecode_encoder_only_full/tuab_bendr_intervention_predictions.csv
results/tuab_full_v3_0_1/bendr_interventions_braindecode_encoder_only_full/tuab_bendr_intervention_subject_bootstrap.csv
results/tuab_full_v3_0_1/bendr_interventions_braindecode_encoder_only_full/tuab_bendr_intervention_subject_bootstrap.md
results/tuab_full_v3_0_1/bendr_interventions_braindecode_encoder_only_full/tuab_bendr_training_log.csv
```

Interpretation:

The full-TUAB BENDR encoder-only run completed successfully and reached
reasonable raw balanced accuracy, but every phase-manipulated condition,
including the phase-sham control, collapsed to chance. This is not a clean
aperiodic-specific effect. It should be treated as evidence that this
Braindecode BENDR pathway is extremely sensitive to the phase-manipulation
pipeline or to distribution shift in the transformed windows; the sham failure
is the key caveat for journal reporting.

## 2026-05-27 - Full TUAB Foundation Model Aperiodic Audit Completion Check

Status check:

```text
checked at: 2026-05-27 16:02 IST
scope: prepared full-TUAB foundation-model aperiodic intervention runs
result root: results/tuab_full_v3_0_1
```

Confirmed full-TUAB result folders with local metrics, metadata, predictions,
training logs, and subject-bootstrap reports:

```text
BIOT:     results/tuab_full_v3_0_1/biot_interventions_prest_full
LaBraM:   results/tuab_full_v3_0_1/labram_interventions_base_full
EEGPT:    results/tuab_full_v3_0_1/eegpt_interventions_braindecode_full
CBraMod:  results/tuab_full_v3_0_1/cbramod_interventions_braindecode_full
REVE:     results/tuab_full_v3_0_1/reve_base_interventions_full
EEGMamba: results/tuab_full_v3_0_1/eegmamba_interventions_official_full
BENDR:    results/tuab_full_v3_0_1/bendr_interventions_braindecode_encoder_only_full
```

Conclusion:

The prepared full-TUAB foundation-model aperiodic audit is complete. BIOT,
LaBraM, EEGPT, CBraMod, REVE-base, and EEGMamba are clean reportable
intervention rows: raw performance is meaningful, phase-sham controls are
neutral, and aperiodic/flattened drops are positive with subject-bootstrap
confidence intervals above zero. BENDR is complete but should be flagged
separately because its phase-sham control collapsed to chance, making it a
cautionary transfer/distribution-shift result rather than a clean
aperiodic-specific finding.

## 2026-05-27 - Full TUAB Age/Sex-Matched Control Implementation

Goal:

Replicate the TUAB-200 age/sex-matched control at full-TUAB scale, using the
same logic and reporting style:

```text
extract EDF header age/sex metadata
build same-sex abnormal-normal pairs within official train/eval split
use a +/-5 year age caliper
run PSD ridge intervention on the matched subject manifest
run raw neural EEGNet/ShallowFBCSPNet/Deep4Net intervention on the same manifest
```

Code added:

```text
code/scripts/launch_tuab_full_age_matched_psd.sh
code/scripts/launch_tuab_full_age_matched_multiseed_neural.sh
```

Code updated:

```text
code/scripts/make_tuab_age_matched_subset.py
```

Update note:

`make_tuab_age_matched_subset.py` now writes the skipped-subject CSV with a
header even if no subjects are skipped. This keeps the matcher robust for the
larger full-TUAB metadata table.

Remote paths:

```text
match root: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/age_matched
PSD run root: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/age_matched/psd_interventions_specparam_tuab_full_age_sex_matched_caliper5
neural run root: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/age_matched/multiseed_neural
PSD log: /mnt/data/aperiodic_confounds/logs/tuab_full_age_matched_psd_20260527_173151.log
neural log: /mnt/data/aperiodic_confounds/logs/tuab_full_age_matched_neural_20260527_173806.log
```

Matched full-TUAB subset:

```text
caliper: +/-5 years
matching: same sex, within official split
train pairs: 834
eval pairs: 87
total pairs: 921
total subject rows: 1842
eval subjects: 174
train mean absolute age difference: 1.97 years
eval mean absolute age difference: 1.32 years
maximum absolute age difference: 5.00 years
```

Completed PSD result:

```text
completed at: 2026-05-27 17:37 IST
local run root: results/tuab_full_v3_0_1/age_matched/psd_interventions_specparam_tuab_full_age_sex_matched_caliper5
remote run root: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/age_matched/psd_interventions_specparam_tuab_full_age_sex_matched_caliper5
```

Subject-stratified eval-subject bootstrap, balanced accuracy:

| Train input | Test input | BA | 95% CI |
| --- | --- | ---: | ---: |
| full PSD | full PSD | 0.715 | [0.663, 0.766] |
| full PSD | aperiodic | 0.567 | [0.525, 0.609] |
| full PSD | flattened | 0.568 | [0.544, 0.593] |
| aperiodic | full PSD | 0.684 | [0.629, 0.737] |
| aperiodic | aperiodic | 0.712 | [0.659, 0.762] |
| aperiodic | flattened | 0.500 | [0.500, 0.500] |
| flattened | full PSD | 0.583 | [0.551, 0.618] |
| flattened | aperiodic | 0.505 | [0.498, 0.513] |
| flattened | flattened | 0.699 | [0.650, 0.746] |

Key drops when trained on full PSD:

| Test intervention | Drop | 95% CI |
| --- | ---: | ---: |
| aperiodic-only | 0.148 | [0.099, 0.197] |
| flattened PSD | 0.148 | [0.101, 0.192] |

Interpretation:

The full-TUAB age/sex-matched PSD control preserves the main TUAB conclusion.
After strict same-sex, +/-5-year matching, aperiodic-only spectra remain
predictive when models are trained and tested on aperiodic spectra, and
full-trained models lose performance under both aperiodic-only and flattened
test-time interventions. This strengthens the claim that age is an important
confound but does not explain away the TUAB aperiodic result.

Completed neural multiseed result:

```text
launched at: 2026-05-27 17:38 IST
completed at: 2026-05-27 19:30 IST
launcher: code/scripts/launch_tuab_full_age_matched_multiseed_neural.sh
run root: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/age_matched/multiseed_neural
log: /mnt/data/aperiodic_confounds/logs/tuab_full_age_matched_neural_20260527_173806.log
models: eegnet, shallow_fbcsp, deep4
seeds: 42, 43, 44
n_bootstrap: 10000
matched neural train epochs: 149471
matched neural eval epochs: 13013
matched neural train subjects: 1624
matched neural eval subjects: 174
```

Completed stage markers:

```text
eegnet_seed42.done        2026-05-27 18:03 IST
shallow_fbcsp_seed42.done 2026-05-27 18:15 IST
deep4_seed42.done         2026-05-27 18:25 IST
eegnet_seed43.done        2026-05-27 18:46 IST
shallow_fbcsp_seed43.done 2026-05-27 18:58 IST
deep4_seed43.done         2026-05-27 19:03 IST
eegnet_seed44.done        2026-05-27 19:17 IST
shallow_fbcsp_seed44.done 2026-05-27 19:25 IST
deep4_seed44.done         2026-05-27 19:30 IST
```

Final neural outputs:

```text
remote run root: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/age_matched/multiseed_neural
aggregate CSV: /mnt/data/aperiodic_confounds/reports/tables/tuab_full_age_matched_multiseed_neural_subject_bootstrap.csv
aggregate Markdown: /mnt/data/aperiodic_confounds/reports/tables/tuab_full_age_matched_multiseed_neural_subject_bootstrap.md
aggregate CSV size: 13K
aggregate Markdown size: 9.1K
```

Hierarchical seed/subject bootstrap, balanced accuracy:

| Model | Baseline BA | Sham BA | Aperiodic BA | Flattened BA | Drop aperiodic | Drop flattened |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EEGNet | 0.740 [0.688, 0.789] | 0.740 [0.688, 0.789] | 0.587 [0.499, 0.671] | 0.653 [0.582, 0.720] | 0.153 [0.066, 0.236] | 0.087 [0.012, 0.156] |
| ShallowFBCSPNet | 0.758 [0.708, 0.807] | 0.758 [0.708, 0.807] | 0.602 [0.537, 0.665] | 0.676 [0.614, 0.736] | 0.156 [0.088, 0.226] | 0.082 [0.019, 0.145] |
| Deep4Net | 0.764 [0.714, 0.811] | 0.763 [0.714, 0.811] | 0.655 [0.582, 0.723] | 0.550 [0.459, 0.640] | 0.109 [0.035, 0.188] | 0.213 [0.117, 0.302] |

Hierarchical seed/subject bootstrap, macro-F1:

| Model | Baseline macro-F1 | Sham macro-F1 | Aperiodic macro-F1 | Flattened macro-F1 | Drop aperiodic | Drop flattened |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EEGNet | 0.393 [0.369, 0.416] | 0.393 [0.369, 0.416] | 0.319 [0.272, 0.363] | 0.342 [0.306, 0.375] | 0.074 [0.025, 0.122] | 0.051 [0.018, 0.082] |
| ShallowFBCSPNet | 0.403 [0.380, 0.425] | 0.403 [0.380, 0.425] | 0.326 [0.294, 0.357] | 0.354 [0.324, 0.383] | 0.077 [0.044, 0.111] | 0.049 [0.021, 0.079] |
| Deep4Net | 0.401 [0.377, 0.424] | 0.401 [0.377, 0.424] | 0.358 [0.320, 0.392] | 0.285 [0.235, 0.337] | 0.043 [0.003, 0.087] | 0.116 [0.067, 0.163] |

Per-seed terminal values from the log:

| Model/seed | Raw BA | Sham BA | Aperiodic BA | Flattened BA |
| --- | ---: | ---: | ---: | ---: |
| EEGNet seed 42 | 0.752 | 0.752 | 0.572 | 0.664 |
| ShallowFBCSPNet seed 42 | 0.761 | 0.761 | 0.577 | 0.692 |
| Deep4Net seed 42 | 0.769 | 0.769 | 0.596 | 0.640 |
| ShallowFBCSPNet seed 44 | 0.746 | 0.746 | 0.599 | 0.676 |
| Deep4Net seed 44 | 0.757 | 0.757 | 0.708 | 0.507 |

Note: the aggregate CSV/Markdown contains the complete per-model multiseed
statistics for balanced accuracy, macro-F1, and accuracy. The per-seed table
above records representative terminal log values; the aggregate report should be
used for paper tables and figures.

Interpretation:

The full-TUAB age/sex-matched neural control strengthens the central TUAB
claim. Across all three raw neural architectures, the sham intervention is
effectively unchanged, while both aperiodic-only and flattened interventions
produce positive drops with 95% confidence intervals above zero. This means the
aperiodic dependence persists after same-sex, +/-5-year age matching and cannot
be dismissed as only an age/sex shortcut.

Compared with the earlier TUAB-200 age/sex-matched neural control, the full
dataset gives the same qualitative result with a larger matched evaluation set
and tighter evidence. TUAB-200 had 54 matched eval subjects and clear
flattening drops, while the full-TUAB matched run has 174 matched eval subjects
and shows significant aperiodic-only as well as flattened drops for every model.

## 2026-05-27 - Full TUAB Foundation-Model Multiseed Sequential Launch

Goal:

Extend the full-TUAB foundation-model audit from the already-completed
single-seed runs to a three-seed multiseed panel. The existing single-seed
outputs are preserved and two additional seeds are launched for each foundation
model.

Foundation models:

```text
BIOT
LaBraM
EEGPT
CBraMod
REVE-base
EEGMamba
BENDR encoder-only
```

New seeds launched:

```text
20260526
20260527
```

Previously completed seed outputs included in aggregation:

```text
BIOT:     seed 20260524, results/tuab_full_v3_0_1/biot_interventions_prest_full
LaBraM:   seed 20260524, results/tuab_full_v3_0_1/labram_interventions_base_full
EEGPT:    seed 20260525, results/tuab_full_v3_0_1/eegpt_interventions_braindecode_full
CBraMod:  seed 20260525, results/tuab_full_v3_0_1/cbramod_interventions_braindecode_full
REVE:     seed 20260525, results/tuab_full_v3_0_1/reve_base_interventions_full
EEGMamba: seed 20260525, results/tuab_full_v3_0_1/eegmamba_interventions_official_full
BENDR:    seed 20260525, results/tuab_full_v3_0_1/bendr_interventions_braindecode_encoder_only_full
```

Code added:

```text
code/scripts/aggregate_foundation_multiseed_predictions.py
code/scripts/launch_tuab_full_foundation_multiseed_sequential.sh
```

Code updated:

```text
code/scripts/launch_tuab_full_biot_intervention.sh
code/scripts/launch_tuab_full_labram_intervention.sh
code/scripts/launch_tuab_full_eegpt_intervention.sh
code/scripts/launch_tuab_full_cbramod_intervention.sh
code/scripts/launch_tuab_full_reve_intervention.sh
code/scripts/launch_tuab_full_eegmamba_intervention.sh
code/scripts/launch_tuab_full_bendr_intervention.sh
```

Update note:

The existing full-TUAB single-seed FM launchers now allow `RUN_ROOT` override.
This lets the multiseed launcher write seed-specific output directories without
overwriting the original single-seed report folders.

Aggregation note:

The foundation-model runners write prediction CSVs and subject-bootstrap
reports, but not the raw-neural `subject_metrics.csv` format. Therefore
`aggregate_foundation_multiseed_predictions.py` aggregates directly from the
prediction CSVs. It computes baseline, sham, aperiodic, flattened, retention,
and drop estimates for balanced accuracy, macro-F1, and accuracy with a
seed/eval-subject hierarchical bootstrap. Eval-subject resampling is stratified
by label within each sampled seed.

Failure behavior:

The sequential launcher does not stop the overnight run when one seed or one
foundation model fails. It records success/failure in a manifest, preserves
completed seed directories and logs, aggregates whatever seed outputs are
available for the just-finished foundation model, refreshes the combined report,
and then proceeds to the next foundation model.

Remote launch:

```text
launched at: 2026-05-27 23:24 IST
remote PID after detach: 2495095
launch command: bash code/scripts/launch_tuab_full_foundation_multiseed_sequential.sh
master log: /mnt/data/aperiodic_confounds/logs/tuab_full_foundation_multiseed_20260527_232428.log
nohup log: /mnt/data/aperiodic_confounds/logs/tuab_full_foundation_multiseed_nohup_20260527_232428.log
run root: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/foundation_multiseed
manifest: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/foundation_multiseed/foundation_multiseed_manifest.csv
per-job logs: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/foundation_multiseed/job_logs
stage markers: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/foundation_multiseed/stage_markers
```

Sequential order:

```text
biot -> labram -> eegpt -> cbramod -> reve -> eegmamba -> bendr
```

Planned aggregate outputs:

```text
per-FM reports:
/mnt/data/aperiodic_confounds/reports/tables/tuab_full_foundation_multiseed_<fm>_subject_bootstrap.csv
/mnt/data/aperiodic_confounds/reports/tables/tuab_full_foundation_multiseed_<fm>_subject_bootstrap.md

combined reports:
/mnt/data/aperiodic_confounds/reports/tables/tuab_full_foundation_multiseed_subject_bootstrap.csv
/mnt/data/aperiodic_confounds/reports/tables/tuab_full_foundation_multiseed_subject_bootstrap.md
```

Validation before launch:

```text
local bash syntax check: pass
remote bash syntax check: pass
remote Python compile: pass
remote aggregation smoke test on existing BIOT/LaBraM predictions: pass
```

Initial status:

```text
status at: 2026-05-27 23:25 IST
current job: BIOT seed 20260526
GPU: NVIDIA H200, approximately 7.1 GiB used, approximately 50% utilization
job log: /mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/foundation_multiseed/job_logs/biot_seed20260526.log
```

Final status:

```text
status checked: 2026-05-29 08:11 IST
completion time: 2026-05-29 01:30:44 IST
launcher process: finished
GPU after completion: NVIDIA H200 idle, 0 MiB allocated, 0% utilization
manifest result: all 14 launched seed jobs completed successfully with exit_code=0
```

Completed seed-job log:

| Foundation model | Seed | Start time | End time | Exit code |
| --- | --- | --- | --- | ---: |
| BIOT | 20260526 | 2026-05-27 23:24:29 | 2026-05-27 23:37:06 | 0 |
| BIOT | 20260527 | 2026-05-27 23:37:06 | 2026-05-27 23:49:02 | 0 |
| LaBraM | 20260526 | 2026-05-27 23:49:19 | 2026-05-28 01:18:48 | 0 |
| LaBraM | 20260527 | 2026-05-28 01:18:48 | 2026-05-28 01:58:24 | 0 |
| EEGPT | 20260526 | 2026-05-28 01:58:43 | 2026-05-28 08:13:54 | 0 |
| EEGPT | 20260527 | 2026-05-28 08:13:54 | 2026-05-28 16:01:37 | 0 |
| CBraMod | 20260526 | 2026-05-28 16:01:58 | 2026-05-28 16:24:43 | 0 |
| CBraMod | 20260527 | 2026-05-28 16:24:43 | 2026-05-28 16:47:13 | 0 |
| REVE-base | 20260526 | 2026-05-28 16:47:35 | 2026-05-28 19:46:44 | 0 |
| REVE-base | 20260527 | 2026-05-28 19:46:44 | 2026-05-28 23:36:53 | 0 |
| EEGMamba | 20260526 | 2026-05-28 23:37:17 | 2026-05-28 23:59:36 | 0 |
| EEGMamba | 20260527 | 2026-05-28 23:59:36 | 2026-05-29 00:49:59 | 0 |
| BENDR encoder-only | 20260526 | 2026-05-29 00:50:24 | 2026-05-29 01:07:03 | 0 |
| BENDR encoder-only | 20260527 | 2026-05-29 01:07:03 | 2026-05-29 01:30:17 | 0 |

Final aggregate outputs:

```text
combined CSV: /mnt/data/aperiodic_confounds/reports/tables/tuab_full_foundation_multiseed_subject_bootstrap.csv
combined Markdown: /mnt/data/aperiodic_confounds/reports/tables/tuab_full_foundation_multiseed_subject_bootstrap.md
per-FM CSV/Markdown reports:
  /mnt/data/aperiodic_confounds/reports/tables/tuab_full_foundation_multiseed_biot_subject_bootstrap.*
  /mnt/data/aperiodic_confounds/reports/tables/tuab_full_foundation_multiseed_labram_subject_bootstrap.*
  /mnt/data/aperiodic_confounds/reports/tables/tuab_full_foundation_multiseed_eegpt_subject_bootstrap.*
  /mnt/data/aperiodic_confounds/reports/tables/tuab_full_foundation_multiseed_cbramod_subject_bootstrap.*
  /mnt/data/aperiodic_confounds/reports/tables/tuab_full_foundation_multiseed_reve_subject_bootstrap.*
  /mnt/data/aperiodic_confounds/reports/tables/tuab_full_foundation_multiseed_eegmamba_subject_bootstrap.*
  /mnt/data/aperiodic_confounds/reports/tables/tuab_full_foundation_multiseed_bendr_subject_bootstrap.*
```

Final balanced-accuracy aggregate, subject-level hierarchical bootstrap:

| Model | n_seeds | n_subjects | Baseline BA | Sham BA | Aperiodic BA | Flattened BA | Drop aperiodic | Drop flattened |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BIOT | 3 | 253 | 0.802 [0.781, 0.823] | 0.800 [0.779, 0.821] | 0.678 [0.651, 0.704] | 0.692 [0.666, 0.719] | 0.124 [0.099, 0.151] | 0.110 [0.088, 0.132] |
| LaBraM | 3 | 253 | 0.786 [0.763, 0.809] | 0.786 [0.763, 0.809] | 0.696 [0.674, 0.722] | 0.711 [0.683, 0.739] | 0.090 [0.062, 0.116] | 0.075 [0.052, 0.100] |
| EEGPT | 3 | 253 | 0.801 [0.779, 0.824] | 0.801 [0.778, 0.823] | 0.685 [0.660, 0.712] | 0.727 [0.702, 0.752] | 0.116 [0.091, 0.139] | 0.075 [0.053, 0.096] |
| CBraMod | 3 | 253 | 0.770 [0.746, 0.795] | 0.770 [0.746, 0.795] | 0.640 [0.612, 0.672] | 0.677 [0.644, 0.709] | 0.130 [0.096, 0.164] | 0.093 [0.071, 0.119] |
| REVE-base | 3 | 253 | 0.780 [0.753, 0.806] | 0.780 [0.753, 0.806] | 0.675 [0.643, 0.702] | 0.723 [0.696, 0.749] | 0.106 [0.071, 0.142] | 0.057 [0.038, 0.076] |
| EEGMamba | 3 | 253 | 0.781 [0.756, 0.807] | 0.781 [0.756, 0.806] | 0.693 [0.668, 0.716] | 0.715 [0.690, 0.739] | 0.088 [0.070, 0.106] | 0.066 [0.050, 0.083] |
| BENDR encoder-only | 3 | 253 | 0.781 [0.742, 0.818] | 0.500 [0.487, 0.513] | 0.500 [0.487, 0.513] | 0.500 [0.487, 0.513] | 0.281 [0.241, 0.320] | 0.281 [0.241, 0.320] |

Final macro-F1 aggregate:

| Model | Baseline macro-F1 | Sham macro-F1 | Aperiodic macro-F1 | Flattened macro-F1 | Drop aperiodic | Drop flattened |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BIOT | 0.802 | 0.800 | 0.657 | 0.687 | 0.145 | 0.115 |
| LaBraM | 0.786 | 0.786 | 0.675 | 0.708 | 0.111 | 0.078 |
| EEGPT | 0.800 | 0.800 | 0.658 | 0.725 | 0.142 | 0.075 |
| CBraMod | 0.770 | 0.770 | 0.612 | 0.671 | 0.158 | 0.099 |
| REVE-base | 0.779 | 0.779 | 0.646 | 0.723 | 0.133 | 0.055 |
| EEGMamba | 0.782 | 0.782 | 0.680 | 0.713 | 0.102 | 0.069 |
| BENDR encoder-only | 0.782 | 0.491 | 0.491 | 0.491 | 0.290 | 0.290 |

Interpretation:

The completed full-TUAB foundation-model multiseed audit supports the paper's
central narrative. For BIOT, LaBraM, EEGPT, CBraMod, REVE-base, and EEGMamba,
the sham condition remains essentially unchanged from baseline, while
aperiodic-only and flattened interventions produce consistent positive drops in
balanced accuracy. The largest aperiodic drops are observed for CBraMod, BIOT,
EEGPT, and REVE-base, and the drop confidence intervals are well above zero.

This is qualitatively consistent with the earlier TUAB-200 foundation-model
subset result and strengthens it by using the full TUAB evaluation set and
three seeds. The stronger full-dataset result is that foundation models retain
high baseline abnormality performance, but that performance systematically
degrades when the aperiodic component is perturbed. This supports the claim
that apparent disease-discriminative performance partly depends on aperiodic EEG
structure rather than being driven only by oscillatory neural signatures.

BENDR should be treated separately in the paper. Its baseline performance is
reasonable, but sham, aperiodic, and flattened conditions all collapse to chance.
Because the sham intervention alone collapses BENDR, the BENDR result indicates
intervention fragility for this implementation rather than clean
aperiodic-specific reliance.

## Formal Bootstrap Hypothesis Tests With BH-FDR Correction

Date: 2026-05-29

Goal:

Add formal hypothesis testing to the intervention audit rather than relying
only on 95% bootstrap confidence intervals. The primary null hypothesis is
H0: flattening drop <= 0, tested against H1: flattening drop > 0. Sham controls
are tested separately with H0: sham drop = 0, H1: sham drop != 0.

Implementation:

- Patched `code/scripts/aggregate_multiseed_subject_bootstrap.py` and
  `code/scripts/aggregate_foundation_multiseed_predictions.py` so each
  bootstrap summary row now includes:
  - `p_one_sided_positive`
  - `p_two_sided_zero`
  - `n_bootstrap_nonpositive`
  - `n_bootstrap_nonnegative`
  - `n_bootstrap_valid`
- Added `code/scripts/collect_formal_hypothesis_tests.py` to collect the
  primary balanced-accuracy flattening tests and sham-control tests and apply
  Benjamini-Hochberg FDR correction at q = 0.05.
- Added `code/scripts/run_formal_hypothesis_tests_from_saved_outputs.sh` as a
  statistics-only H200 launcher. It reruns bootstrap aggregation from saved
  subject-metric/prediction files and does not retrain any model.

H200 run:

```text
log: /mnt/data/aperiodic_confounds/logs/formal_hypothesis_tests_20260529_140252.log
bootstrap resamples: 10,000
primary FDR family: balanced-accuracy drop_flattened tests
sham FDR family: balanced-accuracy drop_sham tests
FDR method: Benjamini-Hochberg, q = 0.05
```

Outputs:

```text
/mnt/data/aperiodic_confounds/reports/tables/formal_hypothesis_tests.csv
/mnt/data/aperiodic_confounds/reports/tables/formal_hypothesis_tests.md
/mnt/data/aperiodic_confounds/reports/tables/formal_hypothesis_tests_primary.csv
/mnt/data/aperiodic_confounds/reports/tables/formal_hypothesis_tests_sham.csv
```

Local copies:

```text
reports/tables/formal_hypothesis_tests.csv
reports/tables/formal_hypothesis_tests.md
reports/tables/formal_hypothesis_tests_primary.csv
reports/tables/formal_hypothesis_tests_sham.csv
```

Primary flattening-test family:

- Total primary tests: 31
- FDR-significant at q = 0.05: 25
- Non-significant primary tests: 6

Non-significant primary flattening tests:

| Domain | Task | Model | Drop flattened BA | Raw p | FDR p |
| --- | --- | --- | ---: | ---: | ---: |
| Sleep-EDF | N2 vs N3 | CNN | 0.002 [-0.021, 0.023] | 0.3832 | 0.4400 |
| Sleep-EDF | N2 vs N3 | EEGNet | -0.001 [-0.024, 0.019] | 0.5418 | 0.5998 |
| Sleep-EDF | N2 vs N3 | ShallowFBCSPNet | -0.005 [-0.023, 0.011] | 0.7093 | 0.7582 |
| Sleep-EDF | N2 vs N3 | Deep4Net | -0.010 [-0.036, 0.013] | 0.7755 | 0.8013 |
| PhysioNet MI | Imagined left vs right fist | EEGNet | 0.005 [-0.002, 0.013] | 0.0844 | 0.1006 |
| PhysioNet MI | Imagined left vs right fist | Deep4Net | -0.010 [-0.020, 0.001] | 0.9620 | 0.9620 |

Selected FDR-significant primary tests:

| Domain | Model/task | Drop flattened BA | Raw p | FDR p |
| --- | --- | ---: | ---: | ---: |
| Sleep-EDF | Wake vs sleep, EEGNet | 0.437 [0.427, 0.446] | <0.0001 | <0.0001 |
| Sleep-EDF | Five-stage, EEGNet | 0.395 [0.377, 0.413] | <0.0001 | <0.0001 |
| TUAB full | EEGNet | 0.097 [0.050, 0.144] | 0.0001 | 0.0001 |
| TUAB full | ShallowFBCSPNet | 0.070 [0.020, 0.119] | 0.0030 | 0.0040 |
| TUAB full | Deep4Net | 0.129 [0.075, 0.184] | <0.0001 | <0.0001 |
| TUAB full age/sex-matched | EEGNet | 0.087 [0.012, 0.156] | 0.0120 | 0.0149 |
| TUAB full age/sex-matched | ShallowFBCSPNet | 0.082 [0.019, 0.145] | 0.0050 | 0.0065 |
| TUAB full age/sex-matched | Deep4Net | 0.213 [0.117, 0.302] | <0.0001 | <0.0001 |
| TUAB full foundation models | BIOT | 0.110 [0.088, 0.132] | <0.0001 | <0.0001 |
| TUAB full foundation models | LaBraM | 0.075 [0.052, 0.100] | <0.0001 | <0.0001 |
| TUAB full foundation models | EEGPT | 0.075 [0.053, 0.096] | <0.0001 | <0.0001 |
| TUAB full foundation models | CBraMod | 0.093 [0.071, 0.119] | <0.0001 | <0.0001 |
| TUAB full foundation models | REVE | 0.057 [0.038, 0.076] | <0.0001 | <0.0001 |
| TUAB full foundation models | EEGMamba | 0.066 [0.050, 0.083] | <0.0001 | <0.0001 |
| PhysioNet MI | ShallowFBCSPNet | 0.014 [0.005, 0.024] | 0.0026 | 0.0037 |

Sham-control family:

- Total sham tests: 28
- FDR-significant at q = 0.05: 1
- The only significant sham result is BENDR on full TUAB:
  - sham drop BA = 0.281 [0.241, 0.320]
  - raw p < 0.0001
  - FDR p < 0.0001

Interpretation:

The formal testing strengthens the main narrative. The large Sleep-EDF
wake-vs-sleep and five-stage flattening effects survive FDR correction. Full
TUAB neural models, full TUAB age/sex-matched neural models, and the six clean
foundation-model audits also survive FDR correction. The expected specificity
checks remain mostly non-significant after correction: Sleep-EDF N2-vs-N3 is
non-significant for CNN, EEGNet, ShallowFBCSPNet and Deep4Net, and PhysioNet MI
is non-significant for EEGNet and Deep4Net. PhysioNet MI ShallowFBCSPNet shows
a small but FDR-significant flattening drop, so the manuscript should describe
MI as showing minimal or near-zero aperiodic reliance overall rather than no
effect in every architecture.

BENDR remains a cautionary fragility case. Its flattening drop is formally
significant, but the sham-control family also flags BENDR as significant
because sham performance collapses to chance. Therefore BENDR should not be
grouped with the six clean full-TUAB foundation-model aperiodic-reliance
results.

## PTB-XL ECG 1/f Demonstration and Architecture Audit (May 29, 2026)

Goal:

- Add a compact non-EEG physiological time-series demonstration using PTB-XL
  normal-versus-abnormal ECG classification.
- Test whether broadband 1/f/aperiodic spectral structure can be an exploitable
  shortcut outside EEG.
- Use established ECG/time-series architectures rather than the initial toy CNN.

Data and preprocessing:

- Dataset: PTB-XL v1.0.3, `records100` 100 Hz 12-lead ECG.
- Download source used on H200: official PhysioNet open S3 mirror,
  `https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3`.
- Local H200 data root:
  `/mnt/data/aperiodic_confounds/data/ptbxl/1.0.3`.
- Prepared cache:
  `/mnt/data/aperiodic_confounds/results/ptbxl_1f_demo/ptbxl_records100_normal_abnormal_raw.npz`
  and
  `/mnt/data/aperiodic_confounds/results/ptbxl_1f_demo/ptbxl_records100_normal_abnormal_psd_fixed.npz`.
- Labeling: records with diagnostic class `NORM` only were labeled normal;
  records with any non-`NORM` diagnostic class were labeled abnormal; ambiguous
  records without diagnostic class were excluded.
- Final prepared cohort:
  - 21,375 records
  - 18,610 patients
  - 9,097 normal records
  - 12,278 abnormal records
  - fold 10 test set: 2,157 records
- Filtering: 0.5--40 Hz bandpass, 12 leads, 10 s windows at 100 Hz.
- PSD representation: Welch PSD over 1--45 Hz.

Download notes:

- Initial file-by-file PhysioNet `wget` download was too slow.
- Official PhysioNet ZIP endpoint was valid but throttled to roughly 4--5 h.
- Switching to the official open S3 mirror with `aria2c` and 96 concurrent jobs
  completed `records100` rapidly.
- Download completion: 43,598 / 43,598 low-resolution ECG files.

Scripts added/updated:

```text
code/scripts/download_ptbxl_records100_s3.sh
code/scripts/download_ptbxl_bulk_zip.sh
code/scripts/prepare_ptbxl_1f_demo.py
code/scripts/run_ptbxl_psd_interventions.py
code/scripts/run_ptbxl_raw_cnn_interventions.py
code/scripts/launch_ptbxl_1f_demo.sh
code/scripts/launch_ptbxl_ecg_architecture_audit.sh
code/scripts/aggregate_ptbxl_prediction_bootstrap.py
code/scripts/run_ptbxl_age_sex_matched_psd.py
code/scripts/launch_ptbxl_age_sex_matched_neural.sh
```

Important aggregation correction:

- The first PTB-XL neural summaries averaged per-patient balanced accuracy.
- This is invalid for PTB-XL normal/abnormal because most patients contribute
  records from only one class; patient-level balanced accuracy therefore
  artificially depresses performance toward ~0.5.
- Corrected ECG neural results use prediction-level pooled confusion matrices
  inside a patient bootstrap. The saved prediction files were valid; no model
  retraining was required.
- Treat the older files
  `reports/tables/ptbxl_resnet1d_wang_multiseed_subject_bootstrap.*`,
  `reports/tables/ptbxl_inception1d_multiseed_subject_bootstrap.*`, and
  `reports/tables/ptbxl_xresnet1d101_multiseed_subject_bootstrap.*` as
  PTB-XL aggregation artifacts for balanced accuracy.
- Corrected table:
  `reports/tables/ptbxl_ecg_architectures_prediction_bootstrap.csv`
  and `.md`.

Initial PSD ridge audit:

| Model | Raw BA | Aperiodic BA | Flattened BA | Flattening drop |
| --- | ---: | ---: | ---: | ---: |
| PTB-XL PSD ridge | 0.707 [0.687, 0.727] | 0.566 [0.543, 0.587] | 0.607 [0.586, 0.629] | 0.100 [0.079, 0.122] |

Corrected ECG neural architecture results:

| Model | Seeds | Raw BA | Sham BA | Aperiodic BA | Flattened BA | Flattening drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet1D-Wang | 3 | 0.860 [0.844, 0.876] | 0.860 [0.844, 0.876] | 0.623 [0.605, 0.641] | 0.512 [0.501, 0.531] | 0.347 [0.331, 0.363] |
| Inception1D | 3 | 0.866 [0.852, 0.879] | 0.866 [0.852, 0.879] | 0.578 [0.537, 0.640] | 0.508 [0.500, 0.517] | 0.358 [0.343, 0.373] |
| XResNet1D101 | 3 | 0.860 [0.846, 0.873] | 0.860 [0.846, 0.873] | 0.601 [0.502, 0.683] | 0.538 [0.506, 0.566] | 0.322 [0.291, 0.354] |

Neural architecture run details:

- H200 master log:
  `/mnt/data/aperiodic_confounds/logs/ptbxl_ecg_architectures_20260529_154243.log`
- Models run sequentially:
  `resnet1d_wang`, `inception1d`, `xresnet1d101`.
- Seeds: 42, 43, 44.
- Epochs: 35.
- Batch size: 192.
- Device: CUDA/H200.
- Each model's seed outputs were written under:
  `/mnt/data/aperiodic_confounds/results/ptbxl_1f_demo/{model}_seed{seed}/`.

Interpretation:

- The ECG neural demonstration is strong after correcting the aggregation:
  all three established ECG/time-series architectures have high raw BA,
  neutral sham, and large flattening drops.
- Flattening moves neural performance close to chance, supporting the broader
  claim that aperiodic/broadband spectral structure can be exploited by
  time-series classifiers outside EEG.
- The result is currently best framed as an ECG proof-of-principle extension,
  not as a full ECG biomarker study.

Demographic confounding check:

- PTB-XL has a TUAB-like demographic imbalance.
- PTB-XL encodes a de-identified oldest-age bucket as age 300; 284 records had
  `age=300` and were excluded from age/sex matching.
- With the `age=300` records excluded, abnormal ECGs are older than normal ECGs:
  - all eligible records: abnormal mean age 64.95 years, median 66.0;
    normal mean age 52.04 years, median 54.0.
  - train folds 1--8 before excluding `age=300`: abnormal mean age 68.89,
    normal mean age 53.20.
  - test fold 10 before excluding `age=300`: abnormal mean age 73.03,
    normal mean age 51.69.
- Sex composition also differs by label. Using PTB-XL sex codes as stored:
  - all records: abnormal sex=0 fraction 0.568, normal sex=0 fraction 0.454.
  - train folds: abnormal sex=0 fraction 0.581, normal sex=0 fraction 0.438.
  - test fold: abnormal sex=0 fraction 0.501, normal sex=0 fraction 0.532.
- This confirms that the ECG extension raises the same demographic-confounding
  question as TUAB, especially for age.

PTB-XL age/sex-matched PSD control:

- Matching script:
  `code/scripts/run_ptbxl_age_sex_matched_psd.py`
- Matching rule: same sex, ±5 years, matched separately within train,
  validation, and test split groups.
- Age handling: excluded `age=300` oldest-age bucket; `max_age=90`.
- Matched pairs:
  - train: 5,524 pairs
  - validation: 692 pairs
  - test: 698 pairs
- Selected records:
  - train: 11,048 records
  - validation: 1,384 records
  - test: 1,396 records
- Output directory:
  `results/ptbxl_1f_demo/age_sex_matched_psd/`.

Age/sex-matched PSD results:

| Model | Raw BA | Aperiodic BA | Flattened BA | Drop aperiodic | Drop flattened |
| --- | ---: | ---: | ---: | ---: | ---: |
| PTB-XL matched PSD ridge | 0.612 [0.586, 0.637] | 0.500 [0.500, 0.500] | 0.580 [0.559, 0.600] | 0.112 [0.087, 0.137] | 0.032 [0.009, 0.056] |

Matched-control interpretation:

- The age/sex-matched PSD result is much weaker than the unmatched PSD result
  for the flattened intervention, but the CI remains above zero.
- This means part of the ECG aperiodic effect is plausibly demographic,
  especially age-linked, but a residual flattening effect persists after
  matching.
- The matched aperiodic-only PSD result collapses to chance in this control,
  while flattened PSD retains above-chance performance. This suggests ECG
  morphology/harmonic structure and broadband slope interact differently than
  in EEG; this should be presented cautiously.

PTB-XL age/sex-matched neural architecture audit:

- Purpose: test whether the large unmatched ECG neural flattening drops
  (0.32--0.36 BA) collapse under the same age/sex-matched control that reduced
  the PSD ridge flattening drop to 0.032.
- Launch script:
  `code/scripts/launch_ptbxl_age_sex_matched_neural.sh`
- Shared runner:
  `code/scripts/run_ptbxl_raw_cnn_interventions.py`
- Prediction-level bootstrap aggregator:
  `code/scripts/aggregate_ptbxl_prediction_bootstrap.py`
- Record filter:
  `results/ptbxl_1f_demo/age_sex_matched_psd/ptbxl_age_sex_matched_records.csv`
- Matching inherited from the PSD control:
  - train: 5,524 pairs / 11,048 records
  - validation: 692 pairs / 1,384 records
  - test: 698 pairs / 1,396 records
- Bootstrap unit for the matched neural audit: `pair_id`, not patient ID. This
  preserves the matched-pair design in the evaluation bootstrap.
- Models: ResNet1D-Wang, Inception1D, XResNet1D101.
- Seeds: 42, 43, 44.
- Epochs: 35.
- Batch size: 192.
- Device: CUDA/H200.
- H200 master log:
  `/mnt/data/aperiodic_confounds/logs/ptbxl_age_sex_matched_neural_20260529_171717.log`
- Local master log:
  `logs/ptbxl_age_sex_matched_neural_20260529_171717.log`
- H200 run root:
  `/mnt/data/aperiodic_confounds/results/ptbxl_1f_demo/age_sex_matched_neural`
- Local run root:
  `results/ptbxl_1f_demo/age_sex_matched_neural/`
- Final combined table:
  `reports/tables/ptbxl_matched_ecg_architectures_prediction_bootstrap.csv`
  and `.md`.

Age/sex-matched neural results:

| Model | Seeds | Raw BA | Sham BA | Aperiodic BA | Flattened BA | Drop aperiodic | Drop flattened |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet1D-Wang | 3 | 0.819 [0.799, 0.839] | 0.819 [0.799, 0.839] | 0.577 [0.529, 0.623] | 0.507 [0.501, 0.514] | 0.243 [0.193, 0.291] | 0.312 [0.293, 0.331] |
| Inception1D | 3 | 0.830 [0.811, 0.848] | 0.830 [0.811, 0.848] | 0.542 [0.523, 0.566] | 0.519 [0.510, 0.529] | 0.287 [0.261, 0.312] | 0.311 [0.291, 0.330] |
| XResNet1D101 | 3 | 0.820 [0.801, 0.840] | 0.820 [0.801, 0.840] | 0.574 [0.535, 0.608] | 0.521 [0.510, 0.533] | 0.247 [0.208, 0.291] | 0.299 [0.279, 0.319] |

Matched neural interpretation:

- This is the key ECG control result. Unlike the PSD ridge, the neural
  flattening drops do not collapse after age/sex matching.
- The matched PSD ridge flattening drop was 0.032 [0.009, 0.056], whereas the
  matched neural drops remain 0.299--0.312 with tight CIs above zero.
- Raw BA is lower in the matched neural cohort than in the unmatched full
  cohort, as expected after balancing age/sex and reducing sample size, but all
  three architectures still classify well before intervention.
- Sham is exactly neutral in the saved predictions, indicating that the
  Fourier round trip itself is not responsible for the effect.
- The ECG interpretation should therefore be nuanced: PSD-level aperiodic
  dependence is partly demographic, but deep ECG models appear to exploit
  additional broadband or morphology-linked spectral structure that survives
  age/sex matching.

SpecParam / fixed aperiodic fit quality on ECG:

- ECG spectra differ from EEG spectra because QRS complexes and heart-rate
  periodicity introduce strong harmonic structure.
- Fixed aperiodic fits are therefore visibly poorer than EEG fits and should be
  reported transparently rather than treated as an EEG-like fit.
- Fit array shape: 21,375 records × 12 leads.
- Channel-level R²:
  - median 0.273
  - p10 0.138
  - p25 0.198
  - p75 0.356
  - p90 0.442
  - p95 0.498
  - p99 0.596
- Record-level median R²:
  - median 0.270
  - p10 0.157
  - p90 0.412
- Fractions of channel fits below thresholds:
  - R² < 0.05: 0.0069
  - R² < 0.10: 0.0424
  - R² < 0.20: 0.2559
- One extreme negative R² outlier makes mean R² uninformative
  (`specparam_fixed_mean_r2 = -141.285` in the summary JSON); the manuscript
  should report medians/quantiles instead of the mean for PTB-XL ECG.

Manuscript relevance:

- PTB-XL adds a non-EEG physiological time-series proof-of-principle.
- It strengthens the NMI framing if presented as an extension and stress test
  of the audit framework, not as a fourth fully developed clinical domain.
- The strongest ECG claim is currently:
  established ECG architectures on PTB-XL show large performance loss after
  flattening broadband spectral structure; unlike the PSD ridge, this neural
  flattening effect remains large after age/sex matching.

## TUAB Full-Dataset Site/Temporal Acquisition-Proxy PSD Audit (May 29, 2026)

Goal:

- Address the site/acquisition-condition confounding objection for the TUAB EEG
  results.
- Test whether the aperiodic PSD ridge effect is merely driven by temporal
  acquisition-condition variation, such as hardware/protocol drift, rather than
  signal that persists within and across acquisition windows.
- Use only non-signal metadata as the acquisition proxy: EDF recording date
  parsed from the header and montage from the TUAB path.

Important metadata caveat:

- The available TUAB EDF headers in our local full-corpus copy expose only
  year-level dates. Most usable recordings fall in 2009--2013.
- A small number of files carry sentinel/outlier years: 1899, 2000 and 2007.
  These are excluded from the primary temporal-bin audit.
- The EDF recording header is largely anonymized beyond the start date and file
  token. The technician/equipment-like suffix is typically `XXX X`, so date is
  the usable acquisition-condition proxy.
- The full-TUAB preprocessing index uses montage `01_tcp_ar` for the analyzed
  cache, so montage variation is not expected to explain the existing
  full-TUAB result.

Implementation:

- Added script:
  `code/scripts/run_tuab_site_temporal_psd_audit.py`
- Inputs:
  - epoch index:
    `/mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/preprocess_20s_100hz/psd_20s_multitaper_index.csv`
  - precomputed specparam decomposition:
    `/mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/preprocess_20s_100hz/specparam/specparam_fixed_20s.npz`
  - EDF header metadata:
    `/mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/age_matched/tuab_full_age_sex_matched_caliper5_header_metadata_files.csv`
- Primary temporal split:
  - early: 2009--2011
  - late: 2012--2013
- Sensitivity temporal terciles:
  - early: 2009--2010
  - middle: 2011--2012
  - late: 2013
- Models:
  - Ridge classifier, balanced class weights, alpha 1.0.
  - Same cached full-log PSD, aperiodic spectrum and flattened residual
    features as the full-TUAB PSD intervention audit.
- Scenarios:
  - within early: train official-train early, evaluate official-eval early.
  - within late: train official-train late, evaluate official-eval late.
  - cross early to late: train official-train early, evaluate official-eval late.
  - cross late to early: train official-train late, evaluate official-eval early.
  - within each year-tercile sensitivity bin.
- Bootstrap:
  - 10,000 stratified eval-subject bootstrap resamples.
  - Same balanced-accuracy drop definition as other PSD intervention audits.

Run status:

- Launched on H200 under nohup at 2026-05-29 18:00 IST.
- H200 PID: 3968973.
- H200 log:
  `/mnt/data/aperiodic_confounds/logs/tuab_site_temporal_psd_audit_20260529_180048.log`
- H200 output directory:
  `/mnt/data/aperiodic_confounds/results/tuab_full_v3_0_1/site_temporal_psd_audit`
- Local script synced to H200 and passed remote compile check.
- Completed successfully. Final small result tables and the log were pulled
  locally. The large prediction CSV remained on H200 after a local rsync timeout,
  but the bootstrap/eval/metadata tables needed for reporting were pulled.
- Local outputs:
  - `results/tuab_full_v3_0_1/site_temporal_psd_audit/tuab_site_temporal_subject_bootstrap.csv`
  - `results/tuab_full_v3_0_1/site_temporal_psd_audit/tuab_site_temporal_subject_bootstrap.md`
  - `results/tuab_full_v3_0_1/site_temporal_psd_audit/tuab_site_temporal_scenario_counts.csv`
  - `logs/tuab_site_temporal_psd_audit_20260529_180048.log`

Primary temporal-bin results:

| Scenario | Eval subjects | Raw BA | Aperiodic BA | Flattened BA | Drop flattened |
| --- | ---: | ---: | ---: | ---: | ---: |
| Within early 2009--2011 | 137 | 0.703 [0.640, 0.762] | 0.609 [0.552, 0.663] | 0.585 [0.547, 0.627] | 0.118 [0.068, 0.165] |
| Within late 2012--2013 | 120 | 0.767 [0.711, 0.819] | 0.544 [0.516, 0.573] | 0.563 [0.539, 0.588] | 0.204 [0.153, 0.253] |
| Train early, eval late | 120 | 0.762 [0.714, 0.807] | 0.644 [0.590, 0.696] | 0.616 [0.581, 0.653] | 0.146 [0.107, 0.184] |
| Train late, eval early | 137 | 0.709 [0.641, 0.775] | 0.543 [0.502, 0.580] | 0.545 [0.517, 0.578] | 0.164 [0.105, 0.221] |

Year-tercile sensitivity:

| Scenario | Eval subjects | Raw BA | Aperiodic BA | Flattened BA | Drop flattened |
| --- | ---: | ---: | ---: | ---: | ---: |
| Within 2009--2010 | 103 | 0.703 [0.636, 0.766] | 0.677 [0.604, 0.749] | 0.604 [0.556, 0.652] | 0.100 [0.061, 0.137] |
| Within 2011--2012 | 99 | 0.720 [0.654, 0.787] | 0.539 [0.488, 0.590] | 0.582 [0.550, 0.617] | 0.138 [0.079, 0.197] |
| Within 2013 | 59 | 0.758 [0.680, 0.829] | 0.606 [0.544, 0.666] | 0.566 [0.533, 0.599] | 0.191 [0.136, 0.245] |

Interpretation:

- The TUAB PSD flattening drop persists within both primary temporal bins.
- The drop also persists in both cross-temporal generalization directions,
  which directly weakens the concern that the full-TUAB aperiodic effect is only
  a non-transferable equipment/protocol artifact from one recording era.
- The year-tercile sensitivity check is consistent: all three within-tercile
  flattening-drop CIs remain above zero.
- The magnitude varies by temporal window, so the conservative manuscript
  framing should say that acquisition period may modulate the effect size, but
  it does not explain the presence of aperiodic reliance.
