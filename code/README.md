# Aperiodic EEG Confound Audit

This is the code workspace for the project described in `../project.md`.

The workflow is local-first and remote-run:

1. Edit code on the Mac in this `code/` folder.
2. Sync the project to H200 with the safe additive command in `../instructions.md`.
3. Run data download, preprocessing, training, and analysis on H200.
4. Keep datasets on H200 only, under `/mnt/data/aperiodic_confounds/data`.

Do not use destructive sync or cleanup commands on the H200 shared storage.

## Current Dataset

The first MVP dataset is a small Sleep-EDF Expanded Sleep-Cassette subset:

```text
/mnt/data/aperiodic_confounds/data/sleep-edf/sleep-cassette
```

The staged subset contains 10 cassette subjects, two nights each where available:

```text
SC4001/SC4002 through SC4091/SC4092
```

Each recording should have:

```text
*PSG.edf
*Hypnogram.edf
```

## Dataset 2: PhysioNet EEG Motor Movement/Imagery

The second-domain dataset is PhysioNet EEG Motor Movement/Imagery
(`eegmmidb`, Schalk et al.). It has 109 subjects, 64 EEG channels, 14 runs per
subject, and a 160 Hz sampling rate. Unlike Sleep-EDF, this is a short
cue-locked motor execution/imagery task, so the analysis unit is a 2-4 second
trial window rather than a 30-second sleep epoch.

The default first task is imagined left fist versus imagined right fist using
runs 4, 8, and 12. We extract `0.5-4.0 s` after cue onset. This avoids the
visual-cue boundary and keeps the window long enough for motor-imagery mu/beta
activity while acknowledging that aperiodic fitting is harder than in
Sleep-EDF. For this dataset, specparam fit quality is a reportable result, not
just a diagnostic.

Additively download the dataset on H200:

```bash
cd /mnt/data/aperiodic_confounds/code
bash scripts/download_physionet_mi.sh
```

Create the imagined-fist trial manifest:

```bash
/mnt/data/aperiodic_confounds/.venv/bin/python scripts/make_physionet_mi_trials.py \
  --data-root /mnt/data/aperiodic_confounds/data/physionet-eegmmidb \
  --output-csv /mnt/data/aperiodic_confounds/results/physionet_mi/imagined_fists_trials.csv \
  --task imagined_fists \
  --tmin 0.5 \
  --tmax 4.0
```

Cache raw cue-locked trials:

```bash
/mnt/data/aperiodic_confounds/.venv/bin/python scripts/extract_physionet_mi_raw_trials.py \
  --trials-csv /mnt/data/aperiodic_confounds/results/physionet_mi/imagined_fists_trials.csv \
  --output-npz /mnt/data/aperiodic_confounds/results/physionet_mi/imagined_fists_raw_160hz.npz \
  --output-index-csv /mnt/data/aperiodic_confounds/results/physionet_mi/imagined_fists_raw_index.csv \
  --channels all \
  --target-sfreq 160
```

Extract short-window multitaper PSDs:

```bash
/mnt/data/aperiodic_confounds/.venv/bin/python scripts/extract_physionet_mi_psd.py \
  --raw-npz /mnt/data/aperiodic_confounds/results/physionet_mi/imagined_fists_raw_160hz.npz \
  --raw-index-csv /mnt/data/aperiodic_confounds/results/physionet_mi/imagined_fists_raw_index.csv \
  --output-npz /mnt/data/aperiodic_confounds/results/physionet_mi/imagined_fists_psd_multitaper.npz \
  --output-index-csv /mnt/data/aperiodic_confounds/results/physionet_mi/imagined_fists_psd_index.csv \
  --freq-min 2 \
  --freq-max 45 \
  --bandwidth 4
```

Fit specparam and save fit-quality diagnostics:

```bash
/mnt/data/aperiodic_confounds/.venv/bin/python scripts/fit_physionet_mi_specparam.py \
  --psd-npz /mnt/data/aperiodic_confounds/results/physionet_mi/imagined_fists_psd_multitaper.npz \
  --output-npz /mnt/data/aperiodic_confounds/results/physionet_mi/specparam/imagined_fists_specparam_fixed.npz \
  --n-jobs 8
```

## Remote Quick Check

From H200:

```bash
cd /mnt/data/aperiodic_confounds/code
/mnt/data/aperiodic_confounds/.venv/bin/python scripts/verify_sleep_edf.py \
  --data-root /mnt/data/aperiodic_confounds/data/sleep-edf/sleep-cassette \
  --expected-pairs 20
```

Create or refresh the project virtual environment on H200:

```bash
cd /mnt/data/aperiodic_confounds/code
bash scripts/setup_h200_env.sh
```

The EDF loading and PSD scripts require `mne`. Use the project virtual environment:

```bash
/mnt/data/aperiodic_confounds/.venv/bin/python scripts/make_sleep_edf_epochs.py
/mnt/data/aperiodic_confounds/.venv/bin/python scripts/extract_sleep_edf_psd.py
```

## Planned Pipeline

The first implementation target is:

```text
Sleep-EDF EDF files
-> epoch labels from hypnograms
-> EEG-only PSD extraction
-> aperiodic/periodic decomposition
-> full PSD vs aperiodic-only vs periodic-residual baselines
-> intervention metrics
```

## Stage 1: Epoch Manifest

Create a 30-second epoch manifest from the Sleep-EDF hypnograms:

```bash
/mnt/data/aperiodic_confounds/.venv/bin/python scripts/make_sleep_edf_epochs.py \
  --data-root /mnt/data/aperiodic_confounds/data/sleep-edf/sleep-cassette \
  --output-csv /mnt/data/aperiodic_confounds/results/sleep_edf_subset/epochs.csv \
  --wake-trim-minutes 30
```

## Stage 2: PSD Extraction

Extract EEG-only Welch PSD features:

```bash
/mnt/data/aperiodic_confounds/.venv/bin/python scripts/extract_sleep_edf_psd.py \
  --data-root /mnt/data/aperiodic_confounds/data/sleep-edf/sleep-cassette \
  --epochs-csv /mnt/data/aperiodic_confounds/results/sleep_edf_subset/epochs.csv \
  --output-npz /mnt/data/aperiodic_confounds/results/sleep_edf_subset/psd_welch_fpz_pz.npz \
  --output-index-csv /mnt/data/aperiodic_confounds/results/sleep_edf_subset/psd_index.csv \
  --channels Fpz-Cz Pz-Oz
```

Expected output for the current subset:

```text
epochs.csv: 21039 rows
psd_welch_fpz_pz.npz: PSD shape (21039, 2, 177)
psd_index.csv: 21039 rows
```

## Stage 3: Aperiodic Audit Baselines

Run subject-level classical baselines for:

```text
full_log_psd
aperiodic_spectrum
aperiodic_params
periodic_residual
```

```bash
/mnt/data/aperiodic_confounds/.venv/bin/python scripts/run_sleep_edf_aperiodic_baselines.py \
  --psd-npz /mnt/data/aperiodic_confounds/results/sleep_edf_subset/psd_welch_fpz_pz.npz \
  --index-csv /mnt/data/aperiodic_confounds/results/sleep_edf_subset/psd_index.csv \
  --output-dir /mnt/data/aperiodic_confounds/results/sleep_edf_subset/baselines \
  --n-splits 5 \
  --classifier ridge
```

Outputs:

```text
fold_metrics.csv
summary_metrics.csv
aperiodic_fit_summary.json
```

## Stage 4: Specparam Sensitivity

Fit conservative fixed-mode specparam models:

```bash
/mnt/data/aperiodic_confounds/.venv/bin/python scripts/fit_sleep_edf_specparam.py \
  --psd-npz /mnt/data/aperiodic_confounds/results/sleep_edf_subset/psd_welch_fpz_pz.npz \
  --output-npz /mnt/data/aperiodic_confounds/results/sleep_edf_subset/specparam/specparam_fixed.npz \
  --n-jobs 8
```

Run the same baselines using the precomputed specparam decomposition:

```bash
/mnt/data/aperiodic_confounds/.venv/bin/python scripts/run_sleep_edf_aperiodic_baselines.py \
  --psd-npz /mnt/data/aperiodic_confounds/results/sleep_edf_subset/psd_welch_fpz_pz.npz \
  --index-csv /mnt/data/aperiodic_confounds/results/sleep_edf_subset/psd_index.csv \
  --decomposition precomputed \
  --decomp-npz /mnt/data/aperiodic_confounds/results/sleep_edf_subset/specparam/specparam_fixed.npz \
  --output-dir /mnt/data/aperiodic_confounds/results/sleep_edf_subset/baselines_specparam \
  --n-splits 5 \
  --classifier ridge
```

## Stage 5: Diagnostics And Matching

Summarize specparam fit quality:

```bash
/mnt/data/aperiodic_confounds/.venv/bin/python scripts/analyze_specparam_diagnostics.py \
  --decomp-npz /mnt/data/aperiodic_confounds/results/sleep_edf_subset/specparam/specparam_fixed.npz \
  --index-csv /mnt/data/aperiodic_confounds/results/sleep_edf_subset/psd_index.csv \
  --output-dir /mnt/data/aperiodic_confounds/results/sleep_edf_subset/specparam/diagnostics
```

Run a first offset/exponent matched-control baseline:

```bash
/mnt/data/aperiodic_confounds/.venv/bin/python scripts/run_sleep_edf_matched_baselines.py \
  --psd-npz /mnt/data/aperiodic_confounds/results/sleep_edf_subset/psd_welch_fpz_pz.npz \
  --index-csv /mnt/data/aperiodic_confounds/results/sleep_edf_subset/psd_index.csv \
  --decomp-npz /mnt/data/aperiodic_confounds/results/sleep_edf_subset/specparam/specparam_fixed.npz \
  --reference-summary-csv /mnt/data/aperiodic_confounds/results/sleep_edf_subset/baselines_specparam/summary_metrics.csv \
  --output-dir /mnt/data/aperiodic_confounds/results/sleep_edf_subset/matched_specparam \
  --n-bins 4 \
  --n-splits 5
```

## Stage 6: Specparam Peak Sensitivity

Run a small peak-constraint sensitivity grid:

```bash
cd /mnt/data/aperiodic_confounds/code
bash scripts/run_specparam_sensitivity_grid.sh
```

Summarize the grid:

```bash
/mnt/data/aperiodic_confounds/.venv/bin/python scripts/summarize_specparam_sensitivity.py
```

## Stage 7: Full-Model Intervention Evaluation

Train on full log-PSD, then evaluate the same trained model on full,
aperiodic-only, and explicitly flattened test inputs:

```bash
/mnt/data/aperiodic_confounds/.venv/bin/python scripts/run_sleep_edf_intervention_eval.py \
  --psd-npz /mnt/data/aperiodic_confounds/results/sleep_edf_subset/psd_welch_fpz_pz.npz \
  --index-csv /mnt/data/aperiodic_confounds/results/sleep_edf_subset/psd_index.csv \
  --decomp-npz /mnt/data/aperiodic_confounds/results/sleep_edf_subset/specparam/specparam_fixed.npz \
  --output-dir /mnt/data/aperiodic_confounds/results/sleep_edf_subset/interventions_specparam_flattening \
  --n-splits 5
```

## Stage 8: Paper-Style Result Artifact

Create a figure and table from the intervention results:

```bash
/mnt/data/aperiodic_confounds/.venv/bin/python scripts/plot_sleep_edf_intervention_results.py \
  --summary-csv /mnt/data/aperiodic_confounds/results/sleep_edf_subset/interventions_specparam_flattening/intervention_summary_metrics.csv
```

Outputs:

```text
reports/figures/sleep_edf_intervention_performance.png
reports/figures/sleep_edf_intervention_performance.pdf
reports/tables/sleep_edf_intervention_summary.csv
reports/tables/sleep_edf_intervention_summary.md
```

## Stage 9: Deep PSD Baseline

Run a small MLP trained on full log-PSD and evaluated on the same intervention
inputs. This uses the shared H200 Torch environment because it already has CUDA
support:

```bash
cd /mnt/data/aperiodic_confounds/code
/mnt/data/.venvs/ml/bin/python3 scripts/run_sleep_edf_deep_intervention_mlp.py \
  --index-csv /mnt/data/aperiodic_confounds/results/sleep_edf_subset/psd_index.csv \
  --decomp-npz /mnt/data/aperiodic_confounds/results/sleep_edf_subset/specparam/specparam_fixed.npz \
  --output-dir /mnt/data/aperiodic_confounds/results/sleep_edf_subset/deep_mlp_interventions_specparam \
  --n-splits 5 \
  --device cuda
```

## Stage 10: Linear Aperiodic Attribution

Project full-spectrum ridge coefficients onto the fixed aperiodic tangent basis
`[1, -log(f)]` and compute AAR:

```bash
/mnt/data/aperiodic_confounds/.venv/bin/python scripts/analyze_linear_aperiodic_attribution.py \
  --index-csv /mnt/data/aperiodic_confounds/results/sleep_edf_subset/psd_index.csv \
  --decomp-npz /mnt/data/aperiodic_confounds/results/sleep_edf_subset/specparam/specparam_fixed.npz \
  --output-dir /mnt/data/aperiodic_confounds/results/sleep_edf_subset/linear_attribution \
  --n-splits 5
```

## Stage 11: Data-Conditioned Logit Contributions

Decompose full-trained ridge logits into aperiodic and flattened input
contributions on held-out subject folds:

```bash
/mnt/data/aperiodic_confounds/.venv/bin/python scripts/analyze_linear_logit_contributions.py \
  --index-csv /mnt/data/aperiodic_confounds/results/sleep_edf_subset/psd_index.csv \
  --decomp-npz /mnt/data/aperiodic_confounds/results/sleep_edf_subset/specparam/specparam_fixed.npz \
  --output-dir /mnt/data/aperiodic_confounds/results/sleep_edf_subset/linear_logit_contributions \
  --n-splits 5
```

## Stage 12: Bootstrap Confidence Intervals

Compute paired fold-bootstrap confidence intervals for the linear and deep
intervention results:

```bash
/mnt/data/aperiodic_confounds/.venv/bin/python scripts/bootstrap_intervention_metrics.py \
  --linear-fold-csv /mnt/data/aperiodic_confounds/results/sleep_edf_subset/interventions_specparam_flattening/intervention_fold_metrics.csv \
  --deep-fold-csv /mnt/data/aperiodic_confounds/results/sleep_edf_subset/deep_mlp_interventions_specparam/deep_mlp_intervention_fold_metrics.csv \
  --output-csv /mnt/data/aperiodic_confounds/reports/tables/intervention_bootstrap_ci.csv \
  --output-md /mnt/data/aperiodic_confounds/reports/tables/intervention_bootstrap_ci.md
```

Create a compact markdown table:

```bash
/mnt/data/aperiodic_confounds/.venv/bin/python scripts/make_intervention_summary_table.py \
  --summary-csv /mnt/data/aperiodic_confounds/results/sleep_edf_subset/interventions_specparam_flattening/intervention_summary_metrics.csv \
  --output-md /mnt/data/aperiodic_confounds/results/sleep_edf_subset/interventions_specparam_flattening/intervention_summary_table.md
```

## Stage 13: Raw EEG CNN Baseline

Cache the same labeled 30-second epochs as raw EEG tensors. This separates EDF
loading from GPU training:

```bash
/mnt/data/aperiodic_confounds/.venv/bin/python scripts/extract_sleep_edf_raw_epochs.py \
  --epochs-csv /mnt/data/aperiodic_confounds/results/sleep_edf_subset/epochs.csv \
  --output-npz /mnt/data/aperiodic_confounds/results/sleep_edf_subset/raw_epochs_fpz_pz_100hz.npz \
  --output-index-csv /mnt/data/aperiodic_confounds/results/sleep_edf_subset/raw_epochs_index.csv \
  --channels Fpz-Cz Pz-Oz
```

Train a lightweight 1D CNN directly on raw EEG epochs using the same
subject-held-out folds and task definitions:

```bash
cd /mnt/data/aperiodic_confounds/code
/mnt/data/.venvs/ml/bin/python3 scripts/run_sleep_edf_raw_cnn.py \
  --raw-npz /mnt/data/aperiodic_confounds/results/sleep_edf_subset/raw_epochs_fpz_pz_100hz.npz \
  --index-csv /mnt/data/aperiodic_confounds/results/sleep_edf_subset/raw_epochs_index.csv \
  --output-dir /mnt/data/aperiodic_confounds/results/sleep_edf_subset/raw_cnn \
  --n-splits 5 \
  --device cuda
```

## Stage 14: Raw CNN Aperiodic Interventions

Train the raw CNN on unmodified epochs, then evaluate the same trained fold
models on phase-preserving FFT edits:

```text
raw_eeg: original held-out epoch
phase_aperiodic: original phase with smoothed aperiodic amplitude envelope
phase_flattened: original phase with centered aperiodic envelope divided out
```

```bash
cd /mnt/data/aperiodic_confounds/code
/mnt/data/.venvs/ml/bin/python3 scripts/run_sleep_edf_raw_cnn_intervention.py \
  --raw-npz /mnt/data/aperiodic_confounds/results/sleep_edf_subset/raw_epochs_fpz_pz_100hz.npz \
  --index-csv /mnt/data/aperiodic_confounds/results/sleep_edf_subset/raw_epochs_index.csv \
  --decomp-npz /mnt/data/aperiodic_confounds/results/sleep_edf_subset/specparam/specparam_fixed.npz \
  --output-dir /mnt/data/aperiodic_confounds/results/sleep_edf_subset/raw_cnn_interventions \
  --n-splits 5 \
  --device cuda
```

## Stage 15: Combined Intervention Report

Refresh fold-bootstrap confidence intervals for all three model families:

```bash
/mnt/data/aperiodic_confounds/.venv/bin/python scripts/bootstrap_intervention_metrics.py \
  --linear-fold-csv /mnt/data/aperiodic_confounds/results/sleep_edf_subset/interventions_specparam_flattening/intervention_fold_metrics.csv \
  --deep-fold-csv /mnt/data/aperiodic_confounds/results/sleep_edf_subset/deep_mlp_interventions_specparam/deep_mlp_intervention_fold_metrics.csv \
  --raw-cnn-fold-csv /mnt/data/aperiodic_confounds/results/sleep_edf_subset/raw_cnn_interventions/raw_cnn_intervention_fold_metrics.csv \
  --braindecode-eegnet-fold-csv /mnt/data/aperiodic_confounds/results/sleep_edf_subset/braindecode_eegnet_interventions/braindecode_eegnet_intervention_fold_metrics.csv \
  --output-csv /mnt/data/aperiodic_confounds/reports/tables/intervention_bootstrap_ci.csv \
  --output-md /mnt/data/aperiodic_confounds/reports/tables/intervention_bootstrap_ci.md \
  --n-bootstrap 10000
```

Create the combined paper-style figure and table:

```bash
/mnt/data/aperiodic_confounds/.venv/bin/python scripts/plot_combined_intervention_results.py \
  --bootstrap-csv /mnt/data/aperiodic_confounds/reports/tables/intervention_bootstrap_ci.csv \
  --figure-png /mnt/data/aperiodic_confounds/reports/figures/combined_intervention_performance.png \
  --figure-pdf /mnt/data/aperiodic_confounds/reports/figures/combined_intervention_performance.pdf \
  --table-csv /mnt/data/aperiodic_confounds/reports/tables/combined_intervention_summary.csv \
  --table-md /mnt/data/aperiodic_confounds/reports/tables/combined_intervention_summary.md
```

## Stage 16: Braindecode EEGNet Intervention

Install Braindecode into the project-local dependency folder on H200 without
modifying the shared Torch virtual environment:

```bash
mkdir -p /mnt/data/aperiodic_confounds/.python_deps
/mnt/data/.venvs/ml/bin/python3 -m pip install \
  --target /mnt/data/aperiodic_confounds/.python_deps \
  --no-deps \
  braindecode skorch docstring_inheritance torchinfo linear_attention_transformer \
  rotary_embedding_torch mne_bids wfdb scikit-learn tabulate joblib threadpoolctl \
  mne lazy-loader decorator pooch tqdm packaging matplotlib contourpy cycler \
  fonttools kiwisolver pillow pyparsing python-dateutil six h5py requests \
  platformdirs certifi charset-normalizer idna urllib3 einops \
  axial-positional-embedding linformer local-attention product-key-memory \
  colt5-attention hyper-connections pydantic annotated-types typing-extensions \
  typing-inspection torchaudio==2.11.0
/mnt/data/.venvs/ml/bin/python3 -m pip install \
  --target /mnt/data/aperiodic_confounds/.python_deps \
  --upgrade --no-deps pydantic-core==2.46.4
```

Run Braindecode's EEGNet with the same raw EEG interventions:

```bash
cd /mnt/data/aperiodic_confounds/code
PYTHONPATH=/mnt/data/aperiodic_confounds/.python_deps \
/mnt/data/.venvs/ml/bin/python3 scripts/run_sleep_edf_braindecode_eegnet_intervention.py \
  --raw-npz /mnt/data/aperiodic_confounds/results/sleep_edf_subset/raw_epochs_fpz_pz_100hz.npz \
  --index-csv /mnt/data/aperiodic_confounds/results/sleep_edf_subset/raw_epochs_index.csv \
  --decomp-npz /mnt/data/aperiodic_confounds/results/sleep_edf_subset/specparam/specparam_fixed.npz \
  --output-dir /mnt/data/aperiodic_confounds/results/sleep_edf_subset/braindecode_eegnet_interventions \
  --n-splits 5 \
  --device cuda
```
