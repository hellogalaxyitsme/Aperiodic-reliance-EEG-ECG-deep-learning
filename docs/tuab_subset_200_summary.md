# TUAB 200-Subject Subset Summary

## Purpose

We prepared a bounded TUAB subset so we can extend the aperiodic-confound
audit into a clinical normal-vs-abnormal EEG benchmark without downloading the
full 60 GB TUAB release.

The subset was designed to follow the official TUAB evaluation protocol:

```text
Use the official train/eval split.
Sample subjects within each official split and label.
Do not mix subjects across train and eval.
Sample by subject, not by recording.
Save exact subject IDs and file paths for reproducibility.
```

## Access

The local TUH SSH key works:

```text
~/.ssh/id_ed25519_tuh
```

Access was verified using read-only rsync dry-runs against:

```text
nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/tuh_eeg_abnormal/v3.0.1/
```

For H200, access was done via SSH agent forwarding:

```text
ssh -A h200
```

No TUH private key was copied to H200.

## Source Dataset

TUAB release:

```text
TUH Abnormal EEG Corpus
version: v3.0.1
remote root: data/tuh_eeg/tuh_eeg_abnormal/v3.0.1/
official EDF root: edf/
```

Official split structure:

```text
edf/train/normal/01_tcp_ar
edf/train/abnormal/01_tcp_ar
edf/eval/normal/01_tcp_ar
edf/eval/abnormal/01_tcp_ar
```

The README reports:

```text
full release size: about 60 GB
EDF files: 2993
subjects: 2383
evaluation subjects: 253
training subjects: 2130 subject-label entries, 2076 unique train subjects
```

Important README note:

```text
The official evaluation and training sets are subject-disjoint.
Some training subjects appear under both normal and abnormal labels.
```

For this subset, we excluded training subjects that appeared under both labels.

## Selected Subset

Subset name:

```text
random_stratified_200
```

Seed:

```text
20260524
```

Target subject counts:

| Official split | Label | Subjects |
| --- | --- | ---: |
| train | normal | 60 |
| train | abnormal | 60 |
| eval | normal | 40 |
| eval | abnormal | 40 |

Final selected data:

```text
unique subjects: 200
EDF files: 247
download size: 5,059,102,966 bytes
download size: 4.71 GiB
selected train label conflicts: 0
```

File counts:

| Official split | Label | EDF files |
| --- | --- | ---: |
| train | normal | 67 |
| train | abnormal | 87 |
| eval | normal | 40 |
| eval | abnormal | 53 |

The subject count is exactly balanced by split and label, while the file count
is higher than 200 because some selected subjects have multiple sessions/files.

## Download

H200 destination:

```text
/mnt/data/aperiodic_confounds/data/tuab/v3.0.1_random_stratified_200
```

Download log:

```text
/mnt/data/aperiodic_confounds/logs/tuab_subset_200_download_20260524_162716.log
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
README present: yes
```

H200 storage after download:

```text
TUAB subset size on disk: 4.8G
/mnt/data available after download: about 594G
```

## Header Metadata Audit

We extracted EDF header metadata without loading signal arrays.

Script:

```text
code/scripts/extract_tuab_header_metadata.py
```

Outputs:

```text
results/tuab_subset_200/tuab_v3_0_1_random_stratified_200_header_metadata_files.csv
results/tuab_subset_200/tuab_v3_0_1_random_stratified_200_header_metadata_subjects.csv
results/tuab_subset_200/tuab_v3_0_1_random_stratified_200_header_metadata_summary.json
```

Sex balance by subject:

| Official split | Label | Female | Male |
| --- | --- | ---: | ---: |
| eval | abnormal | 21 | 19 |
| eval | normal | 19 | 21 |
| train | abnormal | 32 | 28 |
| train | normal | 35 | 25 |

Age availability:

```text
age_available_subject_rows: 0
```

The EDF headers expose anonymized subject ID and sex, but no age/birthday field
was available through MNE for this subset. Therefore, exact age matching is not
possible from the downloaded EDF headers alone.

## Interpretation

This subset is suitable for the first TUAB implementation pass because it:

```text
preserves the official TUAB train/eval split
samples by subject
balances normal/abnormal labels within train and eval
avoids train subjects that appear under both labels
has reproducible subject and file manifests
keeps download size modest at about 4.8G
```

The main limitation is age. The README confirms that TUAB has a known age
imbalance between normal and abnormal EEGs, but this release does not expose
per-subject age in the EDF headers we checked. We can still run the first TUAB
audit and report this limitation. If we later obtain per-subject demographics
from NEDC/TUH or another metadata source, we should build an age-matched subset
as a follow-up control.

## Key Files

Local manifests:

```text
results/tuab_subset_200/tuab_v3_0_1_random_stratified_200_subjects.csv
results/tuab_subset_200/tuab_v3_0_1_random_stratified_200_files.csv
results/tuab_subset_200/tuab_v3_0_1_random_stratified_200_files_from.txt
results/tuab_subset_200/tuab_v3_0_1_random_stratified_200_summary.json
```

Local metadata:

```text
results/tuab_subset_200/tuab_v3_0_1_random_stratified_200_header_metadata_subjects.csv
results/tuab_subset_200/tuab_v3_0_1_random_stratified_200_header_metadata_summary.json
```

Remote data:

```text
/mnt/data/aperiodic_confounds/data/tuab/v3.0.1_random_stratified_200
```

Code:

```text
code/scripts/make_tuab_subset_manifest.py
code/scripts/extract_tuab_header_metadata.py
```

## Preprocessing Cache

Implemented on May 24, 2026.

Code added:

```text
code/src/aperiodic_eeg/tuab.py
code/scripts/audit_tuab_channels.py
code/scripts/make_tuab_epochs.py
code/scripts/extract_tuab_raw_epochs.py
code/scripts/extract_tuab_psd.py
code/scripts/fit_tuab_specparam_qc.py
```

Channel harmonization:

```text
selected EDF files audited: 247
files with full standard 21-channel EEG set: 247
skipped files: 0
sampling rates observed: 250 Hz, 256 Hz, 512 Hz
```

Standard channel set:

```text
FP1 FP2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 A1 A2 FZ CZ PZ
```

Epoching:

```text
window length: 20 s
stride: 20 s
total epochs: 16458
eval abnormal epochs: 3663
eval normal epochs: 2565
train abnormal epochs: 5808
train normal epochs: 4422
```

Remote preprocessing outputs:

```text
/mnt/data/aperiodic_confounds/results/tuab_subset_200/raw_epochs_20s_100hz.npz
/mnt/data/aperiodic_confounds/results/tuab_subset_200/raw_epochs_20s_100hz_index.csv
/mnt/data/aperiodic_confounds/results/tuab_subset_200/psd_20s_multitaper.npz
/mnt/data/aperiodic_confounds/results/tuab_subset_200/psd_20s_multitaper_index.csv
```

Raw cache:

```text
shape: 16458 x 21 x 2000
target sampling rate: 100 Hz
filter: 1-45 Hz
scale: microvolts
remote size: 2.6G
```

PSD cache:

```text
shape: 16458 x 21 x 881
method: multitaper
bandwidth: 2 Hz
frequency range: 1-45 Hz
remote size: 1.1G
```

Specparam fit-quality QC:

```text
sample: 1000 epochs stratified equally across train/eval and normal/abnormal
spectra fit: 21000
ok fraction: 1.000
median R^2: 0.971
mean R^2: 0.951
median MAE: 0.076
mean exponent: 1.552
median exponent: 1.573
```

By-group QC:

| Split | Label | Mean exponent | Median R^2 |
| --- | --- | ---: | ---: |
| eval | abnormal | 1.686 | 0.976 |
| eval | normal | 1.494 | 0.968 |
| train | abnormal | 1.662 | 0.978 |
| train | normal | 1.365 | 0.965 |

Interpretation:

The TUAB subset is now ready for first-domain-transfer baselines and
intervention experiments. The channel audit confirms that the full selected
subset can be harmonized to a common 21-channel referential EEG representation.
The specparam QC indicates that 20-second TUAB windows are long enough for
stable 1-45 Hz fixed-mode aperiodic fits. The abnormal rows have steeper mean
aperiodic exponents than normal rows in the QC sample, which is scientifically
interesting but not yet causal: it could reflect pathology, age imbalance,
medication/state effects, or other TUAB-specific confounds.

## First PSD Baseline

We ran the first TUAB normal-vs-abnormal PSD intervention baseline on the
official train/eval split.

Run:

```text
code/scripts/run_tuab_psd_interventions.py
```

Output:

```text
/mnt/data/aperiodic_confounds/results/tuab_subset_200/psd_interventions_fixed
```

Setup:

```text
decomposition: fast fixed 1/f log-log fit
classifier: balanced ridge classifier
train subjects: 120
eval subjects: 80
train epochs: 10230
eval epochs: 6228
bootstrap: 10000 stratified eval-subject resamples
```

Balanced accuracy on official eval:

| Train input | Test input | Balanced accuracy | 95% CI |
| --- | --- | ---: | --- |
| full PSD | full PSD | 0.656 | 0.590-0.722 |
| full PSD | aperiodic only | 0.530 | 0.481-0.568 |
| full PSD | residual PSD | 0.551 | 0.513-0.586 |
| aperiodic only | aperiodic only | 0.722 | 0.645-0.797 |
| residual PSD | residual PSD | 0.625 | 0.575-0.676 |

Interpretation:

The first TUAB baseline suggests that TUAB normal-vs-abnormal classification
has a strong aperiodic component. A model trained directly on the aperiodic
spectrum outperforms the full-PSD ridge baseline, while the residual/flattened
PSD still remains above chance. In plain terms: TUAB is not behaving like
PhysioNet MI. The classifier is not only using oscillatory peaks; the broad
1/f spectral shape seems highly predictive.

This is promising for the paper because TUAB can become the clinical-domain
case where aperiodic structure may be central to reported benchmark
performance. It is also risky: without age metadata, we cannot yet tell whether
this reflects pathology itself or age/clinical-state confounding.

## Full-Specparam PSD Baseline

We repeated the TUAB PSD baseline using a full specparam fixed-mode
decomposition artifact.

Decomposition:

```text
remote artifact: /mnt/data/aperiodic_confounds/results/tuab_subset_200/specparam/specparam_fixed_20s.npz
shape: 16458 x 21 x 881
spectra fit: 345618
ok fraction: 1.000
median R^2: 0.972
mean R^2: 0.953
p10 R^2: 0.920
median exponent: 1.572
mean exponent: 1.557
```

Baseline output:

```text
/mnt/data/aperiodic_confounds/results/tuab_subset_200/psd_interventions_specparam
```

Balanced accuracy on official eval:

| Train input | Test input | Balanced accuracy | 95% CI |
| --- | --- | ---: | --- |
| full PSD | full PSD | 0.657 | 0.590-0.722 |
| full PSD | aperiodic only | 0.530 | 0.484-0.569 |
| full PSD | residual PSD | 0.570 | 0.531-0.608 |
| aperiodic only | aperiodic only | 0.676 | 0.596-0.756 |
| residual PSD | residual PSD | 0.631 | 0.578-0.684 |

Interpretation:

The full-specparam result confirms the fast fixed-fit result. TUAB
normal-vs-abnormal classification contains a strong aperiodic component, and
the residual/flattened PSD also remains above chance. The exact aperiodic-only
score is lower under specparam than under the simple fixed fit, but the
qualitative conclusion is unchanged: TUAB is a clinical domain where broad
aperiodic structure appears highly predictive and must be audited carefully.

## Raw EEGNet Intervention

We trained Braindecode EEGNet on raw TUAB EEG using the official train/eval
split, then evaluated phase-preserving Fourier interventions on the eval set.

Run:

```text
code/scripts/run_tuab_braindecode_eegnet_intervention.py
```

Output:

```text
/mnt/data/aperiodic_confounds/results/tuab_subset_200/braindecode_eegnet_interventions_specparam
```

Setup:

```text
train subjects: 120
eval subjects: 80
train epochs: 10230
eval epochs: 6228
model: Braindecode EEGNet
device: NVIDIA H200
seed: 42
best epoch: 24
early stopping epoch: 36
intervention band: 1-45 Hz
RMS matched: yes
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

The TUAB aperiodic finding survives in raw EEGNet. The sham control has no
effect, so the intervention pipeline itself is not responsible for the
performance drop. Preserving the aperiodic amplitude shape preserves nearly all
raw EEGNet performance, while flattening the aperiodic shape causes a large
drop. This strengthens the claim that aperiodic structure is not merely a PSD
baseline artifact; a standard raw EEG deep model also uses it.

Current caution:

This is a single-seed EEGNet result. The next robustness step is multi-seed
TUAB neural evaluation, then ShallowConvNet and Deep4Net on the same official
split.

## Multi-Seed Neural Robustness

We ran the TUAB neural intervention suite across three seeds and the classifier
family used in the Sleep-EDF neural experiments.

Run:

```text
code/scripts/launch_tuab_multiseed_neural.sh
```

Outputs:

```text
/mnt/data/aperiodic_confounds/results/tuab_subset_200/multiseed_neural
/mnt/data/aperiodic_confounds/reports/tables/tuab_multiseed_neural_subject_bootstrap.csv
/mnt/data/aperiodic_confounds/reports/tables/tuab_multiseed_neural_subject_bootstrap.md
```

Setup:

```text
seeds: 42, 43, 44
eval subjects: 80
bootstrap: hierarchical seed/subject bootstrap
models: deep MLP, raw CNN, EEGNet, ShallowFBCSPNet, Deep4Net
```

Balanced accuracy:

| Model | Baseline | Aperiodic | Flattened | Flattening drop |
| --- | ---: | ---: | ---: | ---: |
| deep MLP | 0.739 | 0.560 | 0.607 | 0.131 |
| raw CNN | 0.782 | 0.717 | 0.614 | 0.168 |
| EEGNet | 0.763 | 0.702 | 0.605 | 0.159 |
| ShallowFBCSP | 0.754 | 0.702 | 0.589 | 0.165 |
| Deep4Net | 0.759 | 0.713 | 0.529 | 0.231 |

Flattening-drop 95% CIs:

```text
deep MLP: 0.035-0.231
raw CNN: 0.082-0.256
EEGNet: 0.062-0.261
ShallowFBCSP: 0.075-0.258
Deep4Net: 0.140-0.323
```

Interpretation:

The TUAB aperiodic dependence is robust across seeds and architectures. Every
model shows a positive flattening drop with a 95% CI above zero. For raw neural
models, the sham intervention has exactly zero drop, so the Fourier edit
pipeline itself is not causing the performance loss. Aperiodic-shaped raw EEG
keeps high performance, while flattening the aperiodic shape consistently
hurts performance.

This result substantially strengthens the paper narrative: TUAB is not just a
PSD/ridge artifact and not just an EEGNet artifact. Multiple raw EEG deep
architectures rely on broad aperiodic spectral structure for clinical
normal-vs-abnormal classification.

## Age/Sex-Matched Raw Neural Control

Implemented on May 24, 2026.

Purpose:

The age/sex-matched PSD control showed that aperiodic information remains after
matching, but reviewers could still ask whether raw neural models show the same
age-controlled pattern. We therefore reran the raw neural intervention family
on the age/sex-matched TUAB subset.

Run:

```text
code/scripts/launch_tuab_age_matched_multiseed_neural.sh
```

Inputs:

```text
subject filter: results/tuab_subset_200/age_matched/tuab_age_sex_matched_caliper5_subjects.csv
raw cache: results/tuab_subset_200/raw_epochs_20s_100hz.npz
specparam: results/tuab_subset_200/specparam/specparam_fixed_20s.npz
```

Setup:

```text
train pairs: 34
eval pairs: 27
train subjects: 68
eval subjects: 54
matched raw epochs: 10063
train epochs: 5846
eval epochs: 4217
seeds: 42, 43, 44
models: EEGNet, ShallowFBCSPNet, Deep4Net
bootstrap: hierarchical seed/subject bootstrap
```

Outputs:

```text
/mnt/data/aperiodic_confounds/results/tuab_subset_200/age_matched/multiseed_neural
/mnt/data/aperiodic_confounds/reports/tables/tuab_age_matched_multiseed_neural_subject_bootstrap.csv
/mnt/data/aperiodic_confounds/reports/tables/tuab_age_matched_multiseed_neural_subject_bootstrap.md
```

Balanced accuracy:

| Model | Baseline | Aperiodic | Flattened | Flattening drop |
| --- | ---: | ---: | ---: | ---: |
| EEGNet | 0.710 [0.615, 0.799] | 0.641 [0.539, 0.739] | 0.563 [0.448, 0.674] | 0.147 [0.057, 0.245] |
| ShallowFBCSPNet | 0.719 [0.628, 0.807] | 0.675 [0.574, 0.772] | 0.558 [0.445, 0.669] | 0.161 [0.076, 0.250] |
| Deep4Net | 0.717 [0.619, 0.807] | 0.686 [0.581, 0.782] | 0.509 [0.394, 0.625] | 0.209 [0.118, 0.303] |

Interpretation:

The age/sex-matched neural control is strong. Every raw neural model retains
good raw performance on the matched subset, and every model shows a positive
flattening drop with a confidence interval above zero. Sham performance is
unchanged, so the result is not caused by the Fourier reconstruction procedure.

This directly addresses the age-confounding concern for raw neural models. Age
matters and should remain in the discussion, but the aperiodic dependence does
not disappear after age/sex matching. The safest conclusion is that TUAB
contains clinically useful aperiodic structure beyond a simple age shortcut,
while demographic confounding still needs explicit reporting.
