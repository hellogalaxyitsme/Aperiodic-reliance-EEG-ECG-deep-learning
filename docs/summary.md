# Project Summary

## Project

Title:

```text
The Aperiodic Confound in EEG Deep Learning
```

Core idea:

EEG deep-learning models may achieve part of their performance by using
aperiodic 1/f-like spectral structure rather than only the rhythmic oscillatory
features that are often emphasized in interpretation. This does not mean
aperiodic activity is meaningless. It means that model performance and
scientific interpretation need to distinguish broadband aperiodic structure
from narrowband oscillatory peaks.

## Why We Started With Sleep-EDF

We chose the Sleep-EDF Expanded Sleep-Cassette subset as the first MVP dataset.
It is public, small enough for fast iteration, and much cleaner for a first
proof of concept than larger or more heterogeneous datasets.

Dataset used:

```text
10 subjects
20 nights/recordings
21039 labeled 30-second epochs
channels: Fpz-Cz and Pz-Oz
```

Remote storage:

```text
/mnt/data/aperiodic_confounds/data/sleep-edf/sleep-cassette
```

Nothing was deleted from shared cluster storage.

## What We Built

We created a reproducible local-to-H200 workflow:

1. Write code locally on the Mac.
2. Sync code to H200.
3. Keep datasets and heavy outputs on H200.
4. Pull back only small result tables and figures.

We implemented:

- Sleep-EDF verification.
- Epoch manifest creation.
- Welch PSD extraction.
- Fixed 1/f decomposition.
- Specparam decomposition.
- Classical ridge baselines.
- Matched controls.
- Specparam sensitivity analysis.
- Intervention experiments on spectral features.
- Deep MLP PSD baselines.
- Raw EEG cache generation.
- Custom raw CNN baseline.
- Custom raw CNN aperiodic intervention.
- Braindecode EEGNet aperiodic intervention.
- Bootstrap confidence intervals.
- Paper-style tables and figures.
- Full Sleep-EDF multiseed subject-level analysis.
- Sham Fourier intervention controls for raw EEG models.
- PSD train-on-flattened controls.
- Braindecode architecture expansion:
  EEGNet, ShallowFBCSPNet, and Deep4Net.
- Raw intervention distribution diagnostics.
- Synthetic simulation validation.
- PhysioNet Motor Movement/Imagery second-domain setup.
- PhysioNet MI short-window multitaper PSD and specparam fit-quality audit.
- First PhysioNet MI subject-held-out ridge baselines.
- PhysioNet MI multiseed neural interventions with EEGNet,
  ShallowFBCSPNet, and Deep4Net.
- TUAB 200-subject abnormal-vs-normal audit with PSD, raw neural,
  age/sex-matched controls, and foundation-model checks.
- Full TUAB v3.0.1 preprocessing and intervention runs for PSD ridge,
  EEGNet, ShallowFBCSPNet, Deep4Net, BIOT, LaBraM, EEGPT, CBraMod,
  REVE-base, EEGMamba, and BENDR.

## Main Experimental Logic

For PSD models, we compared:

```text
full_log_psd
aperiodic_spectrum
flattened_log_psd
```

For raw EEG models, we trained on original raw EEG and tested on:

```text
raw_eeg
phase_sham
phase_aperiodic
phase_flattened
```

The raw intervention is phase-preserving:

- It keeps the Fourier phase of the held-out raw signal.
- It edits the Fourier amplitude envelope in the 1-45 Hz range.
- `phase_sham` passes through the same FFT/edit/reconstruction path while
  preserving the original spectral shape. It tests whether the intervention
  pipeline itself damages model performance.
- `phase_aperiodic` keeps a smoothed aperiodic amplitude shape.
- `phase_flattened` divides out the centered aperiodic envelope.
- RMS is matched back to the original epoch, so the result is not just a gross
  amplitude scaling artifact.

This lets us ask:

1. Is aperiodic structure sufficient for high performance?
2. Does removing/flattening aperiodic structure hurt a model trained normally?

## Key Results

The strongest current evidence now comes from two full-scale validations:
full Sleep-EDF for the sleep-domain claim and full TUAB for the clinical EEG
claim. Sleep-EDF shows large aperiodic dependence for arousal and five-stage
sleep staging, with an important N2-vs-N3 exception. Full TUAB confirms that
aperiodic reliance is present not only in standard EEG neural networks but also
in multiple EEG foundation models. PhysioNet MI remains the task-specific
contrast case, with much weaker flattening effects.

Full Sleep-EDF uses all available Sleep-Cassette subjects in our copy of
Sleep-EDF, three random seeds and subject-level hierarchical bootstrap
confidence intervals.

Full Sleep-EDF multiseed balanced accuracy:

| Model | Task | Original BA | Aperiodic BA | Flattened BA | Flattening drop |
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

Full Sleep-EDF report files:

```text
reports/tables/full_sleep_edf_multiseed_subject_bootstrap.md
reports/tables/full_sleep_edf_multiseed_subject_bootstrap.csv
```

## Post-Pilot Reviewer-Resistance Runs

After the original 10-subject Sleep-EDF pilot, we ran a larger set of controls
to address the main reviewer concerns: statistical power, raw-intervention
artifacts, PSD flattening interpretation, architecture specificity, and
pipeline validity.

### 1. Full Sleep-EDF scaling

The dataset was expanded from the original 10-subject pilot to all available
Sleep-EDF Sleep-Cassette subjects in the project storage:

```text
78 unique subjects
153 PSG/Hypnogram recording pairs
195469 labeled 30-second epochs
channels: Fpz-Cz and Pz-Oz
seeds: 42, 43, 44
uncertainty: hierarchical bootstrap over seed and subject
```

This addressed the first major critique: the original fold-bootstrap over five
folds was too weak for a serious paper. The current primary Sleep-EDF neural
numbers use subject-level multiseed uncertainty.

### 2. Sham raw-intervention control

The sham control answers a simple question:

> Does the Fourier editing pipeline itself damage the signal?

The answer was no. Sham performance was essentially identical to raw
performance for all raw EEG models. This means the large flattening drops are
not explained by the mechanics of FFT reconstruction.

Balanced accuracy, subject-level hierarchical bootstrap:

| Model | Task | Raw | Sham | Aperiodic | Flattened | Flattening drop |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Raw CNN | Wake vs Sleep | 0.940 [0.932, 0.948] | 0.940 [0.932, 0.948] | 0.871 [0.853, 0.889] | 0.511 [0.505, 0.520] | 0.429 [0.417, 0.439] |
| Raw CNN | N2 vs N3 | 0.877 [0.852, 0.901] | 0.877 [0.852, 0.901] | 0.873 [0.854, 0.890] | 0.875 [0.858, 0.891] | 0.002 [-0.021, 0.023] |
| Raw CNN | Five-stage | 0.736 [0.718, 0.753] | 0.736 [0.718, 0.753] | 0.648 [0.628, 0.666] | 0.281 [0.259, 0.303] | 0.455 [0.427, 0.484] |
| EEGNet | Wake vs Sleep | 0.939 [0.929, 0.948] | 0.939 [0.929, 0.948] | 0.879 [0.860, 0.897] | 0.502 [0.500, 0.505] | 0.437 [0.427, 0.446] |
| EEGNet | N2 vs N3 | 0.881 [0.854, 0.903] | 0.881 [0.854, 0.903] | 0.863 [0.844, 0.881] | 0.882 [0.867, 0.897] | -0.001 [-0.024, 0.019] |
| EEGNet | Five-stage | 0.706 [0.686, 0.725] | 0.706 [0.686, 0.725] | 0.608 [0.588, 0.626] | 0.311 [0.293, 0.329] | 0.395 [0.377, 0.413] |

The key result is the sham column. It is equal to raw within numerical noise.
So the raw intervention pipeline is not creating an artificial failure mode by
itself.

### 3. Architecture expansion

We added two standard Braindecode architectures beyond EEGNet:

```text
ShallowFBCSPNet
Deep4Net
```

These matter because ShallowFBCSPNet behaves like a learned band-power model,
while Deep4Net has a deeper temporal-spatial convolution hierarchy. Both
replicated the same qualitative pattern.

Balanced accuracy:

| Model | Task | Raw | Sham | Aperiodic | Flattened | Flattening drop |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ShallowFBCSPNet | Wake vs Sleep | 0.931 [0.920, 0.941] | 0.931 [0.920, 0.941] | 0.858 [0.833, 0.882] | 0.511 [0.505, 0.518] | 0.420 [0.406, 0.432] |
| ShallowFBCSPNet | N2 vs N3 | 0.871 [0.846, 0.894] | 0.871 [0.846, 0.894] | 0.858 [0.840, 0.875] | 0.876 [0.857, 0.894] | -0.005 [-0.023, 0.011] |
| ShallowFBCSPNet | Five-stage | 0.689 [0.665, 0.711] | 0.689 [0.665, 0.711] | 0.593 [0.570, 0.615] | 0.317 [0.294, 0.341] | 0.372 [0.341, 0.402] |
| Deep4Net | Wake vs Sleep | 0.941 [0.931, 0.948] | 0.941 [0.931, 0.948] | 0.890 [0.872, 0.907] | 0.506 [0.503, 0.510] | 0.434 [0.425, 0.443] |
| Deep4Net | N2 vs N3 | 0.856 [0.826, 0.882] | 0.856 [0.826, 0.882] | 0.861 [0.841, 0.880] | 0.865 [0.847, 0.882] | -0.010 [-0.036, 0.013] |
| Deep4Net | Five-stage | 0.719 [0.699, 0.738] | 0.719 [0.699, 0.738] | 0.639 [0.620, 0.657] | 0.299 [0.272, 0.329] | 0.420 [0.391, 0.448] |

This makes the claim less dependent on one architecture. EEGNet,
ShallowFBCSPNet, Deep4Net, and our custom raw CNN all show:

```text
large flattening drop for Wake vs Sleep
large flattening drop for Five-stage
near-zero flattening drop for N2 vs N3
near-zero sham drop
```

### 4. PSD train-on-flattened controls

This was an important nuance. Earlier PSD intervention results showed that a
model trained on full PSD collapses when tested on flattened PSD. That could be
misread as "flattened PSD has no useful information."

We therefore trained models directly on each representation:

```text
full_log_psd
aperiodic_spectrum
flattened_log_psd
```

Balanced accuracy:

| Task | Full-trained | Aperiodic-trained | Flattened-trained |
| --- | ---: | ---: | ---: |
| Wake vs Sleep | 0.929 [0.918, 0.939] | 0.893 [0.876, 0.908] | 0.913 [0.900, 0.925] |
| N2 vs N3 | 0.850 [0.824, 0.874] | 0.759 [0.727, 0.790] | 0.745 [0.720, 0.768] |
| Five-stage | 0.694 [0.674, 0.712] | 0.572 [0.549, 0.595] | 0.633 [0.614, 0.652] |

Interpretation:

The flattened PSD still contains useful information if the model is trained for
it. Therefore, the PSD test-time collapse should be framed as:

> Models trained on full PSD rely heavily on the dominant aperiodic-shaped
> variance and fail under a large test-time representation shift.

It should not be framed as:

> There is no oscillatory or residual information after flattening.

This makes the paper more precise and harder to attack.

### 5. Raw intervention distribution diagnostics

We computed time-domain diagnostics for raw, sham, aperiodic-only, and
flattened raw interventions. The most important checks were:

```text
phase_sham rmse_vs_raw: 0.0000
phase_sham corr_vs_raw: 1.0000
phase_sham RMS/std/peak-to-peak/kurtosis: identical to raw
```

This confirms that the sham pipeline is an exact no-op control. The edited
aperiodic and flattened signals preserve RMS by construction but change the
waveform shape, as intended. Because sham has no performance drop, the main
flattening drops cannot be blamed on FFT round-trip artifacts.

### 6. Simulation validation

We added synthetic PSD simulations with known ground truth:

```text
aperiodic_only
oscillatory_only
mixed
train_confound_test_unconfounded
```

The simulation behaved as expected:

```text
aperiodic_only:
  full and aperiodic inputs classified well; flattened inputs were chance.

oscillatory_only:
  full and flattened inputs classified well; aperiodic-only inputs were chance.

mixed:
  both aperiodic and flattened/residual inputs carried information.

train_confound_test_unconfounded:
  the pipeline could expose when a training-time aperiodic shortcut no longer
  matched the test-time signal structure.
```

This does not prove the real EEG decomposition is perfect, but it validates
that the audit logic behaves sensibly when the ground truth is known.

Post-pilot reviewer-control files:

```text
reports/tables/sleep_edf_reviewer_resistance_bootstrap.md
reports/tables/sleep_edf_reviewer_resistance_bootstrap.csv
reports/tables/raw_intervention_diagnostics/raw_intervention_distribution_diagnostics.md
reports/tables/raw_intervention_diagnostics/raw_intervention_distribution_diagnostics.csv
reports/tables/aperiodic_simulation_validation/simulation_validation_metrics.md
reports/tables/aperiodic_simulation_validation/simulation_validation_metrics.csv
```

## Second Domain: PhysioNet Motor Imagery

We added PhysioNet EEG Motor Movement/Imagery as the second public dataset.
This matters because it is not sleep staging: it is short cue-locked task EEG
with 109 subjects and 64 channels. The first task is imagined left fist versus
imagined right fist using runs 4, 8, and 12.

Dataset and preprocessing:

```text
dataset: PhysioNet EEG Motor Movement/Imagery
subjects: 109
EDF files: 1526
event files: 1526
trial window: 0.5-4.0 s after cue onset
trials: 4917
condition counts: left_fist=2479, right_fist=2438
raw shape: (4917, 64, 560)
PSD method: multitaper
PSD shape: (4917, 64, 151)
frequency range: 2.00-44.86 Hz
```

The important methodological issue was whether aperiodic fitting would be too
unstable on 3.5-second motor-imagery trials. The first full fit says the setup
is usable:

```text
spectra fit: 314688
ok_fraction: 1.0
mean R2: 0.9520
median R2: 0.9705
10th percentile R2: 0.9066
mean MAE: 0.0771
```

First subject-held-out ridge baseline, balanced accuracy:

| Feature set | Mean | Std |
| --- | ---: | ---: |
| Full log-PSD | 0.565 | 0.018 |
| Aperiodic spectrum | 0.526 | 0.011 |
| Aperiodic params | 0.530 | 0.015 |
| Periodic residual | 0.532 | 0.021 |

Interpretation:

This is not a final motor-imagery benchmark. It is a first cross-subject sanity
check. The result is useful because it does not simply repeat Sleep-EDF. Full
PSD is modestly above chance, while aperiodic-only and residual-only features
are closer to chance. This suggests that motor imagery may have a weaker or
different aperiodic dependence profile than sleep staging. That is good for the
paper: it lets us argue that the audit reveals task-specific reliance patterns
rather than forcing every dataset into the same story.

We then ran the first PhysioNet MI PSD intervention and train-on-representation
control. This used 5-fold subject-held-out ridge models and subject-bootstrap
confidence intervals over all 109 subjects.

Balanced accuracy:

| Train input | Test input | BA [95% CI] |
| --- | --- | ---: |
| Full log-PSD | Full log-PSD | 0.566 [0.550, 0.581] |
| Full log-PSD | Aperiodic spectrum | 0.518 [0.508, 0.530] |
| Full log-PSD | Flattened log-PSD | 0.556 [0.540, 0.572] |
| Aperiodic spectrum | Aperiodic spectrum | 0.527 [0.516, 0.538] |
| Flattened log-PSD | Flattened log-PSD | 0.531 [0.516, 0.547] |

Paired full-trained intervention drops:

```text
aperiodic-only test drop: 0.047 [0.030, 0.065]
flattened test drop:      0.009 [-0.003, 0.022]
```

Interpretation:

This is the first clear second-domain contrast. In Sleep-EDF, flattening the
aperiodic envelope produced large collapses for wake-vs-sleep and five-stage
sleep staging. In PhysioNet MI left-vs-right imagery, flattening barely changes
the full-PSD ridge model, and the CI for the flattening drop includes zero.
Aperiodic-only input does worse, so the model is not simply using the
aperiodic spectrum as a sufficient shortcut.

The train-on-flattened control shows that flattened spectral features retain
weak but above-chance information. So the current MI story is:

```text
Sleep-EDF: strong aperiodic dependence for several tasks.
PhysioNet MI left-vs-right imagery: much weaker aperiodic dependence; residual
spectral structure remains usable.
```

This strengthens the paper by making the claim more precise. The project is
not arguing that every EEG model always relies on aperiodic structure. It is
arguing that aperiodic reliance is measurable, can be large, and varies by
task/domain.

We also ran raw neural interventions on PhysioNet MI using Braindecode EEGNet,
ShallowFBCSPNet, and Deep4Net across three random seeds.

Hierarchical seed/subject bootstrap, balanced accuracy:

| Model | Raw | Sham | Aperiodic | Flattened | Flattening drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| EEGNet | 0.744 [0.720, 0.767] | 0.744 [0.720, 0.767] | 0.737 [0.714, 0.760] | 0.739 [0.717, 0.761] | 0.005 [-0.002, 0.013] |
| ShallowFBCSPNet | 0.655 [0.634, 0.678] | 0.655 [0.634, 0.678] | 0.617 [0.597, 0.639] | 0.641 [0.618, 0.665] | 0.014 [0.005, 0.024] |
| Deep4Net | 0.682 [0.661, 0.703] | 0.682 [0.661, 0.703] | 0.684 [0.663, 0.705] | 0.692 [0.669, 0.715] | -0.010 [-0.020, 0.001] |

This confirms the PSD result in raw neural models. The sham control is exactly
equal to raw, so the Fourier intervention procedure itself is not damaging the
signal. Flattening causes only tiny changes: EEGNet is essentially unchanged,
ShallowFBCSPNet drops by only about 0.014 balanced accuracy, and Deep4Net is
slightly better under flattening. Therefore, for left-vs-right motor imagery,
the core Braindecode convolutional family is largely robust to aperiodic
flattening.

This is a strong contrast with Sleep-EDF and is scientifically useful. It shows
that our method is not just a machine for producing collapse. It can also show
when a task/model is relatively robust to aperiodic removal.

PhysioNet MI files:

```text
results/physionet_mi/baselines_specparam/summary_metrics.csv
results/physionet_mi/baselines_specparam/fold_metrics.csv
results/physionet_mi/specparam/imagined_fists_specparam_fixed.summary.json
results/physionet_mi/psd_interventions/psd_intervention_subject_bootstrap.md
results/physionet_mi/psd_interventions/psd_intervention_subject_bootstrap.csv
results/physionet_mi/raw_braindecode_interventions/raw_braindecode_subject_bootstrap.md
results/physionet_mi/raw_braindecode_interventions/raw_braindecode_subject_bootstrap.csv
results/physionet_mi/multiseed_neural
reports/tables/physionet_mi_multiseed_neural_subject_bootstrap.md
```

The earlier 10-subject MVP results are kept below because they document the
development path and show that the same qualitative pattern appeared before
scaling to the full cohort.

Balanced accuracy with 95% fold-bootstrap confidence intervals:

| Model | Task | Original BA | Aperiodic BA | Flattened BA | Flattening drop |
| --- | --- | ---: | ---: | ---: | ---: |
| Linear ridge PSD | Wake vs Sleep | 0.940 [0.923, 0.952] | 0.798 [0.757, 0.836] | 0.500 [0.500, 0.500] | 0.440 [0.423, 0.452] |
| Linear ridge PSD | N2 vs N3 | 0.895 [0.858, 0.926] | 0.814 [0.733, 0.890] | 0.500 [0.500, 0.500] | 0.395 [0.358, 0.426] |
| Linear ridge PSD | Five-stage | 0.710 [0.679, 0.734] | 0.499 [0.480, 0.515] | 0.200 [0.200, 0.200] | 0.510 [0.479, 0.534] |
| Deep MLP PSD | Wake vs Sleep | 0.946 [0.939, 0.955] | 0.812 [0.762, 0.856] | 0.500 [0.500, 0.500] | 0.446 [0.439, 0.455] |
| Deep MLP PSD | N2 vs N3 | 0.903 [0.872, 0.929] | 0.842 [0.785, 0.883] | 0.501 [0.500, 0.502] | 0.402 [0.370, 0.429] |
| Deep MLP PSD | Five-stage | 0.736 [0.705, 0.769] | 0.576 [0.527, 0.628] | 0.200 [0.200, 0.200] | 0.536 [0.505, 0.569] |
| Custom raw CNN | Wake vs Sleep | 0.957 [0.944, 0.971] | 0.893 [0.865, 0.927] | 0.538 [0.502, 0.594] | 0.420 [0.363, 0.458] |
| Custom raw CNN | N2 vs N3 | 0.936 [0.914, 0.955] | 0.902 [0.835, 0.945] | 0.935 [0.922, 0.948] | 0.001 [-0.010, 0.009] |
| Custom raw CNN | Five-stage | 0.775 [0.738, 0.812] | 0.666 [0.639, 0.710] | 0.346 [0.321, 0.370] | 0.429 [0.410, 0.452] |
| Braindecode EEGNet | Wake vs Sleep | 0.957 [0.945, 0.967] | 0.894 [0.862, 0.927] | 0.576 [0.538, 0.625] | 0.381 [0.325, 0.429] |
| Braindecode EEGNet | N2 vs N3 | 0.927 [0.896, 0.946] | 0.886 [0.804, 0.937] | 0.893 [0.807, 0.942] | 0.033 [-0.002, 0.091] |
| Braindecode EEGNet | Five-stage | 0.752 [0.729, 0.775] | 0.623 [0.609, 0.637] | 0.468 [0.443, 0.493] | 0.285 [0.268, 0.301] |

The current paper-style figure is:

```text
reports/figures/combined_intervention_performance.png
reports/figures/combined_intervention_performance.pdf
```

The current combined result table is:

```text
reports/tables/combined_intervention_summary.md
reports/tables/combined_intervention_summary.csv
```

## Interpretation

### 1. The core hypothesis is supported

Across classical PSD models, deep PSD models, custom raw CNNs, Braindecode
EEGNet, ShallowFBCSPNet, and Deep4Net, aperiodic structure is highly
predictive.

This is clearest in two places:

- Wake vs Sleep
- Five-stage sleep staging

In both settings, the aperiodic-only signal preserves substantial performance,
and flattening the aperiodic envelope causes a large drop.

This supports the paper's main claim: EEG deep models can rely strongly on
broadband aperiodic structure, and this can be missed if results are interpreted
only through oscillatory band-power language.

### 2. The result is not a toy-model artifact

The evidence escalates across model complexity:

```text
linear ridge on PSD
deep MLP on PSD
custom raw EEG CNN
Braindecode EEGNet
Braindecode ShallowFBCSPNet
Braindecode Deep4Net
```

The Braindecode results are especially important because EEGNet,
ShallowFBCSPNet, and Deep4Net are recognized EEG architectures with different
inductive biases. This shows that the finding is not caused by our custom
network design.

### 2a. The raw intervention is not just an FFT artifact

The sham control is central. It passes the signal through the same
phase-preserving FFT reconstruction path while preserving the original spectral
shape. Sham performance was indistinguishable from raw performance across raw
CNN, EEGNet, ShallowFBCSPNet, and Deep4Net.

Therefore, when flattened inputs fail, the failure is tied to removing the
aperiodic envelope, not to the existence of an FFT reconstruction step.

### 3. Wake vs Sleep is strongly aperiodic-sensitive

For Braindecode EEGNet:

```text
Original: 0.939
Sham: 0.939
Aperiodic: 0.879
Flattened: 0.502
Flattening drop: 0.437 [0.427, 0.446]
```

This suggests the model is using broadband sleep-wake spectral structure. That
is scientifically plausible because vigilance and arousal are known to change
the broadband EEG background.

The key interpretive caution is that strong sleep-wake performance should not
automatically be described as discovering narrowband rhythms unless the
aperiodic component is controlled.

### 4. Five-stage sleep staging is also aperiodic-sensitive

For Braindecode EEGNet:

```text
Original: 0.706
Sham: 0.706
Aperiodic: 0.608
Flattened: 0.311
Flattening drop: 0.395 [0.377, 0.413]
```

This is important because five-stage classification is harder and more
realistic than a binary wake-sleep task. The result says that aperiodic
structure is not just a trivial wake detector; it contributes to broader sleep
stage discrimination.

### 5. N2 vs N3 is the important exception

For Braindecode EEGNet:

```text
Original: 0.881
Sham: 0.881
Aperiodic: 0.863
Flattened: 0.882
Flattening drop: -0.001 [-0.024, 0.019]
```

Flattening does not clearly damage N2-vs-N3 performance. The custom raw CNN
showed the same pattern even more strongly.

This strengthens the paper because it shows the method is not blindly claiming
that everything is aperiodic. Instead, the dependence is task-specific.

Likely explanation:

- N2-vs-N3 may use slow-wave morphology.
- It may use phase/time-domain shape.
- It may use residual oscillatory information.
- It may rely on features not fully disrupted by our current amplitude-envelope
  flattening.

This gives us a more nuanced and credible message.

### 6. PSD flattening needs careful wording

The new train-on-flattened control changed the interpretation of the PSD
collapse. A model trained on full PSD fails when tested on flattened PSD, but a
model trained directly on flattened PSD can still perform well.

So the right claim is:

> Full-PSD-trained models rely heavily on aperiodic-dominated spectral variance
> and are brittle when that structure is removed at test time.

The wrong claim would be:

> Flattened PSD contains no useful EEG information.

The latter is false in our controls.

## Current Claim We Can Make Safely

A careful version of the paper claim is:

> In Sleep-EDF sleep staging, aperiodic EEG spectral structure is sufficient to
> preserve substantial classification performance across linear, deep PSD, and
> raw EEG deep models. Removing or flattening the aperiodic envelope at test
> time causes large performance drops for Wake-vs-Sleep and five-stage staging
> across custom CNN, EEGNet, ShallowFBCSPNet, and Deep4Net, while sham
> reconstruction has no effect. However, N2-vs-N3 remains relatively robust to
> flattening, and models trained directly on flattened PSD can recover
> substantial residual information. The result is therefore a task-specific
> aperiodic shortcut and representation-dependence effect, not a claim that all
> non-aperiodic EEG information is absent.

This is strong because it is:

- testable,
- not overstated,
- supported by multiple model families,
- and honest about the N2-vs-N3 exception.

## Limitations

1. Dataset scope:

   We have now scaled from the 10-subject MVP to the full Sleep-EDF
   Sleep-Cassette cohort available in the project storage. The next dataset
   limitation is external replication, not Sleep-EDF sample size.

2. Confidence intervals:

   The strongest current neural CIs are subject-level hierarchical bootstrap
   intervals over random seed and subject. The older 10-subject MVP table used
   fold-bootstrap CIs and should now be treated as developmental evidence.

3. Raw intervention method:

   The raw intervention edits Fourier amplitudes while preserving phase. The
   sham control strongly reduces concern that FFT reconstruction alone causes
   the performance drop, but alternative spectral surgery methods would still
   strengthen the paper.

4. IRASA:

   We implemented and ran a stage-balanced Sleep-EDF IRASA-vs-SpecParam
   agreement check on 5,000 epochs, sampled as 1,000 epochs from each of
   W/N1/N2/N3/REM across 78 subjects and 153 recordings. SpecParam and IRASA
   agreed strongly on aperiodic spectral shape: median shape correlation was
   0.966, mean shape correlation was 0.936, and the 5th percentile was 0.812.
   This supports the claim that the Sleep-EDF decomposition conclusions are not
   specific to SpecParam alone. We corrected for the raw-cache microvolt scale
   before comparison because the Sleep-EDF PSD/SpecParam artifact is in EDF
   voltage units.

   We also ran the downstream IRASA ridge intervention check on the same
   stage-balanced 5,000-epoch sample. Ridge models trained on full IRASA PSD
   retained nearly all performance on IRASA aperiodic-only spectra, while
   IRASA-flattened PSDs collapsed strongly:

   | Task | Full PSD | IRASA aperiodic | IRASA flattened |
   | --- | ---: | ---: | ---: |
   | Wake-vs-Sleep | 0.876 [0.859, 0.893] | 0.867 [0.845, 0.888] | 0.500 [0.500, 0.500] |
   | N2-vs-N3 | 0.842 [0.813, 0.868] | 0.859 [0.832, 0.884] | 0.417 [0.378, 0.455] |
   | Five-stage | 0.615 [0.592, 0.637] | 0.612 [0.589, 0.634] | 0.210 [0.206, 0.215] |

   This is important because it shows the downstream classification conclusion
   is not specific to SpecParam. IRASA gives the same qualitative story:
   aperiodic-only spectra are highly predictive for Sleep-EDF, and removing the
   aperiodic envelope severely damages ridge classification.

5. Specparam settings:

   Sensitivity analysis showed stable classification patterns, but peak fitting
   diagnostics revealed frequent peak-cap hits. This should be discussed.

6. Sleep-specific physiology:

   Aperiodic structure may be physiologically meaningful in sleep. The paper
   should frame this as misattribution risk, not as "aperiodic equals artifact."

7. Architecture scope:

   EEGNet, ShallowFBCSPNet, and Deep4Net are now covered. USleep and
   EEGConformer are implemented in the launcher design but deferred to a later
   staged run.

8. TUAB age confounding:

   We confirmed that subject age is available in the TUAB EDF patient header as
   an `Age:` field, even though MNE did not expose it as a parsed birth-date
   field. In our 200-subject TUAB subset, usable age exists for 198 subjects;
   two train subjects have `Age:999`, treated as missing. The age imbalance is
   real: abnormal subjects average about 56.4 years, while normal subjects
   average about 43.3 years.

   We built a same-sex, +/-5-year age-matched subset within the official
   train/eval split: 34 train pairs and 27 eval pairs, 122 total subjects. The
   matched train means are 50.15 vs 49.29 years, and the matched eval means are
   51.59 vs 50.74 years for abnormal vs normal.

   On this matched subset, PSD ridge results still show aperiodic information:

   | Train input | Test input | Balanced accuracy |
   | --- | --- | ---: |
   | full PSD | full PSD | 0.635 [0.554, 0.713] |
   | full PSD | aperiodic | 0.541 [0.518, 0.572] |
   | full PSD | flattened | 0.544 [0.494, 0.588] |
   | aperiodic | aperiodic | 0.682 [0.598, 0.763] |
   | flattened | flattened | 0.616 [0.546, 0.684] |

   Interpretation: age/sex matching does not erase aperiodic predictive
   information, because a model trained and tested on aperiodic spectra remains
   strong. But the full-trained intervention result becomes more nuanced:
   full-to-aperiodic performance is only 0.541. The safest claim is that TUAB
   contains age-confounded aperiodic structure, but the matched control suggests
   aperiodic pathology-related information remains beyond age alone.

9. Full TUAB dataset scale-up:

   We scaled the TUAB abnormal-vs-normal audit from the 200-subject subset to
   the full TUAB v3.0.1 collection available on H200, preserving the official
   train/eval boundary. This is now the strongest TUAB evidence in the project.
   The full raw-neural and foundation-model runs evaluate 36,945 windows from
   253 official eval subjects, with subject-level bootstrap uncertainty.

   Full-TUAB PSD ridge intervention, balanced accuracy:

   | Train input | Test input | Balanced accuracy |
   | --- | --- | ---: |
   | full PSD | full PSD | 0.752 [0.708, 0.793] |
   | full PSD | aperiodic | 0.588 [0.555, 0.620] |
   | full PSD | flattened | 0.591 [0.566, 0.617] |
   | aperiodic | aperiodic | 0.711 [0.667, 0.754] |
   | flattened | flattened | 0.724 [0.683, 0.762] |

   Full-TUAB multiseed raw-neural intervention, balanced accuracy:

   | Model | Raw | Sham | Aperiodic-shaped | Flattened | Flattening drop |
   | --- | ---: | ---: | ---: | ---: | ---: |
   | EEGNet | 0.804 [0.765, 0.843] | 0.804 [0.765, 0.843] | 0.578 [0.491, 0.658] | 0.707 [0.655, 0.758] | 0.097 [0.050, 0.144] |
   | ShallowFBCSPNet | 0.796 [0.755, 0.836] | 0.796 [0.755, 0.836] | 0.575 [0.511, 0.639] | 0.727 [0.676, 0.774] | 0.070 [0.020, 0.119] |
   | Deep4Net | 0.816 [0.777, 0.854] | 0.816 [0.777, 0.854] | 0.574 [0.512, 0.635] | 0.687 [0.633, 0.739] | 0.129 [0.075, 0.184] |

   Interpretation: the full dataset confirms the TUAB-200 finding at a much
   stronger scale. Raw performance is high, sham controls are exactly neutral,
   and flattening produces positive drops for all three standard neural
   architectures. The drops are smaller than the original subset estimates,
   but they are now estimated across the full official TUAB eval set and three
   random seeds, making the claim more credible for the paper.

10. TUAB age/sex-matched raw neural control:

   We extended the TUAB age/sex-matched control beyond PSD ridge by running
   EEGNet, ShallowFBCSPNet, and Deep4Net on the matched subject subset. This is
   the direct response to the likely reviewer question: does the raw neural
   aperiodic dependence survive after controlling the TUAB age/sex imbalance?

   Matched subset:

   ```text
   train pairs: 34
   eval pairs: 27
   train subjects: 68
   eval subjects: 54
   matched raw epochs: 10,063
   train epochs: 5,846
   eval epochs: 4,217
   seeds: 42, 43, 44
   models: EEGNet, ShallowFBCSPNet, Deep4Net
   ```

   Hierarchical seed/subject bootstrap, balanced accuracy:

   | Model | Raw | Sham | Aperiodic | Flattened | Flattening drop |
   | --- | ---: | ---: | ---: | ---: | ---: |
   | EEGNet | 0.710 [0.615, 0.799] | 0.710 [0.615, 0.799] | 0.641 [0.539, 0.739] | 0.563 [0.448, 0.674] | 0.147 [0.057, 0.245] |
   | ShallowFBCSPNet | 0.719 [0.628, 0.807] | 0.719 [0.628, 0.807] | 0.675 [0.574, 0.772] | 0.558 [0.445, 0.669] | 0.161 [0.076, 0.250] |
   | Deep4Net | 0.717 [0.619, 0.807] | 0.717 [0.619, 0.807] | 0.686 [0.581, 0.782] | 0.509 [0.394, 0.625] | 0.209 [0.118, 0.303] |

   Interpretation: age/sex matching does not remove the TUAB raw-neural
   aperiodic dependence. All three models still show positive flattening drops
   with confidence intervals above zero. The sham control remains unchanged,
   so this is not caused by Fourier reconstruction. This is a major
   reviewer-resistance result: TUAB aperiodic reliance is not explained away by
   the age imbalance alone.

11. Full TUAB foundation-model panel:

   The foundation-model audit has now been scaled from TUAB-200 to full TUAB
   for BIOT, LaBraM, EEGPT, CBraMod, REVE-base, EEGMamba and BENDR. All runs
   use the same high-level logic: fine-tune on original TUAB train windows,
   evaluate the official eval set under raw, phase-sham, aperiodic-shaped and
   flattened inputs, and report subject-stratified bootstrap confidence
   intervals.

   Full-TUAB foundation-model intervention, balanced accuracy:

   | Model | Raw | Sham | Aperiodic-shaped | Flattened | Flattening drop |
   | --- | ---: | ---: | ---: | ---: | ---: |
   | BIOT | 0.799 [0.763, 0.834] | 0.798 [0.762, 0.833] | 0.696 [0.662, 0.729] | 0.682 [0.648, 0.715] | 0.118 [0.088, 0.146] |
   | LaBraM | 0.780 [0.741, 0.818] | 0.780 [0.740, 0.817] | 0.718 [0.684, 0.750] | 0.710 [0.670, 0.750] | 0.070 [0.044, 0.095] |
   | EEGPT | 0.796 [0.757, 0.833] | 0.795 [0.757, 0.832] | 0.665 [0.635, 0.695] | 0.730 [0.684, 0.774] | 0.067 [0.034, 0.099] |
   | CBraMod | 0.782 [0.745, 0.818] | 0.783 [0.745, 0.818] | 0.621 [0.593, 0.649] | 0.700 [0.660, 0.739] | 0.083 [0.056, 0.110] |
   | REVE-base | 0.790 [0.751, 0.827] | 0.790 [0.750, 0.826] | 0.687 [0.652, 0.719] | 0.734 [0.691, 0.775] | 0.056 [0.025, 0.087] |
   | EEGMamba | 0.773 [0.736, 0.809] | 0.773 [0.736, 0.809] | 0.678 [0.643, 0.713] | 0.708 [0.668, 0.746] | 0.066 [0.040, 0.091] |
   | BENDR | 0.744 [0.702, 0.786] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.244 [0.202, 0.286] |

   Implementation notes by model:

   - BIOT uses the official PREST-16 encoder, 16 bipolar TCP-style channels,
     200 Hz, 10-second windows and per-window/channel 95th percentile
     amplitude normalization.
   - LaBraM, EEGPT, REVE-base and BENDR use the full-TUAB 23-channel
     referential 10 s, 200 Hz cache, with model-specific input shaping.
   - CBraMod and EEGMamba derive the official 16-channel longitudinal bipolar
     montage from the same referential cache and apply their model-specific
     scaling.
   - REVE-base uses `brain-bzh/reve-base` plus `brain-bzh/reve-positions`;
     the position bank recognizes 21 of the 23 TUAB referential channels, so
     T1/T2 are dropped rather than assigned invented coordinates.

   Interpretation: the full-TUAB foundation-model evidence is now strong.
   BIOT, LaBraM, EEGPT, CBraMod, REVE-base and EEGMamba all show high raw
   performance, neutral sham controls and positive flattening drops with
   confidence intervals above zero. The magnitude varies by architecture:
   BIOT is most sensitive to flattening, REVE-base and EEGMamba are more
   moderate, and all models retain above-chance flattened performance. This is
   the right paper message: foundation models do not simply ignore aperiodic
   spectral structure, but they also do not rely on it exclusively.

   BENDR should be treated separately. Its raw full-TUAB performance is above
   chance, but the phase-sham condition collapses to chance, so the BENDR row
   is not a clean aperiodic-specific effect. It is useful as a cautionary
   transfer/control result, not as a main confirmatory foundation-model row.

## Recommended Next Steps

1. Build the consolidated full-TUAB paper table:

   - PSD ridge,
   - EEGNet,
   - ShallowFBCSPNet,
   - Deep4Net,
   - BIOT,
   - LaBraM,
   - EEGPT,
   - CBraMod,
   - REVE-base,
   - EEGMamba,
   - BENDR only as a cautionary/supplementary row because sham collapses.

2. Update the manuscript and figure generators so the paper uses the full-TUAB
   results instead of the older TUAB-200 foundation-model values.

3. Add feature-level interpretation:

   - how aperiodic offset/exponent varies by sleep stage,
   - whether age or recording night explains some of the aperiodic signal,
   - whether N2-vs-N3 robustness comes from slow-wave morphology.

4. Prepare a methods figure showing the intervention pipeline:

```text
raw EEG / PSD
-> specparam aperiodic fit
-> original vs aperiodic-only vs flattened input
-> same trained model evaluated under intervention
```

## Files To Show A Colleague

Start with:

```text
summary.md
reports/figures/combined_intervention_performance.png
reports/tables/combined_intervention_summary.md
reports/tables/full_sleep_edf_multiseed_subject_bootstrap.md
reports/tables/sleep_edf_reviewer_resistance_bootstrap.md
reports/tables/raw_intervention_diagnostics/raw_intervention_distribution_diagnostics.md
reports/tables/aperiodic_simulation_validation/simulation_validation_metrics.md
results/sleep_edf_full/irasa/irasa_ridge_interventions_stage_balanced_5k/irasa_ridge_intervention_subject_bootstrap.md
results/tuab_full_v3_0_1/psd_interventions_specparam/tuab_psd_intervention_subject_bootstrap.md
reports/tables/tuab_full_multiseed_neural_subject_bootstrap.md
results/tuab_subset_200/tuab_age_metadata_audit.md
results/tuab_subset_200/age_matched/tuab_age_sex_matched_caliper5_summary.md
results/tuab_subset_200/age_matched/psd_interventions_specparam_age_sex_caliper5/tuab_psd_intervention_subject_bootstrap.md
reports/tables/tuab_age_matched_multiseed_neural_subject_bootstrap.md
results/tuab_full_v3_0_1/biot_interventions_prest_full/tuab_biot_intervention_subject_bootstrap.md
results/tuab_full_v3_0_1/labram_interventions_base_full/tuab_labram_intervention_subject_bootstrap.md
results/tuab_full_v3_0_1/eegpt_interventions_braindecode_full/tuab_eegpt_intervention_subject_bootstrap.md
results/tuab_full_v3_0_1/cbramod_interventions_braindecode_full/tuab_cbramod_intervention_subject_bootstrap.md
results/tuab_full_v3_0_1/reve_base_interventions_full/tuab_reve_intervention_subject_bootstrap.md
results/tuab_full_v3_0_1/eegmamba_interventions_official_full/tuab_eegmamba_intervention_subject_bootstrap.md
results/tuab_full_v3_0_1/bendr_interventions_braindecode_encoder_only_full/tuab_bendr_intervention_subject_bootstrap.md
```

For full details:

```text
experiments.md
code/README.md
project.md
```
