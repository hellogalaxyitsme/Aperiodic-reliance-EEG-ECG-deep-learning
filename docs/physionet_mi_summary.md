# PhysioNet MI Experiment Summary

## Purpose

We added the PhysioNet EEG Motor Movement/Imagery dataset as a second domain
for the project:

```text
The Aperiodic Confound in EEG Deep Learning
```

The reason was strategic. Sleep-EDF showed strong aperiodic dependence for
sleep staging, but a strong paper needs to show whether this is a general EEG
deep-learning issue or a sleep-specific phenomenon. PhysioNet MI is a useful
contrast because it is cue-locked motor imagery rather than sleep staging.

Main question:

```text
Do models trained on motor-imagery EEG rely strongly on aperiodic spectral
structure, or are they robust when the aperiodic envelope is flattened?
```

## Dataset

Dataset:

```text
PhysioNet EEG Motor Movement/Imagery Dataset
Schalk et al.
109 subjects
64 EEG channels
14 runs per subject
160 Hz sampling rate
```

H200 dataset location:

```text
/mnt/data/aperiodic_confounds/data/physionet-eegmmidb
```

Downloaded files:

```text
subjects: 109
EDF files: 1526
event files: 1526
total EDF/event files: 3052
```

No shared datasets were deleted or modified.

## Task Definition

We started with the cleanest canonical motor-imagery contrast:

```text
imagined left fist vs imagined right fist
```

Runs used:

```text
R04, R08, R12
```

Trial window:

```text
0.5-4.0 seconds after cue onset
```

This was chosen because motor-imagery classification usually uses short
cue-locked trials, not 30-second epochs like Sleep-EDF. The 0.5-second offset
avoids the immediate cue boundary, and the 3.5-second window gives enough
signal for mu/beta motor imagery while still matching the task timescale.

Final trial set:

```text
trials: 4917
subjects: 109
left_fist trials: 2479
right_fist trials: 2438
raw tensor shape: (4917, 64, 560)
```

## Short-Window PSD And Aperiodic Fit

Because these trials are much shorter than Sleep-EDF epochs, we did not reuse
the Sleep-EDF Welch PSD setup directly. We used multitaper PSDs:

```text
PSD method: multitaper
frequency range: 2.00-44.86 Hz
bandwidth: 4 Hz
PSD shape: (4917, 64, 151)
```

This directly addressed the concern that specparam fitting may degrade on
2-4 second windows.

Specparam fit quality:

```text
spectra fit: 314688
ok_fraction: 1.0
mean R2: 0.9520
median R2: 0.9705
10th percentile R2: 0.9066
mean MAE: 0.0771
mean number of peaks: 3.5293
```

Interpretation:

The short-window decomposition was much better than expected. Fit quality was
high enough to proceed, but for the paper this still needs to be reported
explicitly because short-window aperiodic fitting is a legitimate reviewer
concern.

## Experiment 1: First PSD Baselines

Script:

```text
code/scripts/run_physionet_mi_aperiodic_baselines.py
```

Output:

```text
/mnt/data/aperiodic_confounds/results/physionet_mi/baselines_specparam
```

Setup:

```text
classifier: balanced RidgeClassifier
evaluation: 5-fold subject-held-out GroupKFold
labels: left_fist vs right_fist
```

Balanced accuracy:

| Feature set | Mean | Std |
| --- | ---: | ---: |
| Full log-PSD | 0.565 | 0.018 |
| Aperiodic spectrum | 0.526 | 0.011 |
| Aperiodic parameters | 0.530 | 0.015 |
| Periodic residual | 0.532 | 0.021 |

Interpretation:

The simple cross-subject PSD ridge baseline is only modestly above chance,
which is plausible for left-vs-right MI across subjects. Full PSD performs
best, but aperiodic-only features do not dominate. This already looked
different from Sleep-EDF.

## Experiment 2: PSD Intervention And Train-On-Representation Controls

Script:

```text
code/scripts/run_physionet_mi_psd_interventions.py
```

Output:

```text
/mnt/data/aperiodic_confounds/results/physionet_mi/psd_interventions
```

Setup:

```text
classifier: balanced RidgeClassifier
evaluation: 5-fold subject-held-out GroupKFold
uncertainty: subject bootstrap, 10000 samples
```

We trained ridge models on three input representations:

```text
full_log_psd
aperiodic_spectrum
flattened_log_psd
```

Then evaluated each trained model on all three representations.

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

Paired full-trained drops:

| Intervention | BA drop [95% CI] |
| --- | ---: |
| Test on aperiodic spectrum | 0.047 [0.030, 0.065] |
| Test on flattened log-PSD | 0.009 [-0.003, 0.022] |

Interpretation:

The full-PSD ridge model did not collapse when the aperiodic envelope was
flattened. The flattening drop was very small and its confidence interval
included zero.

The aperiodic-only test input was worse than full PSD, which means the
aperiodic component alone was not the dominant sufficient signal for this MI
contrast.

The train-on-flattened control also matters: flattened PSD trained and tested
on flattened PSD reached 0.531 BA [0.516, 0.547]. That is weak but above
chance, suggesting residual spectral information remains usable.

## Experiment 3: Raw Neural Multiseed Interventions

Script:

```text
code/scripts/run_physionet_mi_braindecode_intervention.py
code/scripts/launch_physionet_mi_multiseed_neural.sh
code/scripts/aggregate_multiseed_subject_bootstrap.py
```

Output:

```text
/mnt/data/aperiodic_confounds/results/physionet_mi/multiseed_neural
/mnt/data/aperiodic_confounds/reports/tables/physionet_mi_multiseed_neural_subject_bootstrap.md
```

Models:

```text
Braindecode EEGNet
Braindecode ShallowFBCSPNet
Braindecode Deep4Net
```

Setup:

```text
input: raw EEG trials
evaluation: 5-fold subject-held-out
seeds: 42, 43, 44
epochs: 80
early stopping patience: 12
uncertainty: hierarchical seed/subject bootstrap, 10000 samples
subjects: 109
```

Raw intervention conditions:

```text
raw_eeg
phase_sham
phase_aperiodic
phase_flattened
```

The sham condition is important because it passes through the Fourier
reconstruction path while preserving the original spectral shape. If sham
equals raw, the intervention pipeline itself is not damaging the signal.

Hierarchical seed/subject bootstrap, balanced accuracy:

| Model | Raw | Sham | Aperiodic | Flattened | Flattening drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| EEGNet | 0.744 [0.720, 0.767] | 0.744 [0.720, 0.767] | 0.737 [0.714, 0.760] | 0.739 [0.717, 0.761] | 0.005 [-0.002, 0.013] |
| ShallowFBCSPNet | 0.655 [0.634, 0.678] | 0.655 [0.634, 0.678] | 0.617 [0.597, 0.639] | 0.641 [0.618, 0.665] | 0.014 [0.005, 0.024] |
| Deep4Net | 0.682 [0.661, 0.703] | 0.682 [0.661, 0.703] | 0.684 [0.663, 0.705] | 0.692 [0.669, 0.715] | -0.010 [-0.020, 0.001] |

Interpretation:

The sham condition was exactly equal to raw for all models. That strongly
suggests the Fourier intervention procedure itself is not causing a performance
artifact.

Flattening barely changed performance for EEGNet. ShallowFBCSPNet has a small
positive flattening drop, but it is only about 0.014 balanced accuracy. Deep4Net
does not degrade under flattening; if anything, flattened inputs perform
slightly better, with a confidence interval that nearly includes zero.

The aperiodic-shaped intervention causes only a tiny EEGNet drop, a modest
ShallowFBCSPNet drop, and no Deep4Net drop. Overall, the multiseed neural result
confirms that PhysioNet MI left-vs-right imagined fist classification is
largely robust to removing the aperiodic envelope.

This directly addresses the previous limitation that PhysioNet MI only had
single-seed EEGNet/ShallowFBCSPNet results. We now have the same core
Braindecode model family used for Sleep-EDF and TUAB: EEGNet,
ShallowFBCSPNet, and Deep4Net.

## Overall Interpretation

The PhysioNet MI experiments tell a different story from Sleep-EDF.

Sleep-EDF result:

```text
For wake-vs-sleep and five-stage sleep staging, aperiodic flattening caused
large drops in neural model performance.
```

PhysioNet MI result:

```text
For left-vs-right imagined fist classification, PSD ridge, EEGNet,
ShallowFBCSPNet, and Deep4Net were largely robust to aperiodic flattening.
```

This is scientifically useful. It means the project is not claiming that all
EEG models always rely on aperiodic structure. Instead, the stronger and more
defensible claim is:

```text
Aperiodic reliance is measurable and can be large, but it is task- and
domain-dependent.
```

The audit pipeline can detect both vulnerability and robustness:

```text
Sleep-EDF sleep staging: strong aperiodic dependence.
PhysioNet MI left-vs-right imagery: weak aperiodic dependence / robust to
flattening.
```

This strengthens the paper because it avoids overclaiming and shows that the
method is diagnostic rather than merely destructive.

## Current Limitations

1. The current MI task is only left-vs-right imagined fist.
   Other contrasts, such as imagery-vs-rest or fists-vs-feet, may show
   different aperiodic dependence.

2. We have not yet run transformer-style models on PhysioNet MI.
   EEGNet, ShallowFBCSPNet, and Deep4Net now cover the core convolutional
   baseline family, but an attention model would extend the architecture axis.

3. The MI decomposition fit quality is strong, but short-window aperiodic
   fitting should still be reported carefully in the paper.

## Recommended Next Steps

1. Run an additional MI task contrast:

```text
imagery vs rest
or
fists vs feet
```

This will tell us whether the robustness is specific to left-vs-right imagery
or general within the MI dataset.

2. Create a cross-domain paper table:

```text
Sleep-EDF vs PhysioNet MI
models: PSD ridge, EEGNet, ShallowFBCSPNet, Deep4Net
columns: baseline, sham, aperiodic, flattened, flattening drop
```

3. Create a figure showing flattening drop by dataset/task/model. The expected
visual story is:

```text
Sleep-EDF: large positive drops.
PhysioNet MI: near-zero drops.
```

## Key Files

Local summary and logs:

```text
experiments.md
summary.md
physionet_mi_summary.md
```

Local result files:

```text
results/physionet_mi/baselines_specparam/summary_metrics.csv
results/physionet_mi/psd_interventions/psd_intervention_subject_bootstrap.md
results/physionet_mi/raw_braindecode_interventions/raw_braindecode_subject_bootstrap.md
results/physionet_mi/multiseed_neural
reports/tables/physionet_mi_multiseed_neural_subject_bootstrap.md
reports/tables/physionet_mi_multiseed_neural_subject_bootstrap.csv
results/physionet_mi/specparam/imagined_fists_specparam_fixed.summary.json
```

Remote H200 result root:

```text
/mnt/data/aperiodic_confounds/results/physionet_mi
```
