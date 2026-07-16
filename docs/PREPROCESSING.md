# Preprocessing Summary

This file summarizes the main preprocessing settings used by the audit scripts.
The scripts expose these parameters as command-line options where practical.

## Sleep-EDF

- Dataset: Sleep-EDF Expanded Sleep Cassette.
- Channel handling: EEG channels `Fpz-Cz` and `Pz-Oz`.
- Epoching: 30 s scored sleep epochs from hypnograms.
- Wake trimming: configurable through `--wake-trim-minutes`.
- Raw neural filtering: 0.3-45 Hz bandpass in the Sleep-EDF raw pipelines.
- PSD features: Welch spectra over the analysis band used by the relevant
  script, with SpecParam decomposition over 1-45 Hz.

## TUAB

- Dataset: TUAB v3.0.1 normal-versus-abnormal classification.
- Split: official TUAB train/evaluation split.
- Epoching: 10 s windows at 200 Hz.
- EEGNet/ShallowFBCSPNet/Deep4Net: 16-channel TCP bipolar montage.
- LaBraM/EEGPT/BENDR-style caches: 23-channel referential montage.
- SpecParam decomposition: 1-45 Hz.
- Age/sex matching: same sex, within split, with a +/-5 year age caliper.
- Temporal acquisition-proxy audit: EDF recording years binned into early and
  late recording eras, with year-tercile sensitivity checks.

## PhysioNet EEG Motor Movement/Imagery

- Dataset: PhysioNet EEG Motor Movement/Imagery.
- Task: imagined left fist versus imagined right fist.
- Runs: 4, 8 and 12.
- Trial window: 0.5-4.0 s after cue onset.
- Channels: all available EEG channels.
- Sampling rate: 160 Hz.
- PSD features: multitaper spectra over 2-45 Hz.

## PTB-XL ECG

- Dataset: PTB-XL v1.0.3 `records100`.
- Task: normal-versus-abnormal diagnostic classification.
- Input: 12-lead ECG, 10 s windows at 100 Hz.
- Filtering: 0.5-40 Hz bandpass for neural inputs.
- PSD features: Welch spectra over 1-45 Hz.
- Age/sex matching: same sex, within train/validation/test split, with a +/-5
  year age caliper. Records with sentinel age values are excluded before
  matching.
