# Current Stage

Run launches, intermediate checks, completed results and interpretations now
belong in `experiments.md`. This file should stay lightweight: project position,
current priorities and next implementation steps only.

## Project Position

We have moved from the initial Sleep-EDF pilot into a full multi-domain and
full-scale TUAB validation of the central paper idea: EEG models, including
deep neural networks and EEG foundation models, can rely on broadband
aperiodic spectral structure in a task- and architecture-dependent way.

The strongest current evidence is now on full TUAB, not only the original
200-subject subset.

## Completed Work So Far

### Sleep-EDF

We used Sleep-EDF as the first controlled domain for sleep staging. The main
pipeline included PSD/ridge baselines, neural models, sham Fourier controls,
flattened and aperiodic-shaped interventions, train-on-flattened PSD controls,
IRASA agreement, and simulation validation.

Main conclusion: sleep staging shows strong aperiodic dependence, especially
for broad sleep-stage discrimination. Some stage-specific contrasts, such as
N2 vs N3, retain performance after flattening, suggesting that temporal
morphology and slow-wave structure also matter.

### PhysioNet Motor Imagery

We added PhysioNet MI as a second domain with shorter trial windows and
sensorimotor-rhythm structure.

Main conclusion: motor imagery is much less affected by aperiodic flattening
than sleep or TUAB. This supports the paper's task-dependence narrative:
aperiodic reliance is not a universal failure mode across all EEG tasks.

### TUAB 200-Subject Subset

We first implemented the TUAB abnormal-vs-normal audit on a 200-subject
stratified subset, preserving official TUAB train/eval boundaries. This
included PSD baselines, raw neural models, age/sex-matched controls, and
foundation-model audits.

Main conclusion: TUAB showed clear aperiodic dependence. Age matching reduced
but did not eliminate the signal in PSD analyses, supporting the idea that the
aperiodic effect is not only an age shortcut.

### Full TUAB Dataset

We downloaded and preprocessed the full TUAB v3.0.1 EDF collection available on
the H200 storage.

Main conclusion: full TUAB confirms the earlier subset result. Standard EEG
deep learning models show robust aperiodic reliance at full scale, with neutral
sham controls. Numerical details are recorded in `experiments.md` and generated
result tables.

## Full TUAB Foundation Model Results

We scaled multiple EEG foundation-model audits from the TUAB-200 subset to the
full TUAB dataset. All runs use the same raw/sham/aperiodic-shaped/flattened
intervention logic and subject-level bootstrap evaluation.

Main conclusion: the full-TUAB foundation-model evidence is now strong. The
reportable positive rows with neutral sham controls are BIOT, LaBraM, EEGPT,
CBraMod, REVE-base, and EEGMamba. BENDR also completed, but its phase-sham
control collapsed to chance, so it should be treated as a supplementary or
cautionary transfer result rather than a clean aperiodic-specific effect.
Numerical details are recorded in `experiments.md` and generated result tables.

## Current Priority

The full-TUAB foundation-model aperiodic intervention audit is complete across
the prepared model set: BIOT, LaBraM, EEGPT, CBraMod, REVE-base, EEGMamba, and
BENDR. The next priority is consolidation for the manuscript and tables, not
launching another prepared FM run.

## Next Implementation Steps

1. Decide whether BENDR belongs in the main foundation-model table or only as a
   supplementary/cautionary transfer result.

2. Update `summary.md` with the full-TUAB foundation-model results.

3. Prepare a consolidated full-TUAB result table covering:

```text
PSD ridge
EEGNet
ShallowFBCSPNet
Deep4Net
BIOT
LaBraM
EEGPT
CBraMod
REVE-base
BENDR, if reportable
```

4. Update paper figures and table artifacts to replace TUAB-200 foundation-model
   values with full-TUAB values where appropriate.

## Current Paper-Level Interpretation

The paper is now in a much stronger position than before full TUAB scaling.
The results no longer rely on a 200-subject subset. Across full TUAB, standard
deep models and multiple foundation models show measurable aperiodic dependence
with neutral sham controls, while BENDR provides a useful cautionary case where
the sham itself fails.

The emerging message is:

```text
EEG foundation models do not simply ignore aperiodic spectral structure.
They use it, but the degree of reliance differs by architecture.
Flattening does not destroy all performance, which means these models also use
residual waveform, oscillatory, spatial, and temporal information.
```

This is exactly the kind of nuanced result needed for a strong NMI-style paper:
not a blanket criticism of EEG deep learning, but a controlled audit showing a
previously underreported source of model performance and interpretation risk.
