# Project: The Aperiodic Confound in EEG Deep Learning

## One-line thesis

Many EEG deep learning benchmarks may obtain part of their apparent performance from aperiodic 1/f spectral structure rather than from the rhythmic oscillatory band-power mechanisms that their interpretations imply. This project will quantify that dependence, identify when it is scientifically valid versus confounding, and provide corrected evaluation protocols.

## Core claim, stated safely

The project does not assume that band power is useless, that oscillations are not real, or that all EEG deep models are invalid. The falsifiable claim is:

> For several canonical EEG classification benchmarks, the predictive evidence used by common deep models can be decomposed into periodic and aperiodic spectral components, and a nontrivial fraction of reported performance and attribution will be explained by the aperiodic component. In some benchmarks, interpretations framed as band-power or oscillatory effects will not survive aperiodic control.

This wording is important. It makes the work rigorous, testable, and publication-safe.

## Why this matters

EEG spectra contain at least two separable sources of structure:

1. Periodic activity: narrowband peaks above the background spectrum, often interpreted as oscillations such as alpha, beta, theta, or sleep spindles.
2. Aperiodic activity: broadband, non-oscillatory spectral structure, often approximated by a 1/f-like curve.
3. Artifacts and acquisition effects: muscle, eye, electrode impedance, line noise, hardware filters, site effects, and preprocessing choices.

Traditional EEG analysis often computes absolute or relative power in canonical bands. But if the aperiodic background changes, band power can change even when no oscillatory peak changes. Deep models trained on raw EEG, spectrograms, or learned filters may exploit the same background shifts while being interpreted as discovering oscillatory biomarkers.

The scientific danger is not that aperiodic activity is meaningless. It may be physiologically meaningful, for example reflecting synaptic timescales, excitation-inhibition balance, arousal, anesthesia, age, or disease state. The danger is misattribution: calling an aperiodic effect an alpha/beta/gamma oscillatory biomarker, or reporting a model as discovering disease-specific rhythms when it is mostly detecting broadband slope, offset, artifact, age, vigilance, or acquisition-site structure.

## Literature anchors

Primary literature and tools that define the foundation:

1. Donoghue et al. 2020, Nature Neuroscience, "Parameterizing neural power spectra into periodic and aperiodic components." Shows that standard spectral analyses can conflate periodic peak parameters with aperiodic offset and exponent, and introduces the FOOOF/specparam model. URL: https://www.nature.com/articles/s41593-020-00744-x
2. Specparam/FOOOF package. The model treats the power spectrum as an aperiodic component plus a variable number of periodic peaks, returning offset/exponent plus peak center frequency, power, and bandwidth. URL: https://github.com/fooof-tools/fooof
3. Wen and Liu 2016, Brain Topography, IRASA. Separates fractal/aperiodic and oscillatory components using irregular resampling. URL: https://haiguangwen.github.io/papers/Wen-Brain%20Topogr%202016.pdf
4. Gerster et al. 2022, Neuroinformatics, "Separating Neural Oscillations from Aperiodic 1/f Activity: Challenges and Recommendations." Reviews FOOOF and IRASA, warns about fitting-range dependence, low-frequency oscillation contamination, and method-specific failure modes. URL: https://link.springer.com/article/10.1007/s12021-022-09581-8
5. Brake et al. 2024, Nature Communications, "A neurophysiological basis for aperiodic EEG and the background spectral trend." Provides a biophysical account of broadband EEG and shows that aperiodic changes can undermine traditional rhythm interpretation, including under propofol. URL: https://www.nature.com/articles/s41467-024-45922-8
6. Donoghue et al. 2020, eNeuro, "Electrophysiological Frequency Band Ratio Measures Conflate Periodic and Aperiodic Neural Activity." Shows band ratios such as theta/beta can reflect aperiodic exponent rather than intended oscillatory quantities. URL: https://pubmed.ncbi.nlm.nih.gov/32978216/
7. TUH EEG Corpus, Obeid and Picone 2016. Large public clinical EEG corpus underlying TUAB/TUEV-style benchmarks. URL: https://www.frontiersin.org/articles/10.3389/fnins.2016.00196/full
8. TDBRAIN, van Dijk et al. 2022, Scientific Data. Resting EEG with clinical/behavioral metadata, useful for psychiatric and neurophysiological biomarker audits. URL: https://www.nature.com/articles/s41597-022-01409-z
9. BIOT, NeurIPS 2023. A biosignal transformer benchmarked on TUAB/TUEV and other datasets. URL: https://papers.neurips.cc/paper_files/paper/2023/file/f6b30f3e2dd9cb53bbf2024402d02295-Paper-Conference.pdf
10. EEGPT, NeurIPS 2024. EEG foundation model with downstream benchmark claims. URL: https://proceedings.neurips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html
11. LaBraM, ICLR 2024. EEG foundation model benchmarked on abnormal detection, event classification, emotion recognition, and other tasks. URL: https://proceedings.iclr.cc/paper_files/paper/2024/file/47393e8594c82ce8fd83adc672cf9872-Paper-Conference.pdf
12. Braindecode. Practical EEG deep learning library for reproducible baselines. URL: https://braindecode.org/

## Research questions

RQ1. How much of benchmark performance is retained when only aperiodic spectral structure is available?

RQ2. How much performance is retained when aperiodic structure is removed or matched, leaving oscillatory peaks and non-spectral temporal structure as the dominant signal?

RQ3. Do model attributions in spectral space project more strongly onto the aperiodic subspace or the periodic residual subspace?

RQ4. Are results consistent across spectral decomposition methods, especially specparam/FOOOF and IRASA?

RQ5. Is aperiodic reliance scientifically meaningful, for example age/arousal/disease physiology, or is it a shortcut caused by artifacts, montage, site, class imbalance, or leakage?

RQ6. Can an aperiodic-controlled model preserve useful EEG performance while preventing invalid oscillatory interpretations?

## Definitions

Let x_i,c(t) be EEG for sample i, channel c, and time t. Let S_i,c(f) be a power spectral density estimate over a frequency grid F.

Use log power:

```text
y_i,c(f) = log S_i,c(f)
```

Unless otherwise stated, the math uses natural logarithms. Implementations may use log10 because specparam/FOOOF commonly works in log10 units; every inverse transform must use the matching inverse base. For example, if y = log10(S), then S = 10^y, not exp(y).

### Aperiodic model

Fixed 1/f model:

```text
a_i,c(f) = b_i,c - chi_i,c * log(f)
```

where:

```text
b_i,c   = aperiodic offset
chi_i,c = aperiodic exponent/slope
```

Knee model:

```text
a_i,c(f) = b_i,c - log(k_i,c + f ^ chi_i,c)
```

where k_i,c captures a bend/knee in the spectrum. Fixed mode is primary for 1-45 Hz EEG unless diagnostics show strong knee structure; knee mode is a sensitivity analysis.

### Periodic model

Periodic peaks are modeled as localized deviations above the aperiodic background:

```text
p_i,c(f) = sum_{m=1}^{M_i,c} A_i,c,m * exp(-(f - mu_i,c,m)^2 / (2 * sigma_i,c,m^2))
```

where:

```text
mu    = peak center frequency
A     = peak height above aperiodic background in log power
sigma = peak width parameter
```

Total fitted log spectrum:

```text
y_i,c(f) = a_i,c(f) + p_i,c(f) + eps_i,c(f)
```

### Aperiodic-only spectrum

```text
y_i,c^A(f) = a_i,c(f)
```

### Periodic-residual spectrum

```text
y_i,c^P(f) = y_i,c(f) - a_i,c(f)
```

For models requiring raw time series, spectral surgery will produce surrogate time series whose Fourier magnitudes match a target spectrum while preserving phase structure as specified in the intervention protocol.

### Canonical band power

For band I = [f1, f2]:

```text
B_I = integral_{f1}^{f2} S(f) df
```

Relative band power:

```text
RB_I = B_I / integral_{Fmin}^{Fmax} S(f) df
```

Band ratio:

```text
BR_{I,J} = B_I / B_J
```

## Mathematical framework

### Theorem 1: Aperiodic slope changes induce apparent band-power changes

Assume a pure aperiodic spectrum:

```text
S(f) = A * f^(-chi),  f in [fmin, fmax]
```

For a band I = [u, v], u > 0:

```text
B_I(A, chi) = A * integral_u^v f^(-chi) df
```

If chi != 1:

```text
B_I(A, chi) = A * (v^(1-chi) - u^(1-chi)) / (1 - chi)
```

If chi = 1:

```text
B_I(A, 1) = A * log(v / u)
```

Since dB_I / dA > 0 and dB_I / dchi is generally nonzero, absolute band power changes under offset or exponent changes even when no periodic oscillatory peak exists.

Implication: a reported alpha/beta/theta power difference is not automatically an oscillatory difference. It can be generated by a broadband aperiodic shift.

### Theorem 2: Band ratios can be aperiodic-exponent detectors

For two bands I = [u, v] and J = [r, s] under a pure aperiodic spectrum:

```text
BR_{I,J}(chi) =
  integral_u^v f^(-chi) df
  /
  integral_r^s f^(-chi) df
```

The scale A cancels, but chi does not. Therefore band ratios remove offset but remain sensitive to exponent. For theta/beta, a steeper exponent increases low-frequency power relative to high-frequency power, creating a theta/beta increase without a theta oscillation increase or beta oscillation decrease.

Implication: theta/beta, theta/alpha, alpha/beta, and similar ratios must be audited against aperiodic exponent and alpha peak power separately.

### Theorem 3: Local classifier sensitivity decomposes into aperiodic and periodic components

Let a differentiable classifier produce class logit h_theta(y). Treat y as a vectorized log-spectrum across channels and frequencies. Around y, for a perturbation delta:

```text
h_theta(y + delta) - h_theta(y)
  = <grad_y h_theta(y), delta> + O(||delta||^2)
```

Define a finite-dimensional aperiodic tangent space A_i spanned by the derivatives of the aperiodic model with respect to its parameters. For fixed mode:

```text
A_i = span{1, -log(f)} over all channels
```

For knee mode:

```text
A_i = span{d a/db, d a/dk, d a/dchi}
```

Let P_A be the orthogonal projection onto this aperiodic tangent space under a chosen inner product over channel-frequency bins. Let P_P = I - P_A.

Then first-order logit sensitivity decomposes exactly:

```text
grad_y h = P_A grad_y h + P_P grad_y h
```

Define the Aperiodic Attribution Ratio:

```text
AAR_i = ||P_A grad_y h_theta(y_i)||_2^2 / ||grad_y h_theta(y_i)||_2^2
```

and the Oscillatory Attribution Ratio:

```text
OAR_i = 1 - AAR_i
```

If AAR is high and performance is retained under aperiodic-only intervention, the model is primarily aperiodic-sensitive. If OAR is high and performance survives aperiodic flattening, oscillatory or residual structure is important.

This statement is mathematical identity plus first-order Taylor approximation. The empirical question is whether the local approximation predicts actual counterfactual performance changes.

### Principle 4: A valid confound audit requires both necessity and sufficiency tests

Aperiodic sufficiency:

```text
Perf(original model on aperiodic-only inputs) / Perf(original inputs)
```

Aperiodic necessity:

```text
Perf(original inputs) - Perf(aperiodic-flattened inputs)
```

Oscillatory sufficiency:

```text
Perf(periodic-residual-only inputs) / Perf(original inputs)
```

No single test is enough. Aperiodic-only performance can be high because aperiodic features are real biomarkers. Flattened performance can drop because the intervention damaged the signal. Therefore claims require convergence across:

1. aperiodic-only performance,
2. flattened performance,
3. matched-control performance,
4. attribution projection,
5. decomposition-method agreement,
6. simulation ground truth.

### Lemma 5: Decomposition is identifiable only under explicit spectral assumptions

We cannot claim universal periodic/aperiodic identifiability from arbitrary EEG. We can claim identifiability under operational assumptions:

1. The aperiodic component is smooth in log-frequency and belongs to the fixed or knee family over the selected range.
2. Periodic peaks are localized deviations with bounded width.
3. Peaks are sufficiently separated or sparse enough for stable fitting.
4. The PSD estimate has sufficient frequency resolution and signal-to-noise.
5. Line noise, notch artifacts, and muscle contamination are excluded or modeled.

Therefore all conclusions must be reported as conditional on:

```text
frequency range,
PSD estimator,
decomposition method,
fit quality thresholds,
preprocessing pipeline,
and sensitivity analyses.
```

### Theorem 6: Simulation-calibrated audit consistency

Let spectra be generated by:

```text
y_i(f) = a_i(f; z_i^A) + p_i(f; z_i^P) + eps_i(f)
```

where class label Y depends only on z_i^A, only on z_i^P, or on a known mixture of both. Assume:

1. the decomposition estimator is consistent for a_i and p_i over the chosen frequency range,
2. the classifier class is rich enough to approximate the Bayes decision rule,
3. train and test samples are independent at the subject/simulation-unit level,
4. intervention error goes to zero as PSD resolution and sample size increase.

Then, asymptotically:

1. if Y depends only on z_i^A, Retention_A approaches 1 and Retention_P approaches chance-relative performance;
2. if Y depends only on z_i^P, Retention_P approaches 1 and Retention_A approaches chance-relative performance;
3. if Y depends on both, Retention_A, Retention_P, and AAR vary monotonically with the controlled effect-size mixture, up to approximation error.

This theorem is not assumed for real EEG. It is a validation theorem for the simulation suite. Its purpose is to prove that the audit metrics behave correctly when the ground truth is known before applying them to biological data.

## Main empirical hypotheses

H1. On at least one clinical benchmark, aperiodic-only features will retain a large fraction of original model performance.

Operational threshold:

```text
retention_A = balanced_accuracy_A / balanced_accuracy_original >= 0.70
```

This threshold is not a universal truth. It is a pre-registered marker for "substantial aperiodic sufficiency."

H2. For at least one benchmark, aperiodic flattening or class-conditional aperiodic matching will reduce performance enough to change the scientific interpretation.

Operational threshold:

```text
drop_flattened = balanced_accuracy_original - balanced_accuracy_flattened >= 0.05
```

H3. Gradient or perturbation attribution will show significant projection onto the aperiodic subspace beyond chance.

Operational threshold:

```text
mean(AAR_observed) > mean(AAR_label_permuted) with p < 0.05 after FDR correction
```

H4. Some tasks will remain genuinely oscillatory.

This is not a failure. It is essential for credibility. Sleep spindles, alpha reactivity, and event-related rhythmic phenomena may survive aperiodic control. The project should distinguish valid oscillatory effects from confounded ones.

## Datasets

### Tier 1: primary datasets

#### 1. TDBRAIN

Use case:

```text
resting EEG; clinical labels; age/sex/diagnosis/behavioral metadata
```

Why:

TDBRAIN directly matches the psychiatric/biomarker motivation. It includes formal and referral diagnoses, MDD/ADHD/OCD-related cohorts, raw EEG, ECG, and task data. It is ideal for testing whether psychiatric classifiers rely on aperiodic slope, offset, age, arousal, or artifacts.

Primary tasks:

1. MDD versus non-MDD/control-like groups, with conservative inclusion criteria.
2. ADHD versus non-ADHD/control-like groups.
3. Age prediction as a positive-control task, because aperiodic exponent is expected to carry age-related signal.
4. Eyes-open versus eyes-closed if available and cleanly labeled.

Risks:

1. Diagnosis labels are heterogeneous.
2. Medication, referral status, and comorbidities may confound classification.
3. Aperiodic dependence may be physiologically meaningful rather than invalid.

Required controls:

1. Age, sex, medication where available.
2. Recording condition.
3. Artifact burden.
4. Subject-level splits.
5. Matched cohorts by age and sex.

Source:

https://www.nature.com/articles/s41597-022-01409-z

#### 2. TUAB

Use case:

```text
normal versus abnormal clinical EEG
```

Why:

TUAB is a major benchmark for EEG pathology classification and is used by deep learning and foundation-model papers. It is clinically important and large enough for robust testing.

Primary task:

```text
normal vs abnormal
```

Metrics:

```text
balanced accuracy, AUROC, AUPRC, macro-F1, calibration error
```

Risks:

1. Abnormal labels may include broad clinical heterogeneity.
2. Abnormality may include slowing, vigilance, medication, age, or site effects.
3. Aperiodic changes may be clinically valid but not specific.

Required controls:

1. Follow official subject-level train/test split.
2. Use common 10-20 montage channels.
3. Match age/sex if metadata permits.
4. Compare report-level labels with signal-derived shortcuts.

Source:

https://www.frontiersin.org/articles/10.3389/fnins.2016.00196/full

#### 3. Sleep-EDF or ISRUC Sleep

Use case:

```text
sleep staging
```

Why:

Sleep staging has strong canonical band interpretations: delta for N3, spindles/sigma for N2, alpha/theta for wake/N1, etc. It is also a domain where aperiodic slope changes with arousal and sleep depth. This makes it ideal for distinguishing legitimate oscillatory effects from aperiodic arousal signals.

Primary task:

```text
Wake, N1, N2, N3, REM classification
```

Metrics:

```text
macro-F1, Cohen's kappa, balanced accuracy, per-stage F1
```

Risks:

1. Sleep stages are partly defined by visible graphoelements, so periodic features should matter.
2. Epoch labels are noisy and temporally correlated.
3. EMG/EOG can leak sleep stage if included.

Required controls:

1. EEG-only and EEG+EOG/EMG variants reported separately.
2. Subject-level splits.
3. Per-stage analysis of aperiodic reliance.
4. Sensitivity to 30 s epoching and subwindow PSD.

Sources:

1. Sleep-EDF / Sleep-EDF Expanded: https://physionet.org/content/sleep-edfx/
2. ISRUC-Sleep: https://sleeptight.isr.uc.pt/?page_id=48
3. ISRUC-Sleep paper: https://www.sciencedirect.com/science/article/pii/S0169260715002734

#### 4. TUEV

Use case:

```text
clinical event classification
```

Why:

TUEV is a major benchmark in BIOT/LaBraM-style evaluations. It tests whether event classifiers rely on broadband background, artifacts, or true event morphology.

Risks:

1. Some classes include artifact or background labels, making aperiodic/artifact reliance scientifically ambiguous.
2. Event labels may encode duration, amplitude, or annotator conventions.

Use as secondary unless the label taxonomy is handled carefully.

### Tier 2: optional datasets

#### SEED-IV or DEAP

Use case:

```text
emotion recognition
```

Why:

Emotion EEG papers often report high performance and band-specific interpretations. This is a good stress test for aperiodic shortcuts.

Risks:

1. Subject/session leakage is common.
2. Labels are weak because they are affective stimuli or self-report proxies.
3. Dataset access and preprocessing conventions vary.

Use after Tier 1 results are stable.

#### MOABB motor imagery datasets

Use case:

```text
BCI motor imagery
```

Why:

Motor imagery has known sensorimotor rhythm mechanisms. If aperiodic controls do not destroy performance, that supports specificity of the audit.

Expected result:

Oscillatory beta/mu features should retain more importance than in broad clinical labels.

## Models

### Classical interpretable baselines

1. Logistic regression on canonical band powers.
2. Logistic regression on specparam aperiodic parameters only.
3. Logistic regression on periodic peak features only.
4. Elastic net on full log-PSD.
5. XGBoost or random forest on engineered spectral features.

Purpose:

These baselines establish whether the phenomenon exists without deep networks.

### Deep baselines

1. EEGNet.
2. ShallowConvNet.
3. Deep4ConvNet.
4. TCN/TCN-like temporal convolution.
5. Spectrogram CNN.
6. Small transformer encoder on channel-frequency-time patches.

Implementation preference:

Use Braindecode where possible for EEGNet/ShallowConvNet/Deep4ConvNet, and keep custom code minimal.

### Foundation-model audit targets

1. BIOT.
2. LaBraM.
3. EEGPT.

Use these only after the core pipeline is validated. Full reproduction is expensive; the paper can audit their reported benchmark settings by:

1. reproducing accessible checkpoints where possible,
2. applying the same interventions to their input pipeline,
3. comparing with smaller baselines under identical splits.

## Spectral decomposition pipeline

### PSD estimation

Primary:

```text
Welch PSD or multitaper PSD, 2 s to 4 s subwindows, 50 percent overlap
```

Frequency range:

```text
1-45 Hz primary for clinical/sleep EEG
```

Sensitivity ranges:

```text
2-40 Hz
3-40 Hz
1-80 Hz where sampling and line noise permit
40-80 Hz high-frequency slope check where data quality permits
```

Notch exclusion:

Exclude bins around 50/60 Hz and harmonics when high-frequency analysis is used.

### Primary decomposition

Use specparam/FOOOF:

```text
aperiodic_mode: fixed primary, knee sensitivity
peak_width_limits: dataset-tuned but pre-registered
max_n_peaks: limited to prevent overfitting
min_peak_height: set by PSD noise floor
peak_threshold: default plus sensitivity
```

Record:

```text
offset, exponent, knee if used, peak center frequencies, peak heights, peak bandwidths, R^2, fitting error
```

### Secondary decomposition

Use IRASA:

```text
input: time series
output: fractal/aperiodic spectrum and oscillatory residual
```

Purpose:

Confirm that the main conclusions are not an artifact of specparam assumptions.

### Fit-quality exclusion

Exclude or flag spectra if:

```text
fit R^2 < 0.90 for specparam
fit error in worst 5 percent
unphysiological exponent outside pre-specified range
peak count equals max_n_peaks too often
line noise or notch dips dominate fit
```

Do not silently remove hard cases. Report:

```text
included count, excluded count, exclusion reasons, and sensitivity including all samples
```

## Counterfactual interventions

### Intervention 1: Aperiodic-only spectrum

Replace each spectrum by its fitted aperiodic component:

```text
y^A(f) = a(f)
```

Purpose:

Test aperiodic sufficiency.

For spectral models:

Feed y^A directly.

For raw time-domain models:

Construct surrogate signals with target Fourier magnitude from a(f). Use either:

1. original phase preservation,
2. randomized phase,
3. multichannel phase-preserving variant if cross-channel structure is required.

Report which phase protocol is used.

### Intervention 2: Aperiodic-flattened spectrum

Subtract fitted aperiodic component:

```text
y^flat(f) = y(f) - a(f)
```

or divide raw power by fitted aperiodic background in linear power:

```text
S^flat(f) = S(f) / inverse_log(a(f))
```

where inverse_log is exp for natural-log spectra and 10^a for log10 spectra.

Purpose:

Test aperiodic necessity.

Important:

Flattened spectra are not raw physiological spectra. They are diagnostic interventions, not replacement signals.

### Intervention 3: Periodic-residual-only spectrum

Use:

```text
y^P(f) = p(f)
```

or IRASA oscillatory residual.

Purpose:

Test whether isolated peaks carry class signal.

### Intervention 4: Class-conditional aperiodic matching

For each class, match samples so that the distribution of:

```text
mean exponent, mean offset, age, sex, artifact score, recording length
```

is balanced across classes.

Methods:

1. propensity score matching,
2. optimal transport matching,
3. stratified resampling by exponent-offset bins.

Purpose:

Test whether model performance survives when aperiodic shortcuts are statistically unavailable.

### Intervention 5: Aperiodic randomization

Randomly permute exponent/offset across samples while preserving periodic residuals:

```text
y_i^rand(f) = a_{pi(i)}(f) + p_i(f)
```

where pi is a label-preserving or label-breaking permutation depending on the test.

Purpose:

Separate aperiodic class information from periodic information.

### Intervention 6: Controlled exponent perturbation

For each sample:

```text
y_delta(f) = y(f) - delta * log(f)
```

or adjust fitted exponent:

```text
chi' = chi + delta
```

Measure logit response:

```text
d h_theta / d chi
```

Purpose:

Estimate causal sensitivity of the trained classifier to slope changes.

## Attribution framework

### Spectral gradient projection

For differentiable models, compute:

```text
grad_y h_theta(y)
```

Then project onto:

```text
A = aperiodic tangent space
P = orthogonal complement / periodic-residual space
```

Metrics:

```text
AAR = ||P_A grad||^2 / ||grad||^2
OAR = ||P_P grad||^2 / ||grad||^2
```

Compute per:

```text
sample, class, channel, model, dataset
```

### Perturbation attribution

For non-differentiable models or raw models:

1. Perturb offset.
2. Perturb exponent.
3. Remove individual peaks.
4. Flatten selected frequency ranges.
5. Compute change in predicted probability or logit.

Metrics:

```text
Delta logit per unit exponent
Delta logit per unit offset
Delta logit per peak removal
```

### Shapley-style grouped attribution

Groups:

```text
aperiodic offset
aperiodic exponent
delta peak
theta peak
alpha peak
sigma peak
beta peak
gamma/high-frequency residual
channel topography
```

Use grouped permutation/Shapley approximation rather than per-frequency Shapley, because per-bin Shapley is unstable and computationally expensive.

## Evaluation metrics

### Predictive performance

Binary:

```text
balanced accuracy
AUROC
AUPRC
macro-F1
sensitivity
specificity
ECE calibration error
```

Multiclass:

```text
macro-F1
weighted-F1
balanced accuracy
Cohen's kappa
per-class F1
confusion matrix
ECE calibration error
```

Regression, if age prediction is included:

```text
MAE
RMSE
R^2
Spearman correlation
calibration slope
```

### Confound/reliance metrics

Aperiodic retention:

```text
Retention_A = Perf_AperiodicOnly / Perf_Original
```

Periodic retention:

```text
Retention_P = Perf_PeriodicResidualOnly / Perf_Original
```

Flattening drop:

```text
Drop_flat = Perf_Original - Perf_Flattened
```

Aperiodic reliance index:

```text
ARI = 0.5 * Retention_A + 0.5 * normalized(Drop_flat)
```

Attribution aperiodic ratio:

```text
AAR = ||P_A grad||^2 / ||grad||^2
```

Matching robustness:

```text
Robustness_match = Perf_Matched / Perf_Original
```

Decomposition agreement:

```text
Agreement = corr(ARI_specparam, ARI_IRASA)
```

### Statistical testing

Use:

1. subject-level bootstrap confidence intervals,
2. paired bootstrap for performance deltas,
3. permutation tests for attribution ratios,
4. mixed-effects models for repeated windows per subject,
5. FDR correction across datasets/tasks/models,
6. nested CV for hyperparameter selection.

Mixed-effects template:

```text
metric ~ intervention * model * dataset + age + sex + artifact_score + (1 | subject)
```

For sleep:

```text
metric ~ intervention * model * stage + age + sex + (1 | subject)
```

## Simulation experiments

Simulations are mandatory because real EEG has no ground-truth decomposition.

### Simulation A: pure aperiodic classification

Generate two classes differing only in:

```text
offset or exponent
```

No oscillatory peaks differ.

Expected:

1. band-power classifiers perform above chance,
2. aperiodic-only models perform above chance,
3. periodic-residual models perform near chance,
4. attribution projection is aperiodic.

Purpose:

Show the confound mechanism exists.

### Simulation B: pure periodic classification

Generate two classes differing only in alpha/theta/beta peak height.

Expected:

1. periodic-residual models perform above chance,
2. aperiodic-only models near chance,
3. attribution projection is periodic.

Purpose:

Show the audit does not falsely label all classification as aperiodic.

### Simulation C: mixed periodic and aperiodic effects

Generate class differences in both slope and peak height with controlled effect sizes.

Expected:

AAR and retention metrics track the known mixture weights.

Purpose:

Calibrate ARI/AAR against known ground truth.

### Simulation D: artifact confounding

Add EMG-like high-frequency contamination, eye-blink low-frequency transients, and line noise.

Expected:

Artifact-aware preprocessing should reduce false aperiodic reliance. If not, the method is detecting artifacts rather than physiology.

Purpose:

Stress-test the pipeline.

## Preprocessing and leakage controls

### Universal controls

1. Subject-level train/validation/test split.
2. No overlapping windows across train and test from the same recording.
3. Hyperparameters selected only on validation data.
4. Test data untouched until final evaluation.
5. Same preprocessing for all interventions.
6. Random seeds fixed and reported.
7. All metrics aggregated first per subject, then across subjects, unless the task is explicitly event-level.

### EEG preprocessing

1. Load with MNE.
2. Resample to common rate, usually 200 or 256 Hz.
3. Select common channel set.
4. Re-reference consistently, for example average reference or standard bipolar montage depending on dataset.
5. Bandpass only as needed for stability, with all filter settings reported.
6. Notch line noise only if required.
7. Artifact scoring using amplitude, high-frequency power, flat channels, and line noise.
8. Do not apply ICA by default unless comparing artifact-removal sensitivity, because ICA can alter spectral structure.

### Artifact score

Define:

```text
artifact_score =
  z(abs_amplitude_robust) +
  z(high_frequency_power) +
  z(line_noise_power) +
  z(flatline_fraction)
```

Use it for:

1. exclusion sensitivity,
2. covariate adjustment,
3. matching,
4. sanity checking aperiodic reliance.

## Expected figures

Figure 1. Conceptual decomposition:

Original PSD, fitted aperiodic background, periodic peaks, flattened residual.

Figure 2. Simulation proof:

Band power changes under pure 1/f slope change; classifiers appear to discover bands but attribution is aperiodic.

Figure 3. Dataset-level performance retention:

Original vs aperiodic-only vs flattened vs periodic-only for all datasets/models.

Figure 4. Aperiodic Attribution Ratio heatmap:

Models by datasets, with confidence intervals.

Figure 5. Frequency-channel attribution:

Raw attribution map and projection onto aperiodic/periodic subspaces.

Figure 6. Matching analysis:

Performance before and after exponent/offset matching.

Figure 7. Case studies:

One benchmark where aperiodic reliance is high and one where oscillatory information survives.

Figure 8. Recommended protocol:

A concise decision tree for future EEG papers.

## Paper contribution structure

### Contribution 1: Theory

Formalize how aperiodic offset and exponent contaminate band power, band ratios, and learned spectral classifiers.

### Contribution 2: Method

Introduce an audit pipeline:

```text
decompose -> intervene -> attribute -> match -> validate by simulation
```

### Contribution 3: Empirical benchmark

Evaluate across clinical, sleep, and BCI-like datasets using classical and deep models.

### Contribution 4: Corrected reporting protocol

Provide a practical checklist for EEG ML papers:

1. report aperiodic parameters,
2. report periodic-adjusted band power,
3. include aperiodic-only and flattened controls,
4. perform subject-level leakage checks,
5. distinguish physiological aperiodic biomarkers from oscillatory claims.

## What would count as strong evidence

Strong support:

1. Aperiodic-only inputs retain substantial performance in multiple benchmarks.
2. Flattening/matching reduces performance materially.
3. AAR is high and statistically above null.
4. Specparam and IRASA agree qualitatively.
5. Simulations show the metrics recover known ground truth.
6. Controls rule out age, sex, artifact, montage, site, and leakage as sole explanations.

Moderate support:

1. Aperiodic reliance is high in clinical/psychiatric tasks but low in motor imagery or sleep spindle-related tasks.
2. This would still be publishable because it maps where aperiodic confounding matters.

Weak support:

1. Aperiodic-only features perform near chance everywhere.
2. Flattening does not reduce performance.
3. Attribution is mostly periodic.

In that case, the paper becomes a useful negative result and a validation of oscillatory interpretations in modern EEG DL.

## Falsification criteria

The central hypothesis is considered falsified if all are true:

1. On all primary datasets, Retention_A < 0.55.
2. On all primary datasets, Drop_flat < 0.02.
3. AAR does not exceed permutation null after correction.
4. Matching exponent/offset does not change performance.
5. Simulation metrics work correctly, proving the audit itself is sensitive.

This protects the project from confirmation bias.

## Common failure modes and fixes

### Failure mode 1: Low-frequency oscillations distort exponent estimates

Fix:

1. Report multiple fitting ranges.
2. Use 2-40 and 3-40 Hz sensitivity analyses.
3. Compare fixed and knee modes.
4. Compare specparam and IRASA.
5. Avoid overinterpreting delta-range exponent in sleep.

### Failure mode 2: Aperiodic reliance is actually artifact reliance

Fix:

1. Include artifact score.
2. Exclude high-artifact samples.
3. Run sensitivity with/without ICA or artifact cleaning.
4. Inspect high-frequency EMG contamination.
5. Report whether exponent correlates with artifact score.

### Failure mode 3: Flattening creates unnatural inputs

Fix:

1. Treat flattening as diagnostic, not physiological.
2. Use matching and attribution as independent tests.
3. Use simulations to validate intervention behavior.
4. For raw models, compare original-phase and random-phase surrogates.

### Failure mode 4: Dataset labels are broad or noisy

Fix:

1. Use label-noise-aware interpretation.
2. Prefer balanced accuracy, macro-F1, AUROC/AUPRC.
3. Aggregate per subject.
4. Avoid claiming disease specificity unless controlled.

### Failure mode 5: Deep model learns time-domain morphology not captured by PSD

Fix:

1. Compare raw and spectral models.
2. Analyze residual performance after spectral flattening.
3. Include temporal controls and phase-preserving surrogates.
4. State clearly when conclusions apply to spectral evidence only.

## Minimal viable experiment

Start with one dataset:

```text
TDBRAIN or TUAB
```

Implement:

1. PSD extraction.
2. Specparam decomposition.
3. Feature baselines:
   - aperiodic-only logistic regression,
   - periodic-only logistic regression,
   - full PSD elastic net.
4. One deep baseline:
   - EEGNet or ShallowConvNet.
5. Interventions:
   - aperiodic-only,
   - flattened,
   - matched exponent/offset.
6. Metrics:
   - balanced accuracy,
   - AUROC,
   - Retention_A,
   - Drop_flat,
   - AAR if differentiable spectral model is used.

Decision after MVP:

If Retention_A >= 0.70 or Drop_flat >= 0.05, expand to sleep and TUEV.
If not, run simulations and a second dataset before abandoning.

## Full project phases

### Phase 0: Reproducibility scaffold

Deliverables:

1. `configs/` for datasets and models.
2. `scripts/prepare_dataset.py`.
3. `scripts/extract_psd.py`.
4. `scripts/fit_aperiodic.py`.
5. `scripts/train_model.py`.
6. `scripts/run_interventions.py`.
7. `scripts/analyze_attribution.py`.
8. `reports/figures/`.
9. `results/metrics.csv`.

### Phase 1: Simulation validation

Deliverables:

1. Simulated spectra with known ground truth.
2. Recovery plots for aperiodic and periodic effects.
3. Unit tests for ARI/AAR behavior.

### Phase 2: TDBRAIN/TUAB pilot

Deliverables:

1. Subject-level splits.
2. Classical baselines.
3. First deep baseline.
4. Aperiodic-only and flattened results.
5. Preliminary figure set.

### Phase 3: Multi-dataset benchmark

Deliverables:

1. TUAB.
2. TDBRAIN.
3. Sleep-EDF or ISRUC.
4. TUEV secondary.
5. Optional MOABB motor imagery.

### Phase 4: Foundation-model audit

Deliverables:

1. BIOT/LaBraM/EEGPT reproduction where feasible.
2. Same interventions and metrics.
3. Comparison against small baselines.

### Phase 5: Manuscript

Target venues:

1. Nature Machine Intelligence if the multi-dataset deep-learning audit is strong.
2. NeuroImage if the paper emphasizes EEG methodology and spectral interpretation.
3. IEEE TMI if clinical EEG benchmark impact dominates.
4. JMLR/TPAMI only if the theory and benchmark tooling become very strong.

## Pre-registration style analysis plan

Primary outcome:

```text
Retention_A and Drop_flat on TUAB and one sleep dataset
```

Secondary outcomes:

```text
AAR, Retention_P, matching robustness, decomposition agreement
```

Primary statistical test:

```text
paired subject-level bootstrap CI for Perf_Original - Perf_Flattened
```

Correction:

```text
Benjamini-Hochberg FDR across models and datasets
```

Reporting:

1. Every model reports original, aperiodic-only, flattened, and periodic-residual performance.
2. Every dataset reports fit-quality diagnostics.
3. Every claim states whether it is about predictive performance, attribution, or physiological interpretation.

## Implementation notes

Preferred stack:

```text
Python
MNE
NumPy/SciPy
scikit-learn
PyTorch
Braindecode
specparam/FOOOF
YASA or neurodsp for IRASA
pandas
statsmodels
seaborn/matplotlib
```

Use deterministic configs:

```text
dataset, channels, sample_rate, filter_settings, psd_method,
freq_range, decomposition_method, model, split_seed
```

All results should be keyed by a hash of the config to prevent silent reruns with changed settings.

## Scientific interpretation guide

### Case A: aperiodic reliance is high and survives controls

Interpretation:

The model uses broadband physiological state. This may be valid but should not be described as oscillatory band-power discovery.

### Case B: aperiodic reliance is high but disappears after artifact control

Interpretation:

The model likely used artifact or acquisition shortcuts. This is a benchmark validity problem.

### Case C: periodic residual carries most performance

Interpretation:

The task likely depends on genuine oscillatory or event-related spectral structure.

### Case D: both are important

Interpretation:

The task mixes rhythmic and broadband physiology. Future papers must report both.

## Final deliverable

The final paper should provide:

1. a mathematical explanation of the aperiodic confound,
2. a validated audit pipeline,
3. multi-dataset evidence,
4. code and reproducible configs,
5. a reporting checklist for EEG ML papers,
6. a clear distinction between invalid oscillatory interpretation and valid aperiodic biomarkers.

The goal is not to destroy EEG band analysis. The goal is to force EEG ML to say what its models are actually using.
