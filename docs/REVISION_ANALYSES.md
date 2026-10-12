# Revision Analyses

Code for the analyses added during peer review is in `code/revision/`, and the
aggregate outputs are in `results/revision/`. All analyses reuse the original
preprocessing, data splits, training protocols and seeds. The full sequence of
commands is in `code/revision/run_revision_pipeline.sh`.

```bash
export RESULTS_DIR=/path/to/results          # results root of the original analyses
export REVISION_DIR=/path/to/revision_output
export BENDR_CHECKPOINT=/path/to/braindecode-bendr/model.safetensors
bash code/revision/run_revision_pipeline.sh  # STAGES="fits experiments aggregate checks"
```

## Analyses

| Analysis | Scripts | Aggregate output (`results/revision/`) |
|---|---|---|
| Envelope-estimator sensitivity: SpecParam versus fixed linear fit on the same TUAB EEGNet models (Table S5) | `tuab_fit_procedure_sensitivity.py`, `fit_tuab_specparam_eval.py` | `envelope_estimator_sensitivity_tuab_eegnet.csv` |
| PTB-XL envelope estimators: linear, harmonic-masked, SpecParam (with peaks) and filter-bank equalization (Table S6) | `ptbxl_revision_interventions.py`, `fit_ptbxl_specparam_test.py` | `ptbxl_envelope_estimators_*.csv`, `ptbxl_test_specparam_fit_summary.json` |
| Median-beat morphology baseline (Table S6) | `ptbxl_morphology_baseline.py` | `ptbxl_morphology_baseline.csv` |
| Model-free envelope diagnostics for the ECG operators | `ptbxl_envelope_removal_diagnostics.py` | `ptbxl_envelope_removal_diagnostics.csv` |
| Time-domain waveform changes and overlays (Table S7, Fig. S5) | `waveform_distortion_audit.py`, `fig_time_domain_overlays.py` | `waveform_changes.csv` |
| Training-time flattening augmentation (Table S8) | `tuab_flataug_mitigation.py`, `ptbxl_flataug_mitigation.py`, `flataug_torch.py`, `paired_mitigation_bootstrap.py` | `mitigation_tuab_eegnet.csv`, `mitigation_ptbxl_resnet1d_wang.csv` |
| TUAB 250-Hz evaluation subset | `aggregate_revision_bootstrap.py` | `tuab_250hz_subset_*.csv` |
| Conceptual schematic (Fig. S4) | `fig_reliance_schematic.py` | – |

The robustness and mitigation analyses use models retrained with the original
protocol and seeds. Because GPU training is not bit-for-bit reproducible, their
baseline values differ slightly from the main results; every comparison within
these analyses is paired on the same trained models.

## Corrections

**BENDR input scaling.** In the original `code/scripts/run_tuab_bendr_intervention.py`,
the uV-to-V conversion was applied twice to the edited (sham, aperiodic-shaped and
flattened) inputs. The script is corrected in this repository; the reported values
come from `code/revision/run_tuab_bendr_intervention_fixed.py`, which applies the
conversion once, after the intervention, for raw and edited inputs alike.
Outputs: `bendr_corrected.csv`, `sham_integrity.csv`.

**Balanced accuracy.** TUAB conventional-model results (full and age/sex-matched)
and the unmatched PTB-XL PSD ridge baseline are computed as balanced accuracy pooled
over evaluation windows or records and seeds, as for the foundation models and the
PTB-XL neural models. Outputs: `tuab_conventional_full_pooled_ba.csv`,
`tuab_conventional_matched_pooled_ba.csv`, `ptbxl_psd_ridge_pooled_ba.csv`, and the
complete formal test families `formal_tests_primary.csv` and `formal_tests_sham.csv`
(Benjamini-Hochberg FDR). These supersede the corresponding rows in `results/tables/`.

## Checks

- `ba_evaluator_validation.txt` (`validate_ba_evaluators.py`, `ba_bootstrap_sanity.py`):
  agreement with an independent implementation and scikit-learn, and synthetic
  constant- and perfect-prediction tests.
- `shared_subject_bootstrap_check/` (`shared_subject_check.py`): TUAB intervals with
  one subject draw shared across seeds, and both formal test families recomputed.
- `binary_sleep_mi/` (`reconstruct_binary_counts.py`): subject-wise and pooled
  balanced accuracy for the binary Sleep-EDF and PhysioNet MI tasks (Table S9).
