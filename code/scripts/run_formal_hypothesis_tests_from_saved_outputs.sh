#!/usr/bin/env bash
set -euo pipefail

# Statistics-only formal testing pass. This reruns bootstrap aggregation from
# saved subject metrics/predictions, extracts bootstrap p-values, and applies
# Benjamini-Hochberg FDR correction. It does not train or evaluate any model.

PROJECT_ROOT="${APERIODIC_PROJECT_ROOT:-/mnt/data/aperiodic_confounds}"
RESULT_ROOT="${PROJECT_ROOT}/results"
REPORT_ROOT="${PROJECT_ROOT}/reports/tables"
PROJECT_PY="${PROJECT_ROOT}/.venv/bin/python"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"
SEEDS="${SEEDS:-20260526 20260527}"

cd "${PROJECT_ROOT}/code"
mkdir -p "${REPORT_ROOT}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "REPORT_ROOT=${REPORT_ROOT}"
echo "N_BOOTSTRAP=${N_BOOTSTRAP}"

echo "=== Re-aggregating Sleep-EDF full MLP/CNN/EEGNet table ==="
"${PROJECT_PY}" scripts/aggregate_multiseed_subject_bootstrap.py \
  --input "deep_mlp=${RESULT_ROOT}/sleep_edf_full/multiseed_neural/deep_mlp_seed42/deep_mlp_intervention_subject_metrics.csv" \
  --input "raw_cnn=${RESULT_ROOT}/sleep_edf_full/multiseed_neural/raw_cnn_seed42/raw_cnn_intervention_subject_metrics.csv" \
  --input "braindecode_eegnet=${RESULT_ROOT}/sleep_edf_full/multiseed_neural/braindecode_eegnet_seed42/braindecode_eegnet_intervention_subject_metrics.csv" \
  --input "deep_mlp=${RESULT_ROOT}/sleep_edf_full/multiseed_neural/deep_mlp_seed43/deep_mlp_intervention_subject_metrics.csv" \
  --input "raw_cnn=${RESULT_ROOT}/sleep_edf_full/multiseed_neural/raw_cnn_seed43/raw_cnn_intervention_subject_metrics.csv" \
  --input "braindecode_eegnet=${RESULT_ROOT}/sleep_edf_full/multiseed_neural/braindecode_eegnet_seed43/braindecode_eegnet_intervention_subject_metrics.csv" \
  --input "deep_mlp=${RESULT_ROOT}/sleep_edf_full/multiseed_neural/deep_mlp_seed44/deep_mlp_intervention_subject_metrics.csv" \
  --input "raw_cnn=${RESULT_ROOT}/sleep_edf_full/multiseed_neural/raw_cnn_seed44/raw_cnn_intervention_subject_metrics.csv" \
  --input "braindecode_eegnet=${RESULT_ROOT}/sleep_edf_full/multiseed_neural/braindecode_eegnet_seed44/braindecode_eegnet_intervention_subject_metrics.csv" \
  --output-csv "${REPORT_ROOT}/full_sleep_edf_multiseed_subject_bootstrap.csv" \
  --output-md "${REPORT_ROOT}/full_sleep_edf_multiseed_subject_bootstrap.md" \
  --n-bootstrap "${N_BOOTSTRAP}"

echo "=== Re-aggregating Sleep-EDF reviewer-resistance neural table ==="
"${PROJECT_PY}" scripts/aggregate_multiseed_subject_bootstrap.py \
  --input "psd_train_control=${RESULT_ROOT}/sleep_edf_full/reviewer_resistance_controls/psd_train_input_controls_seed42/psd_train_input_control_subject_metrics.csv" \
  --input "raw_cnn_sham=${RESULT_ROOT}/sleep_edf_full/reviewer_resistance_controls/raw_cnn_sham_seed42/raw_cnn_intervention_subject_metrics.csv" \
  --input "braindecode_eegnet=${RESULT_ROOT}/sleep_edf_full/reviewer_resistance_controls/braindecode_eegnet_seed42/braindecode_eegnet_intervention_subject_metrics.csv" \
  --input "braindecode_shallow_fbcsp=${RESULT_ROOT}/sleep_edf_full/reviewer_resistance_controls/braindecode_shallow_fbcsp_seed42/braindecode_shallow_fbcsp_intervention_subject_metrics.csv" \
  --input "braindecode_deep4=${RESULT_ROOT}/sleep_edf_full/reviewer_resistance_controls/braindecode_deep4_seed42/braindecode_deep4_intervention_subject_metrics.csv" \
  --input "psd_train_control=${RESULT_ROOT}/sleep_edf_full/reviewer_resistance_controls/psd_train_input_controls_seed43/psd_train_input_control_subject_metrics.csv" \
  --input "raw_cnn_sham=${RESULT_ROOT}/sleep_edf_full/reviewer_resistance_controls/raw_cnn_sham_seed43/raw_cnn_intervention_subject_metrics.csv" \
  --input "braindecode_eegnet=${RESULT_ROOT}/sleep_edf_full/reviewer_resistance_controls/braindecode_eegnet_seed43/braindecode_eegnet_intervention_subject_metrics.csv" \
  --input "braindecode_shallow_fbcsp=${RESULT_ROOT}/sleep_edf_full/reviewer_resistance_controls/braindecode_shallow_fbcsp_seed43/braindecode_shallow_fbcsp_intervention_subject_metrics.csv" \
  --input "braindecode_deep4=${RESULT_ROOT}/sleep_edf_full/reviewer_resistance_controls/braindecode_deep4_seed43/braindecode_deep4_intervention_subject_metrics.csv" \
  --input "psd_train_control=${RESULT_ROOT}/sleep_edf_full/reviewer_resistance_controls/psd_train_input_controls_seed44/psd_train_input_control_subject_metrics.csv" \
  --input "raw_cnn_sham=${RESULT_ROOT}/sleep_edf_full/reviewer_resistance_controls/raw_cnn_sham_seed44/raw_cnn_intervention_subject_metrics.csv" \
  --input "braindecode_eegnet=${RESULT_ROOT}/sleep_edf_full/reviewer_resistance_controls/braindecode_eegnet_seed44/braindecode_eegnet_intervention_subject_metrics.csv" \
  --input "braindecode_shallow_fbcsp=${RESULT_ROOT}/sleep_edf_full/reviewer_resistance_controls/braindecode_shallow_fbcsp_seed44/braindecode_shallow_fbcsp_intervention_subject_metrics.csv" \
  --input "braindecode_deep4=${RESULT_ROOT}/sleep_edf_full/reviewer_resistance_controls/braindecode_deep4_seed44/braindecode_deep4_intervention_subject_metrics.csv" \
  --output-csv "${REPORT_ROOT}/sleep_edf_reviewer_resistance_bootstrap.csv" \
  --output-md "${REPORT_ROOT}/sleep_edf_reviewer_resistance_bootstrap.md" \
  --n-bootstrap "${N_BOOTSTRAP}"

echo "=== Re-aggregating full-TUAB neural table ==="
"${PROJECT_PY}" scripts/aggregate_multiseed_subject_bootstrap.py \
  --input "eegnet=${RESULT_ROOT}/tuab_full_v3_0_1/multiseed_neural/eegnet_seed42/tuab_full_eegnet_intervention_subject_metrics.csv" \
  --input "shallow_fbcsp=${RESULT_ROOT}/tuab_full_v3_0_1/multiseed_neural/shallow_fbcsp_seed42/tuab_full_shallow_fbcsp_intervention_subject_metrics.csv" \
  --input "deep4=${RESULT_ROOT}/tuab_full_v3_0_1/multiseed_neural/deep4_seed42/tuab_full_deep4_intervention_subject_metrics.csv" \
  --input "eegnet=${RESULT_ROOT}/tuab_full_v3_0_1/multiseed_neural/eegnet_seed43/tuab_full_eegnet_intervention_subject_metrics.csv" \
  --input "shallow_fbcsp=${RESULT_ROOT}/tuab_full_v3_0_1/multiseed_neural/shallow_fbcsp_seed43/tuab_full_shallow_fbcsp_intervention_subject_metrics.csv" \
  --input "deep4=${RESULT_ROOT}/tuab_full_v3_0_1/multiseed_neural/deep4_seed43/tuab_full_deep4_intervention_subject_metrics.csv" \
  --input "eegnet=${RESULT_ROOT}/tuab_full_v3_0_1/multiseed_neural/eegnet_seed44/tuab_full_eegnet_intervention_subject_metrics.csv" \
  --input "shallow_fbcsp=${RESULT_ROOT}/tuab_full_v3_0_1/multiseed_neural/shallow_fbcsp_seed44/tuab_full_shallow_fbcsp_intervention_subject_metrics.csv" \
  --input "deep4=${RESULT_ROOT}/tuab_full_v3_0_1/multiseed_neural/deep4_seed44/tuab_full_deep4_intervention_subject_metrics.csv" \
  --output-csv "${REPORT_ROOT}/tuab_full_multiseed_neural_subject_bootstrap.csv" \
  --output-md "${REPORT_ROOT}/tuab_full_multiseed_neural_subject_bootstrap.md" \
  --n-bootstrap "${N_BOOTSTRAP}"

echo "=== Re-aggregating full-TUAB age/sex-matched neural table ==="
"${PROJECT_PY}" scripts/aggregate_multiseed_subject_bootstrap.py \
  --input "eegnet=${RESULT_ROOT}/tuab_full_v3_0_1/age_matched/multiseed_neural/eegnet_seed42/tuab_full_age_matched_eegnet_intervention_subject_metrics.csv" \
  --input "shallow_fbcsp=${RESULT_ROOT}/tuab_full_v3_0_1/age_matched/multiseed_neural/shallow_fbcsp_seed42/tuab_full_age_matched_shallow_fbcsp_intervention_subject_metrics.csv" \
  --input "deep4=${RESULT_ROOT}/tuab_full_v3_0_1/age_matched/multiseed_neural/deep4_seed42/tuab_full_age_matched_deep4_intervention_subject_metrics.csv" \
  --input "eegnet=${RESULT_ROOT}/tuab_full_v3_0_1/age_matched/multiseed_neural/eegnet_seed43/tuab_full_age_matched_eegnet_intervention_subject_metrics.csv" \
  --input "shallow_fbcsp=${RESULT_ROOT}/tuab_full_v3_0_1/age_matched/multiseed_neural/shallow_fbcsp_seed43/tuab_full_age_matched_shallow_fbcsp_intervention_subject_metrics.csv" \
  --input "deep4=${RESULT_ROOT}/tuab_full_v3_0_1/age_matched/multiseed_neural/deep4_seed43/tuab_full_age_matched_deep4_intervention_subject_metrics.csv" \
  --input "eegnet=${RESULT_ROOT}/tuab_full_v3_0_1/age_matched/multiseed_neural/eegnet_seed44/tuab_full_age_matched_eegnet_intervention_subject_metrics.csv" \
  --input "shallow_fbcsp=${RESULT_ROOT}/tuab_full_v3_0_1/age_matched/multiseed_neural/shallow_fbcsp_seed44/tuab_full_age_matched_shallow_fbcsp_intervention_subject_metrics.csv" \
  --input "deep4=${RESULT_ROOT}/tuab_full_v3_0_1/age_matched/multiseed_neural/deep4_seed44/tuab_full_age_matched_deep4_intervention_subject_metrics.csv" \
  --output-csv "${REPORT_ROOT}/tuab_full_age_matched_multiseed_neural_subject_bootstrap.csv" \
  --output-md "${REPORT_ROOT}/tuab_full_age_matched_multiseed_neural_subject_bootstrap.md" \
  --n-bootstrap "${N_BOOTSTRAP}"

echo "=== Re-aggregating PhysioNet MI neural table ==="
"${PROJECT_PY}" scripts/aggregate_multiseed_subject_bootstrap.py \
  --input "eegnet=${RESULT_ROOT}/physionet_mi/multiseed_neural/eegnet_seed42/physionet_mi_eegnet_intervention_subject_metrics.csv" \
  --input "shallow_fbcsp=${RESULT_ROOT}/physionet_mi/multiseed_neural/shallow_fbcsp_seed42/physionet_mi_shallow_fbcsp_intervention_subject_metrics.csv" \
  --input "deep4=${RESULT_ROOT}/physionet_mi/multiseed_neural/deep4_seed42/physionet_mi_deep4_intervention_subject_metrics.csv" \
  --input "eegnet=${RESULT_ROOT}/physionet_mi/multiseed_neural/eegnet_seed43/physionet_mi_eegnet_intervention_subject_metrics.csv" \
  --input "shallow_fbcsp=${RESULT_ROOT}/physionet_mi/multiseed_neural/shallow_fbcsp_seed43/physionet_mi_shallow_fbcsp_intervention_subject_metrics.csv" \
  --input "deep4=${RESULT_ROOT}/physionet_mi/multiseed_neural/deep4_seed43/physionet_mi_deep4_intervention_subject_metrics.csv" \
  --input "eegnet=${RESULT_ROOT}/physionet_mi/multiseed_neural/eegnet_seed44/physionet_mi_eegnet_intervention_subject_metrics.csv" \
  --input "shallow_fbcsp=${RESULT_ROOT}/physionet_mi/multiseed_neural/shallow_fbcsp_seed44/physionet_mi_shallow_fbcsp_intervention_subject_metrics.csv" \
  --input "deep4=${RESULT_ROOT}/physionet_mi/multiseed_neural/deep4_seed44/physionet_mi_deep4_intervention_subject_metrics.csv" \
  --output-csv "${REPORT_ROOT}/physionet_mi_multiseed_neural_subject_bootstrap.csv" \
  --output-md "${REPORT_ROOT}/physionet_mi_multiseed_neural_subject_bootstrap.md" \
  --n-bootstrap "${N_BOOTSTRAP}"

fm_args=()
for spec in \
  "BIOT:20260524:${RESULT_ROOT}/tuab_full_v3_0_1/biot_interventions_prest_full/tuab_biot_intervention_predictions.csv" \
  "BIOT:20260526:${RESULT_ROOT}/tuab_full_v3_0_1/foundation_multiseed/biot_seed20260526/tuab_biot_intervention_predictions.csv" \
  "BIOT:20260527:${RESULT_ROOT}/tuab_full_v3_0_1/foundation_multiseed/biot_seed20260527/tuab_biot_intervention_predictions.csv" \
  "LaBraM:20260524:${RESULT_ROOT}/tuab_full_v3_0_1/labram_interventions_base_full/tuab_labram_intervention_predictions.csv" \
  "LaBraM:20260526:${RESULT_ROOT}/tuab_full_v3_0_1/foundation_multiseed/labram_seed20260526/tuab_labram_intervention_predictions.csv" \
  "LaBraM:20260527:${RESULT_ROOT}/tuab_full_v3_0_1/foundation_multiseed/labram_seed20260527/tuab_labram_intervention_predictions.csv" \
  "EEGPT:20260525:${RESULT_ROOT}/tuab_full_v3_0_1/eegpt_interventions_braindecode_full/tuab_eegpt_intervention_predictions.csv" \
  "EEGPT:20260526:${RESULT_ROOT}/tuab_full_v3_0_1/foundation_multiseed/eegpt_seed20260526/tuab_eegpt_intervention_predictions.csv" \
  "EEGPT:20260527:${RESULT_ROOT}/tuab_full_v3_0_1/foundation_multiseed/eegpt_seed20260527/tuab_eegpt_intervention_predictions.csv" \
  "CBraMod:20260525:${RESULT_ROOT}/tuab_full_v3_0_1/cbramod_interventions_braindecode_full/tuab_cbramod_intervention_predictions.csv" \
  "CBraMod:20260526:${RESULT_ROOT}/tuab_full_v3_0_1/foundation_multiseed/cbramod_seed20260526/tuab_cbramod_intervention_predictions.csv" \
  "CBraMod:20260527:${RESULT_ROOT}/tuab_full_v3_0_1/foundation_multiseed/cbramod_seed20260527/tuab_cbramod_intervention_predictions.csv" \
  "REVE-base:20260525:${RESULT_ROOT}/tuab_full_v3_0_1/reve_base_interventions_full/tuab_reve_intervention_predictions.csv" \
  "REVE-base:20260526:${RESULT_ROOT}/tuab_full_v3_0_1/foundation_multiseed/reve_seed20260526/tuab_reve_intervention_predictions.csv" \
  "REVE-base:20260527:${RESULT_ROOT}/tuab_full_v3_0_1/foundation_multiseed/reve_seed20260527/tuab_reve_intervention_predictions.csv" \
  "EEGMamba:20260525:${RESULT_ROOT}/tuab_full_v3_0_1/eegmamba_interventions_official_full/tuab_eegmamba_intervention_predictions.csv" \
  "EEGMamba:20260526:${RESULT_ROOT}/tuab_full_v3_0_1/foundation_multiseed/eegmamba_seed20260526/tuab_eegmamba_intervention_predictions.csv" \
  "EEGMamba:20260527:${RESULT_ROOT}/tuab_full_v3_0_1/foundation_multiseed/eegmamba_seed20260527/tuab_eegmamba_intervention_predictions.csv" \
  "BENDR:20260525:${RESULT_ROOT}/tuab_full_v3_0_1/bendr_interventions_braindecode_encoder_only_full/tuab_bendr_intervention_predictions.csv" \
  "BENDR:20260526:${RESULT_ROOT}/tuab_full_v3_0_1/foundation_multiseed/bendr_seed20260526/tuab_bendr_intervention_predictions.csv" \
  "BENDR:20260527:${RESULT_ROOT}/tuab_full_v3_0_1/foundation_multiseed/bendr_seed20260527/tuab_bendr_intervention_predictions.csv"; do
  IFS=":" read -r model seed path <<< "${spec}"
  if [[ -f "${path}" ]]; then
    fm_args+=(--input "${model}:${seed}=${path}")
  else
    echo "WARNING: missing FM prediction file: ${path}"
  fi
done

echo "=== Re-aggregating full-TUAB foundation-model table ==="
"${PROJECT_PY}" scripts/aggregate_foundation_multiseed_predictions.py \
  "${fm_args[@]}" \
  --output-csv "${REPORT_ROOT}/tuab_full_foundation_multiseed_subject_bootstrap.csv" \
  --output-md "${REPORT_ROOT}/tuab_full_foundation_multiseed_subject_bootstrap.md" \
  --n-bootstrap "${N_BOOTSTRAP}"

echo "=== Applying BH-FDR correction ==="
"${PROJECT_PY}" scripts/collect_formal_hypothesis_tests.py \
  --project-root "${PROJECT_ROOT}" \
  --q 0.05 \
  --output-prefix "${REPORT_ROOT}/formal_hypothesis_tests"
