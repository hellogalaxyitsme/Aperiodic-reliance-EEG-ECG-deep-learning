#!/usr/bin/env bash
set -euo pipefail

# Reviewer-resistance package for the full Sleep-EDF run. This script is
# additive only: it creates new result/report folders and never deletes data.

PROJECT_ROOT="${APERIODIC_PROJECT_ROOT:-/mnt/data/aperiodic_confounds}"
RESULT_ROOT="${PROJECT_ROOT}/results/sleep_edf_full"
RUN_ROOT="${RESULT_ROOT}/reviewer_resistance_controls"
REPORT_ROOT="${PROJECT_ROOT}/reports/tables"
TORCH_PY="${TORCH_PY:-/mnt/data/.venvs/ml/bin/python3}"
PROJECT_PY="${PROJECT_ROOT}/.venv/bin/python"
SEEDS="${SEEDS:-42 43 44}"
N_SPLITS="${N_SPLITS:-5}"
DEVICE="${DEVICE:-cuda}"
BRAIND_DEPS="${PROJECT_ROOT}/.python_deps"

RUN_RAW_DIAGNOSTICS="${RUN_RAW_DIAGNOSTICS:-1}"
RUN_SIMULATIONS="${RUN_SIMULATIONS:-1}"
RUN_PSD_CONTROLS="${RUN_PSD_CONTROLS:-1}"
RUN_RAW_CNN_SHAM="${RUN_RAW_CNN_SHAM:-1}"
RUN_BRAINDECODE_ARCHS="${RUN_BRAINDECODE_ARCHS:-1}"
RUN_IRASA="${RUN_IRASA:-1}"
RUN_AGGREGATION="${RUN_AGGREGATION:-1}"

BRAIND_ARCHS="${BRAIND_ARCHS:-eegnet shallow_fbcsp deep4 usleep eegconformer}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"

cd "${PROJECT_ROOT}/code"
mkdir -p "${RUN_ROOT}" "${REPORT_ROOT}"

if [[ "${RUN_RAW_DIAGNOSTICS}" == "1" ]]; then
  echo "=== raw intervention distribution diagnostics ==="
  "${PROJECT_PY}" scripts/analyze_raw_intervention_diagnostics.py \
    --raw-npz "${RESULT_ROOT}/raw_epochs_fpz_pz_100hz.npz" \
    --index-csv "${RESULT_ROOT}/raw_epochs_index.csv" \
    --decomp-npz "${RESULT_ROOT}/specparam/specparam_fixed.npz" \
    --output-dir "${REPORT_ROOT}/raw_intervention_diagnostics"
fi

if [[ "${RUN_SIMULATIONS}" == "1" ]]; then
  echo "=== simulation validation ==="
  "${PROJECT_PY}" scripts/run_aperiodic_simulation_validation.py \
    --output-dir "${PROJECT_ROOT}/results/simulations/aperiodic_validation"
fi

PSD_SUBJECT_INPUTS=()
if [[ "${RUN_PSD_CONTROLS}" == "1" ]]; then
  for seed in ${SEEDS}; do
    echo "=== seed ${seed}: PSD train-input controls ==="
    PSD_DIR="${RUN_ROOT}/psd_train_input_controls_seed${seed}"
    "${TORCH_PY}" scripts/run_sleep_edf_psd_train_input_controls.py \
      --index-csv "${RESULT_ROOT}/psd_index.csv" \
      --decomp-npz "${RESULT_ROOT}/specparam/specparam_fixed.npz" \
      --output-dir "${PSD_DIR}" \
      --n-splits "${N_SPLITS}" \
      --seed "${seed}" \
      --device "${DEVICE}"
    PSD_SUBJECT_INPUTS+=("psd_train_control=${PSD_DIR}/psd_train_input_control_subject_metrics.csv")
  done
fi

RAW_SUBJECT_INPUTS=()
if [[ "${RUN_RAW_CNN_SHAM}" == "1" ]]; then
  for seed in ${SEEDS}; do
    echo "=== seed ${seed}: custom raw CNN with sham control ==="
    RAW_DIR="${RUN_ROOT}/raw_cnn_sham_seed${seed}"
    "${TORCH_PY}" scripts/run_sleep_edf_raw_cnn_intervention.py \
      --raw-npz "${RESULT_ROOT}/raw_epochs_fpz_pz_100hz.npz" \
      --index-csv "${RESULT_ROOT}/raw_epochs_index.csv" \
      --decomp-npz "${RESULT_ROOT}/specparam/specparam_fixed.npz" \
      --output-dir "${RAW_DIR}" \
      --n-splits "${N_SPLITS}" \
      --seed "${seed}" \
      --device "${DEVICE}"
    RAW_SUBJECT_INPUTS+=("raw_cnn_sham=${RAW_DIR}/raw_cnn_intervention_subject_metrics.csv")
  done
fi

BRAINDECODE_SUBJECT_INPUTS=()
if [[ "${RUN_BRAINDECODE_ARCHS}" == "1" ]]; then
  for arch in ${BRAIND_ARCHS}; do
    for seed in ${SEEDS}; do
      echo "=== seed ${seed}: Braindecode ${arch} with sham control ==="
      ARCH_DIR="${RUN_ROOT}/braindecode_${arch}_seed${seed}"
      PYTHONPATH="${BRAIND_DEPS}" "${TORCH_PY}" \
        scripts/run_sleep_edf_braindecode_eegnet_intervention.py \
        --model "${arch}" \
        --raw-npz "${RESULT_ROOT}/raw_epochs_fpz_pz_100hz.npz" \
        --index-csv "${RESULT_ROOT}/raw_epochs_index.csv" \
        --decomp-npz "${RESULT_ROOT}/specparam/specparam_fixed.npz" \
        --output-dir "${ARCH_DIR}" \
        --n-splits "${N_SPLITS}" \
        --seed "${seed}" \
        --device "${DEVICE}"
      BRAINDECODE_SUBJECT_INPUTS+=("braindecode_${arch}=${ARCH_DIR}/braindecode_${arch}_intervention_subject_metrics.csv")
    done
  done
fi

if [[ "${RUN_IRASA}" == "1" ]]; then
  echo "=== IRASA decomposition agreement ==="
  IRASA_ARGS=()
  if [[ -n "${IRASA_MAX_EPOCHS:-}" ]]; then
    IRASA_ARGS+=(--max-epochs "${IRASA_MAX_EPOCHS}")
  fi
  "${PROJECT_PY}" scripts/fit_sleep_edf_irasa.py \
    --raw-npz "${RESULT_ROOT}/raw_epochs_fpz_pz_100hz.npz" \
    --output-npz "${RESULT_ROOT}/irasa/irasa_aperiodic.npz" \
    "${IRASA_ARGS[@]}"
  "${PROJECT_PY}" scripts/compare_specparam_irasa.py \
    --specparam-npz "${RESULT_ROOT}/specparam/specparam_fixed.npz" \
    --irasa-npz "${RESULT_ROOT}/irasa/irasa_aperiodic.npz" \
    --output-dir "${REPORT_ROOT}/irasa_specparam_agreement"
fi

if [[ "${RUN_AGGREGATION}" == "1" ]]; then
  AGG_INPUTS=()
  for item in "${PSD_SUBJECT_INPUTS[@]}" "${RAW_SUBJECT_INPUTS[@]}" "${BRAINDECODE_SUBJECT_INPUTS[@]}"; do
    AGG_INPUTS+=(--input "${item}")
  done
  if [[ "${#AGG_INPUTS[@]}" -gt 0 ]]; then
    "${PROJECT_PY}" scripts/aggregate_multiseed_subject_bootstrap.py \
      "${AGG_INPUTS[@]}" \
      --output-csv "${REPORT_ROOT}/sleep_edf_reviewer_resistance_bootstrap.csv" \
      --output-md "${REPORT_ROOT}/sleep_edf_reviewer_resistance_bootstrap.md" \
      --n-bootstrap "${N_BOOTSTRAP}"
  fi
fi

echo "Reviewer-resistance controls complete: ${RUN_ROOT}"
