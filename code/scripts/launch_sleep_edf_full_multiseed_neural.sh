#!/usr/bin/env bash
set -euo pipefail

# Run all neural intervention models across multiple seeds on the full
# Sleep-EDF preprocessing outputs. This script writes additive result folders
# only and never deletes existing outputs.

PROJECT_ROOT="${APERIODIC_PROJECT_ROOT:-.}"
RESULT_ROOT="${PROJECT_ROOT}/results/sleep_edf_full"
RUN_ROOT="${RESULT_ROOT}/multiseed_neural"
REPORT_ROOT="${PROJECT_ROOT}/reports/tables"
TORCH_PY="${TORCH_PY:-python}"
PROJECT_PY="${PROJECT_ROOT}/.venv/bin/python"
SEEDS="${SEEDS:-42 43 44}"
N_SPLITS="${N_SPLITS:-5}"
DEVICE="${DEVICE:-cuda}"
BRAIND_DEPS="${PROJECT_ROOT}/.python_deps"

cd "${PROJECT_ROOT}/code"
mkdir -p "${RUN_ROOT}" "${REPORT_ROOT}"

DEEP_SUBJECT_INPUTS=()
RAW_SUBJECT_INPUTS=()
EEGNET_SUBJECT_INPUTS=()

for seed in ${SEEDS}; do
  echo "=== seed ${seed}: deep MLP PSD ==="
  DEEP_DIR="${RUN_ROOT}/deep_mlp_seed${seed}"
  "${TORCH_PY}" scripts/run_sleep_edf_deep_intervention_mlp.py \
    --index-csv "${RESULT_ROOT}/psd_index.csv" \
    --decomp-npz "${RESULT_ROOT}/specparam/specparam_fixed.npz" \
    --output-dir "${DEEP_DIR}" \
    --n-splits "${N_SPLITS}" \
    --seed "${seed}" \
    --device "${DEVICE}"
  DEEP_SUBJECT_INPUTS+=("deep_mlp=${DEEP_DIR}/deep_mlp_intervention_subject_metrics.csv")

  echo "=== seed ${seed}: custom raw CNN ==="
  RAW_DIR="${RUN_ROOT}/raw_cnn_seed${seed}"
  "${TORCH_PY}" scripts/run_sleep_edf_raw_cnn_intervention.py \
    --raw-npz "${RESULT_ROOT}/raw_epochs_fpz_pz_100hz.npz" \
    --index-csv "${RESULT_ROOT}/raw_epochs_index.csv" \
    --decomp-npz "${RESULT_ROOT}/specparam/specparam_fixed.npz" \
    --output-dir "${RAW_DIR}" \
    --n-splits "${N_SPLITS}" \
    --seed "${seed}" \
    --device "${DEVICE}"
  RAW_SUBJECT_INPUTS+=("raw_cnn=${RAW_DIR}/raw_cnn_intervention_subject_metrics.csv")

  echo "=== seed ${seed}: Braindecode EEGNet ==="
  EEGNET_DIR="${RUN_ROOT}/braindecode_eegnet_seed${seed}"
  PYTHONPATH="${BRAIND_DEPS}" "${TORCH_PY}" \
    scripts/run_sleep_edf_braindecode_eegnet_intervention.py \
    --raw-npz "${RESULT_ROOT}/raw_epochs_fpz_pz_100hz.npz" \
    --index-csv "${RESULT_ROOT}/raw_epochs_index.csv" \
    --decomp-npz "${RESULT_ROOT}/specparam/specparam_fixed.npz" \
    --output-dir "${EEGNET_DIR}" \
    --n-splits "${N_SPLITS}" \
    --seed "${seed}" \
    --device "${DEVICE}"
  EEGNET_SUBJECT_INPUTS+=("braindecode_eegnet=${EEGNET_DIR}/braindecode_eegnet_intervention_subject_metrics.csv")
done

AGG_INPUTS=()
for item in "${DEEP_SUBJECT_INPUTS[@]}" "${RAW_SUBJECT_INPUTS[@]}" "${EEGNET_SUBJECT_INPUTS[@]}"; do
  AGG_INPUTS+=(--input "${item}")
done

"${PROJECT_PY}" scripts/aggregate_multiseed_subject_bootstrap.py \
  "${AGG_INPUTS[@]}" \
  --output-csv "${REPORT_ROOT}/full_sleep_edf_multiseed_subject_bootstrap.csv" \
  --output-md "${REPORT_ROOT}/full_sleep_edf_multiseed_subject_bootstrap.md" \
  --n-bootstrap "${N_BOOTSTRAP:-10000}"

echo "Multiseed neural runs complete: ${RUN_ROOT}"
