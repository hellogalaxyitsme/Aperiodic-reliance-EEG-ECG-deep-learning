#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${APERIODIC_PROJECT_ROOT:-.}"
RESULT_ROOT="${PROJECT_ROOT}/results/ptbxl_1f_demo"
REPORT_ROOT="${PROJECT_ROOT}/reports/tables"
PROJECT_PY="${PROJECT_ROOT}/.venv/bin/python"
TORCH_PY="${TORCH_PY:-python}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-42 43 44}"
EPOCHS="${EPOCHS:-25}"
BATCH_SIZE="${BATCH_SIZE:-256}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"

cd "${PROJECT_ROOT}/code"
mkdir -p "${RESULT_ROOT}" "${REPORT_ROOT}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "SEEDS=${SEEDS}"
echo "EPOCHS=${EPOCHS}"
echo "DEVICE=${DEVICE}"

if [[ "${PTBXL_DOWNLOAD_MODE:-records100}" == "bulk_zip" ]]; then
  bash scripts/download_ptbxl_bulk_zip.sh
elif [[ "${PTBXL_DOWNLOAD_MODE:-records100}" == "s3_records100" ]]; then
  bash scripts/download_ptbxl_records100_s3.sh
else
  bash scripts/download_ptbxl_records100.sh
fi

"${PROJECT_PY}" scripts/prepare_ptbxl_1f_demo.py \
  --data-root "${PROJECT_ROOT}/data/ptbxl/1.0.3" \
  --output-dir "${RESULT_ROOT}"

"${PROJECT_PY}" scripts/run_ptbxl_psd_interventions.py \
  --psd-npz "${RESULT_ROOT}/ptbxl_records100_normal_abnormal_psd_fixed.npz" \
  --output-dir "${RESULT_ROOT}/psd_interventions" \
  --n-bootstrap "${N_BOOTSTRAP}"

AGG_ARGS=()
for seed in ${SEEDS}; do
  OUT_DIR="${RESULT_ROOT}/raw_cnn_seed${seed}"
  "${TORCH_PY}" scripts/run_ptbxl_raw_cnn_interventions.py \
    --raw-npz "${RESULT_ROOT}/ptbxl_records100_normal_abnormal_raw.npz" \
    --output-dir "${OUT_DIR}" \
    --seed "${seed}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}"
  AGG_ARGS+=(--input "ptbxl_raw_cnn=${OUT_DIR}/ptbxl_raw_cnn_intervention_subject_metrics.csv")
done

"${PROJECT_PY}" scripts/aggregate_multiseed_subject_bootstrap.py \
  "${AGG_ARGS[@]}" \
  --output-csv "${REPORT_ROOT}/ptbxl_raw_cnn_multiseed_subject_bootstrap.csv" \
  --output-md "${REPORT_ROOT}/ptbxl_raw_cnn_multiseed_subject_bootstrap.md" \
  --n-bootstrap "${N_BOOTSTRAP}"

echo "PTB-XL 1/f demo complete."
echo "PSD: ${RESULT_ROOT}/psd_interventions/ptbxl_psd_intervention_subject_bootstrap.csv"
echo "Raw CNN: ${REPORT_ROOT}/ptbxl_raw_cnn_multiseed_subject_bootstrap.csv"
