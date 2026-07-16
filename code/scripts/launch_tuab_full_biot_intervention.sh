#!/usr/bin/env bash
set -euo pipefail

# Full-TUAB BIOT foundation-model audit. This mirrors the TUAB-200 BIOT
# protocol while using a memmap-backed cache so the full dataset does not need
# to be duplicated in host memory.

PROJECT_ROOT="${APERIODIC_PROJECT_ROOT:-.}"
RESULT_ROOT="${PROJECT_ROOT}/results/tuab_full_v3_0_1"
DATA_ROOT="${PROJECT_ROOT}/data/tuab/v3.0.1_random_stratified_200"
SELECTED_FILES_CSV="${RESULT_ROOT}/tuab_v3_0_1_full_edf_files.csv"
RUN_ROOT="${RUN_ROOT:-${RESULT_ROOT}/biot_interventions_prest_full}"
CACHE_NPZ="${RESULT_ROOT}/biot_10s_200hz_cache.npz"
TORCH_PY="${TORCH_PY:-python}"
BIOT_REPO="${PROJECT_ROOT}/external/BIOT"
BIOT_CKPT="${BIOT_REPO}/pretrained-models/EEG-PREST-16-channels.ckpt"

EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-256}"
INTERVENTION_BATCH_SIZE="${INTERVENTION_BATCH_SIZE:-512}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"
SEED="${SEED:-20260524}"
DEVICE="${DEVICE:-cuda}"

cd "${PROJECT_ROOT}"
mkdir -p "${RUN_ROOT}" "${RESULT_ROOT}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "SELECTED_FILES_CSV=${SELECTED_FILES_CSV}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "CACHE_NPZ=${CACHE_NPZ}"
echo "TORCH_PY=${TORCH_PY}"
echo "BIOT_REPO=${BIOT_REPO}"
echo "BIOT_CKPT=${BIOT_CKPT}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "INTERVENTION_BATCH_SIZE=${INTERVENTION_BATCH_SIZE}"
echo "N_BOOTSTRAP=${N_BOOTSTRAP}"
echo "SEED=${SEED}"
echo "DEVICE=${DEVICE}"

"${TORCH_PY}" code/scripts/run_tuab_biot_intervention.py \
  --data-root "${DATA_ROOT}" \
  --selected-files-csv "${SELECTED_FILES_CSV}" \
  --cache-npz "${CACHE_NPZ}" \
  --cache-format npy \
  --output-dir "${RUN_ROOT}" \
  --biot-repo "${BIOT_REPO}" \
  --pretrain-model-path "${BIOT_CKPT}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --intervention-batch-size "${INTERVENTION_BATCH_SIZE}" \
  --n-bootstrap "${N_BOOTSTRAP}" \
  --seed "${SEED}" \
  --device "${DEVICE}"

echo "Full TUAB BIOT intervention complete: ${RUN_ROOT}"
