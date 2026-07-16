#!/usr/bin/env bash
set -euo pipefail

# Full-TUAB LaBraM foundation-model audit. This mirrors the TUAB-200 LaBraM
# protocol while using a memmap-backed cache for the full dataset.

PROJECT_ROOT="${APERIODIC_PROJECT_ROOT:-.}"
RESULT_ROOT="${PROJECT_ROOT}/results/tuab_full_v3_0_1"
DATA_ROOT="${PROJECT_ROOT}/data/tuab/v3.0.1_random_stratified_200"
SELECTED_FILES_CSV="${RESULT_ROOT}/tuab_v3_0_1_full_edf_files.csv"
RUN_ROOT="${RUN_ROOT:-${RESULT_ROOT}/labram_interventions_base_full}"
CACHE_NPZ="${RESULT_ROOT}/labram_10s_200hz_cache.npz"
TORCH_PY="${TORCH_PY:-python}"
LABRAM_REPO="${PROJECT_ROOT}/external/LaBraM"
LABRAM_CKPT="${LABRAM_REPO}/checkpoints/labram-base.pth"

EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
INTERVENTION_BATCH_SIZE="${INTERVENTION_BATCH_SIZE:-256}"
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
echo "LABRAM_REPO=${LABRAM_REPO}"
echo "LABRAM_CKPT=${LABRAM_CKPT}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "INTERVENTION_BATCH_SIZE=${INTERVENTION_BATCH_SIZE}"
echo "N_BOOTSTRAP=${N_BOOTSTRAP}"
echo "SEED=${SEED}"
echo "DEVICE=${DEVICE}"

"${TORCH_PY}" code/scripts/run_tuab_labram_intervention.py \
  --data-root "${DATA_ROOT}" \
  --selected-files-csv "${SELECTED_FILES_CSV}" \
  --cache-npz "${CACHE_NPZ}" \
  --cache-format npy \
  --output-dir "${RUN_ROOT}" \
  --labram-repo "${LABRAM_REPO}" \
  --checkpoint-path "${LABRAM_CKPT}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --intervention-batch-size "${INTERVENTION_BATCH_SIZE}" \
  --n-bootstrap "${N_BOOTSTRAP}" \
  --seed "${SEED}" \
  --device "${DEVICE}"

echo "Full TUAB LaBraM intervention complete: ${RUN_ROOT}"
