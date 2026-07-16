#!/usr/bin/env bash
set -euo pipefail

# Full-TUAB EEGPT foundation-model audit. EEGPT uses the same 23-channel
# LaBraM-style TUAB cache, so this launcher reuses the full-TUAB LaBraM cache.

PROJECT_ROOT="${APERIODIC_PROJECT_ROOT:-.}"
RESULT_ROOT="${PROJECT_ROOT}/results/tuab_full_v3_0_1"
RUN_ROOT="${RUN_ROOT:-${RESULT_ROOT}/eegpt_interventions_braindecode_full}"
CACHE_NPZ="${RESULT_ROOT}/labram_10s_200hz_cache.npz"
TORCH_PY="${TORCH_PY:-python}"
BRAIND_DEPS="${PROJECT_ROOT}/.python_deps"
EEGPT_CKPT="${PROJECT_ROOT}/external/EEGPT/checkpoint/braindecode_eegpt_pretrained_pytorch_model.bin"

EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
INTERVENTION_BATCH_SIZE="${INTERVENTION_BATCH_SIZE:-256}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"
SEED="${SEED:-20260525}"
DEVICE="${DEVICE:-cuda}"

cd "${PROJECT_ROOT}"
mkdir -p "${RUN_ROOT}" "${RESULT_ROOT}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "CACHE_NPZ=${CACHE_NPZ}"
echo "TORCH_PY=${TORCH_PY}"
echo "BRAIND_DEPS=${BRAIND_DEPS}"
echo "EEGPT_CKPT=${EEGPT_CKPT}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "INTERVENTION_BATCH_SIZE=${INTERVENTION_BATCH_SIZE}"
echo "N_BOOTSTRAP=${N_BOOTSTRAP}"
echo "SEED=${SEED}"
echo "DEVICE=${DEVICE}"

PYTHONPATH="${BRAIND_DEPS}" "${TORCH_PY}" code/scripts/run_tuab_eegpt_intervention.py \
  --cache-npz "${CACHE_NPZ}" \
  --cache-format npy \
  --output-dir "${RUN_ROOT}" \
  --checkpoint-path "${EEGPT_CKPT}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --intervention-batch-size "${INTERVENTION_BATCH_SIZE}" \
  --n-bootstrap "${N_BOOTSTRAP}" \
  --seed "${SEED}" \
  --device "${DEVICE}"

echo "Full TUAB EEGPT intervention complete: ${RUN_ROOT}"
