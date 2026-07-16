#!/usr/bin/env bash
set -euo pipefail

# Full-TUAB CBraMod foundation-model audit. CBraMod uses the same 23-channel
# LaBraM-style TUAB cache, then derives its official 16-channel bipolar montage
# batch-by-batch inside the runner.

PROJECT_ROOT="${APERIODIC_PROJECT_ROOT:-.}"
RESULT_ROOT="${PROJECT_ROOT}/results/tuab_full_v3_0_1"
RUN_ROOT="${RUN_ROOT:-${RESULT_ROOT}/cbramod_interventions_braindecode_full}"
CACHE_NPZ="${RESULT_ROOT}/labram_10s_200hz_cache.npz"
TORCH_PY="${TORCH_PY:-python}"
BRAIND_DEPS="${PROJECT_ROOT}/.python_deps"
CBRAMOD_REPO="${CBRAMOD_REPO:-braindecode/cbramod-pretrained}"
CBRAMOD_CKPT="${CBRAMOD_CKPT:-external/checkpoints/models--braindecode--cbramod-pretrained/snapshots/584cdc415913739a05d84bf0c1cb3db397764507/model.safetensors}"

EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
INTERVENTION_BATCH_SIZE="${INTERVENTION_BATCH_SIZE:-256}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"
SEED="${SEED:-20260525}"
DEVICE="${DEVICE:-cuda}"
INPUT_DIVISOR="${INPUT_DIVISOR:-100.0}"

cd "${PROJECT_ROOT}"
mkdir -p "${RUN_ROOT}" "${RESULT_ROOT}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "CACHE_NPZ=${CACHE_NPZ}"
echo "TORCH_PY=${TORCH_PY}"
echo "BRAIND_DEPS=${BRAIND_DEPS}"
echo "CBRAMOD_REPO=${CBRAMOD_REPO}"
echo "CBRAMOD_CKPT=${CBRAMOD_CKPT}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "INTERVENTION_BATCH_SIZE=${INTERVENTION_BATCH_SIZE}"
echo "N_BOOTSTRAP=${N_BOOTSTRAP}"
echo "SEED=${SEED}"
echo "DEVICE=${DEVICE}"
echo "INPUT_DIVISOR=${INPUT_DIVISOR}"

PYTHONPATH="${BRAIND_DEPS}" "${TORCH_PY}" code/scripts/run_tuab_cbramod_intervention.py \
  --cache-npz "${CACHE_NPZ}" \
  --cache-format npy \
  --output-dir "${RUN_ROOT}" \
  --checkpoint-path "${CBRAMOD_CKPT}" \
  --pretrained-repo "${CBRAMOD_REPO}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --intervention-batch-size "${INTERVENTION_BATCH_SIZE}" \
  --n-bootstrap "${N_BOOTSTRAP}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --input-divisor "${INPUT_DIVISOR}"

echo "Full TUAB CBraMod intervention complete: ${RUN_ROOT}"
