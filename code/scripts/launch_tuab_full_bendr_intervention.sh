#!/usr/bin/env bash
set -euo pipefail

# Full-TUAB BENDR foundation-model audit. This mirrors the TUAB-200 reportable
# BENDR run: encoder-only downstream mode, clipped/scaled 23-channel TUAB input,
# and the standard raw/sham/aperiodic/flattened intervention audit.

PROJECT_ROOT="${APERIODIC_PROJECT_ROOT:-.}"
RESULT_ROOT="${PROJECT_ROOT}/results/tuab_full_v3_0_1"
RUN_ROOT="${RUN_ROOT:-${RESULT_ROOT}/bendr_interventions_braindecode_encoder_only_full}"
CACHE_NPZ="${RESULT_ROOT}/labram_10s_200hz_cache.npz"
TORCH_PY="${TORCH_PY:-python}"
BRAIND_DEPS="${PROJECT_ROOT}/.python_deps"
BENDR_REPO="${BENDR_REPO:-braindecode/braindecode-bendr}"
BENDR_CKPT="${BENDR_CKPT:-external/checkpoints/models--braindecode--braindecode-bendr/snapshots/191f221cd56de8203899ea9a8d0f43238724f8b6/model.safetensors}"

EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-128}"
INTERVENTION_BATCH_SIZE="${INTERVENTION_BATCH_SIZE:-256}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"
SEED="${SEED:-20260525}"
DEVICE="${DEVICE:-cuda}"
INPUT_SCALE="${INPUT_SCALE:-1e-6}"
CLIP_UV="${CLIP_UV:-500.0}"

cd "${PROJECT_ROOT}"
mkdir -p "${RUN_ROOT}" "${RESULT_ROOT}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "CACHE_NPZ=${CACHE_NPZ}"
echo "TORCH_PY=${TORCH_PY}"
echo "BRAIND_DEPS=${BRAIND_DEPS}"
echo "BENDR_REPO=${BENDR_REPO}"
echo "BENDR_CKPT=${BENDR_CKPT}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "INTERVENTION_BATCH_SIZE=${INTERVENTION_BATCH_SIZE}"
echo "N_BOOTSTRAP=${N_BOOTSTRAP}"
echo "SEED=${SEED}"
echo "DEVICE=${DEVICE}"
echo "INPUT_SCALE=${INPUT_SCALE}"
echo "CLIP_UV=${CLIP_UV}"

PYTHONPATH="${BRAIND_DEPS}" "${TORCH_PY}" code/scripts/run_tuab_bendr_intervention.py \
  --cache-npz "${CACHE_NPZ}" \
  --cache-format npy \
  --output-dir "${RUN_ROOT}" \
  --checkpoint-path "${BENDR_CKPT}" \
  --pretrained-repo "${BENDR_REPO}" \
  --encoder-only \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --intervention-batch-size "${INTERVENTION_BATCH_SIZE}" \
  --n-bootstrap "${N_BOOTSTRAP}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --input-scale "${INPUT_SCALE}" \
  --clip-uv "${CLIP_UV}"

echo "Full TUAB BENDR intervention complete: ${RUN_ROOT}"
