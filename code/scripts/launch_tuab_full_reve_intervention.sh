#!/usr/bin/env bash
set -euo pipefail

# Full-TUAB REVE-base foundation-model audit. REVE uses the same 23-channel
# LaBraM-style TUAB cache, selecting the 21 channels supported by the REVE
# position bank batch-by-batch inside the runner.

PROJECT_ROOT="${APERIODIC_PROJECT_ROOT:-/mnt/data/aperiodic_confounds}"
RESULT_ROOT="${PROJECT_ROOT}/results/tuab_full_v3_0_1"
RUN_ROOT="${RUN_ROOT:-${RESULT_ROOT}/reve_base_interventions_full}"
CACHE_NPZ="${RESULT_ROOT}/labram_10s_200hz_cache.npz"
TORCH_PY="${TORCH_PY:-/mnt/data/.venvs/ml/bin/python3}"

EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
INTERVENTION_BATCH_SIZE="${INTERVENTION_BATCH_SIZE:-256}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"
SEED="${SEED:-20260525}"
DEVICE="${DEVICE:-cuda}"
SELECTION_METRIC="${SELECTION_METRIC:-val_balanced_accuracy}"
PRETRAINED_REPO="${PRETRAINED_REPO:-brain-bzh/reve-base}"
POSITIONS_REPO="${POSITIONS_REPO:-brain-bzh/reve-positions}"

cd "${PROJECT_ROOT}"
mkdir -p "${RUN_ROOT}" "${RESULT_ROOT}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "CACHE_NPZ=${CACHE_NPZ}"
echo "TORCH_PY=${TORCH_PY}"
echo "PRETRAINED_REPO=${PRETRAINED_REPO}"
echo "POSITIONS_REPO=${POSITIONS_REPO}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "INTERVENTION_BATCH_SIZE=${INTERVENTION_BATCH_SIZE}"
echo "N_BOOTSTRAP=${N_BOOTSTRAP}"
echo "SEED=${SEED}"
echo "DEVICE=${DEVICE}"
echo "SELECTION_METRIC=${SELECTION_METRIC}"

"${TORCH_PY}" code/scripts/run_tuab_reve_intervention.py \
  --cache-npz "${CACHE_NPZ}" \
  --cache-format npy \
  --output-dir "${RUN_ROOT}" \
  --pretrained-repo "${PRETRAINED_REPO}" \
  --positions-repo "${POSITIONS_REPO}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --intervention-batch-size "${INTERVENTION_BATCH_SIZE}" \
  --n-bootstrap "${N_BOOTSTRAP}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --selection-metric "${SELECTION_METRIC}"

echo "Full TUAB REVE-base intervention complete: ${RUN_ROOT}"
