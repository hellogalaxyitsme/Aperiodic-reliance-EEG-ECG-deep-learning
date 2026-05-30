#!/usr/bin/env bash
set -euo pipefail

# Full-TUAB EEGMamba foundation-model audit. EEGMamba needs the isolated
# SSM-capable environment created for the TUAB-200 run; the shared ML venv does
# not provide the compiled mamba_ssm/causal-conv1d stack.

PROJECT_ROOT="${APERIODIC_PROJECT_ROOT:-/mnt/data/aperiodic_confounds}"
RESULT_ROOT="${PROJECT_ROOT}/results/tuab_full_v3_0_1"
RUN_ROOT="${RUN_ROOT:-${RESULT_ROOT}/eegmamba_interventions_official_full}"
CACHE_NPZ="${RESULT_ROOT}/labram_10s_200hz_cache.npz"
EEGMAMBA_PY="${EEGMAMBA_PY:-${PROJECT_ROOT}/.venvs/eegmamba/bin/python}"
EEGMAMBA_REPO="${EEGMAMBA_REPO:-${PROJECT_ROOT}/external/EEGMamba}"
EEGMAMBA_CKPT="${EEGMAMBA_CKPT:-/home/vinay/.cache/huggingface/hub/models--weighting666--EEGMamba/snapshots/0b060d87acd6f23bf1d0b852bf1726064f335f97/pretrained_EEGMamba.pth}"

EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
INTERVENTION_BATCH_SIZE="${INTERVENTION_BATCH_SIZE:-256}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"
SEED="${SEED:-20260525}"
DEVICE="${DEVICE:-cuda}"
INPUT_NORMALIZATION="${INPUT_NORMALIZATION:-divisor}"
INPUT_DIVISOR="${INPUT_DIVISOR:-100.0}"
SELECTION_METRIC="${SELECTION_METRIC:-val_balanced_accuracy}"
FREEZE_BACKBONE="${FREEZE_BACKBONE:-0}"

cd "${PROJECT_ROOT}"
mkdir -p "${RUN_ROOT}" "${RESULT_ROOT}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "CACHE_NPZ=${CACHE_NPZ}"
echo "EEGMAMBA_PY=${EEGMAMBA_PY}"
echo "EEGMAMBA_REPO=${EEGMAMBA_REPO}"
echo "EEGMAMBA_CKPT=${EEGMAMBA_CKPT}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "INTERVENTION_BATCH_SIZE=${INTERVENTION_BATCH_SIZE}"
echo "N_BOOTSTRAP=${N_BOOTSTRAP}"
echo "SEED=${SEED}"
echo "DEVICE=${DEVICE}"
echo "INPUT_NORMALIZATION=${INPUT_NORMALIZATION}"
echo "INPUT_DIVISOR=${INPUT_DIVISOR}"
echo "SELECTION_METRIC=${SELECTION_METRIC}"
echo "FREEZE_BACKBONE=${FREEZE_BACKBONE}"

ARGS=(
  code/scripts/run_tuab_eegmamba_intervention.py
  --cache-npz "${CACHE_NPZ}"
  --cache-format npy
  --output-dir "${RUN_ROOT}"
  --checkpoint-path "${EEGMAMBA_CKPT}"
  --eegmamba-repo "${EEGMAMBA_REPO}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --intervention-batch-size "${INTERVENTION_BATCH_SIZE}"
  --n-bootstrap "${N_BOOTSTRAP}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --input-normalization "${INPUT_NORMALIZATION}"
  --input-divisor "${INPUT_DIVISOR}"
  --selection-metric "${SELECTION_METRIC}"
)

if [[ "${FREEZE_BACKBONE}" == "1" || "${FREEZE_BACKBONE}" == "true" ]]; then
  ARGS+=(--freeze-backbone)
fi

PYTHONPATH="${EEGMAMBA_REPO}" "${EEGMAMBA_PY}" "${ARGS[@]}"

echo "Full TUAB EEGMamba intervention complete: ${RUN_ROOT}"
