#!/usr/bin/env bash
set -euo pipefail

# Full-TUAB multiseed raw neural intervention run. This uses the streamed
# full-cache runner and mirrors the TUAB-200 neural architecture set.

PROJECT_ROOT="${APERIODIC_PROJECT_ROOT:-/mnt/data/aperiodic_confounds}"
RESULT_ROOT="${PROJECT_ROOT}/results/tuab_full_v3_0_1"
PREPROCESS_DIR="${RESULT_ROOT}/preprocess_20s_100hz"
RUN_ROOT="${RESULT_ROOT}/multiseed_neural"
SINGLE_SEED_ROOT="${RESULT_ROOT}/single_seed_neural"
REPORT_ROOT="${PROJECT_ROOT}/reports/tables"
TORCH_PY="${TORCH_PY:-/mnt/data/.venvs/ml/bin/python3}"
PROJECT_PY="${PROJECT_ROOT}/.venv/bin/python"
BRAIND_DEPS="${PROJECT_ROOT}/.python_deps"

SEEDS="${SEEDS:-42 43 44}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE_RAW="${BATCH_SIZE_RAW:-512}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"
RAW_MODELS="${RAW_MODELS:-eegnet shallow_fbcsp deep4}"

RAW_NPZ="${RAW_NPZ:-$PREPROCESS_DIR/raw_epochs_20s_100hz.npz}"
RAW_INDEX_CSV="${RAW_INDEX_CSV:-$PREPROCESS_DIR/raw_epochs_20s_100hz_index.csv}"
DECOMP_NPZ="${DECOMP_NPZ:-$PREPROCESS_DIR/specparam/specparam_fixed_20s.npz}"

cd "${PROJECT_ROOT}/code"
mkdir -p "${RUN_ROOT}" "${REPORT_ROOT}" "${RUN_ROOT}/stage_markers"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "SINGLE_SEED_ROOT=${SINGLE_SEED_ROOT}"
echo "RAW_NPZ=${RAW_NPZ}"
echo "RAW_INDEX_CSV=${RAW_INDEX_CSV}"
echo "DECOMP_NPZ=${DECOMP_NPZ}"
echo "SEEDS=${SEEDS}"
echo "DEVICE=${DEVICE}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE_RAW=${BATCH_SIZE_RAW}"
echo "N_BOOTSTRAP=${N_BOOTSTRAP}"
echo "RAW_MODELS=${RAW_MODELS}"

AGG_INPUTS=()

for seed in ${SEEDS}; do
  for model in ${RAW_MODELS}; do
    echo "=== seed ${seed}: full TUAB ${model} raw intervention ==="
    MODEL_DIR="${RUN_ROOT}/${model}_seed${seed}"
    MARKER="${RUN_ROOT}/stage_markers/${model}_seed${seed}.done"
    SUBJECT_METRICS="${MODEL_DIR}/tuab_full_${model}_intervention_subject_metrics.csv"
    SINGLE_DIR="${SINGLE_SEED_ROOT}/${model}_seed${seed}"
    SINGLE_SUBJECT_METRICS="${SINGLE_DIR}/tuab_full_${model}_intervention_subject_metrics.csv"

    if [[ -f "${MARKER}" && -f "${SUBJECT_METRICS}" ]]; then
      echo "Skipping completed model ${model} seed ${seed}"
    elif [[ -f "${SINGLE_SUBJECT_METRICS}" ]]; then
      echo "Reusing completed single-seed output for ${model} seed ${seed}"
      mkdir -p "${MODEL_DIR}"
      cp -a "${SINGLE_DIR}/." "${MODEL_DIR}/"
      touch "${MARKER}"
    else
      PYTHONPATH="${BRAIND_DEPS}" "${TORCH_PY}" \
        scripts/run_tuab_braindecode_eegnet_intervention.py \
        --raw-npz "${RAW_NPZ}" \
        --index-csv "${RAW_INDEX_CSV}" \
        --decomp-npz "${DECOMP_NPZ}" \
        --model "${model}" \
        --output-dir "${MODEL_DIR}" \
        --output-prefix "tuab_full_${model}" \
        --seed "${seed}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE_RAW}" \
        --device "${DEVICE}" \
        --n-bootstrap "${N_BOOTSTRAP}"
      touch "${MARKER}"
    fi
    AGG_INPUTS+=("${model}=${SUBJECT_METRICS}")
  done
done

AGG_ARGS=()
for item in "${AGG_INPUTS[@]}"; do
  AGG_ARGS+=(--input "${item}")
done

"${PROJECT_PY}" scripts/aggregate_multiseed_subject_bootstrap.py \
  "${AGG_ARGS[@]}" \
  --output-csv "${REPORT_ROOT}/tuab_full_multiseed_neural_subject_bootstrap.csv" \
  --output-md "${REPORT_ROOT}/tuab_full_multiseed_neural_subject_bootstrap.md" \
  --n-bootstrap "${N_BOOTSTRAP}"

echo "Full TUAB multiseed neural run complete: ${RUN_ROOT}"
