#!/usr/bin/env bash
set -euo pipefail

# Full-TUAB age/sex-matched raw neural intervention control. This mirrors the
# TUAB-200 age-matched neural workflow but uses the full-TUAB 20s/100Hz cache.

PROJECT_ROOT="${APERIODIC_PROJECT_ROOT:-/mnt/data/aperiodic_confounds}"
RESULT_ROOT="${PROJECT_ROOT}/results/tuab_full_v3_0_1"
PREPROCESS_DIR="${RESULT_ROOT}/preprocess_20s_100hz"
MATCH_ROOT="${RESULT_ROOT}/age_matched"
RUN_ROOT="${MATCH_ROOT}/multiseed_neural"
REPORT_ROOT="${PROJECT_ROOT}/reports/tables"
TORCH_PY="${TORCH_PY:-/mnt/data/.venvs/ml/bin/python3}"
PROJECT_PY="${PROJECT_ROOT}/.venv/bin/python"
BRAIND_DEPS="${PROJECT_ROOT}/.python_deps"

PREFIX="${PREFIX:-tuab_full_age_sex_matched_caliper5}"
SUBJECT_FILTER_CSV="${SUBJECT_FILTER_CSV:-${MATCH_ROOT}/${PREFIX}_subjects.csv}"
SEEDS="${SEEDS:-42 43 44}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE_RAW="${BATCH_SIZE_RAW:-512}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"
RAW_MODELS="${RAW_MODELS:-eegnet shallow_fbcsp deep4}"

RAW_NPZ="${RAW_NPZ:-${PREPROCESS_DIR}/raw_epochs_20s_100hz.npz}"
RAW_INDEX_CSV="${RAW_INDEX_CSV:-${PREPROCESS_DIR}/raw_epochs_20s_100hz_index.csv}"
DECOMP_NPZ="${DECOMP_NPZ:-${PREPROCESS_DIR}/specparam/specparam_fixed_20s.npz}"
REPORT_PREFIX="${REPORT_PREFIX:-tuab_full_age_matched_multiseed_neural_subject_bootstrap}"

cd "${PROJECT_ROOT}/code"
mkdir -p "${RUN_ROOT}" "${REPORT_ROOT}" "${RUN_ROOT}/stage_markers"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "SUBJECT_FILTER_CSV=${SUBJECT_FILTER_CSV}"
echo "RAW_NPZ=${RAW_NPZ}"
echo "RAW_INDEX_CSV=${RAW_INDEX_CSV}"
echo "DECOMP_NPZ=${DECOMP_NPZ}"
echo "SEEDS=${SEEDS}"
echo "DEVICE=${DEVICE}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE_RAW=${BATCH_SIZE_RAW}"
echo "N_BOOTSTRAP=${N_BOOTSTRAP}"
echo "RAW_MODELS=${RAW_MODELS}"

if [[ ! -f "${SUBJECT_FILTER_CSV}" ]]; then
  echo "Missing subject filter: ${SUBJECT_FILTER_CSV}" >&2
  echo "Run code/scripts/launch_tuab_full_age_matched_psd.sh first." >&2
  exit 1
fi

AGG_INPUTS=()

for seed in ${SEEDS}; do
  for model in ${RAW_MODELS}; do
    echo "=== seed ${seed}: full-TUAB age-matched ${model} raw intervention ==="
    MODEL_DIR="${RUN_ROOT}/${model}_seed${seed}"
    MARKER="${RUN_ROOT}/stage_markers/${model}_seed${seed}.done"
    SUBJECT_METRICS="${MODEL_DIR}/tuab_full_age_matched_${model}_intervention_subject_metrics.csv"

    if [[ -f "${MARKER}" && -f "${SUBJECT_METRICS}" ]]; then
      echo "Skipping completed model ${model} seed ${seed}"
    else
      PYTHONPATH="${BRAIND_DEPS}" "${TORCH_PY}" \
        scripts/run_tuab_braindecode_eegnet_intervention.py \
        --raw-npz "${RAW_NPZ}" \
        --index-csv "${RAW_INDEX_CSV}" \
        --decomp-npz "${DECOMP_NPZ}" \
        --model "${model}" \
        --subject-filter-csv "${SUBJECT_FILTER_CSV}" \
        --output-dir "${MODEL_DIR}" \
        --output-prefix "tuab_full_age_matched_${model}" \
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
  --output-csv "${REPORT_ROOT}/${REPORT_PREFIX}.csv" \
  --output-md "${REPORT_ROOT}/${REPORT_PREFIX}.md" \
  --n-bootstrap "${N_BOOTSTRAP}"

echo "Full-TUAB age-matched multiseed neural run complete: ${RUN_ROOT}"
