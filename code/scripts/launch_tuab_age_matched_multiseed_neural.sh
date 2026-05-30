#!/usr/bin/env bash
set -euo pipefail

# Run TUAB age/sex-matched raw neural intervention models across multiple seeds.
# This preserves the official train/eval boundary and only writes additive outputs.

PROJECT_ROOT="${APERIODIC_PROJECT_ROOT:-/mnt/data/aperiodic_confounds}"
RESULT_ROOT="${PROJECT_ROOT}/results/tuab_subset_200"
MATCH_ROOT="${RESULT_ROOT}/age_matched"
RUN_ROOT="${MATCH_ROOT}/multiseed_neural"
REPORT_ROOT="${PROJECT_ROOT}/reports/tables"
TORCH_PY="${TORCH_PY:-/mnt/data/.venvs/ml/bin/python3}"
PROJECT_PY="${PROJECT_ROOT}/.venv/bin/python"
SUBJECT_FILTER_CSV="${SUBJECT_FILTER_CSV:-${MATCH_ROOT}/tuab_age_sex_matched_caliper5_subjects.csv}"
SEEDS="${SEEDS:-42 43 44}"
DEVICE="${DEVICE:-cuda}"
BRAIND_DEPS="${PROJECT_ROOT}/.python_deps"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE_RAW="${BATCH_SIZE_RAW:-128}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"
RAW_MODELS="${RAW_MODELS:-eegnet shallow_fbcsp deep4}"

cd "${PROJECT_ROOT}/code"
mkdir -p "${RUN_ROOT}" "${REPORT_ROOT}"

AGG_INPUTS=()

for seed in ${SEEDS}; do
  for model in ${RAW_MODELS}; do
    echo "=== seed ${seed}: age-matched TUAB ${model} raw intervention ==="
    RAW_DIR="${RUN_ROOT}/${model}_seed${seed}"
    PYTHONPATH="${BRAIND_DEPS}" "${TORCH_PY}" \
      scripts/run_tuab_braindecode_eegnet_intervention.py \
      --model "${model}" \
      --subject-filter-csv "${SUBJECT_FILTER_CSV}" \
      --output-dir "${RAW_DIR}" \
      --output-prefix "tuab_age_matched_${model}" \
      --seed "${seed}" \
      --epochs "${EPOCHS}" \
      --batch-size "${BATCH_SIZE_RAW}" \
      --device "${DEVICE}"
    AGG_INPUTS+=("${model}=${RAW_DIR}/tuab_age_matched_${model}_intervention_subject_metrics.csv")
  done
done

AGG_ARGS=()
for item in "${AGG_INPUTS[@]}"; do
  AGG_ARGS+=(--input "${item}")
done

"${PROJECT_PY}" scripts/aggregate_multiseed_subject_bootstrap.py \
  "${AGG_ARGS[@]}" \
  --output-csv "${REPORT_ROOT}/tuab_age_matched_multiseed_neural_subject_bootstrap.csv" \
  --output-md "${REPORT_ROOT}/tuab_age_matched_multiseed_neural_subject_bootstrap.md" \
  --n-bootstrap "${N_BOOTSTRAP}"

echo "TUAB age-matched multiseed neural runs complete: ${RUN_ROOT}"
