#!/usr/bin/env bash
set -euo pipefail

# Run PhysioNet MI raw neural intervention models across multiple seeds.
# This script only writes additive result folders and report tables.

PROJECT_ROOT="${APERIODIC_PROJECT_ROOT:-.}"
RESULT_ROOT="${PROJECT_ROOT}/results/physionet_mi"
RUN_ROOT="${RESULT_ROOT}/multiseed_neural"
REPORT_ROOT="${PROJECT_ROOT}/reports/tables"
TORCH_PY="${TORCH_PY:-python}"
PROJECT_PY="${PROJECT_ROOT}/.venv/bin/python"
SEEDS="${SEEDS:-42 43 44}"
DEVICE="${DEVICE:-cuda}"
BRAIND_DEPS="${PROJECT_ROOT}/.python_deps"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-256}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"
RAW_MODELS="${RAW_MODELS:-eegnet shallow_fbcsp deep4}"

cd "${PROJECT_ROOT}/code"
mkdir -p "${RUN_ROOT}" "${REPORT_ROOT}"

AGG_INPUTS=()

for seed in ${SEEDS}; do
  for model in ${RAW_MODELS}; do
    echo "=== seed ${seed}: PhysioNet MI ${model} raw intervention ==="
    MODEL_DIR="${RUN_ROOT}/${model}_seed${seed}"
    PYTHONPATH="${BRAIND_DEPS}" "${TORCH_PY}" \
      scripts/run_physionet_mi_braindecode_intervention.py \
      --model "${model}" \
      --output-dir "${MODEL_DIR}" \
      --output-prefix "physionet_mi_${model}" \
      --seed "${seed}" \
      --epochs "${EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --device "${DEVICE}"
    AGG_INPUTS+=("${model}=${MODEL_DIR}/physionet_mi_${model}_intervention_subject_metrics.csv")
  done
done

AGG_ARGS=()
for item in "${AGG_INPUTS[@]}"; do
  AGG_ARGS+=(--input "${item}")
done

"${PROJECT_PY}" scripts/aggregate_multiseed_subject_bootstrap.py \
  "${AGG_ARGS[@]}" \
  --output-csv "${REPORT_ROOT}/physionet_mi_multiseed_neural_subject_bootstrap.csv" \
  --output-md "${REPORT_ROOT}/physionet_mi_multiseed_neural_subject_bootstrap.md" \
  --n-bootstrap "${N_BOOTSTRAP}"

echo "PhysioNet MI multiseed neural runs complete: ${RUN_ROOT}"
