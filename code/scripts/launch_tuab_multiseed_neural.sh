#!/usr/bin/env bash
set -euo pipefail

# Run TUAB neural intervention models across multiple seeds using the official
# train/eval split. This script only writes additive result folders.

PROJECT_ROOT="${APERIODIC_PROJECT_ROOT:-.}"
RESULT_ROOT="${PROJECT_ROOT}/results/tuab_subset_200"
RUN_ROOT="${RESULT_ROOT}/multiseed_neural"
REPORT_ROOT="${PROJECT_ROOT}/reports/tables"
TORCH_PY="${TORCH_PY:-python}"
PROJECT_PY="${PROJECT_ROOT}/.venv/bin/python"
SEEDS="${SEEDS:-42 43 44}"
DEVICE="${DEVICE:-cuda}"
BRAIND_DEPS="${PROJECT_ROOT}/.python_deps"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE_RAW="${BATCH_SIZE_RAW:-128}"
BATCH_SIZE_MLP="${BATCH_SIZE_MLP:-512}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"
RAW_MODELS="${RAW_MODELS:-raw_cnn eegnet shallow_fbcsp deep4}"

cd "${PROJECT_ROOT}/code"
mkdir -p "${RUN_ROOT}" "${REPORT_ROOT}"

AGG_INPUTS=()

for seed in ${SEEDS}; do
  echo "=== seed ${seed}: TUAB deep MLP PSD ==="
  MLP_DIR="${RUN_ROOT}/deep_mlp_seed${seed}"
  "${TORCH_PY}" scripts/run_tuab_deep_mlp_intervention.py \
    --output-dir "${MLP_DIR}" \
    --seed "${seed}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE_MLP}" \
    --device "${DEVICE}"
  AGG_INPUTS+=("deep_mlp=${MLP_DIR}/tuab_deep_mlp_intervention_subject_metrics.csv")

  for model in ${RAW_MODELS}; do
    echo "=== seed ${seed}: TUAB ${model} raw intervention ==="
    RAW_DIR="${RUN_ROOT}/${model}_seed${seed}"
    PYTHONPATH="${BRAIND_DEPS}" "${TORCH_PY}" \
      scripts/run_tuab_braindecode_eegnet_intervention.py \
      --model "${model}" \
      --output-dir "${RAW_DIR}" \
      --output-prefix "tuab_${model}" \
      --seed "${seed}" \
      --epochs "${EPOCHS}" \
      --batch-size "${BATCH_SIZE_RAW}" \
      --device "${DEVICE}"
    AGG_INPUTS+=("${model}=${RAW_DIR}/tuab_${model}_intervention_subject_metrics.csv")
  done
done

AGG_ARGS=()
for item in "${AGG_INPUTS[@]}"; do
  AGG_ARGS+=(--input "${item}")
done

"${PROJECT_PY}" scripts/aggregate_multiseed_subject_bootstrap.py \
  "${AGG_ARGS[@]}" \
  --output-csv "${REPORT_ROOT}/tuab_multiseed_neural_subject_bootstrap.csv" \
  --output-md "${REPORT_ROOT}/tuab_multiseed_neural_subject_bootstrap.md" \
  --n-bootstrap "${N_BOOTSTRAP}"

echo "TUAB multiseed neural runs complete: ${RUN_ROOT}"
