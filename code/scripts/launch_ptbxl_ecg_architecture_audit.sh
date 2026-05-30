#!/usr/bin/env bash
set -u -o pipefail

PROJECT_ROOT="${APERIODIC_PROJECT_ROOT:-/mnt/data/aperiodic_confounds}"
RESULT_ROOT="${PROJECT_ROOT}/results/ptbxl_1f_demo"
REPORT_ROOT="${PROJECT_ROOT}/reports/tables"
PROJECT_PY="${PROJECT_ROOT}/.venv/bin/python"
TORCH_PY="${TORCH_PY:-/mnt/data/.venvs/ml/bin/python3}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-42 43 44}"
MODELS="${MODELS:-resnet1d_wang inception1d xresnet1d101}"
EPOCHS="${EPOCHS:-35}"
BATCH_SIZE="${BATCH_SIZE:-192}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"
LR="${LR:-0.001}"
FORCE="${FORCE:-0}"

cd "${PROJECT_ROOT}/code" || exit 1
mkdir -p "${RESULT_ROOT}" "${REPORT_ROOT}" "${PROJECT_ROOT}/logs"

RAW_NPZ="${RESULT_ROOT}/ptbxl_records100_normal_abnormal_raw.npz"
if [[ ! -s "${RAW_NPZ}" ]]; then
  echo "Missing prepared PTB-XL raw NPZ: ${RAW_NPZ}" >&2
  exit 2
fi

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "REPORT_ROOT=${REPORT_ROOT}"
echo "MODELS=${MODELS}"
echo "SEEDS=${SEEDS}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "DEVICE=${DEVICE}"

failures=()

for model in ${MODELS}; do
  echo "===== PTB-XL ECG architecture: ${model} ====="
  model_inputs=()

  for seed in ${SEEDS}; do
    out_dir="${RESULT_ROOT}/${model}_seed${seed}"
    metrics_file="${out_dir}/ptbxl_raw_cnn_intervention_subject_metrics.csv"
    seed_log="${PROJECT_ROOT}/logs/ptbxl_${model}_seed${seed}_$(date +%Y%m%d_%H%M%S).log"

    if [[ "${FORCE}" != "1" && -s "${metrics_file}" ]]; then
      echo "exists: ${metrics_file}"
    else
      echo "running model=${model} seed=${seed}; log=${seed_log}"
      if ! "${TORCH_PY}" scripts/run_ptbxl_raw_cnn_interventions.py \
        --raw-npz "${RAW_NPZ}" \
        --output-dir "${out_dir}" \
        --model "${model}" \
        --seed "${seed}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --device "${DEVICE}" \
        > "${seed_log}" 2>&1; then
        echo "FAILED model=${model} seed=${seed}; see ${seed_log}" >&2
        failures+=("${model}:seed${seed}")
        continue
      fi
    fi

    if [[ -s "${metrics_file}" ]]; then
      model_inputs+=(--input "ptbxl_${model}=${metrics_file}")
    fi
  done

  if [[ "${#model_inputs[@]}" -gt 0 ]]; then
    echo "aggregating model=${model} from ${#model_inputs[@]} seed files"
    "${PROJECT_PY}" scripts/aggregate_multiseed_subject_bootstrap.py \
      "${model_inputs[@]}" \
      --output-csv "${REPORT_ROOT}/ptbxl_${model}_multiseed_subject_bootstrap.csv" \
      --output-md "${REPORT_ROOT}/ptbxl_${model}_multiseed_subject_bootstrap.md" \
      --n-bootstrap "${N_BOOTSTRAP}" || failures+=("${model}:aggregate")
  else
    echo "No completed seed files for ${model}; skipping aggregate." >&2
    failures+=("${model}:no_completed_seeds")
  fi
done

if [[ "${#failures[@]}" -gt 0 ]]; then
  echo "Completed with failures: ${failures[*]}" >&2
  exit 3
fi

echo "PTB-XL ECG architecture audit complete."
