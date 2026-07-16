#!/usr/bin/env bash
set -u -o pipefail

PROJECT_ROOT="${APERIODIC_PROJECT_ROOT:-.}"
RESULT_ROOT="${PROJECT_ROOT}/results/ptbxl_1f_demo"
MATCH_ROOT="${RESULT_ROOT}/age_sex_matched_psd"
RUN_ROOT="${RESULT_ROOT}/age_sex_matched_neural"
REPORT_ROOT="${PROJECT_ROOT}/reports/tables"
PROJECT_PY="${PROJECT_ROOT}/.venv/bin/python"
TORCH_PY="${TORCH_PY:-python}"
RAW_NPZ="${RESULT_ROOT}/ptbxl_records100_normal_abnormal_raw.npz"
RECORD_FILTER_CSV="${RECORD_FILTER_CSV:-${MATCH_ROOT}/ptbxl_age_sex_matched_records.csv}"
MODELS="${MODELS:-resnet1d_wang inception1d xresnet1d101}"
SEEDS="${SEEDS:-42 43 44}"
EPOCHS="${EPOCHS:-35}"
BATCH_SIZE="${BATCH_SIZE:-192}"
LR="${LR:-0.001}"
DEVICE="${DEVICE:-cuda}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"
FORCE="${FORCE:-0}"

cd "${PROJECT_ROOT}/code" || exit 1
mkdir -p "${RUN_ROOT}" "${REPORT_ROOT}" "${PROJECT_ROOT}/logs"

if [[ ! -s "${RAW_NPZ}" ]]; then
  echo "Missing raw PTB-XL cache: ${RAW_NPZ}" >&2
  exit 2
fi
if [[ ! -s "${RECORD_FILTER_CSV}" ]]; then
  echo "Missing matched record filter: ${RECORD_FILTER_CSV}" >&2
  exit 2
fi

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "RECORD_FILTER_CSV=${RECORD_FILTER_CSV}"
echo "MODELS=${MODELS}"
echo "SEEDS=${SEEDS}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "DEVICE=${DEVICE}"

failures=()

for model in ${MODELS}; do
  echo "===== PTB-XL age/sex-matched ECG architecture: ${model} ====="
  model_inputs=()

  for seed in ${SEEDS}; do
    out_dir="${RUN_ROOT}/${model}_seed${seed}"
    pred_file="${out_dir}/ptbxl_raw_cnn_intervention_predictions.csv"
    seed_log="${PROJECT_ROOT}/logs/ptbxl_matched_${model}_seed${seed}_$(date +%Y%m%d_%H%M%S).log"

    if [[ "${FORCE}" != "1" && -s "${pred_file}" ]]; then
      echo "exists: ${pred_file}"
    else
      echo "running matched model=${model} seed=${seed}; log=${seed_log}"
      if ! "${TORCH_PY}" scripts/run_ptbxl_raw_cnn_interventions.py \
        --raw-npz "${RAW_NPZ}" \
        --record-filter-csv "${RECORD_FILTER_CSV}" \
        --output-dir "${out_dir}" \
        --model "${model}" \
        --seed "${seed}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --device "${DEVICE}" \
        > "${seed_log}" 2>&1; then
        echo "FAILED matched model=${model} seed=${seed}; see ${seed_log}" >&2
        failures+=("${model}:seed${seed}")
        continue
      fi
    fi

    if [[ -s "${pred_file}" ]]; then
      model_inputs+=(--input "ptbxl_matched_${model}=${pred_file}")
    fi
  done

  if [[ "${#model_inputs[@]}" -gt 0 ]]; then
    echo "aggregating matched model=${model} from ${#model_inputs[@]} seed files"
    "${PROJECT_PY}" scripts/aggregate_ptbxl_prediction_bootstrap.py \
      "${model_inputs[@]}" \
      --unit-column pair_id \
      --output-csv "${REPORT_ROOT}/ptbxl_matched_${model}_prediction_bootstrap.csv" \
      --output-md "${REPORT_ROOT}/ptbxl_matched_${model}_prediction_bootstrap.md" \
      --n-bootstrap "${N_BOOTSTRAP}" || failures+=("${model}:aggregate")
  else
    echo "No completed matched seed files for ${model}; skipping aggregate." >&2
    failures+=("${model}:no_completed_seeds")
  fi
done

combined_inputs=()
for model in ${MODELS}; do
  for seed in ${SEEDS}; do
    pred_file="${RUN_ROOT}/${model}_seed${seed}/ptbxl_raw_cnn_intervention_predictions.csv"
    if [[ -s "${pred_file}" ]]; then
      combined_inputs+=(--input "ptbxl_matched_${model}=${pred_file}")
    fi
  done
done

if [[ "${#combined_inputs[@]}" -gt 0 ]]; then
  "${PROJECT_PY}" scripts/aggregate_ptbxl_prediction_bootstrap.py \
    "${combined_inputs[@]}" \
    --unit-column pair_id \
    --output-csv "${REPORT_ROOT}/ptbxl_matched_ecg_architectures_prediction_bootstrap.csv" \
    --output-md "${REPORT_ROOT}/ptbxl_matched_ecg_architectures_prediction_bootstrap.md" \
    --n-bootstrap "${N_BOOTSTRAP}" || failures+=("combined:aggregate")
fi

if [[ "${#failures[@]}" -gt 0 ]]; then
  echo "Completed with failures: ${failures[*]}" >&2
  exit 3
fi

echo "PTB-XL age/sex-matched ECG architecture audit complete."
