#!/usr/bin/env bash
set -uo pipefail

# Sequential full-TUAB foundation-model multiseed launcher. Each FM runs all
# requested seeds before the next FM starts. Failures are recorded and the
# launcher continues so already-completed outputs remain usable overnight.

PROJECT_ROOT="${APERIODIC_PROJECT_ROOT:-/mnt/data/aperiodic_confounds}"
RESULT_ROOT="${PROJECT_ROOT}/results/tuab_full_v3_0_1"
RUN_ROOT="${RESULT_ROOT}/foundation_multiseed"
REPORT_ROOT="${PROJECT_ROOT}/reports/tables"
LOG_ROOT="${PROJECT_ROOT}/logs"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
SEEDS="${SEEDS:-20260526 20260527}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"
EPOCHS="${EPOCHS:-30}"
DEVICE="${DEVICE:-cuda}"
FM_ORDER="${FM_ORDER:-biot labram eegpt cbramod reve eegmamba bendr}"
REPORT_PREFIX="${REPORT_PREFIX:-tuab_full_foundation_multiseed_subject_bootstrap}"
MASTER_LOG="${MASTER_LOG:-${LOG_ROOT}/tuab_full_foundation_multiseed_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "${RUN_ROOT}/stage_markers" "${RUN_ROOT}/job_logs" "${REPORT_ROOT}" "${LOG_ROOT}"
exec > >(tee -a "${MASTER_LOG}") 2>&1

MANIFEST="${RUN_ROOT}/foundation_multiseed_manifest.csv"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "fm,seed,status,start_time,end_time,run_dir,log_file,exit_code" > "${MANIFEST}"
fi

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "REPORT_ROOT=${REPORT_ROOT}"
echo "MASTER_LOG=${MASTER_LOG}"
echo "FM_ORDER=${FM_ORDER}"
echo "SEEDS=${SEEDS}"
echo "N_BOOTSTRAP=${N_BOOTSTRAP}"
echo "EPOCHS=${EPOCHS}"
echo "DEVICE=${DEVICE}"
echo "started_at=$(date)"

fm_launcher() {
  case "$1" in
    biot) echo "code/scripts/launch_tuab_full_biot_intervention.sh" ;;
    labram) echo "code/scripts/launch_tuab_full_labram_intervention.sh" ;;
    eegpt) echo "code/scripts/launch_tuab_full_eegpt_intervention.sh" ;;
    cbramod) echo "code/scripts/launch_tuab_full_cbramod_intervention.sh" ;;
    reve) echo "code/scripts/launch_tuab_full_reve_intervention.sh" ;;
    eegmamba) echo "code/scripts/launch_tuab_full_eegmamba_intervention.sh" ;;
    bendr) echo "code/scripts/launch_tuab_full_bendr_intervention.sh" ;;
    *) return 1 ;;
  esac
}

fm_existing_seed() {
  case "$1" in
    biot|labram) echo "20260524" ;;
    eegpt|cbramod|reve|eegmamba|bendr) echo "20260525" ;;
    *) return 1 ;;
  esac
}

fm_existing_dir() {
  case "$1" in
    biot) echo "${RESULT_ROOT}/biot_interventions_prest_full" ;;
    labram) echo "${RESULT_ROOT}/labram_interventions_base_full" ;;
    eegpt) echo "${RESULT_ROOT}/eegpt_interventions_braindecode_full" ;;
    cbramod) echo "${RESULT_ROOT}/cbramod_interventions_braindecode_full" ;;
    reve) echo "${RESULT_ROOT}/reve_base_interventions_full" ;;
    eegmamba) echo "${RESULT_ROOT}/eegmamba_interventions_official_full" ;;
    bendr) echo "${RESULT_ROOT}/bendr_interventions_braindecode_encoder_only_full" ;;
    *) return 1 ;;
  esac
}

fm_prediction_file() {
  case "$1" in
    biot) echo "tuab_biot_intervention_predictions.csv" ;;
    labram) echo "tuab_labram_intervention_predictions.csv" ;;
    eegpt) echo "tuab_eegpt_intervention_predictions.csv" ;;
    cbramod) echo "tuab_cbramod_intervention_predictions.csv" ;;
    reve) echo "tuab_reve_intervention_predictions.csv" ;;
    eegmamba) echo "tuab_eegmamba_intervention_predictions.csv" ;;
    bendr) echo "tuab_bendr_intervention_predictions.csv" ;;
    *) return 1 ;;
  esac
}

fm_label() {
  case "$1" in
    biot) echo "BIOT" ;;
    labram) echo "LaBraM" ;;
    eegpt) echo "EEGPT" ;;
    cbramod) echo "CBraMod" ;;
    reve) echo "REVE-base" ;;
    eegmamba) echo "EEGMamba" ;;
    bendr) echo "BENDR" ;;
    *) return 1 ;;
  esac
}

collect_inputs_for_fm() {
  local fm="$1"
  local pred_name existing_dir existing_seed label seed seed_dir pred
  pred_name="$(fm_prediction_file "${fm}")" || return 1
  existing_dir="$(fm_existing_dir "${fm}")" || return 1
  existing_seed="$(fm_existing_seed "${fm}")" || return 1
  label="$(fm_label "${fm}")" || return 1

  if [[ -f "${existing_dir}/${pred_name}" ]]; then
    printf '%s:%s=%s\n' "${label}" "${existing_seed}" "${existing_dir}/${pred_name}"
  fi
  for seed in ${SEEDS}; do
    seed_dir="${RUN_ROOT}/${fm}_seed${seed}"
    pred="${seed_dir}/${pred_name}"
    if [[ -f "${pred}" ]]; then
      printf '%s:%s=%s\n' "${label}" "${seed}" "${pred}"
    fi
  done
}

aggregate_inputs() {
  local out_prefix="$1"
  shift
  local inputs=("$@")
  if [[ "${#inputs[@]}" -eq 0 ]]; then
    echo "No prediction inputs available for aggregate ${out_prefix}; skipping."
    return 0
  fi

  local args=()
  local item
  for item in "${inputs[@]}"; do
    args+=(--input "${item}")
  done

  "${PYTHON}" code/scripts/aggregate_foundation_multiseed_predictions.py \
    "${args[@]}" \
    --output-csv "${REPORT_ROOT}/${out_prefix}.csv" \
    --output-md "${REPORT_ROOT}/${out_prefix}.md" \
    --n-bootstrap "${N_BOOTSTRAP}"
}

run_one_seed() {
  local fm="$1"
  local seed="$2"
  local launcher pred_name run_dir job_log done_marker fail_marker start_time end_time status exit_code
  launcher="$(fm_launcher "${fm}")" || return 1
  pred_name="$(fm_prediction_file "${fm}")" || return 1
  run_dir="${RUN_ROOT}/${fm}_seed${seed}"
  job_log="${RUN_ROOT}/job_logs/${fm}_seed${seed}.log"
  done_marker="${RUN_ROOT}/stage_markers/${fm}_seed${seed}.done"
  fail_marker="${RUN_ROOT}/stage_markers/${fm}_seed${seed}.failed"

  if [[ -f "${done_marker}" && -f "${run_dir}/${pred_name}" ]]; then
    echo "Skipping completed ${fm} seed ${seed}: ${run_dir}"
    return 0
  fi

  mkdir -p "${run_dir}"
  start_time="$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "=== START ${fm} seed ${seed} at ${start_time} ==="
  echo "launcher=${launcher}"
  echo "run_dir=${run_dir}"
  echo "job_log=${job_log}"

  RUN_ROOT="${run_dir}" \
  SEED="${seed}" \
  N_BOOTSTRAP="${N_BOOTSTRAP}" \
  EPOCHS="${EPOCHS}" \
  DEVICE="${DEVICE}" \
  bash "${launcher}" > "${job_log}" 2>&1
  exit_code=$?
  end_time="$(date '+%Y-%m-%d %H:%M:%S %Z')"

  if [[ "${exit_code}" -eq 0 && -f "${run_dir}/${pred_name}" ]]; then
    status="success"
    touch "${done_marker}"
    rm -f "${fail_marker}"
    echo "=== DONE ${fm} seed ${seed} at ${end_time} ==="
  else
    status="failed"
    echo "${exit_code}" > "${fail_marker}"
    echo "=== FAILED ${fm} seed ${seed} at ${end_time}; exit_code=${exit_code} ==="
    echo "Last 80 log lines for ${fm} seed ${seed}:"
    tail -80 "${job_log}" || true
  fi

  printf '%s,%s,%s,"%s","%s","%s","%s",%s\n' \
    "${fm}" "${seed}" "${status}" "${start_time}" "${end_time}" \
    "${run_dir}" "${job_log}" "${exit_code}" >> "${MANIFEST}"
  return 0
}

refresh_combined_aggregate() {
  local all_inputs=()
  local fm item
  for fm in ${FM_ORDER}; do
    while IFS= read -r item; do
      [[ -n "${item}" ]] && all_inputs+=("${item}")
    done < <(collect_inputs_for_fm "${fm}")
  done
  aggregate_inputs "${REPORT_PREFIX}" "${all_inputs[@]}"
}

for fm in ${FM_ORDER}; do
  echo "===== FOUNDATION MODEL ${fm} ====="
  for seed in ${SEEDS}; do
    run_one_seed "${fm}" "${seed}"
  done

  fm_inputs=()
  while IFS= read -r item; do
    [[ -n "${item}" ]] && fm_inputs+=("${item}")
  done < <(collect_inputs_for_fm "${fm}")

  echo "Aggregating ${fm} with ${#fm_inputs[@]} available seed input(s)."
  aggregate_inputs "tuab_full_foundation_multiseed_${fm}_subject_bootstrap" "${fm_inputs[@]}"
  refresh_combined_aggregate
done

echo "completed_at=$(date)"
echo "manifest=${MANIFEST}"
echo "combined_csv=${REPORT_ROOT}/${REPORT_PREFIX}.csv"
echo "combined_md=${REPORT_ROOT}/${REPORT_PREFIX}.md"
echo "master_log=${MASTER_LOG}"
