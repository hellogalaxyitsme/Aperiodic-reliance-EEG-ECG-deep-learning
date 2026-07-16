#!/usr/bin/env bash
set -euo pipefail

# Full-TUAB age/sex-matched PSD intervention control. This mirrors the
# TUAB-200 age-matched PSD workflow, preserving the official train/eval
# boundary and writing only additive result artifacts.

PROJECT_ROOT="${APERIODIC_PROJECT_ROOT:-.}"
RESULT_ROOT="${PROJECT_ROOT}/results/tuab_full_v3_0_1"
PREPROCESS_DIR="${RESULT_ROOT}/preprocess_20s_100hz"
MATCH_ROOT="${RESULT_ROOT}/age_matched"
STAGE_ROOT="${MATCH_ROOT}/stage_markers"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/tuab/v3.0.1_random_stratified_200}"
SELECTED_FILES_CSV="${SELECTED_FILES_CSV:-${RESULT_ROOT}/tuab_v3_0_1_full_edf_files.csv}"
PYTHON="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"

PREFIX="${PREFIX:-tuab_full_age_sex_matched_caliper5}"
CALIPER_YEARS="${CALIPER_YEARS:-5}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"

METADATA_FILE_CSV="${METADATA_FILE_CSV:-${MATCH_ROOT}/${PREFIX}_header_metadata_files.csv}"
METADATA_SUBJECT_CSV="${METADATA_SUBJECT_CSV:-${MATCH_ROOT}/${PREFIX}_header_metadata_subjects.csv}"
METADATA_SUMMARY_JSON="${METADATA_SUMMARY_JSON:-${MATCH_ROOT}/${PREFIX}_header_metadata_summary.json}"
SUBJECT_FILTER_CSV="${SUBJECT_FILTER_CSV:-${MATCH_ROOT}/${PREFIX}_subjects.csv}"
PSD_NPZ="${PSD_NPZ:-${PREPROCESS_DIR}/psd_20s_multitaper.npz}"
PSD_INDEX_CSV="${PSD_INDEX_CSV:-${PREPROCESS_DIR}/psd_20s_multitaper_index.csv}"
SPECPARAM_NPZ="${SPECPARAM_NPZ:-${PREPROCESS_DIR}/specparam/specparam_fixed_20s.npz}"
PSD_INTERVENTION_DIR="${PSD_INTERVENTION_DIR:-${MATCH_ROOT}/psd_interventions_specparam_${PREFIX}}"

cd "${PROJECT_ROOT}"
mkdir -p "${MATCH_ROOT}" "${STAGE_ROOT}" "${PSD_INTERVENTION_DIR}"

run_stage() {
  local name="$1"
  shift
  local marker="${STAGE_ROOT}/${name}.done"
  if [[ -f "${marker}" ]]; then
    echo "[$(date --iso-8601=seconds)] Skipping completed stage: ${name}"
    return 0
  fi
  echo "[$(date --iso-8601=seconds)] Starting stage: ${name}"
  "$@"
  echo "[$(date --iso-8601=seconds)] Completed stage: ${name}"
  touch "${marker}"
}

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "MATCH_ROOT=${MATCH_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "SELECTED_FILES_CSV=${SELECTED_FILES_CSV}"
echo "PREFIX=${PREFIX}"
echo "CALIPER_YEARS=${CALIPER_YEARS}"
echo "PSD_NPZ=${PSD_NPZ}"
echo "PSD_INDEX_CSV=${PSD_INDEX_CSV}"
echo "SPECPARAM_NPZ=${SPECPARAM_NPZ}"
echo "PSD_INTERVENTION_DIR=${PSD_INTERVENTION_DIR}"
echo "N_BOOTSTRAP=${N_BOOTSTRAP}"

run_stage header_metadata_full_tuab \
  "${PYTHON}" code/scripts/extract_tuab_header_metadata.py \
    --data-root "${DATA_ROOT}" \
    --selected-files-csv "${SELECTED_FILES_CSV}" \
    --output-file-csv "${METADATA_FILE_CSV}" \
    --output-subject-csv "${METADATA_SUBJECT_CSV}" \
    --output-summary-json "${METADATA_SUMMARY_JSON}"

run_stage age_sex_match_full_tuab \
  "${PYTHON}" code/scripts/make_tuab_age_matched_subset.py \
    --metadata-subjects-csv "${METADATA_SUBJECT_CSV}" \
    --output-dir "${MATCH_ROOT}" \
    --caliper-years "${CALIPER_YEARS}" \
    --same-sex \
    --prefix "${PREFIX}"

run_stage psd_intervention_full_tuab_age_matched \
  "${PYTHON}" code/scripts/run_tuab_psd_interventions.py \
    --psd-npz "${PSD_NPZ}" \
    --index-csv "${PSD_INDEX_CSV}" \
    --decomposition precomputed \
    --decomp-npz "${SPECPARAM_NPZ}" \
    --subject-filter-csv "${SUBJECT_FILTER_CSV}" \
    --output-dir "${PSD_INTERVENTION_DIR}" \
    --n-bootstrap "${N_BOOTSTRAP}"

echo "[$(date --iso-8601=seconds)] Full-TUAB age/sex-matched PSD control complete: ${MATCH_ROOT}"
