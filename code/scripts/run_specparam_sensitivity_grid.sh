#!/usr/bin/env bash
set -euo pipefail

# Runs conservative specparam peak-constraint sensitivities.
# Writes only under the project results directory.

PROJECT_ROOT="${APERIODIC_PROJECT_ROOT:-.}"
PY="${PROJECT_ROOT}/.venv/bin/python"
CODE_ROOT="${PROJECT_ROOT}/code"
RESULT_ROOT="${PROJECT_ROOT}/results/sleep_edf_subset"
PSD_NPZ="${RESULT_ROOT}/psd_welch_fpz_pz.npz"
INDEX_CSV="${RESULT_ROOT}/psd_index.csv"
SENS_ROOT="${RESULT_ROOT}/specparam_sensitivity"

cd "${CODE_ROOT}"

run_setting() {
  local name="$1"
  local max_peaks="$2"
  local min_height="$3"
  local peak_threshold="$4"
  local width_min="$5"
  local width_max="$6"

  local out_dir="${SENS_ROOT}/${name}"
  local decomp_npz="${out_dir}/specparam_fixed.npz"

  echo "== ${name} =="
  "${PY}" scripts/fit_sleep_edf_specparam.py \
    --psd-npz "${PSD_NPZ}" \
    --output-npz "${decomp_npz}" \
    --max-n-peaks "${max_peaks}" \
    --min-peak-height "${min_height}" \
    --peak-threshold "${peak_threshold}" \
    --peak-width-min "${width_min}" \
    --peak-width-max "${width_max}" \
    --n-jobs 8

  "${PY}" scripts/analyze_specparam_diagnostics.py \
    --decomp-npz "${decomp_npz}" \
    --index-csv "${INDEX_CSV}" \
    --output-dir "${out_dir}/diagnostics"

  "${PY}" scripts/run_sleep_edf_aperiodic_baselines.py \
    --psd-npz "${PSD_NPZ}" \
    --index-csv "${INDEX_CSV}" \
    --decomposition precomputed \
    --decomp-npz "${decomp_npz}" \
    --output-dir "${out_dir}/baselines" \
    --n-splits 5 \
    --classifier ridge
}

run_setting conservative_p4_h015 4 0.15 2.0 1.0 8.0
run_setting moderate_p8_h010 8 0.10 2.0 0.5 8.0
run_setting stricter_p6_h020 6 0.20 2.0 0.5 8.0

echo "Sensitivity grid complete: ${SENS_ROOT}"

