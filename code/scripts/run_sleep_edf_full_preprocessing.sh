#!/usr/bin/env bash
set -euo pipefail

# Additive preprocessing only. This script writes new outputs and never deletes
# datasets or previous result folders.

PROJECT_ROOT="${APERIODIC_PROJECT_ROOT:-/mnt/data/aperiodic_confounds}"
DATA_ROOT="${PROJECT_ROOT}/data/sleep-edf/sleep-cassette"
RESULT_ROOT="${PROJECT_ROOT}/results/sleep_edf_full"
PY="${PROJECT_ROOT}/.venv/bin/python"

cd "${PROJECT_ROOT}/code"
mkdir -p "${RESULT_ROOT}"

"${PY}" scripts/verify_sleep_edf.py \
  --data-root "${DATA_ROOT}"

"${PY}" scripts/make_sleep_edf_epochs.py \
  --data-root "${DATA_ROOT}" \
  --output-csv "${RESULT_ROOT}/epochs.csv" \
  --wake-trim-minutes 30

"${PY}" scripts/extract_sleep_edf_psd.py \
  --data-root "${DATA_ROOT}" \
  --epochs-csv "${RESULT_ROOT}/epochs.csv" \
  --output-npz "${RESULT_ROOT}/psd_welch_fpz_pz.npz" \
  --output-index-csv "${RESULT_ROOT}/psd_index.csv" \
  --channels Fpz-Cz Pz-Oz

"${PY}" scripts/fit_sleep_edf_specparam.py \
  --psd-npz "${RESULT_ROOT}/psd_welch_fpz_pz.npz" \
  --output-npz "${RESULT_ROOT}/specparam/specparam_fixed.npz" \
  --n-jobs "${SPECPARAM_N_JOBS:-12}"

"${PY}" scripts/extract_sleep_edf_raw_epochs.py \
  --epochs-csv "${RESULT_ROOT}/epochs.csv" \
  --output-npz "${RESULT_ROOT}/raw_epochs_fpz_pz_100hz.npz" \
  --output-index-csv "${RESULT_ROOT}/raw_epochs_index.csv" \
  --channels Fpz-Cz Pz-Oz

echo "Full Sleep-EDF preprocessing complete: ${RESULT_ROOT}"
