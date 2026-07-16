#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-.}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
PREPROCESS_DIR="${PREPROCESS_DIR:-$ROOT/results/tuab_full_v3_0_1/preprocess_20s_100hz}"
PSD_NPZ="${PSD_NPZ:-$PREPROCESS_DIR/psd_20s_multitaper.npz}"
PSD_INDEX_CSV="${PSD_INDEX_CSV:-$PREPROCESS_DIR/psd_20s_multitaper_index.csv}"
SPECPARAM_DIR="${SPECPARAM_DIR:-$PREPROCESS_DIR/specparam}"
SPECPARAM_NPZ="${SPECPARAM_NPZ:-$SPECPARAM_DIR/specparam_fixed_20s.npz}"
PSD_INTERVENTION_DIR="${PSD_INTERVENTION_DIR:-$ROOT/results/tuab_full_v3_0_1/psd_interventions_specparam}"
N_JOBS="${N_JOBS:-16}"
CHUNK_SPECTRA="${CHUNK_SPECTRA:-8192}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"

mkdir -p "$SPECPARAM_DIR" "$PSD_INTERVENTION_DIR" "$PREPROCESS_DIR/stage_markers"

run_stage() {
  local name="$1"
  shift
  local marker="$PREPROCESS_DIR/stage_markers/${name}.done"
  if [[ -f "$marker" ]]; then
    echo "[$(date --iso-8601=seconds)] Skipping completed stage: $name"
    return 0
  fi
  echo "[$(date --iso-8601=seconds)] Starting stage: $name"
  "$@"
  echo "[$(date --iso-8601=seconds)] Completed stage: $name"
  touch "$marker"
}

echo "ROOT=$ROOT"
echo "PYTHON=$PYTHON"
echo "PREPROCESS_DIR=$PREPROCESS_DIR"
echo "PSD_NPZ=$PSD_NPZ"
echo "PSD_INDEX_CSV=$PSD_INDEX_CSV"
echo "SPECPARAM_NPZ=$SPECPARAM_NPZ"
echo "PSD_INTERVENTION_DIR=$PSD_INTERVENTION_DIR"
echo "N_JOBS=$N_JOBS"
echo "CHUNK_SPECTRA=$CHUNK_SPECTRA"
echo "N_BOOTSTRAP=$N_BOOTSTRAP"

run_stage specparam_full_20s \
  "$PYTHON" code/scripts/fit_tuab_specparam_full.py \
    --psd-npz "$PSD_NPZ" \
    --output-npz "$SPECPARAM_NPZ" \
    --freq-min 1 \
    --freq-max 45 \
    --max-n-peaks 6 \
    --min-peak-height 0.1 \
    --peak-threshold 2.0 \
    --peak-width-min 0.5 \
    --peak-width-max 8.0 \
    --n-jobs "$N_JOBS" \
    --chunk-spectra "$CHUNK_SPECTRA"

run_stage psd_interventions_specparam_full \
  "$PYTHON" code/scripts/run_tuab_psd_interventions.py \
    --psd-npz "$PSD_NPZ" \
    --index-csv "$PSD_INDEX_CSV" \
    --decomposition precomputed \
    --decomp-npz "$SPECPARAM_NPZ" \
    --output-dir "$PSD_INTERVENTION_DIR" \
    --n-bootstrap "$N_BOOTSTRAP"

echo "[$(date --iso-8601=seconds)] TUAB full specparam + PSD intervention stages complete."
