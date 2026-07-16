#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-.}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/tuab/v3.0.1_random_stratified_200}"
SELECTED_FILES_CSV="${SELECTED_FILES_CSV:-$ROOT/results/tuab_full_v3_0_1/tuab_v3_0_1_full_edf_files.csv}"
OUTDIR="${OUTDIR:-$ROOT/results/tuab_full_v3_0_1/preprocess_20s_100hz}"
N_JOBS="${N_JOBS:-16}"
PSD_BATCH_SIZE="${PSD_BATCH_SIZE:-128}"
SPEC_QC_EPOCHS_PER_SPLIT_LABEL="${SPEC_QC_EPOCHS_PER_SPLIT_LABEL:-250}"

mkdir -p "$OUTDIR" "$OUTDIR/stage_markers"

run_stage() {
  local name="$1"
  shift
  local marker="$OUTDIR/stage_markers/${name}.done"
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
echo "DATA_ROOT=$DATA_ROOT"
echo "SELECTED_FILES_CSV=$SELECTED_FILES_CSV"
echo "OUTDIR=$OUTDIR"
echo "N_JOBS=$N_JOBS"
echo "PSD_BATCH_SIZE=$PSD_BATCH_SIZE"
echo "SPEC_QC_EPOCHS_PER_SPLIT_LABEL=$SPEC_QC_EPOCHS_PER_SPLIT_LABEL"

run_stage channel_audit \
  "$PYTHON" code/scripts/audit_tuab_channels.py \
    --data-root "$DATA_ROOT" \
    --selected-files-csv "$SELECTED_FILES_CSV" \
    --output-file-csv "$OUTDIR/tuab_channel_audit_files.csv" \
    --output-summary-json "$OUTDIR/tuab_channel_audit_summary.json"

run_stage epoch_manifest \
  "$PYTHON" code/scripts/make_tuab_epochs.py \
    --data-root "$DATA_ROOT" \
    --selected-files-csv "$SELECTED_FILES_CSV" \
    --output-csv "$OUTDIR/epochs_20s.csv" \
    --output-summary-json "$OUTDIR/epochs_20s_summary.json" \
    --epoch-seconds 20 \
    --stride-seconds 20

run_stage raw_epochs_20s_100hz \
  "$PYTHON" code/scripts/extract_tuab_raw_epochs.py \
    --epochs-csv "$OUTDIR/epochs_20s.csv" \
    --output-npz "$OUTDIR/raw_epochs_20s_100hz.npz" \
    --output-index-csv "$OUTDIR/raw_epochs_20s_100hz_index.csv" \
    --target-sfreq 100 \
    --bandpass-low 1 \
    --bandpass-high 45

run_stage psd_20s_multitaper \
  "$PYTHON" code/scripts/extract_tuab_psd.py \
    --raw-npz "$OUTDIR/raw_epochs_20s_100hz.npz" \
    --raw-index-csv "$OUTDIR/raw_epochs_20s_100hz_index.csv" \
    --output-npz "$OUTDIR/psd_20s_multitaper.npz" \
    --output-index-csv "$OUTDIR/psd_20s_multitaper_index.csv" \
    --freq-min 1 \
    --freq-max 45 \
    --bandwidth 2 \
    --batch-size "$PSD_BATCH_SIZE"

run_stage specparam_qc_20s \
  "$PYTHON" code/scripts/fit_tuab_specparam_qc.py \
    --psd-npz "$OUTDIR/psd_20s_multitaper.npz" \
    --psd-index-csv "$OUTDIR/psd_20s_multitaper_index.csv" \
    --output-csv "$OUTDIR/specparam_qc_20s_metrics.csv" \
    --output-summary-json "$OUTDIR/specparam_qc_20s_summary.json" \
    --epochs-per-split-label "$SPEC_QC_EPOCHS_PER_SPLIT_LABEL" \
    --n-jobs "$N_JOBS"

echo "[$(date --iso-8601=seconds)] TUAB full preprocessing stages complete."
