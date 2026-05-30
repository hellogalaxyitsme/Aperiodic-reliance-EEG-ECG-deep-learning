#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/data/aperiodic_confounds}
PYTHON=${PYTHON:-$ROOT/.venvs/eegmamba/bin/python}
RUNNER=${RUNNER:-$ROOT/code/scripts/run_tuab_eegmamba_intervention.py}
OUT_ROOT=${OUT_ROOT:-$ROOT/results/tuab_subset_200/eegmamba_sanity_checks}
LOG_DIR=${LOG_DIR:-$ROOT/logs}
EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-64}
N_BOOTSTRAP=${N_BOOTSTRAP:-5000}
PATIENCE=${PATIENCE:-6}

export PYTHONPATH="$ROOT/external/EEGMamba"

mkdir -p "$OUT_ROOT" "$LOG_DIR"

run_one() {
  local name=$1
  local seed=$2
  shift 2
  local out_dir="$OUT_ROOT/${name}_seed${seed}"
  local log="$LOG_DIR/tuab_eegmamba_sanity_${name}_seed${seed}.log"

  if [[ -f "$out_dir/tuab_eegmamba_intervention_subject_bootstrap.csv" ]]; then
    echo "SKIP existing $out_dir"
    return 0
  fi

  echo "RUN $name seed=$seed"
  "$PYTHON" "$RUNNER" \
    --output-dir "$out_dir" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --patience "$PATIENCE" \
    --n-bootstrap "$N_BOOTSTRAP" \
    --seed "$seed" \
    --selection-metric val_balanced_accuracy \
    --device cuda \
    "$@" > "$log" 2>&1
  tail -20 "$log"
}

for seed in 20260525 20260526 20260527; do
  run_one div100_ft_vbacc "$seed" \
    --input-normalization divisor --input-divisor 100

  run_one noscale_ft_vbacc "$seed" \
    --input-normalization divisor --input-divisor 1

  run_one zscore_ft_vbacc "$seed" \
    --input-normalization zscore --input-divisor 1

  run_one div100_frozen_vbacc "$seed" \
    --input-normalization divisor --input-divisor 100 --freeze-backbone
done

echo "DONE $OUT_ROOT"
