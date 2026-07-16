#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-.}
REMOTE_BASE=${REMOTE_BASE:-nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/tuh_eeg_abnormal/v3.0.1/}
DATA_ROOT=${DATA_ROOT:-$ROOT/data/tuab/v3.0.1_random_stratified_200}
FILES_FROM=${FILES_FROM:-$ROOT/results/tuab_full_v3_0_1/tuab_v3_0_1_full_files_from.txt}
LOG_DIR=${LOG_DIR:-$ROOT/logs}

mkdir -p "$DATA_ROOT" "$LOG_DIR"

if [[ ! -f "$FILES_FROM" ]]; then
  echo "Missing files-from manifest: $FILES_FROM" >&2
  exit 2
fi

if [[ -z "${SSH_AUTH_SOCK:-}" ]]; then
  cat >&2 <<'MSG'
SSH_AUTH_SOCK is empty. Run this script from a shell with SSH agent forwarding
enabled, or configure direct key-based access to the authorized TUAB host.
MSG
  exit 3
fi

echo "TUAB full resumable download"
echo "ROOT=$ROOT"
echo "REMOTE_BASE=$REMOTE_BASE"
echo "DATA_ROOT=$DATA_ROOT"
echo "FILES_FROM=$FILES_FROM"
echo "START=$(date -Is)"
echo "Existing EDF files: $(find "$DATA_ROOT/edf" -type f -name '*.edf' 2>/dev/null | wc -l)"

rsync -av \
  --partial \
  --append-verify \
  --human-readable \
  --progress \
  --files-from="$FILES_FROM" \
  -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=60 -o ServerAliveCountMax=10" \
  "$REMOTE_BASE" \
  "$DATA_ROOT/"

echo "END=$(date -Is)"
echo "EDF files now: $(find "$DATA_ROOT/edf" -type f -name '*.edf' 2>/dev/null | wc -l)"
du -sh "$DATA_ROOT" || true
