#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${APERIODIC_PROJECT_ROOT:-.}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/ptbxl/1.0.3}"
BASE_URL="${BASE_URL:-https://physionet.org/files/ptb-xl/1.0.3}"
PTBXL_DOWNLOAD_JOBS="${PTBXL_DOWNLOAD_JOBS:-24}"

mkdir -p "${DATA_ROOT}"

echo "DATA_ROOT=${DATA_ROOT}"
echo "BASE_URL=${BASE_URL}"

download_file() {
  local rel="$1"
  mkdir -p "${DATA_ROOT}/$(dirname "${rel}")"
  if [[ -f "${DATA_ROOT}/${rel}" ]]; then
    echo "exists: ${rel}"
    return 0
  fi
  wget -c -O "${DATA_ROOT}/${rel}" "${BASE_URL}/${rel}"
}

download_file "ptbxl_database.csv"
download_file "scp_statements.csv"
download_file "RECORDS"
download_file "SHA256SUMS.txt"

download_record100() {
  local base="$1"
  local ext rel out url
  for ext in dat hea; do
    rel="${base}.${ext}"
    out="${DATA_ROOT}/${rel}"
    url="${BASE_URL}/${rel}"
    mkdir -p "$(dirname "${out}")"
    if [[ -s "${out}" ]]; then
      continue
    fi
    wget -q -c -O "${out}" "${url}"
  done
}

export DATA_ROOT BASE_URL
export -f download_record100

# Keep this compact demonstration on the 100-Hz records only. PTB-XL has many
# tiny WFDB files; downloading from RECORDS in parallel is much faster than a
# recursive serial wget while remaining resumable.
grep '^records100/' "${DATA_ROOT}/RECORDS" \
  | xargs -n 1 -P "${PTBXL_DOWNLOAD_JOBS}" bash -c 'download_record100 "$0"'

echo "PTB-XL records100 download complete."
