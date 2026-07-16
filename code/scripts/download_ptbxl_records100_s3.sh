#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${APERIODIC_PROJECT_ROOT:-.}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/ptbxl/1.0.3}"
S3_BASE_URL="${PTBXL_S3_BASE_URL:-https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3}"
ARIA2_INPUT="${DATA_ROOT}/records100_s3_aria2.input"
PTBXL_S3_JOBS="${PTBXL_S3_JOBS:-96}"

mkdir -p "${DATA_ROOT}"

echo "DATA_ROOT=${DATA_ROOT}"
echo "S3_BASE_URL=${S3_BASE_URL}"
echo "PTBXL_S3_JOBS=${PTBXL_S3_JOBS}"

records100_bases() {
  python3 - "${DATA_ROOT}/ptbxl_database.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as f:
    reader = csv.DictReader(f)
    seen = set()
    for row in reader:
        base = (row.get("filename_lr") or "").strip()
        if base.startswith("records100/") and base not in seen:
            print(base)
            seen.add(base)
PY
}

complete_records100_count() {
  local expected actual
  expected=$(records100_bases | wc -l)
  actual=$(find "${DATA_ROOT}/records100" -type f \( -name '*_lr.dat' -o -name '*_lr.hea' \) 2>/dev/null | wc -l)
  if [[ "${expected}" -gt 0 && "${actual}" -ge $((expected * 2)) ]]; then
    echo 1
  else
    echo 0
  fi
}

download_metadata() {
  local rel="$1"
  if [[ -s "${DATA_ROOT}/${rel}" ]]; then
    echo "exists: ${rel}"
    return 0
  fi
  curl -L --fail --retry 5 --retry-delay 3 -o "${DATA_ROOT}/${rel}" "${S3_BASE_URL}/${rel}"
}

download_metadata "ptbxl_database.csv"
download_metadata "scp_statements.csv"
download_metadata "RECORDS"
download_metadata "SHA256SUMS.txt"

if [[ "$(complete_records100_count)" == "1" ]]; then
  echo "PTB-XL records100 already complete; skipping S3 download."
  exit 0
fi

echo "Building aria2 manifest: ${ARIA2_INPUT}"
{
  while IFS= read -r base; do
    for ext in dat hea; do
      rel="${base}.${ext}"
      printf '%s/%s\n' "${S3_BASE_URL}" "${rel}"
      printf '  dir=%s\n' "${DATA_ROOT}"
      printf '  out=%s\n' "${rel}"
    done
  done < <(records100_bases)
} > "${ARIA2_INPUT}"

echo "Downloading PTB-XL records100 from S3..."
aria2c -c \
  -j "${PTBXL_S3_JOBS}" \
  -x 1 \
  -s 1 \
  --allow-overwrite=true \
  --auto-file-renaming=false \
  --max-tries=10 \
  --retry-wait=5 \
  --summary-interval=60 \
  --console-log-level=notice \
  -i "${ARIA2_INPUT}"

echo "PTB-XL S3 records100 download complete."
