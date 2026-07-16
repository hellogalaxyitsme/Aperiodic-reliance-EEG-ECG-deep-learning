#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${APERIODIC_PROJECT_ROOT:-.}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/ptbxl/1.0.3}"
ZIP_URL="${PTBXL_ZIP_URL:-https://physionet.org/content/ptb-xl/get-zip/1.0.3/}"
ZIP_DIR="${PTBXL_ZIP_DIR:-${PROJECT_ROOT}/data/ptbxl/bulk_zip}"
EXTRACT_DIR="${PTBXL_EXTRACT_DIR:-${PROJECT_ROOT}/data/ptbxl/bulk_extract}"
ZIP_PATH="${ZIP_DIR}/ptbxl_1.0.3.zip"

mkdir -p "${DATA_ROOT}" "${ZIP_DIR}" "${EXTRACT_DIR}"

echo "DATA_ROOT=${DATA_ROOT}"
echo "ZIP_URL=${ZIP_URL}"
echo "ZIP_PATH=${ZIP_PATH}"

complete_records100_count() {
  if [[ ! -f "${DATA_ROOT}/RECORDS" ]]; then
    echo 0
    return
  fi
  local expected actual
  expected=$(grep -c '^records100/' "${DATA_ROOT}/RECORDS" || true)
  actual=$(find "${DATA_ROOT}/records100" -type f 2>/dev/null | wc -l)
  if [[ "${expected}" -gt 0 && "${actual}" -ge $((expected * 2)) ]]; then
    echo 1
  else
    echo 0
  fi
}

if [[ "$(complete_records100_count)" == "1" && -f "${DATA_ROOT}/ptbxl_database.csv" && -f "${DATA_ROOT}/scp_statements.csv" ]]; then
  echo "PTB-XL records100 already complete; skipping bulk download."
  exit 0
fi

echo "Downloading PTB-XL bulk ZIP..."
if command -v aria2c >/dev/null 2>&1; then
  aria2c -c -x 16 -s 16 -k 1M \
    --summary-interval=60 \
    --console-log-level=notice \
    -d "${ZIP_DIR}" \
    -o "$(basename "${ZIP_PATH}")" \
    "${ZIP_URL}"
else
  curl -L -C - -o "${ZIP_PATH}" "${ZIP_URL}"
fi

echo "Extracting PTB-XL bulk ZIP..."
unzip -q -o "${ZIP_PATH}" -d "${EXTRACT_DIR}"

SRC_FILE=$(find "${EXTRACT_DIR}" -type f -name ptbxl_database.csv -print -quit)
if [[ -z "${SRC_FILE}" ]]; then
  echo "Could not locate ptbxl_database.csv after extracting ${ZIP_PATH}" >&2
  exit 2
fi

SRC_DIR=$(dirname "${SRC_FILE}")
echo "Extracted source tree: ${SRC_DIR}"
echo "Syncing extracted PTB-XL tree into ${DATA_ROOT}..."
rsync -a "${SRC_DIR}/" "${DATA_ROOT}/"

echo "PTB-XL bulk ZIP download complete."
