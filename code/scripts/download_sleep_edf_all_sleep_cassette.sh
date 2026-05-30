#!/usr/bin/env bash
set -euo pipefail

# Additive download only. This script never deletes or moves existing data.

DATA_ROOT="${APERIODIC_DATA_ROOT:-/mnt/data/aperiodic_confounds/data}"
SLEEP_EDF_ROOT="${DATA_ROOT}/sleep-edf"
DEST="${SLEEP_EDF_ROOT}/sleep-cassette"
BASE="https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0"
URL_FILE="${SLEEP_EDF_ROOT}/sleep-cassette-urls.txt"
CHECKSUMS="${SLEEP_EDF_ROOT}/SHA256SUMS.txt"

mkdir -p "${DEST}" "${SLEEP_EDF_ROOT}"

cd "${SLEEP_EDF_ROOT}"
if [[ ! -s RECORDS ]]; then
  curl -fL --retry 3 -o RECORDS "${BASE}/RECORDS"
fi
if [[ ! -s SHA256SUMS.txt ]]; then
  curl -fL --retry 3 -o SHA256SUMS.txt "${BASE}/SHA256SUMS.txt"
fi
if [[ ! -s SC-subjects.xls ]]; then
  curl -fL --retry 3 -o SC-subjects.xls "${BASE}/SC-subjects.xls"
fi

python3 - <<'PY' "${CHECKSUMS}" "${URL_FILE}" "${BASE}"
from pathlib import Path
import sys

checksums_path = Path(sys.argv[1])
url_path = Path(sys.argv[2])
base = sys.argv[3].rstrip("/")

files = []
for line in checksums_path.read_text().splitlines():
    parts = line.strip().split()
    if len(parts) < 2:
        continue
    path = parts[-1]
    if not path.startswith("sleep-cassette/"):
        continue
    if not path.endswith(".edf"):
        continue
    files.append(path)

if not files:
    raise SystemExit("No sleep-cassette EDF files found in SHA256SUMS.txt")

url_path.write_text("\n".join(f"{base}/{path}" for path in sorted(files)) + "\n")

psg_keys = {Path(path).name[:6] for path in files if "PSG.edf" in path}
hyp_keys = {Path(path).name[:6] for path in files if "Hypnogram.edf" in path}
pairs = sorted(psg_keys & hyp_keys)
print(f"sleep-cassette EDF files in RECORDS: {len(files)}")
print(f"expected PSG/Hypnogram pairs: {len(pairs)}")
print(f"first pair: {pairs[0] if pairs else 'NA'}")
print(f"last pair: {pairs[-1] if pairs else 'NA'}")
PY

cd "${DEST}"
if command -v aria2c >/dev/null 2>&1; then
  aria2c -c -x 8 -s 8 -j 6 --retry-wait=5 --max-tries=5 -i "${URL_FILE}"
else
  while IFS= read -r url; do
    file="${url##*/}"
    if [[ -s "${file}" ]]; then
      echo "exists, skipping ${file}"
    else
      curl -fL --retry 3 -C - -O "${url}"
    fi
  done < "${URL_FILE}"
fi

EXPECTED_PAIRS="$(python3 - <<'PY' "${CHECKSUMS}"
from pathlib import Path
import sys

files = [
    line.strip().split()[-1]
    for line in Path(sys.argv[1]).read_text().splitlines()
    if len(line.strip().split()) >= 2
    and line.strip().split()[-1].startswith("sleep-cassette/")
    and line.strip().split()[-1].endswith(".edf")
]
psg_keys = {Path(path).name[:6] for path in files if "PSG.edf" in path}
hyp_keys = {Path(path).name[:6] for path in files if "Hypnogram.edf" in path}
print(len(psg_keys & hyp_keys))
PY
)"

python3 /mnt/data/aperiodic_confounds/code/scripts/verify_sleep_edf.py \
  --data-root "${DEST}" \
  --expected-pairs "${EXPECTED_PAIRS}"
