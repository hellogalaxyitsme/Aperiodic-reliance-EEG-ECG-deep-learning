#!/usr/bin/env bash
set -euo pipefail

# Additive download only. This script never deletes, moves, or overwrites datasets.

DATA_ROOT="${APERIODIC_DATA_ROOT:-/mnt/data/aperiodic_confounds/data}"
MI_ROOT="${DATA_ROOT}/physionet-eegmmidb"
BASE="${PHYSIONET_MI_BASE:-https://physionet.org/files/eegmmidb/1.0.0}"
RECORDS="${MI_ROOT}/RECORDS"
CHECKSUMS="${MI_ROOT}/SHA256SUMS.txt"
URL_FILE="${MI_ROOT}/physionet-mi-files.txt"
ARIA2_FILE="${MI_ROOT}/physionet-mi-aria2.txt"

mkdir -p "${MI_ROOT}"

cd "${MI_ROOT}"
if [[ ! -s RECORDS ]]; then
  curl -fL --retry 3 -o RECORDS "${BASE}/RECORDS"
fi
if [[ ! -s SHA256SUMS.txt ]]; then
  curl -fL --retry 3 -o SHA256SUMS.txt "${BASE}/SHA256SUMS.txt"
fi

python3 - <<'PY' "${CHECKSUMS}" "${URL_FILE}" "${ARIA2_FILE}" "${BASE}"
from pathlib import Path
import sys

checksums_path = Path(sys.argv[1])
url_path = Path(sys.argv[2])
aria2_path = Path(sys.argv[3])
base = sys.argv[4].rstrip("/")

files = []
for line in checksums_path.read_text().splitlines():
    parts = line.strip().split()
    if len(parts) < 2:
        continue
    rel = parts[-1]
    if rel.endswith((".edf", ".event")) and rel.startswith("S"):
        files.append(rel)

if not files:
    raise SystemExit("No PhysioNet MI EDF/event files found in SHA256SUMS.txt")

url_path.write_text("\n".join(f"{base}/{rel}\t{rel}" for rel in sorted(files)) + "\n")
aria2_path.write_text(
    "\n".join(f"{base}/{rel}\n  out={rel}" for rel in sorted(files)) + "\n"
)
subjects = sorted({rel.split("/")[0] for rel in files})
edf_count = sum(rel.endswith(".edf") for rel in files)
event_count = sum(rel.endswith(".event") for rel in files)
print(f"Subjects listed: {len(subjects)}")
print(f"EDF files listed: {edf_count}")
print(f"event files listed: {event_count}")
print(f"First subject: {subjects[0] if subjects else 'NA'}")
print(f"Last subject: {subjects[-1] if subjects else 'NA'}")
PY

if command -v aria2c >/dev/null 2>&1; then
  aria2c \
    --dir="${MI_ROOT}" \
    --continue=true \
    --auto-file-renaming=false \
    --allow-overwrite=false \
    --max-concurrent-downloads=8 \
    --split=8 \
    --max-connection-per-server=8 \
    --retry-wait=5 \
    --max-tries=5 \
    --input-file="${ARIA2_FILE}"
else
  while IFS=$'\t' read -r url rel; do
    [[ -n "${url}" && -n "${rel}" ]] || continue
    dest="${MI_ROOT}/${rel}"
    mkdir -p "$(dirname "${dest}")"
    if [[ -s "${dest}" ]]; then
      echo "exists, skipping ${rel}"
    else
      curl -fL --retry 3 -C - -o "${dest}" "${url}"
    fi
  done < "${URL_FILE}"
fi

python3 - <<'PY' "${MI_ROOT}"
from pathlib import Path
import sys

root = Path(sys.argv[1])
edf_files = sorted(root.glob("S*/S*R*.edf"))
event_files = sorted(root.glob("S*/S*R*.event"))
subjects = sorted({path.parent.name for path in edf_files})
print(f"Downloaded EDF files visible: {len(edf_files)}")
print(f"Downloaded event files visible: {len(event_files)}")
print(f"Subjects visible: {len(subjects)}")
if subjects:
    print(f"Subject range: {subjects[0]} to {subjects[-1]}")
if len(subjects) < 109:
    raise SystemExit("WARNING: fewer than 109 subjects are visible")
if len(edf_files) < 109 * 14:
    raise SystemExit("WARNING: fewer than 1526 EDF files are visible")
PY
