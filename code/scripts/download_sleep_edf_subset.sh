#!/usr/bin/env bash
set -euo pipefail

# Additive download only. This script never deletes or moves existing data.

DATA_ROOT="${APERIODIC_DATA_ROOT:-/mnt/data/aperiodic_confounds/data}"
DEST="${DATA_ROOT}/sleep-edf/sleep-cassette"
BASE="https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/sleep-cassette"

mkdir -p "${DEST}"
cd "${DEST}"

FILES=(
  SC4001E0-PSG.edf SC4001EC-Hypnogram.edf
  SC4002E0-PSG.edf SC4002EC-Hypnogram.edf
  SC4011E0-PSG.edf SC4011EH-Hypnogram.edf
  SC4012E0-PSG.edf SC4012EC-Hypnogram.edf
  SC4021E0-PSG.edf SC4021EH-Hypnogram.edf
  SC4022E0-PSG.edf SC4022EJ-Hypnogram.edf
  SC4031E0-PSG.edf SC4031EC-Hypnogram.edf
  SC4032E0-PSG.edf SC4032EP-Hypnogram.edf
  SC4041E0-PSG.edf SC4041EC-Hypnogram.edf
  SC4042E0-PSG.edf SC4042EC-Hypnogram.edf
  SC4051E0-PSG.edf SC4051EC-Hypnogram.edf
  SC4052E0-PSG.edf SC4052EC-Hypnogram.edf
  SC4061E0-PSG.edf SC4061EC-Hypnogram.edf
  SC4062E0-PSG.edf SC4062EC-Hypnogram.edf
  SC4071E0-PSG.edf SC4071EC-Hypnogram.edf
  SC4072E0-PSG.edf SC4072EH-Hypnogram.edf
  SC4081E0-PSG.edf SC4081EC-Hypnogram.edf
  SC4082E0-PSG.edf SC4082EP-Hypnogram.edf
  SC4091E0-PSG.edf SC4091EC-Hypnogram.edf
  SC4092E0-PSG.edf SC4092EC-Hypnogram.edf
)

URLS=()
for file in "${FILES[@]}"; do
  URLS+=("${BASE}/${file}")
done

if command -v aria2c >/dev/null 2>&1; then
  aria2c -Z -c -x 8 -s 8 -j 4 --retry-wait=5 --max-tries=5 "${URLS[@]}"
else
  for url in "${URLS[@]}"; do
    file="${url##*/}"
    if [[ -s "${file}" ]]; then
      echo "exists, skipping ${file}"
    else
      curl -fL --retry 3 -C - -o "${file}" "${url}"
    fi
  done
fi

cd "${DATA_ROOT}/sleep-edf"
aria2c -Z -c -x 4 -s 4 -j 2 \
  "https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/SC-subjects.xls" \
  "https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/RECORDS"

python3 /mnt/data/aperiodic_confounds/code/scripts/verify_sleep_edf.py \
  --data-root "${DEST}" \
  --expected-pairs 20
