#!/usr/bin/env bash
set -euo pipefail

# Creates a project-local Python environment on H200.
# This writes only inside /mnt/data/aperiodic_confounds.

PROJECT_ROOT="${APERIODIC_PROJECT_ROOT:-/mnt/data/aperiodic_confounds}"
VENV_PATH="${PROJECT_ROOT}/.venv"
CODE_ROOT="${PROJECT_ROOT}/code"

python3 -m venv "${VENV_PATH}"
"${VENV_PATH}/bin/python" -m pip install --upgrade pip
"${VENV_PATH}/bin/python" -m pip install -r "${CODE_ROOT}/requirements.txt"

"${VENV_PATH}/bin/python" - <<'PY'
import mne
import numpy
import pandas
import scipy
print("mne", mne.__version__)
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("scipy", scipy.__version__)
PY

