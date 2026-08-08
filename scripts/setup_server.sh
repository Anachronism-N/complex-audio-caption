#!/usr/bin/env bash
set -euo pipefail

SCENELEDGER_ENV_NAME="${SCENELEDGER_ENV_NAME:-sceneledger}"
SCENELEDGER_PROJECT_ROOT="${SCENELEDGER_PROJECT_ROOT:-$(pwd)}"
SCENELEDGER_THIRD_PARTY="${SCENELEDGER_THIRD_PARTY:-${SCENELEDGER_PROJECT_ROOT}/third_party}"
MOSS_ROOT="${MOSS_ROOT:-${SCENELEDGER_THIRD_PARTY}/MOSS-Audio}"

command -v conda >/dev/null || { echo "conda is required" >&2; exit 1; }
eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "${SCENELEDGER_ENV_NAME}"; then
  conda create -n "${SCENELEDGER_ENV_NAME}" python=3.12 -y
fi
conda activate "${SCENELEDGER_ENV_NAME}"
conda install -c conda-forge "ffmpeg=7" -y
python -m pip install --upgrade pip
python -m pip install -e "${SCENELEDGER_PROJECT_ROOT}[dev,download]"

mkdir -p "${SCENELEDGER_THIRD_PARTY}"
if [[ ! -d "${MOSS_ROOT}/.git" ]]; then
  git clone https://github.com/OpenMOSS/MOSS-Audio.git "${MOSS_ROOT}"
fi
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
  -e "${MOSS_ROOT}[torch-runtime]"

echo "Environment ready: ${SCENELEDGER_ENV_NAME}"
echo "MOSS checkout: ${MOSS_ROOT}"
