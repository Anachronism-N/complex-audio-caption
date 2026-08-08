#!/usr/bin/env bash
set -euo pipefail

SCENELEDGER_WEIGHT_ROOT="${SCENELEDGER_WEIGHT_ROOT:-$(pwd)/weights}"
MOSS_MODEL_ID="${MOSS_MODEL_ID:-OpenMOSS-Team/MOSS-Audio-4B-Instruct}"
MOSS_MODEL_DIR="${MOSS_MODEL_DIR:-${SCENELEDGER_WEIGHT_ROOT}/MOSS-Audio-4B-Instruct}"

command -v hf >/dev/null || { echo "Install huggingface_hub and run hf auth login" >&2; exit 1; }
mkdir -p "${SCENELEDGER_WEIGHT_ROOT}"
hf download "${MOSS_MODEL_ID}" --local-dir "${MOSS_MODEL_DIR}"
echo "Downloaded ${MOSS_MODEL_ID} to ${MOSS_MODEL_DIR}"
