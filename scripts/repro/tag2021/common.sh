#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
TAG_DATA_ROOT="${TAG_DATA_ROOT:-${REPO_ROOT}/external/tag2021}"
TAG_RUN_ROOT="${TAG_RUN_ROOT:-${REPO_ROOT}/runs/tag2021}"

tag_repro() {
  python -m sceneledger.repro.tag2021 \
    --repo-root "${REPO_ROOT}" \
    --data-root "${TAG_DATA_ROOT}" \
    --run-root "${TAG_RUN_ROOT}" \
    "$@"
}
