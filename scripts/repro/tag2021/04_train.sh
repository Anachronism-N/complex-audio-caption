#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

SEED="${1:-1}"
NUM_WORKERS="${TAG_NUM_WORKERS:-4}"
tag_repro train --seed "${SEED}" --num-workers "${NUM_WORKERS}"
