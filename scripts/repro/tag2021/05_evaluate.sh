#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

SEED="${1:-1}"
tag_repro evaluate --seed "${SEED}"
