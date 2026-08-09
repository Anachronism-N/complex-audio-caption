#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

if [[ $# -eq 0 ]]; then
  set -- 1 2 3
fi
tag_repro summarize --seeds "$@" --output "${TAG_RUN_ROOT}/reproduction_summary.json"
