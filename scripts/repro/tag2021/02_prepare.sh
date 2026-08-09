#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

if [[ $# -gt 0 ]]; then
  tag_repro prepare --audio-root "$1"
else
  tag_repro prepare
fi
tag_repro audit
