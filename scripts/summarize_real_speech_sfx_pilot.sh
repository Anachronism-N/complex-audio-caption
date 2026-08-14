#!/usr/bin/env bash
# Validate the completed 30-row human review and emit the final go/no-go report.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="${1:?usage: summarize_real_speech_sfx_pilot.sh OUTPUT_DIR}"
python_bin="${PYTHON_BIN:-python}"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${python_bin}" -m sceneledger.cli.human_audit summarize \
  --review-csv "${output_root}/human_audit_tasks.csv" \
  --metadata "${output_root}/human_audit_tasks.meta.json" \
  --output "${output_root}/human_audit_summary.json" \
  --max-severe 0 \
  --max-total-failures 2 \
  --template-failure-threshold 2

echo "PILOT HUMAN GATE PASS: ${output_root}/human_audit_summary.json"
