#!/usr/bin/env bash
# Validate the 60-row frozen test listening audit.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="${1:?usage: summarize_complex_speech_sfx_experiment.sh OUTPUT_DIR}"
python_bin="${PYTHON_BIN:-python}"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${python_bin}" -m sceneledger.cli.human_audit summarize \
  --review-csv "${output_root}/gate/human_audit_tasks.csv" \
  --metadata "${output_root}/gate/human_audit_tasks.meta.json" \
  --output "${output_root}/gate/human_audit_summary.json" \
  --max-severe 0 \
  --max-total-failures 6 \
  --template-failure-threshold 7

echo "COMPLEX TEST HUMAN GATE PASS: ${output_root}/gate/human_audit_summary.json"
echo "next=bash scripts/run_b3_real_complex_anchor.sh ${output_root} /path/to/moss_weights"
