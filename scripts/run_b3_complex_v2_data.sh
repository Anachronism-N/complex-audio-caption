#!/usr/bin/env bash
# Render source-disjoint B3-complex-v2 folds and build the fail-closed gate.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="${1:-${repo_root}/data/derived/b3_complex_v2}"
python_bin="${PYTHON_BIN:-python}"

export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${output_root}"

# Sample the complete deterministic scene plans first. This catches missing
# complex-template weights and source-count distribution drift before WAV I/O.
"${python_bin}" -m sceneledger.cli.preflight_data \
  --config "${repo_root}/configs/data/b3_complex_v2_train.yaml" \
  --config "${repo_root}/configs/data/b3_complex_v2_val.yaml" \
  --config "${repo_root}/configs/data/b3_complex_v2_test.yaml" \
  --quality-config "${repo_root}/configs/data/mixture_quality.yaml" \
  --profile release \
  --output "${output_root}/scene_plan_preflight.json"

for split in train val test; do
  "${python_bin}" -m sceneledger.cli.render \
    --config "${repo_root}/configs/data/b3_complex_v2_${split}.yaml" \
    --output-dir "${output_root}/${split}" \
    --validate
done

"${python_bin}" -m sceneledger.cli.validate_experiment_data \
  --train-manifest "${output_root}/train/manifest.jsonl" \
  --val-manifest "${output_root}/val/manifest.jsonl" \
  --test-manifest "${output_root}/test/manifest.jsonl" \
  --quality-config "${repo_root}/configs/data/mixture_quality.yaml" \
  --profile release \
  --scene-plan-preflight "${output_root}/scene_plan_preflight.json" \
  --output-dir "${output_root}/gate"

"${python_bin}" -m sceneledger.cli.human_audit prepare \
  --manifest "${output_root}/test/manifest.jsonl" \
  --data-gate-summary "${output_root}/gate/experiment_data_summary.json" \
  --split-contract "${output_root}/gate/split_contract.json" \
  --expected-split test \
  --per-template 5 \
  --max-violation-samples 20 \
  --output-csv "${output_root}/gate/human_audit_tasks.csv" \
  --output-metadata "${output_root}/gate/human_audit_tasks.meta.json"

echo "AUTOMATIC DATA GATE PASS: ${output_root}/gate/experiment_data_summary.json"
echo "HUMAN REVIEW REQUIRED BEFORE TRAINING: ${output_root}/gate/human_audit_tasks.csv"
