#!/usr/bin/env bash
# Render source-disjoint B3-complex-v2 folds and build the fail-closed gate.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="${1:-${repo_root}/data/derived/b3_complex_v2}"
python_bin="${PYTHON_BIN:-python}"

export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${output_root}"

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
  --output-dir "${output_root}/gate"

echo "PASS: ${output_root}/gate/experiment_data_summary.json"
