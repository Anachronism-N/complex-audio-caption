#!/usr/bin/env bash
# Render one frozen rule/LLM recipe arm and prepare a stratified human review.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_path="${1:?usage: run_recipe_mix_arm.sh CONFIG OUTPUT_DIR [PROFILE]}"
output_root="${2:?usage: run_recipe_mix_arm.sh CONFIG OUTPUT_DIR [PROFILE]}"
quality_profile="${3:-recipe_scale}"
python_bin="${PYTHON_BIN:-python}"
quality_config="${repo_root}/configs/data/mixture_quality_real_speech_pilot.yaml"

if [[ -e "${output_root}" ]]; then
  echo "Refusing to reuse output directory: ${output_root}" >&2
  exit 2
fi
mkdir -p "${output_root}"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

git -C "${repo_root}" rev-parse HEAD > "${output_root}/git_commit.txt"
"${python_bin}" --version > "${output_root}/python_version.txt" 2>&1
"${python_bin}" -m pip freeze > "${output_root}/pip_freeze.txt"

"${python_bin}" -m sceneledger.cli.preflight_data \
  --config "${config_path}" \
  --quality-config "${quality_config}" \
  --profile "${quality_profile}" \
  --output "${output_root}/scene_plan_preflight.json"

"${python_bin}" -m sceneledger.cli.render \
  --config "${config_path}" \
  --output-dir "${output_root}/test" \
  --validate

"${python_bin}" -m sceneledger.cli.audit_mixtures \
  --manifest "${output_root}/test/manifest.jsonl" \
  --quality-config "${quality_config}" \
  --profile "${quality_profile}" \
  --output "${output_root}/mixture_quality.json"

dataset_id="recipe-mix-$(sha256sum "${output_root}/test/manifest.jsonl" | cut -c1-16)"
"${python_bin}" -m sceneledger.cli.human_audit prepare-standalone \
  --manifest "${output_root}/test/manifest.jsonl" \
  --quality-report "${output_root}/mixture_quality.json" \
  --dataset-id "${dataset_id}" \
  --split test \
  --per-template 20 \
  --max-violation-samples 30 \
  --output-csv "${output_root}/human_audit_tasks.csv" \
  --output-metadata "${output_root}/human_audit_tasks.meta.json"

echo "AUTOMATIC RECIPE ARM PASS: ${output_root}/mixture_quality.json"
echo "STRATIFIED HUMAN REVIEW REQUIRED: ${output_root}/human_audit_tasks.csv"
