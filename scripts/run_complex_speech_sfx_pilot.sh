#!/usr/bin/env bash
# Render, audit and package the first six-source complex-data anchor.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_path="${1:?usage: run_complex_speech_sfx_pilot.sh CONFIG OUTPUT_DIR}"
output_root="${2:?usage: run_complex_speech_sfx_pilot.sh CONFIG OUTPUT_DIR}"
python_bin="${PYTHON_BIN:-python}"
quality_config="${repo_root}/configs/data/mixture_quality_real_speech_pilot.yaml"
complexity_config="${repo_root}/configs/data/complexity_profiles.yaml"

if [[ -e "${output_root}" ]]; then
  echo "Refusing to reuse output directory: ${output_root}" >&2
  exit 2
fi
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

readarray -t recipe_artifacts < <("${python_bin}" - "${config_path}" <<'PY'
import pathlib
import sys

import yaml

config = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(config["render"]["recipe_plan_path"])
print(config["recipe_experiment"]["human_review_path"])
PY
)
"${python_bin}" -m sceneledger.cli.scene_recipes validate-review \
  --recipes "${recipe_artifacts[0]}" \
  --review-csv "${recipe_artifacts[1]}" \
  --min-pass-rate 1.0 > /dev/null

mkdir -p "${output_root}"
git -C "${repo_root}" rev-parse HEAD > "${output_root}/git_commit.txt"
"${python_bin}" --version > "${output_root}/python_version.txt" 2>&1
"${python_bin}" -m pip freeze > "${output_root}/pip_freeze.txt"
"${python_bin}" - "${repo_root}" > "${output_root}/sceneledger_import_path.txt" <<'PY'
import pathlib
import sys

import sceneledger

root = pathlib.Path(sys.argv[1]).resolve()
module = pathlib.Path(sceneledger.__file__).resolve()
module.relative_to(root / "src")
print(module)
PY
"${python_bin}" -m sceneledger.cli.scene_recipes validate-review \
  --recipes "${recipe_artifacts[0]}" \
  --review-csv "${recipe_artifacts[1]}" \
  --min-pass-rate 1.0 \
  --output "${output_root}/recipe_review_report.json"

"${python_bin}" -m sceneledger.cli.preflight_data \
  --config "${config_path}" \
  --quality-config "${quality_config}" \
  --profile complex_speech_sfx \
  --output "${output_root}/scene_plan_preflight.json"

"${python_bin}" -m sceneledger.cli.render \
  --config "${config_path}" \
  --output-dir "${output_root}/test" \
  --validate

"${python_bin}" -m sceneledger.cli.audit_mixtures \
  --manifest "${output_root}/test/manifest.jsonl" \
  --quality-config "${quality_config}" \
  --profile complex_speech_sfx \
  --output "${output_root}/mixture_quality.json"

"${python_bin}" -m sceneledger.cli.audit_complexity \
  --manifest "${output_root}/test/manifest.jsonl" \
  --config "${complexity_config}" \
  --profile speech_sfx_complex_v1 \
  --output "${output_root}/complexity_audit.json"

dataset_id="complex-speech-sfx-$(sha256sum "${output_root}/test/manifest.jsonl" | cut -c1-16)"
"${python_bin}" -m sceneledger.cli.human_audit prepare-standalone \
  --manifest "${output_root}/test/manifest.jsonl" \
  --quality-report "${output_root}/mixture_quality.json" \
  --dataset-id "${dataset_id}" \
  --split test \
  --per-template 60 \
  --max-violation-samples 30 \
  --output-csv "${output_root}/human_audit_tasks.csv" \
  --output-metadata "${output_root}/human_audit_tasks.meta.json"

echo "AUTOMATIC COMPLEX-DATA GATES PASS"
echo "complexity=${output_root}/complexity_audit.json"
echo "quality=${output_root}/mixture_quality.json"
echo "HUMAN REVIEW REQUIRED: ${output_root}/human_audit_tasks.csv"
