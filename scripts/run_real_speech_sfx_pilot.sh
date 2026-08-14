#!/usr/bin/env bash
# Run the test-only real-source evidence pilot.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_path="${1:?usage: run_real_speech_sfx_pilot.sh SERVER_CONFIG OUTPUT_DIR}"
output_root="${2:?usage: run_real_speech_sfx_pilot.sh SERVER_CONFIG OUTPUT_DIR}"
quality_profile="${3:-pilot}"
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
"${python_bin}" - "${repo_root}" > "${output_root}/sceneledger_import_path.txt" <<'PY'
import pathlib
import sys

import sceneledger

repo_root = pathlib.Path(sys.argv[1]).resolve()
import_path = pathlib.Path(sceneledger.__file__).resolve()
try:
    import_path.relative_to(repo_root / "src")
except ValueError as exc:
    raise SystemExit(
        f"sceneledger resolves outside this checkout: {import_path}"
    ) from exc
print(import_path)
PY

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

dataset_id="real-speech-sfx-pilot-$(sha256sum "${output_root}/test/manifest.jsonl" | cut -c1-16)"
"${python_bin}" -m sceneledger.cli.human_audit prepare-standalone \
  --manifest "${output_root}/test/manifest.jsonl" \
  --quality-report "${output_root}/mixture_quality.json" \
  --dataset-id "${dataset_id}" \
  --split test \
  --per-template 5 \
  --all-samples \
  --max-violation-samples 20 \
  --output-csv "${output_root}/human_audit_tasks.csv" \
  --output-metadata "${output_root}/human_audit_tasks.meta.json"

echo "AUTOMATIC PILOT GATE PASS: ${output_root}/mixture_quality.json"
echo "HUMAN REVIEW REQUIRED: ${output_root}/human_audit_tasks.csv"
echo "AFTER REVIEW: bash scripts/summarize_real_speech_sfx_pilot.sh ${output_root}"
