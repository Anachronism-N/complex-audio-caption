#!/usr/bin/env bash
# Render and freeze source-disjoint train/val/test folds for the real complex anchor.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
spec_root="${1:?usage: run_complex_speech_sfx_experiment.sh SPEC_DIR OUTPUT_DIR}"
output_root="${2:?usage: run_complex_speech_sfx_experiment.sh SPEC_DIR OUTPUT_DIR}"
python_bin="${PYTHON_BIN:-python}"
quality_config="${repo_root}/configs/data/mixture_quality_real_speech_pilot.yaml"
complexity_config="${repo_root}/configs/data/complexity_profiles.yaml"

if [[ -e "${output_root}" ]]; then
  echo "Refusing to reuse output directory: ${output_root}" >&2
  exit 2
fi
for split in train val test; do
  test -f "${spec_root}/${split}.yaml"
  test -f "${spec_root}/${split}.rule_recipes.jsonl"
  test -f "${spec_root}/${split}.rule_recipe_review.csv"
done

export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Review all three frozen recipe sets before doing any expensive rendering.
for split in train val test; do
  "${python_bin}" -m sceneledger.cli.scene_recipes validate-review \
    --recipes "${spec_root}/${split}.rule_recipes.jsonl" \
    --review-csv "${spec_root}/${split}.rule_recipe_review.csv" \
    --min-pass-rate 1.0 > /dev/null
done

mkdir -p "${output_root}/gate"
git -C "${repo_root}" rev-parse HEAD > "${output_root}/git_commit.txt"
"${python_bin}" --version > "${output_root}/python_version.txt" 2>&1
"${python_bin}" -m pip freeze > "${output_root}/pip_freeze.txt"
cp "${spec_root}/experiment_spec.json" "${output_root}/experiment_spec.json"

"${python_bin}" - "${repo_root}" > "${output_root}/sceneledger_import_path.txt" <<'PY'
import pathlib
import sys

import sceneledger

root = pathlib.Path(sys.argv[1]).resolve()
module = pathlib.Path(sceneledger.__file__).resolve()
module.relative_to(root / "src")
print(module)
PY

for split in train val test; do
  "${python_bin}" -m sceneledger.cli.scene_recipes validate-review \
    --recipes "${spec_root}/${split}.rule_recipes.jsonl" \
    --review-csv "${spec_root}/${split}.rule_recipe_review.csv" \
    --min-pass-rate 1.0 \
    --output "${output_root}/gate/${split}_recipe_review.json"
done

"${python_bin}" -m sceneledger.cli.preflight_data \
  --config "${spec_root}/train.yaml" \
  --config "${spec_root}/val.yaml" \
  --config "${spec_root}/test.yaml" \
  --quality-config "${quality_config}" \
  --profile complex_speech_sfx \
  --output "${output_root}/scene_plan_preflight.json"

for split in train val test; do
  "${python_bin}" -m sceneledger.cli.render \
    --config "${spec_root}/${split}.yaml" \
    --output-dir "${output_root}/${split}" \
    --validate
done

"${python_bin}" -m sceneledger.cli.validate_experiment_data \
  --train-manifest "${output_root}/train/manifest.jsonl" \
  --val-manifest "${output_root}/val/manifest.jsonl" \
  --test-manifest "${output_root}/test/manifest.jsonl" \
  --quality-config "${quality_config}" \
  --profile complex_speech_sfx \
  --complexity-config "${complexity_config}" \
  --complexity-profile speech_sfx_complex_v1 \
  --recipe-review-dir "${output_root}/gate" \
  --scene-plan-preflight "${output_root}/scene_plan_preflight.json" \
  --output-dir "${output_root}/gate"

"${python_bin}" -m sceneledger.cli.human_audit prepare \
  --manifest "${output_root}/test/manifest.jsonl" \
  --data-gate-summary "${output_root}/gate/experiment_data_summary.json" \
  --split-contract "${output_root}/gate/split_contract.json" \
  --expected-split test \
  --per-template 60 \
  --max-violation-samples 30 \
  --output-csv "${output_root}/gate/human_audit_tasks.csv" \
  --output-metadata "${output_root}/gate/human_audit_tasks.meta.json"

echo "AUTOMATIC THREE-FOLD DATA GATE PASS"
echo "contract=${output_root}/gate/split_contract.json"
echo "data_gate=${output_root}/gate/experiment_data_summary.json"
echo "HUMAN REVIEW REQUIRED: ${output_root}/gate/human_audit_tasks.csv"
echo "next=bash scripts/summarize_complex_speech_sfx_experiment.sh ${output_root}"
