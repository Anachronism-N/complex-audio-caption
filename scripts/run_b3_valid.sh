#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
MOSS_DIR="${MOSS_DIR:-${PROJECT_ROOT}/third_party/MOSS-Audio}"
MOSS_COMMIT="${MOSS_COMMIT:-5cbb1d823937cd5b5de3d8fa4d3a7253ebd3b883}"
MODEL_DIR="${MODEL_DIR:-/tmp/moss_weights}"
SOURCE_CATALOG="${SOURCE_CATALOG:?set SOURCE_CATALOG to CSV/JSONL metadata}"
SOURCE_AUDIO_ROOT="${SOURCE_AUDIO_ROOT:?set SOURCE_AUDIO_ROOT to the waveform root}"
WORK_DIR="${WORK_DIR:-${PROJECT_ROOT}/runs/b3_valid}"
N_SAMPLES="${N_SAMPLES:-10000}"
MAX_STEPS="${MAX_STEPS:-10000}"

if [[ ! -d "${MOSS_DIR}/.git" ]]; then
  echo "MOSS-Audio checkout not found at ${MOSS_DIR}" >&2
  exit 2
fi
actual_commit="$(git -C "${MOSS_DIR}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${MOSS_COMMIT}" ]]; then
  echo "MOSS-Audio revision mismatch: ${actual_commit} != ${MOSS_COMMIT}" >&2
  exit 2
fi

mkdir -p "${WORK_DIR}"

python -m sceneledger.cli.prepare_sources \
  --input "${SOURCE_CATALOG}" \
  --audio-root "${SOURCE_AUDIO_ROOT}" \
  --output "${WORK_DIR}/source_catalog.jsonl" \
  --report "${WORK_DIR}/source_catalog_report.json" \
  --require-kind speech \
  --require-kind vocal \
  --require-kind music \
  --require-kind sfx \
  --require-kind ambience

python -m sceneledger.cli.render \
  --config "${PROJECT_ROOT}/configs/data/b3_real.yaml" \
  --source-catalog "${WORK_DIR}/source_catalog.jsonl" \
  --output-dir "${WORK_DIR}/data" \
  --limit "${N_SAMPLES}" \
  --validate

python -m sceneledger.cli.prepare_moss_sft \
  --manifest "${WORK_DIR}/data/manifest.jsonl" \
  --audio-base "${WORK_DIR}/data" \
  --output-dir "${WORK_DIR}/sft" \
  --target-mode atomic \
  --style brief \
  --include-lyrics \
  --include-tracks \
  --val-fraction 0.1 \
  --seed 20260808

python -m sceneledger.cli.train \
  --config "${PROJECT_ROOT}/configs/model/b3_valid.yaml" \
  --model-path "${MODEL_DIR}" \
  --manifest "${WORK_DIR}/data/manifest.jsonl" \
  --audio-base "${WORK_DIR}/data" \
  --output-dir "${WORK_DIR}/model" \
  --max-steps "${MAX_STEPS}"

python -m sceneledger.cli.infer \
  --manifest "${WORK_DIR}/sft/val_manifest.jsonl" \
  --audio-base "${WORK_DIR}/data" \
  --backend moss \
  --model-path "${MODEL_DIR}" \
  --lora-path "${WORK_DIR}/model/lora" \
  --greedy \
  --target-mode atomic \
  --include-lyrics \
  --include-tracks \
  --max-events 16 \
  --output "${WORK_DIR}/val_predictions.jsonl" \
  --report "${WORK_DIR}/val_infer_report.json"

# Legacy temporal-only event matching is retained for comparison.
python -m sceneledger.cli.evaluate \
  --prediction "${WORK_DIR}/val_predictions.jsonl" \
  --reference "${WORK_DIR}/sft/val_references.jsonl" \
  --parse-report "${WORK_DIR}/val_infer_report.json" \
  --min-text-similarity 0.0 \
  --output "${WORK_DIR}/val_metrics_temporal.json" \
  --pretty

# The lexical-semantic gate prevents a type/time-only true positive.  Add a
# multilingual embedding metric separately before making final paper claims.
python -m sceneledger.cli.evaluate \
  --prediction "${WORK_DIR}/val_predictions.jsonl" \
  --reference "${WORK_DIR}/sft/val_references.jsonl" \
  --parse-report "${WORK_DIR}/val_infer_report.json" \
  --min-text-similarity 0.1 \
  --output "${WORK_DIR}/val_metrics_text_gated.json" \
  --pretty

echo "B3-valid completed: ${WORK_DIR}" >&2
