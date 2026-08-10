#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

MOSS_DIR="${MOSS_DIR:-${PROJECT_ROOT}/third_party/MOSS-Audio}"
MOSS_COMMIT="${MOSS_COMMIT:-5cbb1d823937cd5b5de3d8fa4d3a7253ebd3b883}"
MODEL_DIR="${MODEL_DIR:-/tmp/moss_weights}"
WORK_DIR="${WORK_DIR:-${PROJECT_ROOT}/runs/b3_valid}"
N_SAMPLES="${N_SAMPLES:-10000}"
MAX_STEPS="${MAX_STEPS:-10000}"
STAGE="${STAGE:-all}"
TAG_SUMMARY="${TAG_SUMMARY:-${PROJECT_ROOT}/runs/tag2021/reproduction_summary.json}"
DATA_SUMMARY="${B3_DATA_SUMMARY:-${WORK_DIR}/data_reproduction_summary.json}"

python "${PROJECT_ROOT}/scripts/repro/require_anchor_pass.py" "${TAG_SUMMARY}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "missing required experiment artifact: $1" >&2
    exit 2
  fi
}

require_data() {
  local args=("${DATA_SUMMARY}")
  if [[ -n "${B3_DATASET_ID:-}" ]]; then
    args+=(--dataset-id "${B3_DATASET_ID}")
  fi
  python "${PROJECT_ROOT}/scripts/data/require_b3_data_pass.py" "${args[@]}"
}

require_moss() {
  if [[ ! -d "${MOSS_DIR}/.git" ]]; then
    echo "MOSS-Audio checkout not found at ${MOSS_DIR}" >&2
    exit 2
  fi
  actual_commit="$(git -C "${MOSS_DIR}" rev-parse HEAD)"
  if [[ "${actual_commit}" != "${MOSS_COMMIT}" ]]; then
    echo "MOSS-Audio revision mismatch: ${actual_commit} != ${MOSS_COMMIT}" >&2
    exit 2
  fi
  if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "MOSS model directory not found: ${MODEL_DIR}" >&2
    exit 2
  fi
}

run_data() {
  WORK_DIR="${WORK_DIR}" N_SAMPLES="${N_SAMPLES}" STAGE=all \
    bash "${PROJECT_ROOT}/scripts/run_b3_data.sh"
}

run_train() {
  require_data
  require_moss
  python -m sceneledger.cli.train \
    --config "${PROJECT_ROOT}/configs/model/b3_valid.yaml" \
    --model-path "${MODEL_DIR}" \
    --train-manifest "${WORK_DIR}/sft/train_manifest.jsonl" \
    --audio-base "${WORK_DIR}/data" \
    --output-dir "${WORK_DIR}/model" \
    --max-steps "${MAX_STEPS}"
}

run_infer() {
  require_data
  require_moss
  require_file "${WORK_DIR}/model/lora/adapter_config.json"
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
}

run_evaluate() {
  require_data
  require_file "${WORK_DIR}/val_predictions.jsonl"
  require_file "${WORK_DIR}/val_infer_report.json"
  require_file "${WORK_DIR}/sft/val_references.jsonl"
  python -m sceneledger.cli.evaluate \
    --prediction "${WORK_DIR}/val_predictions.jsonl" \
    --reference "${WORK_DIR}/sft/val_references.jsonl" \
    --parse-report "${WORK_DIR}/val_infer_report.json" \
    --min-text-similarity 0.0 \
    --output "${WORK_DIR}/val_metrics_temporal.json" \
    --pretty

  python -m sceneledger.cli.evaluate \
    --prediction "${WORK_DIR}/val_predictions.jsonl" \
    --reference "${WORK_DIR}/sft/val_references.jsonl" \
    --parse-report "${WORK_DIR}/val_infer_report.json" \
    --min-text-similarity 0.1 \
    --output "${WORK_DIR}/val_metrics_text_gated.json" \
    --pretty
}

case "${STAGE}" in
  data) run_data ;;
  train) run_train ;;
  infer) run_infer ;;
  evaluate) run_evaluate ;;
  all)
    run_data
    run_train
    run_infer
    run_evaluate
    ;;
  *)
    echo "STAGE must be one of: data, train, infer, evaluate, all" >&2
    exit 2
    ;;
esac

echo "B3-valid stage ${STAGE} completed: ${WORK_DIR}" >&2
