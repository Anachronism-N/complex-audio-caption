#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to MOSS-Audio-4B-Instruct weights}"
B3_WORK_DIR="${B3_WORK_DIR:-${PROJECT_ROOT}/runs/b3_valid}"
S1_WORK_DIR="${S1_WORK_DIR:-${PROJECT_ROOT}/runs/s1_valid}"
S1_CONFIG="${S1_CONFIG:-${PROJECT_ROOT}/configs/model/s1_event_slots.yaml}"
DEVICE="${DEVICE:-cuda:0}"
STAGE="${STAGE:-all}"
TAG_SUMMARY="${TAG_SUMMARY:-${PROJECT_ROOT}/runs/tag2021/reproduction_summary.json}"

python "${PROJECT_ROOT}/scripts/repro/require_anchor_pass.py" "${TAG_SUMMARY}"

TRAIN_MANIFEST="${B3_WORK_DIR}/sft/train_manifest.jsonl"
VAL_MANIFEST="${B3_WORK_DIR}/sft/val_manifest.jsonl"
AUDIO_BASE="${B3_WORK_DIR}/data"
FEATURE_CACHE="${S1_FEATURE_CACHE:-${S1_WORK_DIR}/features}"
OUTPUT_DIR="${S1_WORK_DIR}/model"
extra_args=("$@")

for required in "${TRAIN_MANIFEST}" "${VAL_MANIFEST}"; do
  if [[ ! -f "${required}" ]]; then
    echo "missing B3-valid split: ${required}; run scripts/run_b3_valid.sh first" >&2
    exit 2
  fi
done
if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "MOSS model directory not found: ${MODEL_DIR}" >&2
  exit 2
fi

mkdir -p "${S1_WORK_DIR}"
common=(
  --config "${S1_CONFIG}"
  --model-path "${MODEL_DIR}"
  --device "${DEVICE}"
  --train-manifest "${TRAIN_MANIFEST}"
  --val-manifest "${VAL_MANIFEST}"
  --audio-base "${AUDIO_BASE}"
  --feature-cache "${FEATURE_CACHE}"
  --output-dir "${OUTPUT_DIR}"
)

case "${STAGE}" in
  cache)
    python -m sceneledger.cli.train_slots "${common[@]}" "${extra_args[@]}" --cache-only
    ;;
  train)
    python -m sceneledger.cli.train_slots "${common[@]}" "${extra_args[@]}"
    ;;
  resume)
    checkpoint="${RESUME_CHECKPOINT:-${OUTPUT_DIR}/last.pt}"
    python -m sceneledger.cli.train_slots \
      "${common[@]}" "${extra_args[@]}" --resume "${checkpoint}"
    ;;
  evaluate)
    checkpoint="${EVAL_CHECKPOINT:-${OUTPUT_DIR}/best.pt}"
    python -m sceneledger.cli.train_slots \
      "${common[@]}" "${extra_args[@]}" --evaluate-checkpoint "${checkpoint}"
    ;;
  all)
    python -m sceneledger.cli.train_slots "${common[@]}" "${extra_args[@]}"
    ;;
  *)
    echo "STAGE must be one of: cache, train, resume, evaluate, all" >&2
    exit 2
    ;;
esac

echo "S1a-valid ${STAGE} completed: ${S1_WORK_DIR}" >&2
