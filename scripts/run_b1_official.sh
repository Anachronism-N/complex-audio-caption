#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
MOSS_DIR="${MOSS_DIR:-${PROJECT_ROOT}/third_party/MOSS-Audio}"
MOSS_COMMIT="${MOSS_COMMIT:-5cbb1d823937cd5b5de3d8fa4d3a7253ebd3b883}"
MODEL_DIR="${MODEL_DIR:-/tmp/moss_weights}"
AUDIO_DIR="${AUDIO_DIR:-/tmp/tac_mini}"
SFT_DIR="${SFT_DIR:-/tmp/sceneledger_b1_sft}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/b1_official}"
TAG_SUMMARY="${TAG_SUMMARY:-${PROJECT_ROOT}/runs/tag2021/reproduction_summary.json}"

python "${PROJECT_ROOT}/scripts/repro/require_anchor_pass.py" "${TAG_SUMMARY}"

if [[ ! -d "${MOSS_DIR}/.git" ]]; then
  echo "MOSS-Audio checkout not found at ${MOSS_DIR}" >&2
  exit 2
fi

actual_commit="$(git -C "${MOSS_DIR}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${MOSS_COMMIT}" ]]; then
  echo "MOSS-Audio revision mismatch: ${actual_commit} != ${MOSS_COMMIT}" >&2
  echo "Checkout the pinned revision or explicitly override MOSS_COMMIT." >&2
  exit 2
fi

python -m sceneledger.cli.prepare_moss_sft \
  --manifest "${AUDIO_DIR}/manifest.jsonl" \
  --audio-base "${AUDIO_DIR}" \
  --output-dir "${SFT_DIR}" \
  --target-mode atomic \
  --style brief \
  --val-fraction 0.1 \
  --seed 20260808

accelerate launch "${MOSS_DIR}/finetune/finetune.py" \
  --model_dir "${MODEL_DIR}" \
  --data_path "${SFT_DIR}/train.jsonl" \
  --eval_data_path "${SFT_DIR}/val.jsonl" \
  --output_dir "${OUTPUT_DIR}" \
  --use_lora \
  --lora_rank 128 \
  --lora_alpha 256 \
  --bf16 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_steps 1000 \
  --learning_rate 5e-5 \
  --warmup_steps 50 \
  --logging_steps 10 \
  --save_steps 125 \
  --eval_steps 125 \
  --eval_strategy steps \
  --save_strategy steps \
  --load_best_model_at_end \
  --metric_for_best_model eval_loss \
  --greater_is_better false \
  --report_to none \
  --seed 20260808

python -m sceneledger.cli.infer \
  --manifest "${SFT_DIR}/val_manifest.jsonl" \
  --audio-base "${AUDIO_DIR}" \
  --backend moss \
  --model-path "${MODEL_DIR}" \
  --lora-path "${OUTPUT_DIR}" \
  --greedy \
  --target-mode atomic \
  --output "${OUTPUT_DIR}/val_predictions.jsonl" \
  --report "${OUTPUT_DIR}/val_infer_report.json"

python -m sceneledger.cli.evaluate \
  --prediction "${OUTPUT_DIR}/val_predictions.jsonl" \
  --reference "${SFT_DIR}/val_references.jsonl" \
  --parse-report "${OUTPUT_DIR}/val_infer_report.json" \
  --output "${OUTPUT_DIR}/val_metrics.json" \
  --pretty
