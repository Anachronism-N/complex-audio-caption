#!/usr/bin/env bash
set -euo pipefail

: "${MOSS_ROOT:?Set MOSS_ROOT to the official MOSS-Audio checkout}"
: "${MOSS_MODEL_DIR:?Set MOSS_MODEL_DIR to the downloaded checkpoint}"
: "${TRAIN_JSONL:?Set TRAIN_JSONL to the converted MOSS conversation JSONL}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for checkpoints}"

ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
EPOCHS="${EPOCHS:-3}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"

accelerate launch --num_processes "${NUM_PROCESSES}" "${MOSS_ROOT}/finetune/finetune.py" \
  --model_dir "${MOSS_MODEL_DIR}" \
  --data_path "${TRAIN_JSONL}" \
  --output_dir "${OUTPUT_DIR}" \
  --use_lora \
  --lora_rank 32 \
  --bf16 \
  --attn_implementation "${ATTN_IMPLEMENTATION}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --num_train_epochs "${EPOCHS}" \
  --learning_rate "${LEARNING_RATE}" \
  --logging_steps 10 \
  --save_steps 500
