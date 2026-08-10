#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

WORK_DIR="${WORK_DIR:-${PROJECT_ROOT}/runs/b3_valid}"
DATA_CONFIG="${DATA_CONFIG:-${PROJECT_ROOT}/configs/data/b3_real.yaml}"
SOURCE_READINESS_CONFIG="${SOURCE_READINESS_CONFIG:-${PROJECT_ROOT}/configs/data/source_readiness.yaml}"
N_SAMPLES="${N_SAMPLES:-10000}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
SPLIT_SEED="${SPLIT_SEED:-20260808}"
STAGE="${STAGE:-all}"
TAG_SUMMARY="${TAG_SUMMARY:-${PROJECT_ROOT}/runs/tag2021/reproduction_summary.json}"
if [[ -z "${SOURCE_PROFILE:-}" ]]; then
  if (( N_SAMPLES <= 100 )); then
    SOURCE_PROFILE="smoke"
  else
    SOURCE_PROFILE="release"
  fi
fi

CANONICAL_CATALOG="${WORK_DIR}/source_catalog.jsonl"
SOURCE_REPORT="${WORK_DIR}/source_catalog_report.json"
SOURCE_INVENTORY="${WORK_DIR}/source_inventory.jsonl"
SOURCE_READINESS_REPORT="${WORK_DIR}/source_readiness_report.json"
DATA_DIR="${WORK_DIR}/data"
RENDER_REPORT="${DATA_DIR}/validation_report.json"
SFT_DIR="${WORK_DIR}/sft"
DATA_SUMMARY="${WORK_DIR}/data_reproduction_summary.json"

mkdir -p "${WORK_DIR}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "missing required data artifact: $1" >&2
    exit 2
  fi
}

require_anchor() {
  python "${PROJECT_ROOT}/scripts/repro/require_anchor_pass.py" "${TAG_SUMMARY}"
}

run_sources() {
  : "${SOURCE_CATALOG:?set SOURCE_CATALOG to source CSV/JSONL metadata}"
  : "${SOURCE_AUDIO_ROOT:?set SOURCE_AUDIO_ROOT to the waveform root}"
  python -m sceneledger.cli.prepare_sources \
    --input "${SOURCE_CATALOG}" \
    --audio-root "${SOURCE_AUDIO_ROOT}" \
    --output "${CANONICAL_CATALOG}" \
    --report "${SOURCE_REPORT}" \
    --require-kind speech \
    --require-kind vocal \
    --require-kind music \
    --require-kind sfx \
    --require-kind ambience

  run_source_audit
}

run_source_audit() {
  require_file "${CANONICAL_CATALOG}"
  python -m sceneledger.cli.audit_sources \
    --catalog "${CANONICAL_CATALOG}" \
    --config "${SOURCE_READINESS_CONFIG}" \
    --profile "${SOURCE_PROFILE}" \
    --inventory "${SOURCE_INVENTORY}" \
    --report "${SOURCE_READINESS_REPORT}"
}

require_source_readiness() {
  python "${PROJECT_ROOT}/scripts/data/require_source_readiness_pass.py" \
    "${SOURCE_READINESS_REPORT}" \
    --profile "${SOURCE_PROFILE}"
}

run_render() {
  require_anchor
  require_file "${CANONICAL_CATALOG}"
  require_file "${SOURCE_REPORT}"
  require_source_readiness
  python -m sceneledger.cli.render \
    --config "${DATA_CONFIG}" \
    --source-catalog "${CANONICAL_CATALOG}" \
    --output-dir "${DATA_DIR}" \
    --limit "${N_SAMPLES}" \
    --validate \
    --validation-report "${RENDER_REPORT}"
}

run_export() {
  require_anchor
  require_file "${DATA_DIR}/manifest.jsonl"
  require_file "${RENDER_REPORT}"
  python -m sceneledger.cli.prepare_moss_sft \
    --manifest "${DATA_DIR}/manifest.jsonl" \
    --audio-base "${DATA_DIR}" \
    --output-dir "${SFT_DIR}" \
    --target-mode atomic \
    --style brief \
    --include-lyrics \
    --include-tracks \
    --group-key source_id \
    --val-fraction "${VAL_FRACTION}" \
    --seed "${SPLIT_SEED}"
}

run_audit() {
  require_anchor
  require_file "${SOURCE_REPORT}"
  require_source_readiness
  require_file "${RENDER_REPORT}"
  require_file "${SFT_DIR}/metadata.json"
  require_file "${SFT_DIR}/train_manifest.jsonl"
  require_file "${SFT_DIR}/val_manifest.jsonl"
  extra_args=()
  if [[ "${ALLOW_UNKNOWN_LICENSE:-0}" == "1" ]]; then
    extra_args+=(--allow-unknown-license)
  fi
  python -m sceneledger.cli.validate_b3_data \
    --source-report "${SOURCE_REPORT}" \
    --source-readiness-report "${SOURCE_READINESS_REPORT}" \
    --render-report "${RENDER_REPORT}" \
    --sft-metadata "${SFT_DIR}/metadata.json" \
    --train-manifest "${SFT_DIR}/train_manifest.jsonl" \
    --val-manifest "${SFT_DIR}/val_manifest.jsonl" \
    --expected-samples "${N_SAMPLES}" \
    --output "${DATA_SUMMARY}" \
    "${extra_args[@]}"
}

case "${STAGE}" in
  sources) run_sources ;;
  source-audit) run_source_audit ;;
  render) run_render ;;
  export) run_export ;;
  audit) run_audit ;;
  all)
    run_sources
    run_render
    run_export
    run_audit
    ;;
  *)
    echo "STAGE must be one of: sources, source-audit, render, export, audit, all" >&2
    exit 2
    ;;
esac

echo "B3-valid data stage ${STAGE} completed: ${WORK_DIR}" >&2
