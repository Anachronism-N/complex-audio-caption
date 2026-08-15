#!/usr/bin/env bash
# Train B3 on the passed train fold and evaluate exactly the frozen test fold.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
experiment_root="${1:?usage: run_b3_real_complex_anchor.sh EXPERIMENT_ROOT MODEL_PATH [STEPS]}"
model_path="${2:?usage: run_b3_real_complex_anchor.sh EXPERIMENT_ROOT MODEL_PATH [STEPS]}"
steps="${3:-1000}"
python_bin="${PYTHON_BIN:-python}"
gate="${experiment_root}/gate"
runtime_config="${gate}/b3_real_complex_anchor.yaml"
model_output="${experiment_root}/model/b3_real_complex_anchor"
eval_root="${experiment_root}/evaluation"

if [[ -e "${runtime_config}" || -e "${model_output}" || -e "${eval_root}" ]]; then
  echo "Refusing to reuse model/evaluation artifacts under ${experiment_root}" >&2
  exit 2
fi
test -f "${gate}/human_audit_summary.json"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${python_bin}" "${repo_root}/scripts/make_b3_real_complex_anchor_config.py" \
  --experiment-root "${experiment_root}" \
  --model-path "${model_path}" \
  --steps "${steps}" \
  --output "${runtime_config}"

# Reject invalid data before either zero-shot inference or training can reserve
# a GPU.  The trainer repeats the same check immediately before model loading.
"${python_bin}" -m sceneledger.cli.train_preflight \
  --config "${runtime_config}" \
  --repo-root "${repo_root}" \
  --output "${gate}/training_preflight.json"

run_frozen_test() {
  local arm="$1"
  shift
  local arm_root="${eval_root}/${arm}"
  mkdir -p "${arm_root}"
  "${python_bin}" -m sceneledger.cli.infer \
    --manifest "${experiment_root}/test/manifest.jsonl" \
    --audio-base "${experiment_root}/test" \
    --backend moss \
    --model-path "${model_path}" \
    --device cuda:0 \
    --dtype bfloat16 \
    --greedy \
    --style detailed \
    --track-aware \
    --split-contract "${gate}/split_contract.json" \
    --expected-split test \
    --data-gate-summary "${gate}/experiment_data_summary.json" \
    --output "${arm_root}/predictions.jsonl" \
    --report "${arm_root}/inference_report.json" \
    "$@"

  "${python_bin}" -m sceneledger.cli.evaluate \
    --prediction "${arm_root}/predictions.jsonl" \
    --reference "${gate}/test_references.jsonl" \
    --inference-report "${arm_root}/inference_report.json" \
    --split-contract "${gate}/split_contract.json" \
    --expected-split test \
    --data-gate-summary "${gate}/experiment_data_summary.json" \
    --output "${arm_root}/metrics.json" \
    --pretty
}

# Evaluate the unchanged foundation model first.  This gives the tuned arm a
# real, same-test-set baseline instead of comparing against legacy v6.
run_frozen_test zero_shot

"${python_bin}" -m sceneledger.cli.train \
  --config "${runtime_config}" \
  --preflight-report "${gate}/training_preflight.json"
run_frozen_test b3_tuned --lora-path "${model_output}/lora"

"${python_bin}" -m sceneledger.cli.audit_result \
  --train-config "${runtime_config}" \
  --eval-manifest "${experiment_root}/test/manifest.jsonl" \
  --metrics "${eval_root}/b3_tuned/metrics.json" \
  --inference-report "${eval_root}/b3_tuned/inference_report.json" \
  --repo-root "${repo_root}" \
  --split-contract "${gate}/split_contract.json" \
  --data-gate-summary "${gate}/experiment_data_summary.json" \
  --output "${eval_root}/b3_tuned/validity_audit.json" \
  --require-pass

"${python_bin}" - \
  "${eval_root}/zero_shot/metrics.json" \
  "${eval_root}/b3_tuned/metrics.json" \
  "${eval_root}/b3_tuned/validity_audit.json" \
  "${eval_root}/comparison.json" <<'PY'
import hashlib
import json
import pathlib
import sys

zero = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
tuned = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
validity_path = pathlib.Path(sys.argv[3])
validity = json.loads(validity_path.read_text(encoding="utf-8"))
if validity.get("status") != "certified_generalization" or validity.get("pass") is not True:
    raise SystemExit("refusing to write comparison without passed result certification")
dataset_id = validity.get("dataset_id")
for arm, metrics in (("zero_shot", zero), ("b3_tuned", tuned)):
    evidence = metrics.get("experiment_contract", {})
    if evidence.get("dataset_id") != dataset_id or evidence.get("split") != "test":
        raise SystemExit(f"{arm} metrics are not bound to the certified test dataset")
keys = (
    "n_samples",
    "strict_format_success_rate",
    "macro_event_f1",
    "macro_caption_token_f1",
    "mean_onset_mae",
    "macro_tolerance_acc_010",
    "total_hallucination",
    "total_omission",
    "mean_source_count_mae",
    "mean_pointer_accuracy",
)
payload = {
    "schema_version": "sceneledger.anchor_comparison.v1",
    "validity_audit": {
        "path": str(validity_path.resolve()),
        "sha256": hashlib.sha256(validity_path.read_bytes()).hexdigest(),
        "status": validity["status"],
    },
    "metric_artifacts": {
        "zero_shot_sha256": hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest(),
        "b3_tuned_sha256": hashlib.sha256(pathlib.Path(sys.argv[2]).read_bytes()).hexdigest(),
    },
    "zero_shot": {key: zero.get(key) for key in keys},
    "b3_tuned": {key: tuned.get(key) for key in keys},
    "delta_tuned_minus_zero": {
        key: (
            round(tuned[key] - zero[key], 6)
            if isinstance(tuned.get(key), (int, float))
            and isinstance(zero.get(key), (int, float))
            else None
        )
        for key in keys
    },
}
pathlib.Path(sys.argv[4]).write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY

"${python_bin}" -m sceneledger.cli.model_review prepare \
  --manifest "${experiment_root}/test/manifest.jsonl" \
  --audio-base "${experiment_root}/test" \
  --zero-predictions "${eval_root}/zero_shot/predictions.jsonl" \
  --zero-inference-report "${eval_root}/zero_shot/inference_report.json" \
  --tuned-predictions "${eval_root}/b3_tuned/predictions.jsonl" \
  --tuned-inference-report "${eval_root}/b3_tuned/inference_report.json" \
  --validity-audit "${eval_root}/b3_tuned/validity_audit.json" \
  --split-contract "${gate}/split_contract.json" \
  --data-gate-summary "${gate}/experiment_data_summary.json" \
  --sample-count 60 \
  --output-csv "${eval_root}/human_model_review.csv" \
  --output-metadata "${eval_root}/human_model_review.metadata.json" \
  --output-key "${eval_root}/human_model_review.key.json"

echo "FROZEN TEST COMPARISON COMPLETE: ${eval_root}/comparison.json"
echo "BLINDED MODEL REVIEW READY: ${eval_root}/human_model_review.csv"
