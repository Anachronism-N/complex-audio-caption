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

"${python_bin}" -m sceneledger.cli.train --config "${runtime_config}"
run_frozen_test b3_tuned --lora-path "${model_output}/lora"

"${python_bin}" - \
  "${eval_root}/zero_shot/metrics.json" \
  "${eval_root}/b3_tuned/metrics.json" \
  "${eval_root}/comparison.json" <<'PY'
import json
import pathlib
import sys

zero = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
tuned = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
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
pathlib.Path(sys.argv[3]).write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY

echo "FROZEN TEST COMPARISON COMPLETE: ${eval_root}/comparison.json"
