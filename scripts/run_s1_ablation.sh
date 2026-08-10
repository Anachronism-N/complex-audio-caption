#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ABLATION_ROOT="${ABLATION_ROOT:-${PROJECT_ROOT}/runs/s1_ablation}"
SHARED_CACHE="${S1_FEATURE_CACHE:-${ABLATION_ROOT}/features}"

experiments=("$@")
if [[ ${#experiments[@]} -eq 0 ]]; then
  experiments=(
    main slots8 slots16 slots32 no_temporal_embedding no_positive_weight \
    activity_only boundary_only
  )
fi

run_experiment() {
  local name="$1"
  shift
  echo "running S1 ablation: ${name}" >&2
  S1_WORK_DIR="${ABLATION_ROOT}/${name}" \
  S1_FEATURE_CACHE="${SHARED_CACHE}" \
  STAGE=all \
    bash "${PROJECT_ROOT}/scripts/run_s1_valid.sh" "$@"
}

for experiment in "${experiments[@]}"; do
  case "${experiment}" in
    main) run_experiment main ;;
    slots8) run_experiment slots8 --n-slots 8 ;;
    slots16) run_experiment slots16 --n-slots 16 ;;
    slots32) run_experiment slots32 --n-slots 32 ;;
    no_temporal_embedding)
      run_experiment no_temporal_embedding --disable-temporal-embedding
      ;;
    no_positive_weight)
      run_experiment no_positive_weight --positive-weight-scale 0
      ;;
    activity_only)
      run_experiment activity_only --boundary-weight 0 --boundary-cost-weight 0 \
        --primary-decode-mode activity
      ;;
    boundary_only)
      run_experiment boundary_only --activity-weight 0 --activity-cost-weight 0 \
        --primary-decode-mode boundary
      ;;
    *)
      echo "unknown S1 ablation: ${experiment}" >&2
      exit 2
      ;;
  esac
done

echo "S1 ablations completed: ${ABLATION_ROOT}" >&2
