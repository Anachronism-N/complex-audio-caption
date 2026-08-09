#!/usr/bin/env bash
#SBATCH --job-name=tag2021
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=1-3
#SBATCH --output=runs/tag2021/slurm/%A_%a.out
#SBATCH --error=runs/tag2021/slurm/%A_%a.err

set -euo pipefail
mkdir -p runs/tag2021/slurm

# Set TAG_CONDA_ENV to an absolute conda environment path if your cluster does
# not expose `conda run -n tag2021-legacy`.
if [[ -n "${TAG_CONDA_ENV:-}" ]]; then
  PYTHON_BIN="${TAG_CONDA_ENV}/bin/python"
  export PATH="${TAG_CONDA_ENV}/bin:${PATH}"
else
  PYTHON_BIN="python"
fi

export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" -m sceneledger.repro.tag2021 train \
  --seed "${SLURM_ARRAY_TASK_ID}" --num-workers "${SLURM_CPUS_PER_TASK}"
"${PYTHON_BIN}" -m sceneledger.repro.tag2021 evaluate \
  --seed "${SLURM_ARRAY_TASK_ID}"
