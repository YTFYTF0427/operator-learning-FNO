#!/bin/bash
# SLURM job script: zero-shot super-resolution evaluation on Avon.
#
# Generates test data at every resolution in cfg.eval.test_resolutions
# (PDE solves on fine_grid=311, subsample to coarse_grid=G), then runs
# the trained checkpoint at each grid and prints a Table 1-style summary.
#
# Submit:
#     sbatch slurm/super_resolution_eval.slurm
#
# Override config:
#     CONFIG=configs/square_32.yaml sbatch slurm/super_resolution_eval.slurm
#
# Force regeneration (otherwise reuses runs/<name>/eval/super_res_test_<G>.csv):
#     REGENERATE=1 sbatch slurm/super_resolution_eval.slurm
#
# Generation is CPU-bound (FD solves) and inference is fast on the GPU,
# so the same node handles both. For 1000 test PDEs at three resolutions
# on a 10-CPU GPU node, generation takes ~30 minutes; inference is seconds.

#SBATCH --job-name=neuropaths-super-res
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4000
#SBATCH --time=01:00:00

# ── Environment ───────────────────────────────────────────────────────
cd "${SLURM_SUBMIT_DIR}"
[ -f slurm/env.sh ] && source slurm/env.sh
source "${VENV_DIR:-.venv}/bin/activate"

set -euo pipefail

CONFIG="${CONFIG:-configs/square_32.yaml}"
REGENERATE_FLAG=""
if [ "${REGENERATE:-0}" = "1" ]; then
    REGENERATE_FLAG="--regenerate"
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "[$(date -Is)] host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} gpu=$(nvidia-smi -L | head -1) config=${CONFIG}"

uv run python scripts/super_resolution_eval.py \
    --config "${CONFIG}" \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --device cuda \
    ${REGENERATE_FLAG}

echo "[$(date -Is)] === super-res eval done ==="
