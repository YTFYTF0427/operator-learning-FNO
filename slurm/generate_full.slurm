#!/bin/bash
# SLURM job script: generate train + test splits for a full training run.
#
# Designed for CPU-heavy parallelism: with rho_min_max=0.01 and
# oversample_factor=1.5 at fine_grid=311, expect ~1500 train + ~300 test
# PDE solves at ~2 s/solve serially, ~10 s wall on a 168-CPU node.
#
# Submit:
#     sbatch slurm/generate_full.slurm
#
# Override config:
#     CONFIG=configs/square_32.yaml sbatch slurm/generate_full.slurm
#
# Chain into training:
#     gen=$(sbatch --parsable slurm/generate_full.slurm)
#     sbatch --dependency=afterok:$gen slurm/train_full.slurm

#SBATCH --job-name=neuropaths-generate
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00

# ── Environment ───────────────────────────────────────────────────────
# Submit from the repo root: `${SLURM_SUBMIT_DIR}` is the directory you
# ran `sbatch` from, so this works on any cluster (Blythe, Avon, ...)
# where you've cloned the repo and run `uv sync --frozen` once.
cd "${SLURM_SUBMIT_DIR}"
# Optional per-machine knobs (cache redirects, MLflow URI, etc.); the
# file is gitignored so each clone can carry its own.
[ -f slurm/env.sh ] && source slurm/env.sh
source "${VENV_DIR:-.venv}/bin/activate"

set -euo pipefail

CONFIG="${CONFIG:-configs/square_32.yaml}"

export PYTHONUNBUFFERED=1
# Single-threaded BLAS per worker; otherwise 168 workers * MKL fan-out
# would oversubscribe the node and slow everything down.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "[$(date -Is)] host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} config=${CONFIG}"

echo "[$(date -Is)] === generate train split ==="
uv run neuropaths-generate \
    --config "${CONFIG}" \
    --split  train \
    --num-workers "${SLURM_CPUS_PER_TASK}"

echo "[$(date -Is)] === generate test split ==="
uv run neuropaths-generate \
    --config "${CONFIG}" \
    --split  test \
    --num-workers "${SLURM_CPUS_PER_TASK}"

echo "[$(date -Is)] === generation done ==="
