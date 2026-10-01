#!/bin/bash
# SLURM job script: neuropaths-generate with multiprocessing fan-out.
#
# Submit:
#     sbatch slurm/generate.slurm
#
# Override config / split:
#     CONFIG=configs/section2_laminar_32.yaml SPLIT=test sbatch slurm/generate.slurm
#
# Blythe conventions follow scrtp-docs (hpc-jobscript, hpc-slurm-examples).
# Default partition on Blythe is the standard CPU partition; pass -p <name>
# to sbatch to use a non-default one.

#SBATCH --job-name=neuropaths-generate
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
# For a whole Blythe node: --cpus-per-task=168 (matches the docs' blythe-example).
# For a quick smoke test: --cpus-per-task=4  --time=00:30:00.

# ── Environment ───────────────────────────────────────────────────────
# Submit from the repo root; works on any cluster where you've cloned
# the repo and run `uv sync --frozen` once.
cd "${SLURM_SUBMIT_DIR}"
[ -f slurm/env.sh ] && source slurm/env.sh
source "${VENV_DIR:-.venv}/bin/activate"

set -euo pipefail

CONFIG="${CONFIG:-configs/section2_laminar_32.yaml}"
SPLIT="${SPLIT:-train}"

# Progress output: tqdm writes to stderr, which SLURM otherwise buffers
# aggressively -- tail -f would show nothing until the job ended.
export PYTHONUNBUFFERED=1

# Each multiprocessing worker runs single-threaded BLAS; without these,
# 32 workers * N BLAS threads oversubscribes the node and slows everything
# down. Scipy's SuperLU is single-threaded regardless, but numpy linalg and
# any MKL/OpenBLAS call inside autograd would otherwise fan out.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "[$(date -Is)] host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} config=${CONFIG} split=${SPLIT}"

uv run neuropaths-generate \
    --config "${CONFIG}" \
    --split  "${SPLIT}" \
    --num-workers "${SLURM_CPUS_PER_TASK}"
