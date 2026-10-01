#!/bin/bash
# SLURM job script: train the FNO on Avon (Quadro RTX 6000 GPU nodes).
#
# Submit:
#     sbatch slurm/train_full_avon.slurm
#
# Override config:
#     CONFIG=configs/square_32.yaml sbatch slurm/train_full_avon.slurm
#
# Chain after generation:
#     gen=$(sbatch --parsable slurm/generate_full.slurm)
#     sbatch --dependency=afterok:$gen slurm/train_full_avon.slurm
#
# Output:
#   runs/<name>/<stem>_state_dict.pt  (clean state_dict)
#   runs/<name>/<stem>_metadata.pkl   (init kwargs; loaded via FNO.from_checkpoint)
#
# Avon hardware (per scrtp-docs):
#   - GPU nodes: 3x NVIDIA Quadro RTX 6000 (24 GB) per node, 48 cores,
#     192 GB RAM (so 4 GB / core, 16 cores per GPU "slot").
#   - gres name: quadro_rtx_6000.
# For Blythe (lovelace_l40), use slurm/train_full.slurm instead.

#SBATCH --job-name=neuropaths-train
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4000
#SBATCH --time=04:00:00

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
# Pin BLAS even though training is GPU-bound; prevents PyTorch dataloader
# workers from accidentally fanning out via numpy ops.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "[$(date -Is)] host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} gpu=$(nvidia-smi -L | head -1) config=${CONFIG}"

uv run neuropaths-train --config "${CONFIG}"

echo "[$(date -Is)] === training done ==="
