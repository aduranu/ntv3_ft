#!/bin/bash
#SBATCH --job-name=ntv3_ft_leaf
#SBATCH --partition=gpuq
#SBATCH --qos=bio_ai
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --chdir=/grid/koo/home/duran/ntv3_ft
#SBATCH --output=/grid/koo/home/duran/ntv3_ft/ntv3_ft_leaf_%j.out
#SBATCH --error=/grid/koo/home/duran/ntv3_ft/ntv3_ft_leaf_%j.out

set -euo pipefail

echo "=== job $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=== start: $(date) ==="

PYBIN=/grid/koo/home/duran/miniforge3/envs/autotune_ntv3/bin/python
export LD_LIBRARY_PATH=/grid/koo/home/duran/miniforge3/envs/autotune_ntv3/lib:${LD_LIBRARY_PATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false

export NTV3_TISSUE=leaf
export NTV3_MODE=combined
# skip stage 1 — resume head from the saved stage-1 checkpoint, run only stage 2 (now early-stopped)
export NTV3_RESUME_STAGE1=/grid/koo/home/duran/ntv3_ft/best_stage1_leaf_combined.pkl

# writes best_leaf_combined.pkl (stage-2 best)
"$PYBIN" finetune.py

echo "=== end: $(date) ==="
