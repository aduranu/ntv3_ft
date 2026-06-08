#!/bin/bash
#SBATCH --job-name=ntv3_ft
#SBATCH --partition=gpuq
#SBATCH --qos=bio_ai
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --chdir=/grid/koo/home/duran/ntv3_ft
#SBATCH --output=/grid/koo/home/duran/ntv3_ft/ntv3_ft_%j.out

set -euo pipefail
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# edit config.json (tissue/mode) to pick the experiment; HPs are hardcoded in finetune.py
PYBIN=/grid/koo/home/duran/miniforge3/envs/autotune_ntv3/bin/python
export LD_LIBRARY_PATH=/grid/koo/home/duran/miniforge3/envs/autotune_ntv3/lib:${LD_LIBRARY_PATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false

"$PYBIN" finetune.py
