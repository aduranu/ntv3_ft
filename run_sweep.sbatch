#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos=bio_ai
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --chdir=/grid/koo/home/duran/ntv3_ft

set -euo pipefail

echo "=== job $SLURM_JOB_ID $SLURM_JOB_NAME on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=== start: $(date) ==="

PYBIN=/grid/koo/home/duran/miniforge3/envs/autotune_ntv3/bin/python
export LD_LIBRARY_PATH=/grid/koo/home/duran/miniforge3/envs/autotune_ntv3/lib:${LD_LIBRARY_PATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false

export NTV3_MODE=combined
echo "TISSUE=$NTV3_TISSUE TAG=$NTV3_RUN_TAG BASE_LR=$NTV3_S2_BASE_LR BB_SCALE=$NTV3_S2_BB_SCALE BATCH=$NTV3_BATCH_SIZE BUDGET=$NTV3_S2_TIME_BUDGET"

"$PYBIN" finetune.py

echo "=== end: $(date) ==="
