#!/bin/bash
# Stage-2 hyperparameter sweep for NTv3 fine-tuning.
# Resumes each trial from the saved stage-1 head, applies the ported wall-time
# LR schedule, and sweeps peak LR x backbone discriminative scale on both tissues.
set -euo pipefail
cd /grid/koo/home/duran/ntv3_ft
mkdir -p sweep_logs

BASES=(1e-4 2e-4 4e-4)
SCALES=(0.1 0.3 1.0)
BATCH=256
BUDGET=2400

: > sweep_jobs.txt
for tissue in leaf proto; do
  resume=/grid/koo/home/duran/ntv3_ft/best_stage1_${tissue}_combined.pkl
  for base in "${BASES[@]}"; do
    for scale in "${SCALES[@]}"; do
      tag="lr${base}_bb${scale}"
      name="s2_${tissue}_${tag}"
      jid=$(sbatch --parsable \
        --job-name="$name" \
        --output="/grid/koo/home/duran/ntv3_ft/sweep_logs/${name}_%j.out" \
        --export=ALL,NTV3_TISSUE=${tissue},NTV3_RESUME_STAGE1=${resume},NTV3_S2_BASE_LR=${base},NTV3_S2_BB_SCALE=${scale},NTV3_BATCH_SIZE=${BATCH},NTV3_S2_TIME_BUDGET=${BUDGET},NTV3_RUN_TAG=${tag} \
        run_sweep.sbatch)
      echo "$jid $tissue $tag" | tee -a sweep_jobs.txt
    done
  done
done
echo "submitted $(wc -l < sweep_jobs.txt) trials"
