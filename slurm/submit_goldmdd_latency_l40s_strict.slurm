#!/bin/bash
# Strict L40S latency sweep: single-GPU + node exclusive + idle guard.

#SBATCH --job-name=goldmdd_lat_l40s_strict
#SBATCH --output=/deac/csc/yangGrp/cuij/GoldMDD/experiments/logs/slurm_goldmdd_lat_l40s_strict_%j.out
#SBATCH --error=/deac/csc/yangGrp/cuij/GoldMDD/experiments/logs/slurm_goldmdd_lat_l40s_strict_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --partition=yangGrp
#SBATCH --account=yanggrp
#SBATCH --gres=gpu:L40S:1

set -euo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "CUDA_VISIBLE_DEVICES(before): ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

# Keep polling tight; if Slurm exposes multiple GPUs, let runner pick any truly idle one.
if [[ "${CUDA_VISIBLE_DEVICES:-}" == *","* ]]; then
  unset LATENCY_GPU_INDEX
else
  export LATENCY_GPU_INDEX="${CUDA_VISIBLE_DEVICES%%,*}"
fi
export LATENCY_IDLE_POLL_SEC=15
export LATENCY_IDLE_TIMEOUT_SEC=0

bash /deac/csc/alqahtaniGrp/cuij/run_goldmdd_latency_l40s.sh

echo "Done: $(date)"
