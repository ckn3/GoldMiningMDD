#!/bin/bash
# Usage:
#   sbatch /deac/csc/alqahtaniGrp/cuij/submit_goldmdd_latency_l40s.slurm

#SBATCH --job-name=goldmdd_latency_l40s
#SBATCH --output=/deac/csc/yangGrp/cuij/GoldMDD/experiments/logs/slurm_goldmdd_latency_l40s_%j.out
#SBATCH --error=/deac/csc/yangGrp/cuij/GoldMDD/experiments/logs/slurm_goldmdd_latency_l40s_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --partition=yangGrp
#SBATCH --account=yanggrp
#SBATCH --gres=gpu:L40S:1

set -euo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

bash /deac/csc/alqahtaniGrp/cuij/run_goldmdd_latency_l40s.sh

echo "Done: $(date)"

