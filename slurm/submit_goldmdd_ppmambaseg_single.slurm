#!/bin/bash
# GoldMDD PPMambaSeg single-GPU launcher
#
# Smoke:
#   sbatch --export=ALL,MODE=smoke /deac/csc/alqahtaniGrp/cuij/submit_goldmdd_ppmambaseg_single.slurm
# Full:
#   sbatch --export=ALL,MODE=full /deac/csc/alqahtaniGrp/cuij/submit_goldmdd_ppmambaseg_single.slurm

#SBATCH --job-name=goldmdd_ppmambaseg_1x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --output=/deac/csc/yangGrp/cuij/GoldMDD/experiments/logs/slurm_%x_%j.out
#SBATCH --error=/deac/csc/yangGrp/cuij/GoldMDD/experiments/logs/slurm_%x_%j.err
#SBATCH --account=paucagrp
#SBATCH --partition=gpu

set -euo pipefail

ROOT=/deac/csc/yangGrp/cuij
REPO=${ROOT}/third_party/PPMambaSeg/GeoSeg
CONDA_SH=/deac/csc/alqahtaniGrp/cuij/miniconda3/etc/profile.d/conda.sh
CONDA_ENV=bench_mfmamba
MODE=${MODE:-full}  # full | smoke
BASELINE=${BASELINE:-b1}  # b1=ce+dice, b2=weighted_ce+dice, b3=focal+dice

if [[ "${MODE}" == "smoke" ]]; then
  case "${BASELINE}" in
    b1) CONFIG=config/goldmdd/ppmamba_smoke.py ;;
    b2) CONFIG=config/goldmdd/ppmamba_weighted_smoke.py ;;
    b3) CONFIG=config/goldmdd/ppmamba_focal_smoke.py ;;
    *) echo "Unknown BASELINE=${BASELINE} (use b1|b2|b3)"; exit 2 ;;
  esac
else
  case "${BASELINE}" in
    b1) CONFIG=config/goldmdd/ppmamba.py ;;
    b2) CONFIG=config/goldmdd/ppmamba_weighted.py ;;
    b3) CONFIG=config/goldmdd/ppmamba_focal.py ;;
    *) echo "Unknown BASELINE=${BASELINE} (use b1|b2|b3)"; exit 2 ;;
  esac
fi

mkdir -p ${ROOT}/GoldMDD/experiments/logs
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${REPO}"

export PYTHONUNBUFFERED=1

echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Mode: ${MODE}"
echo "Baseline: ${BASELINE}"
echo "Config: ${CONFIG}"
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

python train_supervision.py -c "${CONFIG}"

echo "End: $(date)"
