#!/bin/bash
# GoldMDD rssegmentation single-GPU launcher
#
# Example:
# sbatch --partition=gpu --account=paucagrp --gres=gpu:A100_80:1 \
#   --cpus-per-task=12 --mem=64G --time=3-00:00:00 \
#   --export=ALL,MODE=full \
#   /deac/csc/alqahtaniGrp/cuij/submit_goldmdd_rsseg_single.slurm

#SBATCH --job-name=goldmdd_rsseg_1x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/deac/csc/yangGrp/cuij/GoldMDD/experiments/logs/slurm_%x_%j.out
#SBATCH --error=/deac/csc/yangGrp/cuij/GoldMDD/experiments/logs/slurm_%x_%j.err
#SBATCH --account=paucagrp
#SBATCH --partition=gpu

set -euo pipefail

ROOT=/deac/csc/yangGrp/cuij
REPO=${ROOT}/third_party/rssegmentation
CONDA_SH=/deac/csc/alqahtaniGrp/cuij/miniconda3/etc/profile.d/conda.sh
CONDA_ENV=bench_rsseg

MODE=${MODE:-full}      # full | smoke
METHOD=${METHOD:-logcan}  # logcan | docnet | sacanet | logcanplus

case "${METHOD}" in
  logcan)
    if [[ "${MODE}" == "smoke" ]]; then
      CONFIG=configs/goldmdd/logcan_r50_goldmdd_smoke.py
    else
      CONFIG=configs/goldmdd/logcan_r50_goldmdd.py
    fi
    ;;
  docnet)
    if [[ "${MODE}" == "smoke" ]]; then
      CONFIG=configs/goldmdd/docnet_hrnetw32_goldmdd_smoke.py
    else
      CONFIG=configs/goldmdd/docnet_hrnetw32_goldmdd.py
    fi
    ;;
  sacanet)
    if [[ "${MODE}" == "smoke" ]]; then
      CONFIG=configs/goldmdd/sacanet_hrnetw32_goldmdd_smoke.py
    else
      CONFIG=configs/goldmdd/sacanet_hrnetw32_goldmdd.py
    fi
    ;;
  logcanplus)
    if [[ "${MODE}" == "smoke" ]]; then
      CONFIG=configs/goldmdd/logcanplus_repvitm23_goldmdd_smoke.py
    else
      CONFIG=configs/goldmdd/logcanplus_repvitm23_goldmdd.py
    fi
    ;;
  *)
    echo "Unknown METHOD=${METHOD}"
    exit 2
    ;;
esac

mkdir -p /deac/csc/yangGrp/cuij/GoldMDD/experiments/logs

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${REPO}"

echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Method: ${METHOD}"
echo "Mode: ${MODE}"
echo "Config: ${CONFIG}"
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

python train.py -c "${CONFIG}"

echo "End: $(date)"
