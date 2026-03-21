#!/bin/bash
#SBATCH --job-name=goldmdd_sessrs_1x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=10-00:00:00
#SBATCH --output=/deac/csc/yangGrp/cuij/GoldMDD/experiments/logs/slurm_%x_%j.out
#SBATCH --error=/deac/csc/yangGrp/cuij/GoldMDD/experiments/logs/slurm_%x_%j.err
#SBATCH --account=paucagrp
#SBATCH --partition=gpu

set -euo pipefail

ROOT="/deac/csc/yangGrp/cuij"
REPO="${ROOT}/third_party/SESSRS/GeoSeg"
CONDA_SH="/deac/csc/alqahtaniGrp/cuij/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="bench_sessrs"
LOG_DIR="${ROOT}/GoldMDD/experiments/logs"

MODE="${MODE:-full}"          # full | smoke
MODEL="${MODEL:-unetformer}"  # unetformer|abcnet|dcswin|ftunetformer|manet|segformer|fanet|shelfnet|swinunet
STRATEGY="${STRATEGY:-auto}"

mkdir -p "${LOG_DIR}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

if [[ "${MODE}" == "smoke" ]]; then
  CFG="${REPO}/config/goldmdd/${MODEL}_smoke.py"
  RUN_NAME="${RUN_NAME:-sessrs_${MODEL}_goldmdd_smoke}"
else
  CFG="${REPO}/config/goldmdd/${MODEL}.py"
  RUN_NAME="${RUN_NAME:-sessrs_${MODEL}_goldmdd_b8_e80}"
fi

if [[ ! -f "${CFG}" ]]; then
  echo "Config not found: ${CFG}" >&2
  exit 2
fi

echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Mode: ${MODE}"
echo "Model: ${MODEL}"
echo "Config: ${CFG}"
echo "Python: $(which python)"
nvidia-smi || true

export PYTHONUNBUFFERED=1
cd "${REPO}"
python -u train_supervision.py -c "${CFG}" -s "${STRATEGY}"

echo "End: $(date)"
