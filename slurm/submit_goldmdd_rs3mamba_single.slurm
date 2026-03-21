#!/bin/bash
# RS3Mamba single-GPU runner for GoldMDD.
#
# Smoke test:
#   sbatch --export=ALL,MODE=smoke /deac/csc/alqahtaniGrp/cuij/submit_goldmdd_rs3mamba_single.slurm
# Full baseline (b1/b2/b3):
#   sbatch --export=ALL,MODE=full,BASELINE=b1 /deac/csc/alqahtaniGrp/cuij/submit_goldmdd_rs3mamba_single.slurm

#SBATCH --job-name=goldmdd_rs3mamba_1x
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

ROOT="/deac/csc/yangGrp/cuij"
DATA_ROOT="${ROOT}/GoldMDD/data-cropped"
WORK_DIR="${ROOT}/GoldMDD/experiments"
SCRIPT="${ROOT}/GoldMDD/misc/train_semseg_rs3mamba.py"
RS3MAMBA_REPO="${RS3MAMBA_REPO:-${ROOT}/third_party/SSRS/RS3Mamba}"
VMAMBA_PRETRAINED="${VMAMBA_PRETRAINED:-${RS3MAMBA_REPO}/pretrain/vmamba_tiny_e292.pth}"
CONDA_SH="/deac/csc/alqahtaniGrp/cuij/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="bench_ssrs"
LOG_DIR="${WORK_DIR}/logs"

MODE="${MODE:-full}"   # smoke | full
BASELINE="${BASELINE:-b1}"

BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-80}"
LR="${LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-2}"
TRAIN_LOG_INTERVAL="${TRAIN_LOG_INTERVAL:-200}"
VAL_LOG_INTERVAL="${VAL_LOG_INTERVAL:-100}"

mkdir -p "${LOG_DIR}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

if [[ "${MODE}" == "smoke" ]]; then
  RUN_NAME="${RUN_NAME:-rs3mamba_smoke_augv2_ce_dice}"
  LOSS_ARGS=(--loss-mode ce_dice)
  EXTRA_ARGS=(--limit-train 64 --limit-val 32 --limit-test 32 --epochs 2 --batch-size 2 --num-workers 4 --train-log-interval 20 --val-log-interval 20)
else
  case "${BASELINE}" in
    b1)
      RUN_NAME="${RUN_NAME:-rs3mamba_baseline1_augv2_ce_dice}"
      LOSS_ARGS=(--loss-mode ce_dice)
      ;;
    b2)
      RUN_NAME="${RUN_NAME:-rs3mamba_baseline2_augv2_weighted_ce_dice}"
      LOSS_ARGS=(--loss-mode weighted_ce_dice --class-weight-power 0.5)
      ;;
    b3)
      RUN_NAME="${RUN_NAME:-rs3mamba_baseline3_augv2_focal_dice}"
      LOSS_ARGS=(--loss-mode focal_dice --focal-gamma 2.0)
      ;;
    *)
      echo "ERROR: BASELINE must be b1/b2/b3" >&2
      exit 1
      ;;
  esac
  EXTRA_ARGS=(--epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" --train-log-interval "${TRAIN_LOG_INTERVAL}" --val-log-interval "${VAL_LOG_INTERVAL}")
fi

RUN_LOG="${LOG_DIR}/${RUN_NAME}.log"

echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Mode: ${MODE}, baseline: ${BASELINE}"
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -u "${SCRIPT}" \
  --data-root "${DATA_ROOT}" \
  --work-dir "${WORK_DIR}" \
  --rs3mamba-repo "${RS3MAMBA_REPO}" \
  --vmamba-pretrained "${VMAMBA_PRETRAINED}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --ce-weight 1.0 \
  --dice-weight 1.0 \
  --aug-preset goldmdd_v2 \
  --val-every 1 \
  --save-every 0 \
  --run-name "${RUN_NAME}" \
  "${EXTRA_ARGS[@]}" \
  "${LOSS_ARGS[@]}" \
  2>&1 | tee -a "${RUN_LOG}"

echo "End: $(date)"
