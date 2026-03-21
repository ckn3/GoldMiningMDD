#!/bin/bash
# Submit one EfficientViT-Seg baseline per job (1 GPU / task).
#
# Examples:
#   sbatch --export=ALL,BASELINE=b1 submit_goldmdd_efficientvit_single.slurm
#   sbatch --export=ALL,BASELINE=b2 submit_goldmdd_efficientvit_single.slurm
#   sbatch --export=ALL,BASELINE=b3 submit_goldmdd_efficientvit_single.slurm

#SBATCH --job-name=goldmdd_efficientvit_1x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/deac/csc/yangGrp/cuij/GoldMDD/experiments/logs/slurm_%x_%j.out
#SBATCH --error=/deac/csc/yangGrp/cuij/GoldMDD/experiments/logs/slurm_%x_%j.err
#SBATCH --account=paucagrp
#SBATCH --partition=gpu

set -euo pipefail

ROOT="/deac/csc/yangGrp/cuij"
DATA_ROOT="${ROOT}/GoldMDD/data-cropped"
WORK_DIR="${ROOT}/GoldMDD/experiments"
SCRIPT="${ROOT}/GoldMDD/misc/train_semseg_efficientvit.py"
EFFICIENTVIT_REPO="${ROOT}/third_party/efficientvit"
CONDA_SH="/deac/csc/alqahtaniGrp/cuij/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="bench_efficientvit"
LOG_DIR="${WORK_DIR}/logs"

BASELINE="${BASELINE:-}"
if [[ -z "${BASELINE}" ]]; then
  echo "ERROR: BASELINE is required (b1, b2, or b3)." >&2
  exit 1
fi

MODEL_NAME="${MODEL_NAME:-efficientvit-seg-b2-ade20k}"
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

case "${BASELINE}" in
  b1)
    RUN_NAME="${RUN_NAME:-efficientvit_b2_baseline1_augv2_ce_dice}"
    LOSS_ARGS=(--loss-mode ce_dice)
    ;;
  b2)
    RUN_NAME="${RUN_NAME:-efficientvit_b2_baseline2_augv2_weighted_ce_dice}"
    LOSS_ARGS=(--loss-mode weighted_ce_dice --class-weight-power 0.5)
    ;;
  b3)
    RUN_NAME="${RUN_NAME:-efficientvit_b2_baseline3_augv2_focal_dice}"
    LOSS_ARGS=(--loss-mode focal_dice --focal-gamma 2.0)
    ;;
  *)
    echo "ERROR: Unsupported BASELINE='${BASELINE}'. Use b1, b2, or b3." >&2
    exit 1
    ;;
esac

RUN_LOG="${LOG_DIR}/${RUN_NAME}.log"

echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Baseline: ${BASELINE} -> ${RUN_NAME}"
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -u "${SCRIPT}" \
  --data-root "${DATA_ROOT}" \
  --work-dir "${WORK_DIR}" \
  --efficientvit-repo "${EFFICIENTVIT_REPO}" \
  --model-name "${MODEL_NAME}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --ce-weight 1.0 \
  --dice-weight 1.0 \
  --aug-preset goldmdd_v2 \
  --val-every 1 \
  --save-every 0 \
  --train-log-interval "${TRAIN_LOG_INTERVAL}" \
  --val-log-interval "${VAL_LOG_INTERVAL}" \
  --run-name "${RUN_NAME}" \
  "${LOSS_ARGS[@]}" \
  2>&1 | tee -a "${RUN_LOG}"

echo "End: $(date)"
