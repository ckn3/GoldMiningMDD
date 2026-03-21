#!/bin/bash
# SAM_RS single-GPU runner for GoldMDD.
#
# Smoke:
#   sbatch --export=ALL,MODE=smoke /deac/csc/alqahtaniGrp/cuij/submit_goldmdd_samrs_single.slurm
# Full:
#   sbatch --export=ALL,MODE=full /deac/csc/alqahtaniGrp/cuij/submit_goldmdd_samrs_single.slurm

#SBATCH --job-name=goldmdd_samrs_1x
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
DATA_ROOT="${ROOT}/GoldMDD/data-cropped"
PRIORS_ROOT="${ROOT}/GoldMDD/data-cropped-samrs-priors"
WORK_DIR="${ROOT}/GoldMDD/experiments"
SCRIPT="${ROOT}/GoldMDD/misc/train_semseg_sam_rs.py"
SSRS_REPO="${SSRS_REPO:-${ROOT}/third_party/SSRS/SAM_RS}"
SEGANY_PATH="${SEGANY_PATH:-${ROOT}/GeCo/sam-hq}"
SAM_CKPT="${SAM_CKPT:-${ROOT}/GeCo/MODEL_folder/sam_vit_h_4b8939.pth}"
CONDA_SH="/deac/csc/alqahtaniGrp/cuij/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="bench_ssrs"
LOG_DIR="${WORK_DIR}/logs"

MODE="${MODE:-full}"   # smoke | full
MODEL="${MODEL:-unetformer}"  # unetformer | ftunetformer | abcnet | cmtfnet

BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-80}"
LR="${LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-2}"
TRAIN_LOG_INTERVAL="${TRAIN_LOG_INTERVAL:-200}"
VAL_LOG_INTERVAL="${VAL_LOG_INTERVAL:-100}"
USE_AMP="${USE_AMP:-1}"   # 1=--amp(default), 0=--no-amp

mkdir -p "${LOG_DIR}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

if [[ "${MODE}" == "smoke" ]]; then
  RUN_NAME="${RUN_NAME:-sam_rs_${MODEL}_smoke_augv2_seg_bdy_obj}"
  EXTRA_ARGS=(
    --epochs 1
    --batch-size 2
    --num-workers 4
    --limit-train 64
    --limit-val 32
    --limit-test 32
    --prior-max-images 0
    --train-log-interval 20
    --val-log-interval 20
  )
else
  RUN_NAME="${RUN_NAME:-sam_rs_${MODEL}_baseline1_augv2_seg_bdy_obj}"
  EXTRA_ARGS=(
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --train-log-interval "${TRAIN_LOG_INTERVAL}"
    --val-log-interval "${VAL_LOG_INTERVAL}"
  )
fi

AMP_ARGS=()
if [[ "${USE_AMP}" == "0" ]]; then
  AMP_ARGS+=(--no-amp)
fi

RUN_LOG="${LOG_DIR}/${RUN_NAME}.log"

echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Mode: ${MODE}"
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -u "${SCRIPT}" \
  --data-root "${DATA_ROOT}" \
  --priors-root "${PRIORS_ROOT}" \
  --work-dir "${WORK_DIR}" \
  --run-name "${RUN_NAME}" \
  --model "${MODEL}" \
  --ssrs-repo "${SSRS_REPO}" \
  --segment-anything-path "${SEGANY_PATH}" \
  --sam-checkpoint "${SAM_CKPT}" \
  --sam-model-type vit_h \
  --sam-device cuda \
  --sam-min-area 50 \
  --sam-max-objects 50 \
  --sam-pred-iou-thresh 0.96 \
  --sam-box-nms-thresh 0.5 \
  --sam-crop-nms-thresh 0.5 \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --aug-preset goldmdd_v2 \
  --val-every 1 \
  --save-every 0 \
  --loss-mode seg_bdy_obj \
  --lambda-boundary 0.1 \
  --lambda-object 1.0 \
  "${AMP_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "${RUN_LOG}"

echo "End: $(date)"
