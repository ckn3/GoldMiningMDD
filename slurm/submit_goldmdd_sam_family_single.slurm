#!/bin/bash
# SAM-family (SAM2.1 / SAM3 / HQ-SAM) single-GPU runner for GoldMDD.
#
# Smoke:
#   sbatch --export=ALL,MODE=smoke,FAMILY=sam2_1 /deac/csc/alqahtaniGrp/cuij/submit_goldmdd_sam_family_single.slurm
#   sbatch --export=ALL,MODE=smoke,FAMILY=sam3 /deac/csc/alqahtaniGrp/cuij/submit_goldmdd_sam_family_single.slurm
#   sbatch --export=ALL,MODE=smoke,FAMILY=hq_sam /deac/csc/alqahtaniGrp/cuij/submit_goldmdd_sam_family_single.slurm
#
# Full:
#   sbatch --export=ALL,MODE=full,FAMILY=sam2_1 /deac/csc/alqahtaniGrp/cuij/submit_goldmdd_sam_family_single.slurm

#SBATCH --job-name=goldmdd_samfam_1x
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
WORK_DIR="${ROOT}/GoldMDD/experiments"
SCRIPT="${ROOT}/GoldMDD/misc/train_semseg_sam_family.py"
CONDA_SH="/deac/csc/alqahtaniGrp/cuij/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="${CONDA_ENV:-bench_sam_family}"
LOG_DIR="${WORK_DIR}/logs"

MODE="${MODE:-full}"         # smoke | full
FAMILY="${FAMILY:-sam2_1}"   # sam2_1 | sam3 | hq_sam
LOSS_MODE="${LOSS_MODE:-ce_dice}"
FREEZE_BACKBONE="${FREEZE_BACKBONE:-0}"  # 1=frozen backbone, 0=full fine-tune
DECODER_TYPE="${DECODER_TYPE:-fpn_multiscale}"  # fpn_multiscale | tiny_single

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

COMMON_ARGS=(
  --data-root "${DATA_ROOT}"
  --work-dir "${WORK_DIR}"
  --family "${FAMILY}"
  --loss-mode "${LOSS_MODE}"
  --decoder-type "${DECODER_TYPE}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --aug-preset goldmdd_v2
  --val-every 1
  --save-every 0
)

if [[ "${FREEZE_BACKBONE}" == "1" ]]; then
  COMMON_ARGS+=(--freeze-backbone)
fi

if [[ "${MODE}" == "smoke" ]]; then
  RUN_NAME="${RUN_NAME:-${FAMILY}_smoke_augv2_${LOSS_MODE}}"
  EXTRA_ARGS=(
    --epochs 1
    --batch-size 2
    --num-workers 4
    --limit-train 64
    --limit-val 32
    --limit-test 32
    --train-log-interval 20
    --val-log-interval 20
  )
else
  RUN_NAME="${RUN_NAME:-${FAMILY}_baseline_augv2_${LOSS_MODE}}"
  EXTRA_ARGS=(
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --train-log-interval "${TRAIN_LOG_INTERVAL}"
    --val-log-interval "${VAL_LOG_INTERVAL}"
  )
fi

FAMILY_ARGS=()
if [[ "${FAMILY}" == "sam2_1" ]]; then
  SAM2_ROOT="${SAM2_ROOT:-${ROOT}/third_party/sam2}"
  SAM2_CFG="${SAM2_CFG:-configs/sam2.1/sam2.1_hiera_b+.yaml}"
  SAM2_CKPT="${SAM2_CKPT:-}"
  FAMILY_ARGS+=(--sam2-root "${SAM2_ROOT}" --sam2-config "${SAM2_CFG}" --sam2-image-size 512)
  if [[ -n "${SAM2_CKPT}" ]]; then
    FAMILY_ARGS+=(--sam2-checkpoint "${SAM2_CKPT}")
  fi
elif [[ "${FAMILY}" == "sam3" ]]; then
  SAM3_ROOT="${SAM3_ROOT:-${ROOT}/third_party/sam3}"
  SAM3_CKPT="${SAM3_CKPT:-}"
  FAMILY_ARGS+=(--sam3-root "${SAM3_ROOT}" --sam3-image-size 1008)
  if [[ -n "${SAM3_CKPT}" ]]; then
    FAMILY_ARGS+=(--sam3-checkpoint "${SAM3_CKPT}")
  else
    FAMILY_ARGS+=(--sam3-load-from-hf)
  fi
elif [[ "${FAMILY}" == "hq_sam" ]]; then
  HQSAM_ROOT="${HQSAM_ROOT:-${ROOT}/third_party/sam-hq}"
  HQSAM_MODEL_TYPE="${HQSAM_MODEL_TYPE:-vit_b}"
  HQSAM_CKPT="${HQSAM_CKPT:-}"
  FAMILY_ARGS+=(--hqsam-root "${HQSAM_ROOT}" --hqsam-model-type "${HQSAM_MODEL_TYPE}")
  if [[ -n "${HQSAM_CKPT}" ]]; then
    FAMILY_ARGS+=(--hqsam-checkpoint "${HQSAM_CKPT}")
  fi
else
  echo "Unsupported FAMILY=${FAMILY}" >&2
  exit 2
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
echo "Family: ${FAMILY}"
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -u "${SCRIPT}" \
  --run-name "${RUN_NAME}" \
  "${COMMON_ARGS[@]}" \
  "${FAMILY_ARGS[@]}" \
  "${AMP_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "${RUN_LOG}"

echo "End: $(date)"
