#!/bin/bash
# GoldMDD FarSeg/FarSeg++ single-GPU launcher.
#
# Examples:
# sbatch --partition=gpu --account=paucagrp --gres=gpu:A100_80:1 \
#   --cpus-per-task=12 --mem=64G --time=3-00:00:00 \
#   --export=ALL,MODEL=farseg,MODE=full \
#   /deac/csc/alqahtaniGrp/cuij/submit_goldmdd_farseg_single.slurm
#
# sbatch --partition=gpu --account=paucagrp --gres=gpu:V100_32:1 \
#   --cpus-per-task=8 --mem=32G --time=01:00:00 \
#   --export=ALL,MODEL=farsegpp,MODE=smoke \
#   /deac/csc/alqahtaniGrp/cuij/submit_goldmdd_farseg_single.slurm

#SBATCH --job-name=goldmdd_farseg_1x
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
GOLDMDD=${ROOT}/GoldMDD
SCRIPT=${GOLDMDD}/misc/train_semseg_farseg.py
CONDA_SH=/deac/csc/alqahtaniGrp/cuij/miniconda3/etc/profile.d/conda.sh
CONDA_ENV=bench_farseg

MODEL=${MODEL:-farseg}       # farseg | farsegpp
MODE=${MODE:-full}           # full | smoke
BASELINE=${BASELINE:-b1}     # b1=ce+dice, b2=weighted ce+dice, b3=focal+dice
FARSEGPP_BACKBONE=${FARSEGPP_BACKBONE:-mit_b2}  # mit_b2 | resnet50
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-8}
EPOCHS=${EPOCHS:-80}
PRETRAINED=${PRETRAINED:-1}  # 1 true, 0 false
RUN_NAME=${RUN_NAME:-}

mkdir -p "${GOLDMDD}/experiments/logs"
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

cd "${GOLDMDD}"

echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Model: ${MODEL}"
echo "Baseline: ${BASELINE}"
echo "Mode: ${MODE}"
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

LOSS_ARGS=()
if [[ "${MODE}" != "smoke" ]]; then
  case "${BASELINE}" in
    b1)
      LOSS_ARGS=(--loss-mode ce_dice --ce-weight 1.0 --dice-weight 1.0)
      if [[ -z "${RUN_NAME}" ]]; then
        RUN_NAME="${MODEL}_baseline1_augv2_ce_dice"
      fi
      ;;
    b2)
      LOSS_ARGS=(--loss-mode weighted_ce_dice --ce-weight 1.0 --dice-weight 1.0 --class-weight-power 0.5)
      if [[ -z "${RUN_NAME}" ]]; then
        RUN_NAME="${MODEL}_baseline2_augv2_weighted_ce_dice"
      fi
      ;;
    b3)
      LOSS_ARGS=(--loss-mode focal_dice --ce-weight 1.0 --dice-weight 1.0 --focal-gamma 2.0)
      if [[ -z "${RUN_NAME}" ]]; then
        RUN_NAME="${MODEL}_baseline3_augv2_focal_dice"
      fi
      ;;
    *)
      echo "Unknown BASELINE=${BASELINE}, expected b1|b2|b3" >&2
      exit 2
      ;;
  esac
fi

ARGS=(
  --model "${MODEL}"
  --data-root "${GOLDMDD}/data-cropped"
  --work-dir "${GOLDMDD}/experiments"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --epochs "${EPOCHS}"
  --aug-preset goldmdd_v2
  --train-log-interval 200
  --val-log-interval 200
)
ARGS+=("${LOSS_ARGS[@]}")

if [[ "${MODEL}" == "farsegpp" ]]; then
  ARGS+=(--farsegpp-backbone "${FARSEGPP_BACKBONE}")
fi

if [[ -n "${RUN_NAME}" ]]; then
  ARGS+=(--run-name "${RUN_NAME}")
fi

if [[ "${PRETRAINED}" == "0" ]]; then
  ARGS+=(--no-pretrained)
fi

if [[ "${MODE}" == "smoke" ]]; then
  ARGS+=(
    --epochs 1
    --batch-size 2
    --num-workers 0
    --limit-train 8
    --limit-val 4
    --limit-test 4
    --no-amp
  )
  if [[ -z "${RUN_NAME}" ]]; then
    ARGS+=(--run-name "${MODEL}_smoke")
  fi
fi

python -u "${SCRIPT}" "${ARGS[@]}"

echo "End: $(date)"
