#!/bin/bash
# Spike2Former (MMseg) single-GPU runner for GoldMDD.
#
# Examples:
#   sbatch --export=ALL,MODE=smoke submit_goldmdd_spike2former_single.slurm
#   sbatch --export=ALL,MODE=full submit_goldmdd_spike2former_single.slurm
#   sbatch --export=ALL,MODE=full,PRETRAIN_CKPT=/path/to/ckpt.pth submit_goldmdd_spike2former_single.slurm

#SBATCH --job-name=goldmdd_spike2f_1x
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

ROOT="/deac/csc/yangGrp/cuij"
REPO_DIR="${ROOT}/third_party/Spike2Former/Segmentation"
WORK_ROOT="${ROOT}/GoldMDD/experiments"
LOG_ROOT="${WORK_ROOT}/logs"
CONDA_SH="/deac/csc/alqahtaniGrp/cuij/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="bench_spike2former"

CONFIG_REL="${CONFIG_REL:-configs/goldmdd/spike2former_sdtv3_b_goldmdd_512x512.py}"
RUN_NAME="${RUN_NAME:-spike2former_sdtv3_b_goldmdd}"
MODE="${MODE:-full}"  # full | smoke
PRETRAIN_CKPT="${PRETRAIN_CKPT:-}"
SPIKE2F_ENABLE_COMPILE="${SPIKE2F_ENABLE_COMPILE:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"

mkdir -p "${LOG_ROOT}" "${WORK_ROOT}/${RUN_NAME}"

module load nvidia/cuda12/cuda/12.8.1 >/dev/null 2>&1 || true
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

cd "${REPO_DIR}"

echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Mode: ${MODE}"
echo "Config: ${CONFIG_REL}"
echo "Run: ${RUN_NAME}"
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SPIKE2F_ENABLE_COMPILE
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

CFG_OPTS=()
if [[ -n "${PRETRAIN_CKPT}" ]]; then
  CFG_OPTS+=(
    model.backbone.init_cfg._delete_=True
    model.backbone.init_cfg.type=Pretrained
    "model.backbone.init_cfg.checkpoint=${PRETRAIN_CKPT}"
  )
fi

if [[ "${MODE}" == "smoke" ]]; then
  CFG_OPTS+=(
    train_cfg.max_iters=20
    train_cfg.val_interval=10
    train_dataloader.batch_size=1
    default_hooks.logger.interval=1
    train_dataloader.num_workers=1
    val_dataloader.num_workers=1
    test_dataloader.num_workers=1
    train_dataloader.dataset.indices=8
    val_dataloader.dataset.indices=8
    test_dataloader.dataset.indices=8
  )
else
  CFG_OPTS+=(
    "train_dataloader.batch_size=${BATCH_SIZE}"
  )
fi

python -u tools/train.py "${CONFIG_REL}" \
  --work-dir "${WORK_ROOT}/${RUN_NAME}" \
  --cfg-options "${CFG_OPTS[@]}"

echo "End: $(date)"
