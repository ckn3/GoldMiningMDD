#!/bin/bash
# Generic SSA-Seg (GoldMDD) single-GPU launcher.
#
# Examples:
#   sbatch --partition=gpu --gres=gpu:A100_80:1 --export=ALL,MODEL=upernet_swin_tiny submit_goldmdd_ssaseg_single.slurm
#   sbatch --partition=gpu --gres=gpu:A100_80:1 --export=ALL,MODEL=segnext_tiny,MODE=smoke submit_goldmdd_ssaseg_single.slurm

#SBATCH --job-name=goldmdd_ssaseg_1x
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
REPO=${ROOT}/third_party/SSA-Seg
WORK_ROOT=${ROOT}/GoldMDD/experiments
LOG_ROOT=${WORK_ROOT}/logs
CONDA_SH=/deac/csc/alqahtaniGrp/cuij/miniconda3/etc/profile.d/conda.sh
CONDA_ENV=bench_ssaseg

MODEL=${MODEL:-upernet_swin_tiny}
VARIANT=${VARIANT:-ssa}  # ssa | base
MODE=${MODE:-full}  # full | smoke
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_ITERS=${MAX_ITERS:-658000}
VAL_INTERVAL=${VAL_INTERVAL:-8225}
PRETRAIN_CKPT=${PRETRAIN_CKPT:-}

mkdir -p "${LOG_ROOT}" "${WORK_ROOT}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${REPO}"

if [[ "${VARIANT}" == "ssa" ]]; then
  case "${MODEL}" in
    ocrnet_hr48)
      CONFIG=configs/goldmdd/ocrnet_hr48_ssa_goldmdd.py
      RUN_NAME=ocrnet_hr48_ssa_goldmdd
      ;;
    upernet_swin_tiny)
      CONFIG=configs/goldmdd/upernet_swin_tiny_ssa_goldmdd.py
      RUN_NAME=upernet_swin_tiny_ssa_goldmdd
      ;;
    afformer_base)
      CONFIG=configs/goldmdd/afformer_base_ssa_goldmdd.py
      RUN_NAME=afformer_base_ssa_goldmdd
      ;;
    seaformer_base)
      CONFIG=configs/goldmdd/seaformer_base_ssa_goldmdd.py
      RUN_NAME=seaformer_base_ssa_goldmdd
      ;;
    segnext_tiny)
      CONFIG=configs/goldmdd/segnext_tiny_ssa_goldmdd.py
      RUN_NAME=segnext_tiny_ssa_goldmdd
      ;;
    cgrseg_b)
      CONFIG=configs/goldmdd/cgrseg_b_ssa_goldmdd.py
      RUN_NAME=cgrseg_b_ssa_goldmdd
      ;;
    *)
      echo "Unknown MODEL=${MODEL}"
      exit 2
      ;;
  esac
elif [[ "${VARIANT}" == "base" ]]; then
  case "${MODEL}" in
    ocrnet_hr48)
      CONFIG=configs/goldmdd/ocrnet_hr48_goldmdd.py
      RUN_NAME=ocrnet_hr48_goldmdd
      ;;
    upernet_swin_tiny)
      CONFIG=configs/goldmdd/upernet_swin_tiny_goldmdd.py
      RUN_NAME=upernet_swin_tiny_goldmdd
      ;;
    afformer_base)
      CONFIG=configs/goldmdd/afformer_base_goldmdd.py
      RUN_NAME=afformer_base_goldmdd
      ;;
    seaformer_base)
      CONFIG=configs/goldmdd/seaformer_base_goldmdd.py
      RUN_NAME=seaformer_base_goldmdd
      ;;
    segnext_tiny)
      CONFIG=configs/goldmdd/segnext_tiny_goldmdd.py
      RUN_NAME=segnext_tiny_goldmdd
      ;;
    cgrseg_b)
      CONFIG=configs/goldmdd/cgrseg_b_goldmdd.py
      RUN_NAME=cgrseg_b_goldmdd
      ;;
    *)
      echo "Unknown MODEL=${MODEL}"
      exit 2
      ;;
  esac
else
  echo "Unknown VARIANT=${VARIANT}, expected ssa or base"
  exit 2
fi

WORK_DIR=${WORK_ROOT}/${RUN_NAME}
mkdir -p "${WORK_DIR}"

echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Mode: ${MODE}"
echo "Model: ${MODEL}"
echo "Variant: ${VARIANT}"
echo "Config: ${CONFIG}"
echo "Work dir: ${WORK_DIR}"
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

CFG_OPTS=(
  "data.samples_per_gpu=${BATCH_SIZE}"
  data.workers_per_gpu=8
  "runner.max_iters=${MAX_ITERS}"
  "evaluation.interval=${VAL_INTERVAL}"
  checkpoint_config.interval=658001
  log_config.interval=200
  optimizer_config.grad_clip.max_norm=1.0
  optimizer_config.grad_clip.norm_type=2
)

if [[ -n "${PRETRAIN_CKPT}" ]]; then
  case "${MODEL}" in
    ocrnet_hr48|afformer_base)
      CFG_OPTS+=("model.pretrained=${PRETRAIN_CKPT}")
      ;;
    *)
      CFG_OPTS+=(
        model.backbone.init_cfg._delete_=True
        model.backbone.init_cfg.type=Pretrained
        "model.backbone.init_cfg.checkpoint=${PRETRAIN_CKPT}"
      )
      ;;
  esac
fi

EXTRA_ARGS=()
if [[ "${MODE}" == "smoke" ]]; then
  CFG_OPTS=(
    data.samples_per_gpu=2
    data.workers_per_gpu=1
    runner.max_iters=4
    checkpoint_config.interval=999999
    log_config.interval=1
    optimizer_config.grad_clip.max_norm=1.0
    optimizer_config.grad_clip.norm_type=2
  )
  EXTRA_ARGS+=(--no-validate)
fi

python -u train.py "${CONFIG}" "${EXTRA_ARGS[@]}" \
  --work-dir "${WORK_DIR}" \
  --cfg-options "${CFG_OPTS[@]}"

echo "End: $(date)"
