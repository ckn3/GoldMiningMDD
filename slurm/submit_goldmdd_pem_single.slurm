#!/bin/bash
#SBATCH --job-name=goldmdd_pem_1x
#SBATCH --output=/deac/csc/yangGrp/cuij/GoldMDD/experiments/logs/slurm_goldmdd_pem_1x_%j.out
#SBATCH --error=/deac/csc/yangGrp/cuij/GoldMDD/experiments/logs/slurm_goldmdd_pem_1x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --account=paucagrp
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00

set -euo pipefail

ROOT=/deac/csc/yangGrp/cuij
REPO_DIR="$ROOT/third_party/PEM"
WORK_ROOT="$ROOT/GoldMDD/experiments"
LOG_ROOT="$WORK_ROOT/logs"
CONFIG_REL=${CONFIG_REL:-configs/goldmdd/semantic-segmentation/pem_R50_bs8_658k.yaml}
RUN_NAME=${RUN_NAME:-pem_r50_baseline_semseg_default}
MODE=${MODE:-full}
NUM_GPUS=${NUM_GPUS:-1}
NUM_WORKERS=${NUM_WORKERS:-8}
IMS_PER_BATCH=${IMS_PER_BATCH:-8}
MAX_ITER=${MAX_ITER:-658000}
EVAL_PERIOD=${EVAL_PERIOD:-8225}
CHECKPOINT_PERIOD=${CHECKPOINT_PERIOD:-999999999}

mkdir -p "$LOG_ROOT" "$WORK_ROOT/$RUN_NAME"

module load nvidia/cuda12/cuda/12.8.1 >/dev/null 2>&1
source /deac/csc/alqahtaniGrp/cuij/miniconda3/etc/profile.d/conda.sh
conda activate bench_pem

cd "$REPO_DIR"

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Mode: $MODE"
echo "Config: $CONFIG_REL"
echo "Run dir: $WORK_ROOT/$RUN_NAME"
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

OPTS=(
  OUTPUT_DIR "$WORK_ROOT/$RUN_NAME"
  DATALOADER.NUM_WORKERS "$NUM_WORKERS"
  SOLVER.IMS_PER_BATCH "$IMS_PER_BATCH"
  SOLVER.MAX_ITER "$MAX_ITER"
  TEST.EVAL_PERIOD "$EVAL_PERIOD"
  SOLVER.CHECKPOINT_PERIOD "$CHECKPOINT_PERIOD"
)

if [[ "$MODE" == "smoke" ]]; then
  OPTS=(
    OUTPUT_DIR "$WORK_ROOT/$RUN_NAME"
    DATALOADER.NUM_WORKERS 0
    SOLVER.IMS_PER_BATCH 2
    SOLVER.MAX_ITER 2
    TEST.EVAL_PERIOD 0
    SOLVER.CHECKPOINT_PERIOD 999999999
  )
fi

python -u train_net.py \
  --num-gpus "$NUM_GPUS" \
  --config-file "$CONFIG_REL" \
  "${OPTS[@]}"
