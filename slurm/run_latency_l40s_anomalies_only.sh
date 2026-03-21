#!/bin/bash
set -euo pipefail

ROOT=/deac/csc/yangGrp/cuij/GoldMDD
SCRIPT="$ROOT/misc/benchmark_semseg_model.py"
OUTDIR="$ROOT/experiments/diagnostics/model_stats"

mkdir -p "$OUTDIR"
source /deac/csc/alqahtaniGrp/cuij/miniconda3/etc/profile.d/conda.sh

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

# 1) CGRSeg anomaly entry
conda activate bench_ssaseg
python -u "$SCRIPT" --family ssaseg_cgrseg_b --device cuda:0 --output "$OUTDIR/cgrseg_b.json"

# 2) PyramidMamba anomaly entry (L40S rerun)
conda activate bench_mfmamba
python -u "$SCRIPT" --family geoseg_pyramidmamba --device cuda:0 --output "$OUTDIR/geoseg_pyramidmamba.json"

# 3) HQ-SAM model stats for summary table filling
conda activate bench_sam_family
python -u "$SCRIPT" --family hq_sam_vit_b --device cuda:0 --output "$OUTDIR/hq_sam_vit_b_msfpn.json"

# Sync tables
conda activate bench_segformer
python -u "$ROOT/misc/sync_experiment_tables.py"

echo "Latency anomalies suite done: $(date)"
