#!/bin/bash
set -euo pipefail

ROOT=/deac/csc/yangGrp/cuij/GoldMDD
SCRIPT="$ROOT/misc/benchmark_semseg_model.py"
OUTDIR="$ROOT/experiments/diagnostics/model_stats"
LOGDIR="$ROOT/experiments/logs"
STAMP=$(date +%Y%m%d_%H%M%S)
FAIL_FILE="$LOGDIR/latency_h200_retry_failures_${STAMP}.txt"

mkdir -p "$OUTDIR" "$LOGDIR"
: > "$FAIL_FILE"

source /deac/csc/alqahtaniGrp/cuij/miniconda3/etc/profile.d/conda.sh

run_bench() {
  local env_name="$1"; shift
  local out_file="$1"; shift
  local family="$1"; shift

  echo "\n===== RETRY [$family] env=$env_name -> $out_file ====="
  conda activate "$env_name"
  if python -u "$SCRIPT" --family "$family" --device cuda:0 --output "$OUTDIR/$out_file" "$@"; then
    echo "[OK] $out_file"
  else
    echo "[FAIL] $out_file"
    echo "$out_file|$env_name|$family" >> "$FAIL_FILE"
  fi
}

# 1) Afformer: retry latency with FLOPs skipped (previous run hit floating-point exception during full benchmark)
run_bench bench_ssaseg ssaseg_afformer_base.json ssaseg_afformer_base --skip-flops

# 2) PyramidMamba: run under env where mamba_ssm CUDA extension resolves correctly
run_bench bench_mfmamba geoseg_pyramidmamba.json geoseg_pyramidmamba

# 3) SAM2.1 benchmark (lazy-parameter safe after script fix)
run_bench bench_sam_family sam2_1_hierabplus_msfpn.json sam2_1_hierabplus

# 4) RSAM-Seg benchmark (fixed repo path)
run_bench bench_sam_family rsamseg_vit_b.json rsamseg_vit_b

# Sync summary with updated latency/efficiency stats.
conda activate bench_segformer
python -u "$ROOT/misc/sync_experiment_tables.py"

FAIL_COUNT=$(grep -c . "$FAIL_FILE" || true)
echo "\nRetry suite finished. failures=$FAIL_COUNT"
if [[ "$FAIL_COUNT" -eq 0 ]]; then
  rm -f "$FAIL_FILE"
  echo "All retry benchmarks succeeded."
else
  echo "Failure list: $FAIL_FILE"
fi
