#!/bin/bash
set -euo pipefail

ROOT=/deac/csc/yangGrp/cuij/GoldMDD
SCRIPT="$ROOT/misc/benchmark_semseg_model.py"
OUTDIR="$ROOT/experiments/diagnostics/model_stats"
LOGDIR="$ROOT/experiments/logs"
STAMP=$(date +%Y%m%d_%H%M%S)
FAIL_FILE="$LOGDIR/latency_h200_failures_${STAMP}.txt"

mkdir -p "$OUTDIR" "$LOGDIR"
: > "$FAIL_FILE"

source /deac/csc/alqahtaniGrp/cuij/miniconda3/etc/profile.d/conda.sh

run_bench() {
  local env_name="$1"; shift
  local out_file="$1"; shift
  local family="$1"; shift

  echo "\n===== [$family] env=$env_name -> $out_file ====="
  conda activate "$env_name"
  if python -u "$SCRIPT" --family "$family" --device cuda:0 --output "$OUTDIR/$out_file" "$@"; then
    echo "[OK] $out_file"
  else
    echo "[FAIL] $out_file"
    echo "$out_file|$env_name|$family" >> "$FAIL_FILE"
  fi
}

# SMP baselines
run_bench bench_base smp_deeplabv3p_convnext_tiny.json smp --smp-arch deeplabv3plus --smp-encoder convnext_tiny --smp-encoder-weights imagenet
run_bench bench_base smp_deeplabv3p_resnet50.json smp --smp-arch deeplabv3plus --smp-encoder resnet50 --smp-encoder-weights imagenet

# Core baselines
run_bench bench_segformer segformer_b2.json segformer
run_bench bench_efficientvit efficientvit_b2.json efficientvit
run_bench bench_farseg farseg_r50.json farseg_r50
run_bench bench_farseg farsegpp_mitb2.json farsegpp_mitb2

# RS segmentation families
run_bench bench_rsseg rsseg_logcan_r50.json rsseg_logcan_r50
run_bench bench_rsseg rsseg_logcanplus_repvitm23.json rsseg_logcanplus_repvitm23
run_bench bench_rsseg rsseg_docnet_hrnetw32.json rsseg_docnet_hrnetw32
run_bench bench_rsseg rsseg_sacanet_hrnetw32.json rsseg_sacanet_hrnetw32

run_bench bench_pem pem_r50.json pem
run_bench bench_mask2former mask2former_r50.json mask2former

# SSA-Seg
run_bench bench_ssaseg ssaseg_afformer_base.json ssaseg_afformer_base
run_bench bench_ssaseg ssaseg_seaformer_base.json ssaseg_seaformer_base
run_bench bench_ssaseg ssaseg_segnext_tiny.json ssaseg_segnext_tiny
run_bench bench_ssaseg ssaseg_cgrseg_b.json ssaseg_cgrseg_b
run_bench bench_ssaseg ssaseg_upernet_swin_tiny.json ssaseg_upernet_swin_tiny
run_bench bench_ssaseg ssaseg_ocrnet_hr48.json ssaseg_ocrnet_hr48

# GeoSeg/PPMamba/Mamba variants
run_bench bench_geoseg geoseg_unetformer.json geoseg_unetformer
run_bench bench_geoseg geoseg_a2fpn.json geoseg_a2fpn
run_bench bench_geoseg geoseg_abcnet_r18.json geoseg_abcnet
run_bench bench_geoseg geoseg_banet.json geoseg_banet
run_bench bench_geoseg geoseg_manet.json geoseg_manet
run_bench bench_geoseg geoseg_dcswin.json geoseg_dcswin
run_bench bench_geoseg_mamba geoseg_pyramidmamba.json geoseg_pyramidmamba
run_bench bench_mfmamba ppmambaseg_ppmamba.json ppmambaseg_ppmamba
run_bench bench_ssrs rs3mamba_vmamba_tiny.json rs3mamba
run_bench bench_mfmamba mfmamba_hrnetw18.json mfmamba
run_bench bench_mcpnet mcpnet_r50.json mcpnet

# SAM-based
run_bench bench_ssrs sam_rs_unetformer.json sam_rs --samrs-model unetformer
run_bench bench_ssrs sam_rs_ftunetformer.json sam_rs --samrs-model ftunetformer
run_bench bench_ssrs sam_rs_abcnet.json sam_rs --samrs-model abcnet
run_bench bench_ssrs sam_rs_cmtfnet.json sam_rs --samrs-model cmtfnet
run_bench bench_sam_family sam2_1_hierabplus_msfpn.json sam2_1_hierabplus
run_bench bench_sam_family rsamseg_vit_b.json rsamseg_vit_b

# Sync latency/efficiency fields into summary tables.
conda activate bench_base
python -u "$ROOT/misc/sync_experiment_tables.py"

FAIL_COUNT=$(grep -c . "$FAIL_FILE" || true)
echo "\nLatency benchmark suite finished. failures=$FAIL_COUNT"
if [[ "$FAIL_COUNT" -eq 0 ]]; then
  rm -f "$FAIL_FILE"
  echo "All benchmarks succeeded."
else
  echo "Failure list: $FAIL_FILE"
fi
