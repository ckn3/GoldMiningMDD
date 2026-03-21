#!/bin/bash
set -euo pipefail

ROOT=/deac/csc/yangGrp/cuij/GoldMDD
SCRIPT="$ROOT/misc/benchmark_semseg_model.py"
OUTDIR="$ROOT/experiments/diagnostics/model_stats"
LOGDIR="$ROOT/experiments/logs"
STAMP=$(date +%Y%m%d_%H%M%S)
FAIL_FILE="$LOGDIR/latency_l40s_failures_${STAMP}.txt"

mkdir -p "$OUTDIR" "$LOGDIR"
: > "$FAIL_FILE"

source /deac/csc/alqahtaniGrp/cuij/miniconda3/etc/profile.d/conda.sh

wait_for_idle_gpu() {
  local target_gpu="${LATENCY_GPU_INDEX:-}"
  if [[ -z "${target_gpu}" && -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "unset" && "${CUDA_VISIBLE_DEVICES}" != *","* ]]; then
    target_gpu="${CUDA_VISIBLE_DEVICES%%,*}"
  fi
  local poll_sec="${LATENCY_IDLE_POLL_SEC:-60}"
  local timeout_sec="${LATENCY_IDLE_TIMEOUT_SEC:-0}"
  local waited=0
  local bus_id all_indices

  all_indices=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | tr -d ' ')
  if [[ -z "${target_gpu}" ]]; then
    echo "Waiting for any idle GPU among indices: ${all_indices}"
  else
    echo "Waiting for idle GPU index=${target_gpu}"
  fi

  while true; do
    local candidates
    if [[ -n "${target_gpu}" ]]; then
      candidates="${target_gpu}"
    else
      candidates="${all_indices}"
    fi

    local found=""
    for idx in ${candidates}; do
      bus_id=$(nvidia-smi --query-gpu=index,gpu_bus_id --format=csv,noheader,nounits | awk -F',' -v i="$idx" '
        {
          gsub(/^ +| +$/, "", $1);
          gsub(/^ +| +$/, "", $2);
          if ($1 == i) print $2;
        }'
      )
      if [[ -z "${bus_id}" ]]; then
        continue
      fi

      local busy_pids
      busy_pids=$(nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader,nounits 2>/dev/null | awk -F',' -v bus="$bus_id" '
        {
          gsub(/^ +| +$/, "", $1);
          gsub(/^ +| +$/, "", $2);
          if ($1 == bus) print $2;
        }'
      )
      if [[ -z "${busy_pids}" ]]; then
        found="$idx"
        break
      fi
    done

    if [[ -n "${found}" ]]; then
      target_gpu="${found}"
      break
    fi

    echo "No idle GPU yet; sleep ${poll_sec}s"
    sleep "${poll_sec}"
    waited=$((waited + poll_sec))
    if [[ "${timeout_sec}" -gt 0 && "${waited}" -ge "${timeout_sec}" ]]; then
      echo "[ERROR] Timed out waiting for an idle GPU."
      exit 2
    fi
  done

  export CUDA_VISIBLE_DEVICES="${target_gpu}"
  echo "GPU ${target_gpu} is idle. Starting latency suite."
}

current_gpu_busy_pids() {
  local idx="${CUDA_VISIBLE_DEVICES%%,*}"
  local bus_id
  bus_id=$(nvidia-smi --query-gpu=index,gpu_bus_id --format=csv,noheader,nounits | awk -F',' -v i="$idx" '
    {
      gsub(/^ +| +$/, "", $1);
      gsub(/^ +| +$/, "", $2);
      if ($1 == i) print $2;
    }'
  )
  if [[ -z "${bus_id}" ]]; then
    return 0
  fi
  nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader,nounits 2>/dev/null | awk -F',' -v bus="$bus_id" '
    {
      gsub(/^ +| +$/, "", $1);
      gsub(/^ +| +$/, "", $2);
      if ($1 == bus) print $2;
    }'
}

wait_for_current_gpu_idle() {
  local poll_sec="${LATENCY_IDLE_POLL_SEC:-60}"
  while true; do
    local busy
    busy=$(current_gpu_busy_pids | tr '\n' ' ' | xargs || true)
    if [[ -z "${busy}" ]]; then
      return 0
    fi
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}: busy pids=${busy}; waiting ${poll_sec}s"
    sleep "${poll_sec}"
  done
}

wait_for_idle_gpu

run_bench() {
  local env_name="$1"; shift
  local out_file="$1"; shift
  local family="$1"; shift

  wait_for_current_gpu_idle
  echo
  echo "===== [$family] env=$env_name -> $out_file ====="
  conda activate "$env_name"
  if python -u "$SCRIPT" --family "$family" --device cuda:0 --skip-flops --output "$OUTDIR/$out_file" "$@"; then
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
run_bench bench_sam_family hq_sam_vit_b_msfpn.json hq_sam_vit_b
run_bench bench_sam_family rsamseg_vit_b.json rsamseg_vit_b

# Sync latency into summary tables.
conda activate bench_segformer
python -u "$ROOT/misc/sync_experiment_tables.py"

FAIL_COUNT=$(grep -c . "$FAIL_FILE" || true)
echo
echo "Latency benchmark suite finished. failures=$FAIL_COUNT"
if [[ "$FAIL_COUNT" -eq 0 ]]; then
  rm -f "$FAIL_FILE"
  echo "All benchmarks succeeded."
else
  echo "Failure list: $FAIL_FILE"
fi
