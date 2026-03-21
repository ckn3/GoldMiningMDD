#!/usr/bin/env bash
set -euo pipefail

source /deac/csc/alqahtaniGrp/cuij/miniconda3/etc/profile.d/conda.sh
conda activate bench_ssaseg

REPO=/deac/csc/yangGrp/cuij/third_party/SSA-Seg
OUT=/deac/csc/yangGrp/cuij/GoldMDD/experiments/ssa_base_smoke
mkdir -p "${OUT}"

cd "${REPO}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

configs=(
  ocrnet_hr48_goldmdd.py
  upernet_swin_tiny_goldmdd.py
  afformer_base_goldmdd.py
  seaformer_base_goldmdd.py
  segnext_tiny_goldmdd.py
  cgrseg_b_goldmdd.py
)

for c in "${configs[@]}"; do
  name="${c%.py}"
  wdir="${OUT}/${name}"
  log="${wdir}/smoke.log"
  mkdir -p "${wdir}"
  echo "===== ${name} ====="
  python train.py "configs/goldmdd/${c}" --no-validate \
    --work-dir "${wdir}" \
    --cfg-options \
      runner.max_iters=4 \
      checkpoint_config.interval=999999 \
      data.samples_per_gpu=2 \
      data.workers_per_gpu=1 \
      log_config.interval=1 \
      optimizer_config.grad_clip.max_norm=1.0 \
      optimizer_config.grad_clip.norm_type=2 \
      > "${log}" 2>&1
  echo "${name} PASS"
done

echo "Baseline smoke tests completed."
