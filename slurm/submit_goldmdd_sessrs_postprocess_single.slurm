#!/bin/bash
# Re-run official-style SESSRS post-process for one finished GeoSeg run.
#
# Example:
# sbatch --partition=large --account=paucagrp --cpus-per-task=24 --mem=96G --time=5-00:00:00 \
#   --job-name=goldmdd_sessrs_post_unet \
#   --export=ALL,RUN=geoseg_unetformer_goldmdd_b8_e80_poly_ce_dice \
#   /deac/csc/alqahtaniGrp/cuij/submit_goldmdd_sessrs_postprocess_single.slurm

#SBATCH --job-name=goldmdd_sessrs_post_1x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=5-00:00:00
#SBATCH --output=/deac/csc/yangGrp/cuij/GoldMDD/experiments/logs/slurm_%x_%j.out
#SBATCH --error=/deac/csc/yangGrp/cuij/GoldMDD/experiments/logs/slurm_%x_%j.err
#SBATCH --account=paucagrp
#SBATCH --partition=large

set -euo pipefail

ROOT="/deac/csc/yangGrp/cuij"
SCRIPT="${ROOT}/GoldMDD/misc/run_sessrs_postprocess_geoseg_official.py"
CONDA_SH="/deac/csc/alqahtaniGrp/cuij/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="bench_geoseg"

RUN="${RUN:-}"
POST_WORKERS="${POST_WORKERS:-24}"
WORKERS="${WORKERS:-8}"
OVERWRITE="${OVERWRITE:-1}"
SKIP_INFERENCE="${SKIP_INFERENCE:-1}"
SEARCH_MAX_IMAGES="${SEARCH_MAX_IMAGES:-0}"   # 0 means full val split
MAX_SAMPLES="${MAX_SAMPLES:-0}"               # 0 means full split

if [[ -z "${RUN}" ]]; then
  echo "RUN is required, e.g. geoseg_unetformer_goldmdd_b8_e80_poly_ce_dice" >&2
  exit 2
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Run: ${RUN}"
echo "Python: $(which python)"

ARGS=(
  --repo-root "${ROOT}/third_party/GeoSeg"
  --experiments-root "${ROOT}/GoldMDD/experiments"
  --data-root "${ROOT}/GoldMDD/data-cropped"
  --sam-prior-root "${ROOT}/GoldMDD/data-cropped-samrs-priors"
  --runs "${RUN}"
  --splits val test
  --device cpu
  --workers "${WORKERS}"
  --post-workers "${POST_WORKERS}"
)

if [[ "${SKIP_INFERENCE}" == "1" ]]; then
  ARGS+=(--skip-inference)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi
if [[ "${SEARCH_MAX_IMAGES}" != "0" ]]; then
  ARGS+=(--search-max-images "${SEARCH_MAX_IMAGES}")
fi
if [[ "${MAX_SAMPLES}" != "0" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

python -u "${SCRIPT}" "${ARGS[@]}"

echo "End: $(date)"
