#!/bin/bash
#SBATCH --job-name=lat_l40s_anom
#SBATCH --output=/deac/csc/yangGrp/cuij/GoldMDD/experiments/logs/slurm_lat_l40s_anom_%j.out
#SBATCH --error=/deac/csc/yangGrp/cuij/GoldMDD/experiments/logs/slurm_lat_l40s_anom_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --partition=yangGrp
#SBATCH --account=yanggrp
#SBATCH --gres=gpu:L40S:1

set -euo pipefail

bash /deac/csc/alqahtaniGrp/cuij/run_latency_l40s_anomalies_only.sh
