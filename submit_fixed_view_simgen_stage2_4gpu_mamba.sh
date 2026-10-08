#!/usr/bin/env bash
#SBATCH --job-name=aether_fixed_view_simgen_stage2_4gpu
#SBATCH --partition=gpu_requeue
#SBATCH --constraint=h200&holyndr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/n/lab_storage/ydu_lab/jaysonzlin/Aether_edited/logs/fixed_view_simgen_stage2_%j.out
#SBATCH --error=/n/lab_storage/ydu_lab/jaysonzlin/Aether_edited/logs/fixed_view_simgen_stage2_%j.err

set -euo pipefail
module load Mambaforge cuda/12.4.1 gcc/9.5.0-fasrc01
PROJECT_DIR="${PROJECT_DIR:-/n/lab_storage/ydu_lab/jaysonzlin/Aether_edited}"
MAMBA_ENV_PREFIX="${MAMBA_ENV_PREFIX:-/n/holylabs/ydu_lab/Lab/jaysonzlin/aether_env}"
PYTHON_BIN="${MAMBA_ENV_PREFIX}/bin/python"
ACCELERATE_BIN="${MAMBA_ENV_PREFIX}/bin/accelerate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
cd "${PROJECT_DIR}"
"${PYTHON_BIN}" scripts/train_fixed_view_simgen_stage2.py --preflight \
  --override aether_model_id="${PROJECT_DIR}/models/AetherV1" \
  --override cogvideox_model_id="${PROJECT_DIR}/models/CogVideoX-5b-I2V" \
  --override stage1_checkpoint="${PROJECT_DIR}/outputs/fixed_view_simgen/checkpoint-010000"
exec "${ACCELERATE_BIN}" launch --config_file configs/accelerate/h200_4gpu.yaml \
  scripts/train_fixed_view_simgen_stage2.py --resume latest \
  --override aether_model_id="${PROJECT_DIR}/models/AetherV1" \
  --override cogvideox_model_id="${PROJECT_DIR}/models/CogVideoX-5b-I2V" \
  --override stage1_checkpoint="${PROJECT_DIR}/outputs/fixed_view_simgen/checkpoint-010000"
