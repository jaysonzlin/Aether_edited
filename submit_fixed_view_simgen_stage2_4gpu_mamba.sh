#!/usr/bin/env bash
#SBATCH --mail-user=jlin3@college.harvard.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=aether_fixed_view_simgen_stage2_4gpu
#SBATCH --partition=gpu_requeue
#SBATCH --constraint=h200&holyndr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/n/lab_storage/ydu_lab/jaysonzlin/Aether_edited/logs/fixed_view_simgen_stage2_%j.out
#SBATCH --error=/n/lab_storage/ydu_lab/jaysonzlin/Aether_edited/logs/fixed_view_simgen_stage2_%j.err

set -euo pipefail
module load Mambaforge cuda/12.4.1 gcc/9.5.0-fasrc01
PROJECT_DIR="${PROJECT_DIR:-/n/lab_storage/ydu_lab/jaysonzlin/Aether_edited}"
MAMBA_ENV_PREFIX="${MAMBA_ENV_PREFIX:-/n/holylabs/ydu_lab/Lab/jaysonzlin/aether_env}"
PYTHON_BIN="${MAMBA_ENV_PREFIX}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "The Aether Mamba environment is incomplete: ${MAMBA_ENV_PREFIX}" >&2
  exit 1
fi
cd "${PROJECT_DIR}"
mkdir -p logs outputs/fixed_view_simgen_stage2
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
export NCCL_DEBUG_FILE="${PROJECT_DIR}/logs/nccl-aether-fixed-view-stage2-${SLURM_JOB_ID}/nccl.%h.%p.log"
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_SOCKET_FAMILY=AF_INET
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
mkdir -p "${PROJECT_DIR}/logs/nccl-aether-fixed-view-stage2-${SLURM_JOB_ID}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Restart count: ${SLURM_RESTART_COUNT:-0}"
echo "Node: $(hostname)"
echo "Environment: ${MAMBA_ENV_PREFIX}"
echo "Stage-1 checkpoint: ${PROJECT_DIR}/outputs/fixed_view_simgen/checkpoint-010000"
echo "Start time: $(date)"
nvidia-smi
"${PYTHON_BIN}" -c '
import accelerate, diffusers, h5py, tiktoken, torch, torchmetrics, transformers, wandb
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
'
"${PYTHON_BIN}" scripts/train_fixed_view_simgen_stage2.py --preflight \
  --override aether_model_id="${PROJECT_DIR}/models/AetherV1" \
  --override cogvideox_model_id="${PROJECT_DIR}/models/CogVideoX-5b-I2V" \
  --override stage1_checkpoint="${PROJECT_DIR}/outputs/fixed_view_simgen/checkpoint-010000"
exec "${PYTHON_BIN}" -m accelerate.commands.accelerate_cli launch --config_file configs/accelerate/h200_4gpu.yaml \
  scripts/train_fixed_view_simgen_stage2.py --resume latest \
  --override aether_model_id="${PROJECT_DIR}/models/AetherV1" \
  --override cogvideox_model_id="${PROJECT_DIR}/models/CogVideoX-5b-I2V" \
  --override stage1_checkpoint="${PROJECT_DIR}/outputs/fixed_view_simgen/checkpoint-010000"
