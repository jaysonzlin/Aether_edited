#!/usr/bin/env bash
#SBATCH --mail-user=jlin3@college.harvard.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=aether_fixed_view_simgen_4gpu
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
#SBATCH --output=/n/lab_storage/ydu_lab/jaysonzlin/Aether_edited/logs/fixed_view_simgen_4gpu_%j.out
#SBATCH --error=/n/lab_storage/ydu_lab/jaysonzlin/Aether_edited/logs/fixed_view_simgen_4gpu_%j.err

set -euo pipefail

module load Mambaforge
module load cuda/12.4.1
module load gcc/9.5.0-fasrc01

PROJECT_DIR="${PROJECT_DIR:-/n/lab_storage/ydu_lab/jaysonzlin/Aether_edited}"
MAMBA_ENV_PREFIX="${MAMBA_ENV_PREFIX:-/n/holylabs/ydu_lab/Lab/jaysonzlin/aether_env}"
PYTHON_BIN="${MAMBA_ENV_PREFIX}/bin/python"
ACCELERATE_BIN="${MAMBA_ENV_PREFIX}/bin/accelerate"
AETHER_MODEL_DIR="${PROJECT_DIR}/models/AetherV1"
COGVIDEOX_MODEL_DIR="${PROJECT_DIR}/models/CogVideoX-5b-I2V"

if [[ ! -x "${PYTHON_BIN}" || ! -x "${ACCELERATE_BIN}" ]]; then
    echo "The Aether Mamba environment is incomplete: ${MAMBA_ENV_PREFIX}" >&2
    exit 1
fi

for required_path in \
    "${AETHER_MODEL_DIR}/transformer/config.json" \
    "${COGVIDEOX_MODEL_DIR}/tokenizer" \
    "${COGVIDEOX_MODEL_DIR}/text_encoder" \
    "${COGVIDEOX_MODEL_DIR}/vae" \
    "${COGVIDEOX_MODEL_DIR}/scheduler"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Missing required local model component: ${required_path}" >&2
        exit 1
    fi
done

cd "${PROJECT_DIR}"
mkdir -p logs outputs/fixed_view_simgen

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
export NCCL_DEBUG_FILE="${PROJECT_DIR}/logs/nccl-aether-fixed-view-${SLURM_JOB_ID}/nccl.%h.%p.log"
export OMP_NUM_THREADS=1
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_SOCKET_FAMILY=AF_INET
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "${PROJECT_DIR}/logs/nccl-aether-fixed-view-${SLURM_JOB_ID}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Restart count: ${SLURM_RESTART_COUNT:-0}"
echo "Node: $(hostname)"
echo "Environment: ${MAMBA_ENV_PREFIX}"
echo "Aether model: ${AETHER_MODEL_DIR}"
echo "CogVideoX model: ${COGVIDEOX_MODEL_DIR}"
echo "Start time: $(date)"
nvidia-smi

"${PYTHON_BIN}" -c '
import torch
import accelerate
import diffusers
import h5py
import transformers
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 4
print(f"torch={torch.__version__} cuda={torch.version.cuda} gpus={torch.cuda.device_count()}")
'

exec "${ACCELERATE_BIN}" launch \
    --config_file configs/accelerate/h200_4gpu.yaml \
    scripts/train_fixed_view_simgen.py \
    --resume latest \
    --override aether_model_id=/n/lab_storage/ydu_lab/jaysonzlin/Aether_edited/models/AetherV1 \
    --override cogvideox_model_id=/n/lab_storage/ydu_lab/jaysonzlin/Aether_edited/models/CogVideoX-5b-I2V
