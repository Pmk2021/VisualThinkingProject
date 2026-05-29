#!/bin/bash
#SBATCH --job-name=unet-keypoints-waymo
#SBATCH --output=logs/unet_keypoints_waymo_%j.out
#SBATCH --error=logs/unet_keypoints_waymo_%j.err
#SBATCH --time=08:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --signal=TERM@120

set -eo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

cd /home/santanto/vi/VisualThinkingProject_backup
export PYTHONPATH=/home/santanto/vi/VisualThinkingProject_backup:${PYTHONPATH:-}
mkdir -p logs checkpoints

module load gcc/11.3.0 cuda/12.1.1

source /home/santanto/miniconda3/etc/profile.d/conda.sh
conda activate vtp-gpu

nvidia-smi

python -u - <<'PYCHECK'
import torch

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available(), "device_count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this environment; refusing to start CPU training on a GPU allocation.")
print("device_name", torch.cuda.get_device_name(0))
PYCHECK

python -u AnytimeTrajectoryPredictor/scripts/train_unet_keypoints.py \
  --config configs/unet_keypoints_waymo.yml
