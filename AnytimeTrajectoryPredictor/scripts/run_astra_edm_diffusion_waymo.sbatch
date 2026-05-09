#!/bin/bash
#SBATCH --job-name=astra-edm-waymo
#SBATCH --output=logs/astra_edm_waymo_%j.out
#SBATCH --error=logs/astra_edm_waymo_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -eo pipefail
export PYTHONUNBUFFERED=1

cd /home/santanto/vi/VisualThinkingProject
export PYTHONPATH=/home/santanto/vi/VisualThinkingProject:${PYTHONPATH:-}
mkdir -p logs checkpoints

module load gcc/11.3.0 cuda/12.1.1

source /home/santanto/miniconda3/etc/profile.d/conda.sh
conda activate vtp-gpu

nvidia-smi

python -u - <<'PY'
import torch

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available(), "device_count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this environment; refusing to start CPU training on a GPU allocation.")
print("device_name", torch.cuda.get_device_name(0))
PY

python -u AnytimeTrajectoryPredictor/scripts/train_trajectory_model.py \
  --config configs/astra_edm_diffusion_waymo.yml
