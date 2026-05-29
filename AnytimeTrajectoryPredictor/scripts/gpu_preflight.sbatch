#!/bin/bash
#SBATCH --job-name=vtp-gpu-preflight
#SBATCH --output=logs/gpu_preflight_%j.out
#SBATCH --error=logs/gpu_preflight_%j.err
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -eo pipefail
export PYTHONUNBUFFERED=1

cd /home/santanto/vi/VisualThinkingProject
mkdir -p logs

module load gcc/11.3.0 cuda/12.1.1
source /home/santanto/miniconda3/etc/profile.d/conda.sh
conda activate vtp-gpu

nvidia-smi
python -u - <<'PY'
import torch

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available(), "device_count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible inside the Slurm GPU allocation.")
print("device_name", torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("matmul_ok", float(y[0, 0].detach().cpu()))
PY
