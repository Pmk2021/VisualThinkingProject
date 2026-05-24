#!/bin/bash
#SBATCH --job-name=traj_gnn
#SBATCH --time=10:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ---------------------------
# conda setup
# ---------------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate geometric

# ---------------------------
# move to project directory
# ---------------------------
cd /home/muralikr/VisualThinkingProject/AnytimeTrajectoryPredictor

echo "Running on: $(hostname)"
echo "Python: $(which python)"

# ---------------------------
# run training
# ---------------------------
python scripts/train_trajectory_model.py \
  --config /home/muralikr/VisualThinkingProject/configs/gnn.yml