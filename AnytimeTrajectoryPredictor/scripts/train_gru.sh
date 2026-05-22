#!/bin/bash
#SBATCH --job-name=traj_gru
#SBATCH --time=10:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ---------------------------
# conda setup
# ---------------------------
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vtp

# ---------------------------
# move to project directory
# ---------------------------
cd /home/lamsler/VisualThinkingProject/AnytimeTrajectoryPredictor

echo "Running on: $(hostname)"
echo "Python: $(which python)"

# ---------------------------
# run training
# ---------------------------
echo "Starting training with GRU model..."
python scripts/train_trajectory_model.py \
  --config /home/lamsler/VisualThinkingProject/configs/gru_config.yml