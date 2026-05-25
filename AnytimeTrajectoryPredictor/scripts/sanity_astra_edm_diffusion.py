import argparse
import sys
from pathlib import Path

import torch
import yaml
from box import Box
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from AnytimeTrajectoryPredictor.Data.feature_extractor import WaymoPredictionDataset
from AnytimeTrajectoryPredictor.models.TrajectoryPredictor import TrajectoryPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sampling-steps", type=int, default=2)
    args_cli = parser.parse_args()

    with open(args_cli.config, "r") as f:
        args = Box(yaml.safe_load(f))

    train_dataset = WaymoPredictionDataset(args.feature_extractor, split="train")
    val_dataset = WaymoPredictionDataset(args.feature_extractor, split="val")
    args.model.trajectory_mean = train_dataset.target_mean
    args.model.trajectory_std = train_dataset.target_std
    args.model.history_steps_H = train_dataset.history_steps
    args.model.future_horizon_T = train_dataset.future_steps

    loader = DataLoader(train_dataset, batch_size=args_cli.batch_size, shuffle=False)
    batch = next(iter(loader))
    model = TrajectoryPredictor.create_model(args)
    model.eval()

    with torch.no_grad():
        losses = model.compute_loss(batch)
        samples = model(batch, num_sampling_steps=args_cli.sampling_steps)

    print(f"train_samples={len(train_dataset)} val_samples={len(val_dataset)}")
    print(f"target_mean={train_dataset.target_mean} target_std={train_dataset.target_std}")
    print(f"features_shape={tuple(batch['features'].shape)}")
    print(f"trajectory_shape={tuple(batch['trajectory'].shape)}")
    print(f"future_mask_shape={tuple(batch['future_mask'].shape)}")
    for key, value in losses.items():
        if torch.is_tensor(value):
            value = value.detach()
            if value.numel() == 1:
                value = float(value.cpu())
        print(f"{key}={value}")
    print(f"sample_mu_shape={tuple(samples.mu.shape)}")
    print(f"sample_logits_shape={tuple(samples.mode_logits.shape)}")
    print(f"sample_cov_cholesky_shape={tuple(samples.cov_cholesky.shape)}")
    assert torch.isfinite(losses["loss"]).all()
    assert torch.isfinite(samples.mu).all()
    assert torch.isfinite(samples.cov_cholesky).all()


if __name__ == "__main__":
    main()
