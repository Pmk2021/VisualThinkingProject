import argparse
import sys
from pathlib import Path

import torch
import yaml
from box import Box
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from AnytimeTrajectoryPredictor.Data.feature_extractor import WaymoImagePlaneDataset
from AnytimeTrajectoryPredictor.models.TrajectoryPredictor import TrajectoryPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/astra_edm_diffusion_waymo_image_plane_sanity.yml")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sampling-steps", type=int, default=2)
    args_cli = parser.parse_args()

    with open(args_cli.config, "r") as f:
        args = Box(yaml.safe_load(f))

    train_dataset = WaymoImagePlaneDataset(args.feature_extractor, split="train")
    val_dataset = WaymoImagePlaneDataset(args.feature_extractor, split="val")
    args.model.use_rgb_context = True
    args.model.input_dim = 4
    args.model.trajectory_mean = train_dataset.target_mean
    args.model.trajectory_std = train_dataset.target_std
    args.model.history_steps_H = train_dataset.history_steps
    args.model.future_horizon_T = train_dataset.future_steps

    loader = DataLoader(train_dataset, batch_size=args_cli.batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    model = TrajectoryPredictor.create_model(args)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    output = model.compute_loss(batch)
    assert torch.isfinite(output["loss"]).all()
    optimizer.zero_grad(set_to_none=True)
    output["loss"].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert torch.isfinite(grad_norm)
    optimizer.step()

    model.eval()
    with torch.no_grad():
        params = model(batch, num_sampling_steps=args_cli.sampling_steps)
    assert torch.isfinite(params.mu).all()
    assert torch.isfinite(params.cov_cholesky).all()

    print(f"train_samples={len(train_dataset)} val_samples={len(val_dataset)}")
    print(f"rgb_history_shape={tuple(batch['rgb_history'].shape)}")
    print(f"box_history_shape={tuple(batch['box_history'].shape)}")
    print(f"trajectory_shape={tuple(batch['trajectory'].shape)}")
    print(f"observed_mask_shape={tuple(batch['observed_mask'].shape)}")
    print(f"future_mask_shape={tuple(batch['future_mask'].shape)}")
    for key, value in output.items():
        if torch.is_tensor(value) and value.numel() == 1:
            print(f"{key}={float(value.detach())}")
    print(f"sample_mu_shape={tuple(params.mu.shape)}")
    print(f"sample_mu_range=({float(params.mu.min()):.4f}, {float(params.mu.max()):.4f})")
    print(f"sample_logits_shape={tuple(params.mode_logits.shape)}")
    print(f"grad_norm={float(grad_norm):.4f}")


if __name__ == "__main__":
    main()
