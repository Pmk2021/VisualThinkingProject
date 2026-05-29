#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from box import Box
from torch.utils.data import DataLoader

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AnytimeTrajectoryPredictor").is_dir())
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from AnytimeTrajectoryPredictor.Data.feature_extractor import WaymoImagePlaneDataset
from AnytimeTrajectoryPredictor.models.TrajectoryPredictor import TrajectoryPredictor
from visualize_image_plane_diffusion import _maneuver, _pil_from_history, _render_step


def _load(config_path, checkpoint, device):
    with open(config_path, "r") as f:
        args = Box(yaml.safe_load(f))
    args.model.use_rgb_context = True
    args.model.input_dim = 4
    dataset = WaymoImagePlaneDataset(args.feature_extractor, split="val")
    args.model.trajectory_mean = dataset.target_mean
    args.model.trajectory_std = dataset.target_std
    args.model.history_steps_H = dataset.history_steps
    args.model.future_horizon_T = dataset.future_steps
    model = TrajectoryPredictor.create_model(args).to(device)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.eval()
    return model, dataset


def _score_turn(batch):
    traj = batch["trajectory"][0].numpy()
    mask = batch["future_mask"][0].numpy().astype(bool)
    pts = traj[mask]
    if len(pts) < 3:
        return 0.0
    total_dx = abs(float(pts[-1, 0] - pts[0, 0]))
    second = np.diff(pts[:, 0], n=2)
    curvature = float(np.abs(second).mean()) if len(second) else 0.0
    return total_dx + 2.0 * curvature


def _mode_maneuvers(mu):
    return [_maneuver(mu[k]) for k in range(mu.shape[0])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/astra_edm_diffusion_waymo_image_plane_sanity.yml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_gif", default="turn_commitment.gif")
    parser.add_argument("--output_csv", default="turn_commitment.csv")
    parser.add_argument("--max_samples", type=int, default=48)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--frame_ms", type=int, default=450)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, dataset = _load(args.config, args.checkpoint, device)

    best_idx, best_score, best_batch = 0, -1.0, None
    for idx in range(min(len(dataset), args.max_samples)):
        sample = dataset[idx]
        score = _score_turn(sample)
        if score > best_score:
            best_idx, best_score, best_batch = idx, score, sample

    loader = DataLoader(torch.utils.data.Subset(dataset, [best_idx]), batch_size=1, shuffle=False)
    batch = next(iter(loader))
    batch_d = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    with torch.no_grad():
        final_params, frames = model(batch_d, num_sampling_steps=args.num_steps, capture_steps=True)
    if not frames:
        frames = [{"sigma": 0.0, "params": final_params}]

    gt = batch["trajectory"][0, 0].numpy()
    mask = batch["future_mask"][0, 0].numpy().astype(bool)
    gt_maneuver = _maneuver(gt[mask])
    rows = []
    threshold_hits = {0.5: None, 0.7: None}
    for step_idx, frame in enumerate(frames, start=1):
        params = frame["params"]
        mu = params.mu[0, :, 0].detach().cpu().numpy()
        probs = params.mode_probs[0].detach().cpu().numpy()
        maneuvers = _mode_maneuvers(mu)
        correct_prob = float(sum(prob for prob, man in zip(probs, maneuvers) if man == gt_maneuver))
        for threshold in threshold_hits:
            if threshold_hits[threshold] is None and correct_prob >= threshold:
                threshold_hits[threshold] = step_idx
        rows.append(
            {
                "sample_index": best_idx,
                "step": step_idx,
                "sigma": frame["sigma"],
                "gt_maneuver": gt_maneuver,
                "top_mode": int(np.argmax(probs)),
                "top_prob": float(probs.max()),
                "top_maneuver": maneuvers[int(np.argmax(probs))],
                "correct_maneuver_prob": correct_prob,
            }
        )

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    base = _pil_from_history(batch)
    images = [_render_step(base, frame["params"], batch, i, len(frames), frame["sigma"]) for i, frame in enumerate(frames, start=1)]
    images += [images[-1]] * 2
    images[0].save(args.output_gif, save_all=True, append_images=images[1:], duration=args.frame_ms, loop=0, optimize=False)
    print(f"sample_index={best_idx} score={best_score:.4f} gt_maneuver={gt_maneuver}")
    print(f"threshold_hits={threshold_hits}")
    print(f"saved {args.output_csv}")
    print(f"saved {args.output_gif}")


if __name__ == "__main__":
    main()
