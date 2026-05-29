#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from box import Box
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AnytimeTrajectoryPredictor").is_dir())
sys.path.insert(0, str(PROJECT_ROOT))

from AnytimeTrajectoryPredictor.Data.feature_extractor import WaymoImagePlaneDataset
from AnytimeTrajectoryPredictor.scripts.train_astra_image_plane import ASTRAImagePlaneAdapter, make_astra_cfg
from visualizations.visualize_image_plane_diffusion import (
    AGENT_COLORS,
    _draw_poly,
    _draw_target_boxes,
    _draw_gt_trajectories,
    _maneuver,
    _mode_ade,
    _pil_from_history,
    _unit_to_px,
)


def _cfg_get(config, key, default=None):
    return getattr(config, key, default) if config is not None else default


def _parse_indices(value):
    indices = []
    for part in str(value).split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            lo, hi = part.split('-', 1)
            indices.extend(range(int(lo), int(hi) + 1))
        else:
            indices.append(int(part))
    return indices


def _best_checkpoint_from_save_to(save_to):
    if not save_to:
        return None
    path = Path(save_to)
    suffix = path.suffix or '.pth'
    best = path.with_name(f'{path.stem}.best{suffix}')
    return best if best.exists() else path


def _move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def _load_model_and_dataset(config_path, checkpoint_path, device):
    with open(config_path) as f:
        args = Box(yaml.safe_load(f))
    dataset = WaymoImagePlaneDataset(args.feature_extractor, split='val')
    args.training.batch_size = 1
    astra_cfg = make_astra_cfg(args, device, dataset.history_steps, dataset.future_steps)
    model = ASTRAImagePlaneAdapter(
        astra_cfg,
        unet_weights=_cfg_get(args.astra, 'unet_weights', 'checkpoints/unet_keypoints_waymo_latest.pth'),
        freeze_unet=bool(_cfg_get(args.astra, 'freeze_unet', True)),
    ).to(device)
    if checkpoint_path is None:
        checkpoint_path = _best_checkpoint_from_save_to(_cfg_get(args.training, 'save_to', None))
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        raise FileNotFoundError(f'ASTRA checkpoint not found: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
    print(f'Loaded ASTRA checkpoint {checkpoint_path} missing={len(missing)} unexpected={len(unexpected)}')
    model.eval()
    return model, dataset


def _final_valid_future_step(batch):
    mask = batch['future_mask'][0].detach().cpu().numpy().astype(bool)
    valid = np.flatnonzero(mask.any(axis=0) if mask.ndim == 2 else mask)
    return int(valid[-1]) if len(valid) else 0


def _tensor_or_image_to_rgba(value):
    if torch.is_tensor(value):
        arr = (value.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr).convert('RGBA')
    return value.convert('RGBA')


def _future_frame_base(dataset, index, batch, future_step=None):
    samples = getattr(dataset, 'samples', None)
    if samples is None:
        return _pil_from_history(batch), 0
    sample = samples[index]
    future = sample.get('future', [])
    if not future:
        return _pil_from_history(batch), 0
    if future_step is None:
        future_step = _final_valid_future_step(batch)
    future_step = max(0, min(int(future_step), len(future) - 1))
    try:
        segment_images = dataset._load_segment_images(sample['segment'])
        row = future[future_step]
        return _tensor_or_image_to_rgba(dataset._decode_image(segment_images[str(row['image_id'])])), future_step
    except Exception as exc:
        print(f'Falling back to last observed frame for index {index}: {exc}', flush=True)
        return _pil_from_history(batch), future_step


def _render_astra_overlay(base, pred_modes, batch, index, future_step):
    image = base.copy().convert('RGBA')
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, 'RGBA')
    width, height = image.size

    mu = pred_modes[0].detach().cpu().numpy()  # [K A T 2]
    gt = batch['trajectory'][0].detach().cpu().numpy()  # [A T 2]
    mask = batch['future_mask'][0].detach().cpu().numpy().astype(bool)
    if mu.ndim == 3:
        mu = mu[:, None]

    _draw_gt_trajectories(draw, batch, width, height, line_width=4)
    _draw_target_boxes(draw, batch, width, height)
    mode_ade = _mode_ade(mu, gt, mask)
    shown_mode = int(np.nanargmin(mode_ade))

    for agent_idx in range(mu.shape[1]):
        pred_pts = _unit_to_px(mu[shown_mode, agent_idx], width, height)
        color = AGENT_COLORS[agent_idx % len(AGENT_COLORS)]
        _draw_poly(draw, pred_pts, color, alpha=245, width=5)
        for x, y in pred_pts[::max(1, len(pred_pts) // 8)]:
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color + (235,))

    gt_valid = gt[0][mask[0]] if mask.ndim == 2 else gt[mask]
    pred_maneuver = _maneuver(mu[shown_mode, 0])
    gt_maneuver = _maneuver(gt_valid)
    minade = float(mode_ade[shown_mode])
    final_pred = mu[shown_mode, 0, -1]
    final_gt = gt_valid[-1] if len(gt_valid) else gt[0, -1]
    fde = float(np.linalg.norm(final_pred - final_gt))

    panel = [
        f'ASTRA baseline  dataset index={index}',
        f'future frame step={future_step}  minADE={minade:.3f}  FDE={fde:.3f}',
        f'gt={gt_maneuver}  pred={pred_maneuver}',
    ]
    draw.rectangle([8, 8, 520, 78], fill=(0, 0, 0, 165))
    for i, text in enumerate(panel):
        draw.text((16, 15 + i * 19), text, fill=(255, 255, 255, 235))

    legend_y = height - 54
    draw.rectangle([8, legend_y, 360, height - 8], fill=(0, 0, 0, 145))
    draw.text((16, legend_y + 8), 'green/GT path, colored/ASTRA prediction', fill=(255, 255, 255, 235))
    draw.text((16, legend_y + 27), 'box = last observed bbox / target region', fill=(255, 255, 255, 235))
    return Image.alpha_composite(image, overlay).convert('RGB'), {
        'sample_index': int(index),
        'scene_id': batch.get('scene_id', [''])[0],
        'trajectory_row_id': batch.get('trajectory_row_id', [''])[0],
        'minade_norm': minade,
        'fde_norm': fde,
        'gt_maneuver': gt_maneuver,
        'pred_maneuver': pred_maneuver,
        'future_step': int(future_step),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/astra_waymo_image_plane_baseline.yml')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--indices', default='1090-1100')
    parser.add_argument('--output_dir', default='visualizations/outs/astra_baseline_1090_1100')
    parser.add_argument('--future_step', type=int, default=None)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, dataset = _load_model_and_dataset(args.config, args.checkpoint, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for index in _parse_indices(args.indices):
        loader = DataLoader(Subset(dataset, [index]), batch_size=1, shuffle=False)
        batch = next(iter(loader))
        batch_d = _move_batch(batch, device)
        with torch.no_grad():
            pred_modes = model.predict_modes(batch_d)
        base, future_step = _future_frame_base(dataset, index, batch, args.future_step)
        image, summary = _render_astra_overlay(base, pred_modes, batch, index, future_step)
        out_path = output_dir / f'astra_{index:04d}.png'
        image.save(out_path)
        summaries.append(summary)
        print(f'saved {out_path} minADE={summary["minade_norm"]:.4f} FDE={summary["fde_norm"]:.4f}')

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summaries, f, indent=2)
    print(f'wrote {output_dir / "summary.json"}')


if __name__ == '__main__':
    main()
