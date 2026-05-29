#!/usr/bin/env python3
"""Render concatenated refinement GIFs for selected Waymo image-plane tracks.

This script is self-contained and is meant to be copied into a notebook cell or
run directly. It loads the image-plane ASTRA-EDM model, forces each scene
segment explicitly, filters samples by exact `trajectory_row_id`, and exports
one GIF per selected track.

Each output GIF is built by concatenating the refinement sequences of all samples
that belong to the same track. For every sample, the sequence is:
- one ground-truth/context frame, then
- one frame per refinement step, rendered from scratch with Matplotlib.

Rendering requirements:
- use our own Matplotlib-based annotations, not the old ImageDraw helpers;
- draw all modes;
- make the best mode fully opaque;
- render the other modes with lower opacity.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from box import Box
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import Rectangle
from PIL import Image
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AnytimeTrajectoryPredictor").is_dir())
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from AnytimeTrajectoryPredictor.Data.feature_extractor import WaymoImagePlaneDataset
from AnytimeTrajectoryPredictor.models.TrajectoryPredictor import TrajectoryPredictor


DEFAULT_CHECKPOINT = "/work/cs-503/santanto/checkpoints/astra_edm_diffusion_waymo_image_plane_prior_correction_knots_regularized_diversity_aug_latest.best.pth"
DEFAULT_CONFIG = "configs/astra_edm_diffusion_waymo_image_plane.yml"
DEFAULT_OUTPUT_DIR = "visualizations/image_plane_track_refinement_gifs"

TRACK_IDS_BY_SCENE: Dict[str, str] = {
    "18446264979321894359_3700_000_3720_000": "18446264979321894359_3700_000_3720_000;WMpNISKr4naKH705Q9BUWA",
    "15224741240438106736_960_000_980_000": "15224741240438106736_960_000_980_000;uHByocSaacA0JoGLQhcqkA",
    "14300007604205869133_1160_000_1180_000": "14300007604205869133_1160_000_1180_000;iXXWDtTk8ZulKstE2Mt4jg",
    "1906113358876584689_1359_560_1379_560": "1906113358876584689_1359_560_1379_560;qo0sk0esFAv2q_Imz__lAg",
    "3731719923709458059_1540_000_1560_000": "3731719923709458059_1540_000_1560_000;8_LsdpXxbep9pedo66bR1g",
}

def load_model(config_path, checkpoint, device):
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

def _sanitize_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def _track_id_from_value(value) -> str:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return str(value)


def _batch_track_id(sample: Mapping) -> str:
    return _track_id_from_value(sample.get("trajectory_row_id", ""))


def _force_dataset_to_scene(dataset, scene_name: str):
    scene_segment = Path(f"/work/cs-503/santanto/waymo/validation__{scene_name}")
    dataset.segments = [scene_segment]
    dataset.split_source = "seeded_fraction"
    dataset.samples = sorted(
        dataset._load_samples(dataset.segments),
        key=lambda sample: f'{sample["trajectory_row_id"]}{int(sample["history"][0]["frame_timestamp_micros"])}',
    )
    return dataset


def _collect_track_indices(dataset, track_id: str, max_samples: Optional[int] = None) -> List[int]:
    samples = getattr(dataset, "samples", None)
    if samples is None:
        raise ValueError("The dataset must expose a samples attribute to filter by trajectory_row_id.")

    track_id = str(track_id)
    indices: List[int] = []
    for idx, sample in enumerate(samples):
        if _batch_track_id(sample) != track_id:
            continue
        indices.append(idx)
        if max_samples is not None and len(indices) >= int(max_samples):
            break
    return indices


def _sample_to_batch(dataset, index: int):
    loader = DataLoader(Subset(dataset, [index]), batch_size=1, shuffle=False)
    return next(iter(loader))


def _batch_to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _latest_image_array(batch) -> np.ndarray:
    rgb = batch["rgb_history"][0, -1].detach().cpu().clamp(0, 1)
    return (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _unit_to_px(points, width: int, height: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    out = np.empty_like(pts)
    out[..., 0] = (pts[..., 0] + 1.0) * 0.5 * width
    out[..., 1] = (pts[..., 1] + 1.0) * 0.5 * height
    return out


def _inside_image_mask(points_px: np.ndarray, width: int, height: int) -> np.ndarray:
    points_px = np.asarray(points_px, dtype=np.float32)
    return (
        np.isfinite(points_px[..., 0])
        & np.isfinite(points_px[..., 1])
        & (points_px[..., 0] >= 0.0)
        & (points_px[..., 0] <= float(width - 1))
        & (points_px[..., 1] >= 0.0)
        & (points_px[..., 1] <= float(height - 1))
    )


def _clip_rect_to_image(rect, width: int, height: int):
    if rect is None:
        return None
    left = max(0.0, float(rect[0]))
    top = max(0.0, float(rect[1]))
    right = min(float(width - 1), float(rect[2]))
    bottom = min(float(height - 1), float(rect[3]))
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def _trajectory_points_px(batch, width: int, height: int) -> np.ndarray:
    trajectory = batch["trajectory"][0].detach().cpu().numpy()
    future_mask = batch["future_mask"][0].detach().cpu().numpy().astype(bool)
    pts = _unit_to_px(trajectory, width, height)
    pts[~future_mask] = np.nan
    return pts


def _observed_points_px(batch, width: int, height: int) -> np.ndarray:
    box_history = batch["box_history"][0, 0].detach().cpu().numpy()
    observed_mask = batch["observed_mask"][0, 0].detach().cpu().numpy().astype(bool)
    pts = _unit_to_px(box_history[:, :2], width, height)
    pts[~observed_mask] = np.nan
    return pts


def _last_observed_box_px(batch, width: int, height: int):
    box_history = batch["box_history"][0, 0].detach().cpu().numpy()
    observed_mask = batch["observed_mask"][0, 0].detach().cpu().numpy().astype(bool)
    valid_indices = np.flatnonzero(observed_mask)
    if len(valid_indices) == 0:
        return None
    box = box_history[int(valid_indices[-1])]
    cx = (float(box[0]) + 1.0) * 0.5 * width
    cy = (float(box[1]) + 1.0) * 0.5 * height
    bw = max(float(box[2]) * width, 2.0)
    bh = max(float(box[3]) * height, 2.0)
    return [cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5]


def _trajectory_ade_per_mode(mu, gt, mask) -> np.ndarray:
    mu = np.asarray(mu, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if mu.ndim == 3:
        valid_count = max(int(mask.sum()), 1)
        dists = np.linalg.norm(mu - gt[None], axis=-1)
        return (dists * mask[None]).sum(axis=-1) / valid_count
    valid_count = max(int(mask.sum()), 1)
    dists = np.linalg.norm(mu - gt[None], axis=-1)
    return (dists * mask[None]).sum(axis=(1, 2)) / valid_count


def _best_mode_index(params, batch) -> int:
    mu = params.mu[0].detach().cpu().numpy()
    gt = batch["trajectory"][0].detach().cpu().numpy()
    mask = batch["future_mask"][0].detach().cpu().numpy().astype(bool)
    if mu.ndim == 3:
        mu = mu[:, None]
    return int(np.nanargmin(_trajectory_ade_per_mode(mu, gt, mask)))


def _render_single_frame(
    batch,
    params,
    step_idx: int,
    total_steps: int,
    sample_pos: int,
    total_samples: int,
    scene_name: str,
    track_id: str,
    show_modes: bool = True,
    mode_selection: str = "best",
    frame_kind: str = "refine",
    heatmap_alpha: float = 0.0,
):
    base = _latest_image_array(batch)
    height, width = base.shape[:2]

    mu = params.mu[0].detach().cpu().numpy()
    probs = params.mode_probs[0].detach().cpu().numpy()
    gt = batch["trajectory"][0].detach().cpu().numpy()
    mask = batch["future_mask"][0].detach().cpu().numpy().astype(bool)
    if mu.ndim == 3:
        mu = mu[:, None]

    best_mode = _best_mode_index(params, batch)
    top_mode = int(np.argmax(probs))
    shown_mode = top_mode if mode_selection == "top" else best_mode

    fig, ax = plt.subplots(figsize=(12, 8), dpi=240)
    ax.imshow(base)
    ax.set_axis_off()

    observed_pts = _observed_points_px(batch, width, height)
    observed_valid = np.isfinite(observed_pts).all(axis=1)
    if observed_valid.any():
        ax.plot(observed_pts[observed_valid, 0], observed_pts[observed_valid, 1], color="white", linewidth=2.6, alpha=0.95, zorder=3)
        ax.scatter(observed_pts[observed_valid, 0], observed_pts[observed_valid, 1], s=24, c="white", edgecolors="black", linewidths=0.6, zorder=4)

    box_px = _last_observed_box_px(batch, width, height)
    box_px = _clip_rect_to_image(box_px, width, height)
    if box_px is not None:
        ax.add_patch(
            Rectangle(
                (box_px[0], box_px[1]),
                box_px[2] - box_px[0],
                box_px[3] - box_px[1],
                fill=False,
                edgecolor="#ff9100",
                linewidth=2.8,
                zorder=6,
            )
        )

    gt_pts = _unit_to_px(gt[0], width, height)
    gt_valid = mask[0]
    gt_pts = gt_pts[gt_valid]
    gt_inside = _inside_image_mask(gt_pts, width, height)
    gt_pts = gt_pts[gt_inside]
    if len(gt_pts) >= 2:
        ax.plot(gt_pts[:, 0], gt_pts[:, 1], color="white", linewidth=2.0, alpha=0.35, linestyle="--", zorder=2)

    if show_modes:
        for mode_idx in range(mu.shape[0]):
            mode_pts = _unit_to_px(mu[mode_idx, 0], width, height)
            valid = _inside_image_mask(mode_pts, width, height)
            if not valid.any():
                continue
            color = "#ffd600"
            alpha = 1.0 if mode_idx == shown_mode else 0.22
            linewidth = 3.0 if mode_idx == shown_mode else 2.0
            ax.plot(
                mode_pts[valid, 0],
                mode_pts[valid, 1],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                zorder=5 if mode_idx == shown_mode else 4,
            )
            ax.scatter(
                mode_pts[valid, 0],
                mode_pts[valid, 1],
                s=20 if mode_idx == shown_mode else 12,
                c=color,
                alpha=alpha,
                edgecolors="white",
                linewidths=0.4,
                zorder=6 if mode_idx == shown_mode else 5,
            )
            end_x, end_y = mode_pts[valid][-1]
            ax.scatter(
                [end_x],
                [end_y],
                s=42 if mode_idx == shown_mode else 20,
                c=color,
                alpha=alpha,
                edgecolors="white",
                linewidths=0.5,
                zorder=7 if mode_idx == shown_mode else 5,
            )

    pred_maneuver = "unknown"
    gt_maneuver = "unknown"
    if len(gt_pts) >= 2:
        gt_dx = float(gt_pts[-1, 0] - gt_pts[0, 0])
        if gt_dx < -0.12 * width:
            gt_maneuver = "left"
        elif gt_dx > 0.12 * width:
            gt_maneuver = "right"
        else:
            gt_maneuver = "straight"
    shown_pts = _unit_to_px(mu[shown_mode, 0], width, height)
    shown_valid = _inside_image_mask(shown_pts, width, height)
    if shown_valid.any():
        shown_pts = shown_pts[shown_valid]
        dx = float(shown_pts[-1, 0] - shown_pts[0, 0])
        if dx < -0.12 * width:
            pred_maneuver = "left"
        elif dx > 0.12 * width:
            pred_maneuver = "right"
        else:
            pred_maneuver = "straight"

    panel_lines = [
        f"{scene_name}",
        f"track={track_id}",
        f"sample {sample_pos + 1}/{total_samples}  step {step_idx}/{total_steps}  kind={frame_kind}",
        f"shown_mode={shown_mode}  top={top_mode}  gt={gt_maneuver}  pred={pred_maneuver}",
    ]
    ax.text(
        0.015,
        0.985,
        "\n".join(panel_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        color="white",
        bbox=dict(facecolor="black", alpha=0.45, pad=3, edgecolor="none"),
        zorder=10,
    )

    if heatmap_alpha > 0:
        # Deliberately no heatmap overlay: the request is to keep annotations high quality
        # and focus on clean mode/trajectory rendering with Matplotlib only.
        pass

    canvas = FigureCanvas(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(rgba).convert("RGB")


def _render_sample_sequence(
    model,
    dataset,
    sample_index: int,
    sample_pos: int,
    total_samples: int,
    scene_name: str,
    track_id: str,
    num_steps: Optional[int],
    gt_only_frames: int,
    mode_selection: str,
    heatmap_alpha: int,
    sample_seed: Optional[int] = None,
):
    batch = _sample_to_batch(dataset, sample_index)
    device = next(model.parameters()).device
    batch_d = _batch_to_device(batch, device)

    if sample_seed is not None:
        torch.manual_seed(int(sample_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(sample_seed))

    with torch.no_grad():
        final_params, frames = model(batch_d, num_sampling_steps=num_steps, capture_steps=True)

    if not frames:
        frames = [{"sigma": 0.0, "params": final_params}]

    images: List[Image.Image] = []
    initial_frames = max(1, int(gt_only_frames))
    for i in range(initial_frames):
        images.append(
            _render_single_frame(
                batch,
                final_params,
                step_idx=0,
                total_steps=len(frames),
                sample_pos=sample_pos,
                total_samples=total_samples,
                scene_name=scene_name,
                track_id=track_id,
                show_modes=False,
                mode_selection=mode_selection,
                frame_kind="context",
                heatmap_alpha=heatmap_alpha,
            )
        )

    for step_idx, frame in enumerate(frames, start=1):
        images.append(
            _render_single_frame(
                batch,
                frame["params"],
                step_idx=step_idx,
                total_steps=len(frames),
                sample_pos=sample_pos,
                total_samples=total_samples,
                scene_name=scene_name,
                track_id=track_id,
                show_modes=True,
                mode_selection=mode_selection,
                frame_kind="refine",
                heatmap_alpha=heatmap_alpha,
            )
        )

    return images


def _save_gif(images: Sequence[Image.Image], output_path: Path, frame_ms: int):
    if not images:
        raise ValueError(f"No images were rendered for {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output_path, save_all=True, append_images=list(images[1:]), duration=frame_ms, loop=0, optimize=False)


def render_selected_tracks(
    config: str = DEFAULT_CONFIG,
    checkpoint: str = DEFAULT_CHECKPOINT,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    device: Optional[str] = None,
    num_steps: Optional[int] = None,
    frame_ms: int = 450,
    gt_only_frames: int = 1,
    mode_selection: str = "best",
    heatmap_alpha: int = 0,
    max_samples_per_track: int = 6,
    seed: Optional[int] = None,
):
    device_obj = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, dataset = load_model(config, checkpoint, device_obj)
    output_root = Path(output_dir)

    for scene_name, track_id in TRACK_IDS_BY_SCENE.items():
        _force_dataset_to_scene(dataset, scene_name)
        track_indices = _collect_track_indices(dataset, track_id, max_samples=max_samples_per_track)
        if not track_indices:
            print(f"{scene_name}: no samples found for {track_id}", flush=True)
            continue

        print(f"{scene_name}: rendering {len(track_indices)} samples for {track_id}", flush=True)
        track_images: List[Image.Image] = []
        for sample_pos, sample_index in enumerate(track_indices):
            batch = _sample_to_batch(dataset, sample_index)
            sample_id = str(batch.get("trajectory_row_id", [""])[0])
            print(f"  sample {sample_pos}: index={sample_index} trajectory_row_id={sample_id}", flush=True)
            sequence_images = _render_sample_sequence(
                model,
                dataset,
                sample_index,
                sample_pos,
                total_samples=len(track_indices),
                scene_name=scene_name,
                track_id=track_id,
                num_steps=num_steps,
                gt_only_frames=gt_only_frames,
                mode_selection=mode_selection,
                heatmap_alpha=heatmap_alpha,
                sample_seed=None if seed is None else int(seed) + int(sample_index),
            )
            track_images.extend(sequence_images)

        out_path = output_root / f"{_sanitize_filename(track_id)}.gif"
        print(f"  saving -> {out_path}", flush=True)
        _save_gif(track_images, out_path, frame_ms=frame_ms)


def main():
    parser = argparse.ArgumentParser(description="Render concatenated refinement GIFs for selected Waymo tracks.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_steps", type=int, default=8)
    parser.add_argument("--frame_ms", type=int, default=250)
    parser.add_argument("--gt_only_frames", type=int, default=1)
    parser.add_argument("--mode_selection", choices=["best", "top"], default="top")
    parser.add_argument("--heatmap_alpha", type=int, default=0)
    parser.add_argument("--max_samples_per_track", type=int, default=60)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    render_selected_tracks(
        config=args.config,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        num_steps=args.num_steps,
        frame_ms=args.frame_ms,
        gt_only_frames=args.gt_only_frames,
        mode_selection=args.mode_selection,
        heatmap_alpha=args.heatmap_alpha,
        max_samples_per_track=args.max_samples_per_track,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
