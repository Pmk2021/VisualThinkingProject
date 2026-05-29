#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from box import Box
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import DataLoader

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AnytimeTrajectoryPredictor").is_dir())
sys.path.insert(0, str(PROJECT_ROOT))

from AnytimeTrajectoryPredictor.Data.feature_extractor import WaymoImagePlaneDataset
from AnytimeTrajectoryPredictor.models.TrajectoryPredictor import TrajectoryPredictor


MODE_COLORS = [
    (0, 229, 255),
    (255, 109, 0),
    (0, 255, 76),
    (255, 23, 68),
    (213, 0, 249),
    (255, 255, 0),
]

AGENT_COLORS = [
    (255, 235, 59),
    (0, 229, 255),
    (255, 109, 0),
    (0, 255, 76),
    (213, 0, 249),
    (255, 23, 68),
    (124, 255, 0),
    (255, 255, 255),
]


def _unit_to_px(points, width, height):
    pts = np.asarray(points, dtype=np.float32)
    out = np.empty_like(pts)
    out[..., 0] = (pts[..., 0] + 1.0) * 0.5 * width
    out[..., 1] = (pts[..., 1] + 1.0) * 0.5 * height
    return out


def _pil_from_history(batch):
    rgb = batch["rgb_history"][0, -1].detach().cpu().clamp(0, 1)
    arr = (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGBA")




def _box_unit_to_px(box, width, height):
    b = np.asarray(box, dtype=np.float32)
    cx = (float(b[0]) + 1.0) * 0.5 * width
    cy = (float(b[1]) + 1.0) * 0.5 * height
    bw = max(float(b[2]) * width, 2.0)
    bh = max(float(b[3]) * height, 2.0)
    return [cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5]


def _draw_target_boxes(draw, batch, width, height):
    if "box_history" not in batch:
        return
    boxes = batch["box_history"][0].detach().cpu().numpy()
    observed = batch.get("observed_mask")
    observed_np = observed[0].detach().cpu().numpy() > 0 if observed is not None else None
    for agent_idx in range(boxes.shape[0]):
        if observed_np is not None:
            valid = np.flatnonzero(observed_np[agent_idx])
            if len(valid) == 0:
                continue
            box = boxes[agent_idx, int(valid[-1])]
        else:
            box = boxes[agent_idx, -1]
        rect = _box_unit_to_px(box, width, height)
        color = AGENT_COLORS[agent_idx % len(AGENT_COLORS)]
        draw.rectangle(rect, outline=color + (255,), width=3)
        label = f"agent {agent_idx}" if boxes.shape[0] > 1 else "target box"
        label_w = max(74, 10 + 7 * len(label))
        label_y1 = max(0, rect[1])
        label_y0 = max(0, label_y1 - 16)
        if label_y1 > label_y0:
            draw.rectangle([rect[0], label_y0, rect[0] + label_w, label_y1], fill=(0, 0, 0, 150))
        draw.text((rect[0] + 4, max(0, label_y1 - 15)), label, fill=color + (255,))


def _turbo_color(values):
    v = np.clip(values, 0.0, 1.0)[..., None]
    colors = np.array(
        [
            [48, 18, 59],
            [50, 101, 181],
            [42, 174, 128],
            [246, 230, 32],
            [240, 101, 34],
            [122, 4, 3],
        ],
        dtype=np.float32,
    )
    x = v * (len(colors) - 1)
    lo = np.floor(x).astype(np.int32).clip(0, len(colors) - 1)
    hi = np.ceil(x).astype(np.int32).clip(0, len(colors) - 1)
    frac = x - lo
    return (colors[lo[..., 0]] * (1.0 - frac) + colors[hi[..., 0]] * frac).astype(np.uint8)


def _gmm_heatmap(params, width, height, shown_mode=None, modes="all", max_alpha=105, stride=4, timesteps=None):
    if modes == "off" or not hasattr(params, "cov_cholesky"):
        return None
    mu = params.mu[0].detach().cpu().numpy()
    probs = params.mode_probs[0].detach().cpu().numpy()
    chol = params.cov_cholesky[0].detach().cpu().numpy()
    if mu.ndim == 3:
        mu = mu[:, None]
        chol = chol[:, None]
    if modes == "shown" and shown_mode is not None:
        mode_ids = [int(shown_mode)]
    elif modes == "top":
        mode_ids = [int(np.argmax(probs))]
    else:
        mode_ids = list(range(mu.shape[0]))

    small_w = max(1, int(np.ceil(width / stride)))
    small_h = max(1, int(np.ceil(height / stride)))
    xs = np.linspace(-1.0, 1.0, small_w, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, small_h, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.stack([gx, gy], axis=-1)
    heat = np.zeros((small_h, small_w), dtype=np.float32)

    if timesteps is None:
        time_ids = range(mu.shape[2])
    elif isinstance(timesteps, (list, tuple, np.ndarray)):
        time_ids = [int(t) for t in timesteps if 0 <= int(t) < mu.shape[2]]
    else:
        t = int(timesteps)
        time_ids = [t] if 0 <= t < mu.shape[2] else []

    for mode_idx in mode_ids:
        weight = float(probs[mode_idx])
        if modes == "shown":
            weight = 1.0
        for agent_idx in range(mu.shape[1]):
            for t in time_ids:
                mean = mu[mode_idx, agent_idx, t]
                if not np.isfinite(mean).all():
                    continue
                L = chol[mode_idx, agent_idx, t].astype(np.float32)
                cov = L @ L.T
                cov[0, 0] = max(float(cov[0, 0]), 1e-4)
                cov[1, 1] = max(float(cov[1, 1]), 1e-4)
                cov[0, 1] = cov[1, 0] = float(np.clip(cov[0, 1], -0.08, 0.08))
                inv = np.linalg.pinv(cov)
                delta = grid - mean
                maha = delta[..., 0] * (inv[0, 0] * delta[..., 0] + inv[0, 1] * delta[..., 1])
                maha += delta[..., 1] * (inv[1, 0] * delta[..., 0] + inv[1, 1] * delta[..., 1])
                heat += weight * np.exp(-0.5 * np.clip(maha, 0.0, 40.0)).astype(np.float32)

    if float(heat.max()) <= 1e-8:
        return None
    heat = heat / float(heat.max())
    rgb = _turbo_color(heat)
    alpha = (np.clip(heat, 0.0, 1.0) ** 0.65 * int(max_alpha)).astype(np.uint8)
    arr = np.dstack([rgb, alpha])
    image = Image.fromarray(arr, mode="RGBA").resize((width, height), resample=Image.Resampling.BILINEAR)
    return image.filter(ImageFilter.GaussianBlur(radius=1.0))


def _draw_poly(draw, pts, color, alpha=220, width=4):
    finite = np.isfinite(pts).all(axis=-1)
    pts = pts[finite]
    if len(pts) < 2:
        return
    draw.line([tuple(p) for p in pts], fill=(255, 255, 255, min(alpha, 120)), width=width + 2, joint="curve")
    draw.line([tuple(p) for p in pts], fill=color + (alpha,), width=width, joint="curve")
    x, y = pts[-1]
    r = max(3, width + 1)
    draw.ellipse([x - r, y - r, x + r, y + r], fill=color + (alpha,), outline=(255, 255, 255, 220), width=2)


def _normalize_knot_indices(knot_indices, horizon):
    if knot_indices is None:
        return None
    if hasattr(knot_indices, "detach"):
        knot_indices = knot_indices.detach().cpu().tolist()
    else:
        knot_indices = list(knot_indices)
    out = [int(i) for i in knot_indices if 0 <= int(i) < int(horizon)]
    return out or None


def _draw_knot_points(draw, mu_for_mode, knot_indices, width, height, color, radius=4):
    """Overlay the predicted knot positions for one mode on top of the polyline.

    `mu_for_mode` is `[A, T, 2]` in unit coords; `knot_indices` are integers in
    [0, T). Filled circles are drawn in the agent color, outlined in white so
    they pop on top of the spline.
    """
    if knot_indices is None:
        return
    for agent_idx in range(mu_for_mode.shape[0]):
        pts_unit = np.asarray([mu_for_mode[agent_idx, t] for t in knot_indices], dtype=np.float32)
        pts_px = _unit_to_px(pts_unit, width, height)
        for (px, py) in pts_px:
            if not (np.isfinite(px) and np.isfinite(py)):
                continue
            draw.ellipse(
                [px - radius, py - radius, px + radius, py + radius],
                fill=color + (235,),
                outline=(255, 255, 255, 235),
                width=2,
            )


def _maneuver(points):
    pts = np.asarray(points)
    valid = np.isfinite(pts).all(axis=-1)
    pts = pts[valid]
    if len(pts) < 2:
        return "unknown"
    dx = float(pts[-1, 0] - pts[0, 0])
    if dx < -0.12:
        return "left"
    if dx > 0.12:
        return "right"
    return "straight"


def _mode_ade(mu, gt, mask):
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


def _draw_gt_trajectories(draw, batch, width, height, line_width=3):
    gt = batch["trajectory"][0].detach().cpu().numpy()
    masks = batch["future_mask"][0].detach().cpu().numpy().astype(bool)
    for agent_idx in range(gt.shape[0]):
        pts = _unit_to_px(gt[agent_idx], width, height)
        pts[~masks[agent_idx]] = np.nan
        color = AGENT_COLORS[agent_idx % len(AGENT_COLORS)]
        _draw_poly(draw, pts, color, alpha=245, width=line_width)


def _render_gt_only(base, batch, step_idx, total_steps):
    image = base.copy()
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    width, height = image.size
    gt = batch["trajectory"][0].detach().cpu().numpy()
    mask = batch["future_mask"][0].detach().cpu().numpy().astype(bool)
    _draw_gt_trajectories(draw, batch, width, height, line_width=4)
    _draw_target_boxes(draw, batch, width, height)
    draw.rectangle([8, 8, 360, 54], fill=(0, 0, 0, 150))
    draw.text((16, 15), f"ground truth only {step_idx}/{total_steps}", fill=(255, 255, 255, 235))
    draw.text((16, 33), f"gt maneuver={_maneuver(gt[0][mask[0]])}", fill=(255, 255, 255, 235))
    return Image.alpha_composite(image, overlay).convert("RGB")


def _render_step(base, params, batch, step_idx, total_steps, sigma, mode_selection="best", gmm_heatmap="all", heatmap_alpha=105, knot_indices=None):
    image = base.copy()
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    width, height = image.size
    mu = params.mu[0].detach().cpu().numpy()
    probs = params.mode_probs[0].detach().cpu().numpy()
    gt = batch["trajectory"][0].detach().cpu().numpy()
    mask = batch["future_mask"][0].detach().cpu().numpy().astype(bool)
    if mu.ndim == 3:
        mu = mu[:, None]
    _draw_gt_trajectories(draw, batch, width, height, line_width=3)
    _draw_target_boxes(draw, batch, width, height)
    mode_ade = _mode_ade(mu, gt, mask)
    shown_mode = _select_shown_mode(params, batch, mode_selection=mode_selection)
    heatmap = _gmm_heatmap(params, width, height, shown_mode=shown_mode, modes=gmm_heatmap, max_alpha=heatmap_alpha)
    if heatmap is not None:
        overlay = Image.alpha_composite(overlay, heatmap)
        draw = ImageDraw.Draw(overlay, "RGBA")
    prob = float(probs[shown_mode])
    horizon = mu.shape[2]
    knots = _normalize_knot_indices(knot_indices, horizon)
    for agent_idx in range(mu.shape[1]):
        pts = _unit_to_px(mu[shown_mode, agent_idx], width, height)
        color = AGENT_COLORS[agent_idx % len(AGENT_COLORS)]
        _draw_poly(draw, pts, color, alpha=235, width=5)
    if knots is not None:
        for agent_idx in range(mu.shape[1]):
            color = AGENT_COLORS[agent_idx % len(AGENT_COLORS)]
            _draw_knot_points(draw, mu[shown_mode], knots, width, height, color, radius=5)
    entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0))))
    top_mode = int(np.argmax(probs))
    pred_maneuver = _maneuver(mu[shown_mode, 0])
    gt_maneuver = _maneuver(gt[0][mask[0]])
    minade = float(mode_ade[shown_mode])
    panel = [
        f"step {step_idx}/{total_steps}  sigma={sigma:.4f}",
        f"shown={shown_mode} p={prob:.2f} top={top_mode} entropy={entropy:.2f}",
        f"minADE(norm)={minade:.3f}  gt={gt_maneuver} pred={pred_maneuver}",
    ]
    draw.rectangle([8, 8, 430, 72], fill=(0, 0, 0, 150))
    for i, text in enumerate(panel):
        draw.text((16, 15 + i * 18), text, fill=(255, 255, 255, 235))
    return Image.alpha_composite(image, overlay).convert("RGB")


def _select_shown_mode(params, batch, mode_selection="best"):
    mu = params.mu[0].detach().cpu().numpy()
    probs = params.mode_probs[0].detach().cpu().numpy()
    gt = batch["trajectory"][0].detach().cpu().numpy()
    mask = batch["future_mask"][0].detach().cpu().numpy().astype(bool)
    if mu.ndim == 3:
        mu = mu[:, None]
    if mode_selection == "top":
        shown_mode = int(np.argmax(probs))
    else:
        shown_mode = int(np.nanargmin(_mode_ade(mu, gt, mask)))
    return shown_mode


def _render_gmm_png(
    base,
    params,
    batch,
    mode_selection="best",
    gmm_heatmap="all",
    heatmap_alpha=135,
    draw_observed_boxes=True,
    future_step=None,
    knot_indices=None,
):
    image = base.copy()
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    width, height = image.size
    shown_mode = _select_shown_mode(params, batch, mode_selection=mode_selection)
    heatmap = _gmm_heatmap(
        params,
        width,
        height,
        shown_mode=shown_mode,
        modes=gmm_heatmap,
        max_alpha=heatmap_alpha,
        timesteps=future_step,
    )
    if heatmap is not None:
        overlay = Image.alpha_composite(overlay, heatmap)
        draw = ImageDraw.Draw(overlay, "RGBA")
    if draw_observed_boxes:
        _draw_target_boxes(draw, batch, width, height)
    mu = params.mu[0].detach().cpu().numpy()
    if mu.ndim == 3:
        mu = mu[:, None]
    horizon = mu.shape[2]
    knots = _normalize_knot_indices(knot_indices, horizon)
    if knots is not None:
        for agent_idx in range(mu.shape[1]):
            color = AGENT_COLORS[agent_idx % len(AGENT_COLORS)]
            _draw_knot_points(draw, mu[shown_mode], knots, width, height, color, radius=5)
    draw.rectangle([8, 8, 430, 54], fill=(0, 0, 0, 150))
    step_text = "all steps" if future_step is None else f"future step={int(future_step)}"
    draw.text((16, 15), f"GMM heatmap  {step_text}  shown mode={shown_mode}", fill=(255, 255, 255, 235))
    draw.text((16, 33), f"heatmap={gmm_heatmap}  alpha={heatmap_alpha}", fill=(255, 255, 255, 235))
    return Image.alpha_composite(image, overlay).convert("RGB")


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/astra_edm_diffusion_waymo_image_plane_sanity.yml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="image_plane_refinement.gif")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--frame_ms", type=int, default=450)
    parser.add_argument("--gt_only_frames", type=int, default=2)
    parser.add_argument("--mode_selection", choices=["best", "top"], default="best")
    parser.add_argument("--gmm_heatmap", choices=["all", "shown", "top", "off"], default="all")
    parser.add_argument("--heatmap_alpha", type=int, default=105)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, dataset = load_model(args.config, args.checkpoint, device)
    loader = DataLoader(torch.utils.data.Subset(dataset, [args.sample_index]), batch_size=1, shuffle=False)
    batch = next(iter(loader))
    batch_d = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    with torch.no_grad():
        final_params, frames = model(batch_d, num_sampling_steps=args.num_steps, capture_steps=True)
    if not frames:
        frames = [{"sigma": 0.0, "params": final_params}]
    base = _pil_from_history(batch)
    images = [_render_gt_only(base, batch, i, args.gt_only_frames) for i in range(1, args.gt_only_frames + 1)]
    total = len(frames)
    knot_indices = getattr(model, "knot_indices", None) if getattr(model, "use_control_points", False) else None
    for i, frame in enumerate(frames, start=1):
        images.append(
            _render_step(
                base,
                frame["params"],
                batch,
                i,
                total,
                frame["sigma"],
                mode_selection=args.mode_selection,
                gmm_heatmap=args.gmm_heatmap,
                heatmap_alpha=args.heatmap_alpha,
                knot_indices=knot_indices,
            )
        )
    images += [images[-1]] * 2
    images[0].save(args.output, save_all=True, append_images=images[1:], duration=args.frame_ms, loop=0, optimize=False)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
