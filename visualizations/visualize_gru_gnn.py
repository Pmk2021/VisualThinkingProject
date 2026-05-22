#!/usr/bin/env python3
"""Render refinement-style visualizations for GRU and GNN polynomial GMMs.

GRU and GNN predict polynomial coefficients for image-space bbox centers rather
than EDM world-space trajectories. This script keeps RGB rendering in that
native pixel coordinate system and builds an approximate world-space copy using
the RGB context projector so the BEV GIF and occupancy heatmap can reuse the
renderers from ``visualize_refinement.py``.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import yaml
from box import Box
from PIL import Image, ImageDraw

PROJECT_ROOT = next(
    p for p in Path(__file__).resolve().parents
    if (p / "AnytimeTrajectoryPredictor").is_dir()
)
sys.path.insert(0, str(PROJECT_ROOT))

from AnytimeTrajectoryPredictor.Data.feature_extractor import LATENT_FEATURE_COLS
from AnytimeTrajectoryPredictor.models.TrajectoryPredictor import TrajectoryPredictor

import visualizations.visualize_polynomial_gmm as poly
import visualizations.visualize_refinement as refinement


BBOX_FEATURES = ["bbox_center_x", "bbox_center_y", "bbox_width", "bbox_height"]
LATENT_KEYS = ["trajectory_row_id", "frame_timestamp_micros"] + LATENT_FEATURE_COLS


def _config(path: str) -> Box:
    with open(path, "r", encoding="utf-8") as handle:
        return Box(yaml.safe_load(handle))


def _cfg_value(box: Any, *keys: str) -> Any | None:
    value = box
    for key in keys:
        value = getattr(value, key, None)
        if value is None:
            return None
    return value


def _dataset_root(cfg: Box, override: str | None) -> str:
    root = override or _cfg_value(cfg, "feature_extractor", "waymo_root")
    root = root or _cfg_value(cfg, "feature_extractor", "dataset_root")
    if root is None:
        raise ValueError("Provide --waymo_root or set feature_extractor.dataset_root in the config")
    return str(root)


def _feature_root(cfg: Box, override: str | None) -> Path:
    root = override or _cfg_value(cfg, "feature_extractor", "feature_root")
    root = root or _cfg_value(cfg, "feature_extractor", "dataset_root")
    if root is None:
        raise ValueError("Provide --feature_root or a feature_extractor dataset root in the config")
    return Path(str(root))


def _latent_candidates(cfg: Box, feature_root: Path, seg_path: Path) -> list[Path]:
    names = []
    configured = _cfg_value(cfg, "feature_extractor", "tables", "fe_gt_local_latent_features")
    configured = configured or _cfg_value(cfg, "feature_extractor", "tables", "latent_features")
    if configured:
        names.append(str(configured))
    names.extend(["fe_gt_local_latent_features.parquet", "fe_local_latent_features.parquet"])
    seen = set()
    candidates = []
    for name in names:
        for path in (feature_root / seg_path.name / name, seg_path / name):
            key = str(path)
            if key not in seen:
                seen.add(key)
                candidates.append(path)
    return candidates


def _read_latents(cfg: Box, feature_root: Path, seg_path: Path) -> dict[tuple[str, int], np.ndarray]:
    for path in _latent_candidates(cfg, feature_root, seg_path):
        if not path.exists():
            continue
        table = refinement._read_table_existing(path, LATENT_KEYS)
        if not set(LATENT_KEYS).issubset(table.schema.names):
            continue
        data = table.to_pydict()
        rows = {}
        for idx, rid in enumerate(data["trajectory_row_id"]):
            key = (str(rid), int(data["frame_timestamp_micros"][idx]))
            rows[key] = np.asarray([data[col][idx] for col in LATENT_FEATURE_COLS], dtype=np.float32)
        if rows:
            return rows
    checked = ", ".join(str(path) for path in _latent_candidates(cfg, feature_root, seg_path))
    raise FileNotFoundError(f"No usable local latent feature parquet found. Checked: {checked}")


def _camera_links(seg_path: Path, camera_name: int | None) -> list[dict]:
    table = refinement._read_table_existing(seg_path / "image_trajectories.parquet", refinement.LINK_COLS)
    data = table.to_pydict()
    links = [{key: data[key][idx] for key in data} for idx in range(len(data.get("image_id", [])))]
    if camera_name is not None and "camera_name" in data:
        links = [link for link in links if int(link.get("camera_name", -1)) == int(camera_name)]
    return links


def _feature_history(
    seg_path: Path,
    agents: list[dict],
    latents: dict[tuple[str, int], np.ndarray],
    camera_name: int | None,
    window: int,
    anchor_timestamp: int,
) -> tuple[torch.Tensor, list[dict], list[int]]:
    """Build (window, 1, N, 196) features and keep agents with anchor inputs."""
    wanted_ids = {str(agent.get("trajectory_row_id")) for agent in agents}
    links_by_key = {}
    time_counts = defaultdict(int)
    for link in _camera_links(seg_path, camera_name):
        rid = str(link.get("trajectory_row_id"))
        timestamp = int(link.get("frame_timestamp_micros", -1))
        key = (rid, timestamp)
        if rid not in wanted_ids or timestamp > anchor_timestamp or key not in latents:
            continue
        links_by_key[key] = link
        time_counts[timestamp] += 1

    anchor_agents = [
        agent for agent in agents
        if (str(agent.get("trajectory_row_id")), anchor_timestamp) in links_by_key
    ]
    if not anchor_agents:
        raise RuntimeError("Selected agents have no camera link and latent feature at the prediction timestamp")

    times = sorted(time for time in time_counts if time <= anchor_timestamp)[-window:]
    if anchor_timestamp not in times:
        times.append(anchor_timestamp)
        times = sorted(times)[-window:]
    if not times:
        raise RuntimeError("Could not assemble a timestamp window for GRU/GNN inputs")

    N = len(anchor_agents)
    features = np.zeros((window, 1, N, 196), dtype=np.float32)
    padded_times = [None] * (window - len(times)) + times
    for time_idx, timestamp in enumerate(padded_times):
        if timestamp is None:
            continue
        for agent_idx, agent in enumerate(anchor_agents):
            key = (str(agent.get("trajectory_row_id")), int(timestamp))
            link = links_by_key.get(key)
            latent = latents.get(key)
            if link is None or latent is None:
                continue
            bbox = [float(link.get(col, 0.0) or 0.0) for col in BBOX_FEATURES]
            features[time_idx, 0, agent_idx, :4] = bbox
            features[time_idx, 0, agent_idx, 4:] = latent
    return torch.from_numpy(features), anchor_agents, times


def _run_budget_predictions(
    model: torch.nn.Module,
    features: torch.Tensor,
    steps: int,
    device: torch.device,
) -> list[tuple[int, torch.Tensor, torch.Tensor, poly.PolynomialGMMSpec]]:
    spec = poly._spec_from_model(model)
    features = features.to(device)
    out = []
    with torch.no_grad():
        for budget in range(1, max(1, int(steps)) + 1):
            predictions = model(features, [budget] * int(features.shape[0]))
            means, covs = poly.extract_polynomial_gmm_params(predictions, spec)
            out.append((budget, means[:, -1].cpu(), covs[:, -1].cpu(), spec))
    return out


def _unproject_points(projector: dict, image_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(image_xy, dtype=np.float64)
    flat = pts.reshape(-1, 2)
    if projector["type"] in {"affine", "temporal_affine"}:
        B = projector["matrix"]
        jac = refinement._projection_jacobian(projector, np.zeros(2))
        offset = np.asarray(B[-1, :2], dtype=np.float64)
        world = (flat - offset) @ np.linalg.pinv(jac).T
    else:
        H_inv = np.linalg.pinv(projector["matrix"])
        uv1 = np.column_stack([flat, np.ones(len(flat))])
        xyw = uv1 @ H_inv.T
        denom = np.where(np.abs(xyw[:, 2:3]) < 1e-8, np.nan, xyw[:, 2:3])
        world = xyw[:, :2] / denom
    return world.reshape(*pts.shape[:-1], 2)


def _unprojection_jacobian(projector: dict, image_xy: np.ndarray, eps: float = 1.0) -> np.ndarray:
    center = _unproject_points(projector, np.asarray(image_xy, dtype=np.float64)[None])[0]
    dx = _unproject_points(projector, (np.asarray(image_xy) + np.array([eps, 0.0]))[None])[0]
    dy = _unproject_points(projector, (np.asarray(image_xy) + np.array([0.0, eps]))[None])[0]
    return np.column_stack([(dx - center) / eps, (dy - center) / eps])


def _cholesky(cov: np.ndarray, min_var: float) -> np.ndarray:
    cov = np.nan_to_num(0.5 * (cov + cov.T), nan=0.0, posinf=0.0, neginf=0.0)
    vals, vecs = np.linalg.eigh(cov)
    cov = vecs @ np.diag(np.maximum(vals, min_var)) @ vecs.T
    return np.linalg.cholesky(cov + np.eye(2) * min_var)


def _world_history_and_future(agent: dict) -> tuple[np.ndarray, np.ndarray]:
    features = agent["features"][0].numpy()
    observed_mask = agent["observed_mask"][0].numpy().astype(bool)
    trajectory = agent["trajectory"][0].numpy()
    future_mask = agent["future_mask"][0].numpy().astype(bool)
    hist = features[:, :2].copy()
    fut = trajectory.copy()
    hist[~observed_mask] = np.nan
    fut[~future_mask] = np.nan
    return (
        refinement.to_world(hist, agent["anchor_x"], agent["anchor_y"], agent["anchor_heading"]),
        refinement.to_world(fut, agent["anchor_x"], agent["anchor_y"], agent["anchor_heading"]),
    )


def _pixel_gmm_frame(
    budget: int,
    means: torch.Tensor,
    covs: torch.Tensor,
    spec: poly.PolynomialGMMSpec,
    agents: list[dict],
    anchor_ctx: dict,
    future_points: int,
) -> dict:
    adata = []
    projector = anchor_ctx["projector"]
    for agent_idx, agent in enumerate(agents):
        rid = agent.get("trajectory_row_id")
        offset = np.asarray(anchor_ctx.get("agent_offsets", {}).get(rid, np.zeros(2)), dtype=np.float64)
        mode_px = []
        chol_px = []
        mode_world = []
        chol_world = []
        for mode_idx in range(spec.num_modes):
            coeffs = means[0, agent_idx, mode_idx].numpy()
            coeff_cov = covs[0, agent_idx, mode_idx].numpy()
            curve_px = poly.evaluate_xy_coeffs(coeffs, future_points)
            cov_px = poly._xy_covariances(coeff_cov, future_points, spec.num_coeffs)
            curve_world = _unproject_points(projector, curve_px - offset)
            world_covs = []
            for point_px, point_cov_px in zip(curve_px - offset, cov_px):
                jac = _unprojection_jacobian(projector, point_px)
                world_covs.append(jac @ point_cov_px @ jac.T)
            mode_px.append(curve_px)
            chol_px.append(np.stack([_cholesky(cov, min_var=1.0) for cov in cov_px]))
            mode_world.append(curve_world)
            chol_world.append(np.stack([_cholesky(cov, min_var=1e-4) for cov in world_covs]))

        hist_w, fut_w = _world_history_and_future(agent)
        adata.append({
            "mu": np.stack(mode_world),
            "mu_px": np.stack(mode_px),
            "mode_probs": np.full(spec.num_modes, 1.0 / spec.num_modes, dtype=np.float64),
            "cov_chol_w": np.stack(chol_world),
            "cov_chol_px": np.stack(chol_px),
            "history_w": hist_w,
            "future_w": fut_w,
            "anchor_x": agent["anchor_x"],
            "anchor_y": agent["anchor_y"],
            "anchor_heading": agent["anchor_heading"],
            "anchor_vx": agent.get("anchor_vx", 0.0),
            "anchor_vy": agent.get("anchor_vy", 0.0),
            "anchor_timestamp": agent.get("anchor_timestamp"),
            "anchor_index": agent.get("anchor_index"),
            "trajectory_id": agent.get("trajectory_id", ""),
            "trajectory_row_id": rid,
            "scene_id": agent.get("scene_id", ""),
            "object_type": agent["object_type"],
            "box_length": agent.get("box_length"),
            "box_width": agent.get("box_width"),
        })
    return {
        "sigma": 0.0,
        "title": f"GRU/GNN polynomial GMM budget {budget} | {len(adata)} agents",
        "budget": budget,
        "agents_data": adata,
    }


def _view_box(agents: list[dict]) -> tuple[float, float, float]:
    anchors = np.asarray([[agent["anchor_x"], agent["anchor_y"]] for agent in agents], dtype=np.float64)
    cx, cy = anchors.mean(axis=0)
    spread = float(np.max(np.linalg.norm(anchors - np.array([cx, cy]), axis=1)))
    return float(cx), float(cy), max(spread * 1.5 + 30.0, 40.0)


def _save_bev_gif(frames: list[dict], agents: list[dict], output: str, frame_ms: int):
    cx, cy, bev_half = _view_box(agents)
    images = [
        refinement.render_frame(frame, idx, len(frames) - 1, idx == len(frames) - 1, cx, cy, bev_half)
        for idx, frame in enumerate(frames)
    ]
    images += [images[-1]] * 2
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output, save_all=True, append_images=images[1:], duration=frame_ms, loop=0, optimize=False)
    print(f"BEV GIF saved -> {output}")


def _save_gmm_png(frame: dict, agents: list[dict], output: str, min_std_m: float):
    cx, cy, bev_half = _view_box(agents)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    refinement.render_gmm_png(frame, cx, cy, bev_half, output_path=output, heatmap_min_std_m=min_std_m)


def _scaled_xy(points: np.ndarray, scale: float) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) * float(scale)


def _draw_pixel_history(draw: ImageDraw.ImageDraw, ctx: dict, ad: dict, color: tuple[int, int, int]):
    refinement._draw_history_dots(
        draw, ad, ctx["projector"], float(ctx["pixel_scale"]), ctx["image"].size, color, ctx=ctx
    )


def _render_rgb_frame(frame: dict, ctx: dict, step_idx: int, total_steps: int) -> Image.Image:
    base = ctx["image"].convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    w, h = base.size
    scale = float(ctx["pixel_scale"])
    for agent_idx, ad in enumerate(frame["agents_data"]):
        rid = ad.get("trajectory_row_id")
        link = ctx["links_by_agent"].get(rid)
        if link is None:
            continue
        color = refinement._rgb_refinement_color(agent_idx)
        refinement._draw_scaled_box(draw, link, scale, color, width=4, alpha=245)
        _draw_pixel_history(draw, ctx, ad, color)
        probs = np.asarray(ad["mode_probs"], dtype=np.float64)
        for mode_idx in np.argsort(probs):
            points = _scaled_xy(ad["mu_px"][mode_idx], scale)
            finite = np.isfinite(points).all(axis=1)
            points = points[finite]
            if len(points) < 2:
                continue
            points[:, 0] = np.clip(points[:, 0], -w, 2 * w)
            points[:, 1] = np.clip(points[:, 1], -h, 2 * h)
            prob = float(probs[mode_idx])
            width = max(3, int(round(3 + 7 * prob)))
            alpha = int(np.clip(140 + 115 * prob, 140, 255))
            for segment in refinement._projected_polyline_segments(points, max_jump_px=max(w, h) * 0.45):
                draw.line([tuple(point) for point in segment], fill=color + (alpha,), width=width, joint="curve")
            ex, ey = points[-1]
            radius = 3.0 + 7.0 * prob
            draw.ellipse([ex - radius, ey - radius, ex + radius, ey + radius],
                         fill=color + (alpha,), outline=(255, 255, 255, 230), width=2)
    title = f"polynomial GMM budget {frame.get('budget', step_idx + 1)}/{max(1, total_steps + 1)}"
    draw.rectangle([8, 8, min(360, w - 8), 34], fill=(0, 0, 0, 120))
    draw.text((16, 14), title, fill=(255, 255, 255, 235))
    return Image.alpha_composite(base, overlay).convert("RGB")


def _save_rgb_gif(frames: list[dict], ctx: dict, output: str, frame_ms: int):
    images = [_render_rgb_frame(frame, ctx, idx, len(frames) - 1) for idx, frame in enumerate(frames)]
    images += [images[-1]] * 2
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output, save_all=True, append_images=images[1:], duration=frame_ms, loop=0, optimize=False)
    print(f"RGB GIF saved -> {output}")


def _pixel_heatmap(frame: dict, ctx: dict, future_step: int, min_std_px: float) -> np.ndarray:
    w, h = ctx["image"].size
    scale = float(ctx["pixel_scale"])
    xs = np.arange(w, dtype=np.float64)
    ys = np.arange(h, dtype=np.float64)
    density = np.zeros((h, w), dtype=np.float64)
    step_idx = min(max(0, int(future_step) - 1), frame["agents_data"][0]["mu_px"].shape[1] - 1)
    for ad in frame["agents_data"]:
        if ad.get("trajectory_row_id") not in ctx["links_by_agent"]:
            continue
        probs = np.asarray(ad["mode_probs"], dtype=np.float64)
        for mode_idx, prob in enumerate(probs):
            mu_px = _scaled_xy(ad["mu_px"][mode_idx, step_idx], scale)
            chol = ad["cov_chol_px"][mode_idx, step_idx] * scale
            refinement._add_gaussian_to_heatmap(
                density, xs, ys, mu_px, chol @ chol.T, float(prob), min_std=min_std_px
            )
    return density


def _save_rgb_heatmap(
    frame: dict,
    ctx: dict,
    output: str,
    future_step: int,
    heatmap_alpha: float,
    min_std_px: float,
):
    base = ctx["image"].convert("RGBA")
    density = _pixel_heatmap(frame, ctx, future_step, min_std_px)
    if density.max() > 0:
        scale_den = np.percentile(density[density > 0], 98.5)
        norm = np.clip(density / max(scale_den, 1e-12), 0.0, 1.0)
        rgba = matplotlib.colormaps["inferno"](norm)
        rgba[..., 3] = heatmap_alpha * np.power(norm, 0.42)
        base = Image.alpha_composite(base, Image.fromarray((rgba * 255).astype(np.uint8), mode="RGBA"))
    draw = ImageDraw.Draw(base, "RGBA")
    scale = float(ctx["pixel_scale"])
    for agent_idx, ad in enumerate(frame["agents_data"]):
        rid = ad.get("trajectory_row_id")
        link = ctx["links_by_agent"].get(rid)
        if link is None:
            continue
        color = refinement._rgb_refinement_color(agent_idx)
        _draw_pixel_history(draw, ctx, ad, color)
        refinement._draw_scaled_box(draw, link, scale, color, width=3, alpha=245)
    w, _ = base.size
    label = (
        f"RGB @ prediction+{future_step} | heatmap: polynomial predicted occupancy @ +{future_step} | "
        "dots: history | boxes: target objects"
    )
    draw.rectangle([8, 8, min(w - 8, 790), 38], fill=(0, 0, 0, 135))
    draw.text((16, 16), label, fill=(255, 255, 255, 235))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    base.convert("RGB").save(output)
    print(f"RGB heatmap saved -> {output}")


def _load_scene_and_frames(args, device: torch.device):
    cfg = _config(args.config)
    waymo_root = _dataset_root(cfg, args.waymo_root)
    feature_root = _feature_root(cfg, args.feature_root)
    if args.checkpoint:
        model = poly._load_model(args.config, args.checkpoint, device)
    else:
        model = TrajectoryPredictor.create_model(cfg).to(device).eval()
        print("Checkpoint omitted; using random model parameters for visualization smoke testing.")
    model_type = str(_cfg_value(cfg, "model", "type"))
    if model_type.lower() not in {"gru", "gnn"}:
        raise ValueError(f"Expected a GRU or GNN config, got model.type={model_type!r}")

    seg_path, scene_id, agents = refinement.find_scene(
        waymo_root,
        max_segs=args.max_segments,
        min_agents=args.min_agents,
        max_agents=args.max_agents,
        H=args.history_steps,
        T=args.future_points,
        scene_id=str(args.scene_id) if args.scene_id else None,
        anchor_timestamp=args.anchor_timestamp,
    )
    anchor_times = [agent["anchor_timestamp"] for agent in agents if agent.get("anchor_timestamp") is not None]
    if not anchor_times:
        raise RuntimeError("Scene selection did not provide a prediction timestamp")
    timestamp = int(max(set(anchor_times), key=anchor_times.count))
    ctx = refinement._load_rgb_context(
        seg_path, agents, camera_name=args.rgb_camera, max_width=args.rgb_max_width,
        target_timestamp=timestamp, require_selected_visible=True,
    )
    if ctx is None:
        raise RuntimeError("Could not build a prediction-time RGB context for the selected scene")

    latents = _read_latents(cfg, feature_root, seg_path)
    feature_history, agents, used_times = _feature_history(
        seg_path, agents, latents, args.rgb_camera, args.window, timestamp
    )
    ctx = refinement._load_rgb_context(
        seg_path, agents, camera_name=args.rgb_camera, max_width=args.rgb_max_width,
        target_timestamp=timestamp, require_selected_visible=True,
    )
    if ctx is None:
        raise RuntimeError("No selected GRU/GNN input agents are visible in prediction-time RGB")
    budgets = _run_budget_predictions(model, feature_history, args.num_steps, device)
    frames = [
        _pixel_gmm_frame(budget, means, covs, spec, agents, ctx, args.future_points)
        for budget, means, covs, spec in budgets
    ]
    print(
        f"Scene: {scene_id} | agents={len(agents)} | timestamp={timestamp} | "
        f"input_frames={len(used_times)} | projector={ctx['projector']['type']} "
        f"residual={ctx['projector']['residual_px']:.1f}px"
    )
    return seg_path, agents, ctx, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint; omit for random-weight smoke tests")
    parser.add_argument("--output", default="scene_refinement.gif")
    parser.add_argument("--output_gmm", default="gmm_output.png")
    parser.add_argument("--output_rgb", default="rgb_refinement.gif")
    parser.add_argument("--output_rgb_final", default="rgb_heatmap.png")
    parser.add_argument("--mode", default="all", choices=["gif", "gmm", "rgb", "rgb_gif", "both", "all"])
    parser.add_argument("--scene_id", default=None)
    parser.add_argument("--anchor_timestamp", type=int, default=None)
    parser.add_argument("--waymo_root", default=None)
    parser.add_argument("--feature_root", default=None)
    parser.add_argument("--min_agents", type=int, default=3)
    parser.add_argument("--max_agents", type=int, default=5)
    parser.add_argument("--max_segments", type=int, default=5)
    parser.add_argument("--window", type=int, default=10, help="GRU/GNN feature frames fed to the model")
    parser.add_argument("--history_steps", type=int, default=10, help="World-space GT history steps drawn in BEV/RGB")
    parser.add_argument("--future_points", type=int, default=90, help="Polynomial samples drawn per mode")
    parser.add_argument("--num_steps", type=int, default=3, help="Inference budgets rendered into GIF frames")
    parser.add_argument("--frame_ms", type=int, default=450)
    parser.add_argument("--device", default=None)
    parser.add_argument("--rgb_camera", type=int, default=1)
    parser.add_argument("--rgb_max_width", type=int, default=1280)
    parser.add_argument("--rgb_heatmap_future_step", type=int, default=5)
    parser.add_argument("--gmm_heatmap_min_std_m", type=float, default=1.75)
    parser.add_argument("--rgb_heatmap_alpha", type=float, default=0.62)
    parser.add_argument("--rgb_heatmap_min_std_px", type=float, default=8.0)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    seg_path, agents, anchor_ctx, frames = _load_scene_and_frames(args, device)
    if args.mode in {"gif", "both", "all"}:
        _save_bev_gif(frames, agents, args.output, args.frame_ms)
    if args.mode in {"gmm", "both", "all"}:
        output = args.output_gmm if args.mode in {"both", "all"} else args.output
        _save_gmm_png(frames[-1], agents, output, args.gmm_heatmap_min_std_m)
    if args.mode in {"rgb", "rgb_gif", "all"}:
        _save_rgb_gif(frames, anchor_ctx, args.output_rgb, args.frame_ms)
        if args.mode != "rgb_gif":
            timestamp = refinement._target_future_timestamp(seg_path, agents, args.rgb_heatmap_future_step)
            heatmap_ctx = refinement._load_rgb_context(
                seg_path, agents, camera_name=args.rgb_camera, max_width=args.rgb_max_width,
                target_timestamp=timestamp, require_selected_visible=False,
            ) or anchor_ctx
            _save_rgb_heatmap(
                frames[-1], heatmap_ctx, args.output_rgb_final, args.rgb_heatmap_future_step,
                args.rgb_heatmap_alpha, args.rgb_heatmap_min_std_px,
            )


if __name__ == "__main__":
    main()
