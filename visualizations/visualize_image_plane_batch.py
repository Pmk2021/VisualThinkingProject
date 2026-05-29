#!/usr/bin/env python3
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AnytimeTrajectoryPredictor").is_dir())
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from visualize_image_plane_diffusion import (
    AGENT_COLORS,
    _box_unit_to_px,
    _maneuver,
    _mode_ade,
    _pil_from_history,
    _render_gmm_png,
    _render_gt_only,
    _render_step,
    load_model,
)


MOVING_OBJECT_TYPES = {1, 2, 4}  # Waymo: vehicle, pedestrian, cyclist. Excludes signs.


def _parse_indices(value):
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)):
        parts = [str(part).strip() for part in value if str(part).strip()]
    else:
        parts = [part.strip() for part in str(value).split(",") if part.strip()]
    out = []
    for part in parts:
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def _cfg_get(section, key, default=None):
    return section.get(key, default) if isinstance(section, dict) else default


def _best_checkpoint_from_save_to(save_to):
    if not save_to:
        return None
    path = Path(save_to)
    suffix = path.suffix or ".pth"
    best = path.with_name(f"{path.stem}.best{suffix}")
    return str(best if best.exists() else path)


def _outs_output_dir(subdir):
    if subdir is None or str(subdir).strip() == "":
        return None
    rel = Path(str(subdir).strip())
    if rel.is_absolute() or any(part == ".." for part in rel.parts):
        raise ValueError("--outs_subdir must be a relative path inside visualizations/outs/.")
    return str(Path("visualizations") / "outs" / rel)


def _load_training_config(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("training", {}) or {}


def _apply_config_defaults(args):
    training = _load_training_config(args.config)
    args.checkpoint = args.checkpoint or _cfg_get(
        training,
        "visualization_checkpoint",
        _best_checkpoint_from_save_to(_cfg_get(training, "save_to", None)),
    )
    args.output_dir = args.output_dir or _cfg_get(training, "visualization_output_dir", "visualizations/image_plane_results")
    args.indices = args.indices if args.indices is not None else _cfg_get(training, "visualization_indices", None)
    args.start_index = 0 if args.start_index is None else args.start_index
    args.num_samples = int(args.num_samples if args.num_samples is not None else _cfg_get(training, "visualization_num_samples", 8))
    args.random = bool(args.random if args.random is not None else _cfg_get(training, "visualization_random", False))
    args.random_scene = bool(
        args.random_scene if args.random_scene is not None else _cfg_get(training, "visualization_random_scene", False)
    )
    args.random_indices = args.random_indices if args.random_indices is not None else _cfg_get(
        training, "visualization_random_indices", None
    )
    args.moving_only = bool(args.moving_only if args.moving_only is not None else _cfg_get(training, "visualization_moving_only", True))
    args.min_motion_px = float(
        args.min_motion_px if args.min_motion_px is not None else _cfg_get(training, "visualization_min_motion_px", 12.0)
    )
    args.seed = args.seed if args.seed is not None else _cfg_get(training, "visualization_seed", None)
    args.num_steps = args.num_steps if args.num_steps is not None else _cfg_get(training, "visualization_num_steps", None)
    args.show_initial_noise = bool(args.show_initial_noise if args.show_initial_noise is not None else _cfg_get(training, "visualization_show_initial_noise", False))
    args.frame_ms = int(args.frame_ms if args.frame_ms is not None else _cfg_get(training, "visualization_frame_ms", 450))
    args.gt_only_frames = int(
        args.gt_only_frames if args.gt_only_frames is not None else _cfg_get(training, "visualization_gt_only_frames", 2)
    )
    args.mode_selection = args.mode_selection or _cfg_get(training, "visualization_mode_selection", "best")
    args.gmm_heatmap = args.gmm_heatmap or _cfg_get(training, "visualization_gmm_heatmap", "all")
    args.heatmap_alpha = int(
        args.heatmap_alpha if args.heatmap_alpha is not None else _cfg_get(training, "visualization_heatmap_alpha", 105)
    )
    args.save_gmm_png = bool(
        args.save_gmm_png if args.save_gmm_png is not None else _cfg_get(training, "visualization_save_gmm_png", True)
    )
    args.gmm_png_future_step = args.gmm_png_future_step if args.gmm_png_future_step is not None else _cfg_get(
        training, "visualization_gmm_png_future_step", None
    )
    return args


def _batch_to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _center_xy(row):
    try:
        return float(row["bbox_center_x"]), float(row["bbox_center_y"])
    except (KeyError, TypeError, ValueError):
        return None


def _sample_motion_px(sample):
    points = []
    for row in list(sample.get("history", [])) + list(sample.get("future", [])):
        center = _center_xy(row)
        if center is not None and np.isfinite(center).all():
            points.append(center)
    if len(points) < 2:
        return 0.0
    pts = np.asarray(points, dtype=np.float32)
    displacement = float(np.linalg.norm(pts[-1] - pts[0]))
    span = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    return max(displacement, span)


def _sample_is_moving_object(sample, min_motion_px):
    try:
        object_type = int(sample.get("object_type", -1))
    except (TypeError, ValueError):
        return False
    return object_type in MOVING_OBJECT_TYPES and _sample_motion_px(sample) >= float(min_motion_px)


def _candidate_indices(dataset, moving_only=True, min_motion_px=12.0):
    samples = getattr(dataset, "samples", None)
    if samples is None:
        return list(range(len(dataset)))
    candidates = []
    for idx, sample in enumerate(samples):
        if not moving_only or _sample_is_moving_object(sample, min_motion_px):
            candidates.append(idx)
    return candidates


def _quality_requested(args):
    return int(args.bad or 0) > 0 or int(args.good or 0) > 0


def _select_indices(args, dataset):
    candidates = _candidate_indices(dataset, moving_only=args.moving_only, min_motion_px=args.min_motion_px)
    if not candidates:
        raise ValueError("No candidate moving vehicle/pedestrian/cyclist samples found. Lower --min_motion_px or pass --no-moving_only.")

    explicit = _parse_indices(args.indices)
    if explicit is not None:
        # Explicit indices mean exactly those dataset rows; do not silently drop
        # non-moving/sign/static objects through the candidate filter.
        return [idx for idx in explicit if 0 <= idx < len(dataset)]

    rng = random.Random(args.seed)
    sample_count = args.random_indices if args.random_indices is not None else args.num_samples
    pool = candidates
    if args.random_scene or _quality_requested(args):
        samples = getattr(dataset, "samples", None)
        if samples is None:
            raise ValueError("--random_scene/--bad/--good requires a dataset with a samples attribute.")
        scenes = sorted({str(samples[idx].get("scene_id", "")) for idx in candidates})
        scene_id = rng.choice(scenes)
        pool = [idx for idx in candidates if str(samples[idx].get("scene_id", "")) == scene_id]
        print(f"Selected random scene {scene_id} with {len(pool)} moving candidates", flush=True)

    if _quality_requested(args):
        return pool

    if args.random or args.random_indices is not None or args.random_scene:
        return rng.sample(pool, k=min(int(sample_count), len(pool)))

    end = min(args.start_index + args.num_samples, len(pool))
    return pool[args.start_index:end]


def _sample_seed(base_seed, index):
    if base_seed is None:
        return None
    return int((int(base_seed) + int(index) * 1_000_003) % (2**31 - 1))


def _set_torch_seed(seed):
    if seed is None:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _top_prediction_quality(index, params, batch):
    mu = params.mu[0].detach().cpu().numpy()
    probs = params.mode_probs[0].detach().cpu().numpy()
    gt = batch["trajectory"][0].detach().cpu().numpy()
    mask = batch["future_mask"][0].detach().cpu().numpy().astype(bool)
    if mu.ndim == 3:
        mu = mu[:, None]
    mode_ade = _mode_ade(mu, gt, mask)
    top_mode = int(np.argmax(probs))
    best_mode = int(np.nanargmin(mode_ade))
    gt_maneuver = _maneuver(gt[0][mask[0]])
    top_pred_maneuver = _maneuver(mu[top_mode, 0])
    return {
        "sample_index": int(index),
        "scene_id": batch.get("scene_id", [""])[0],
        "trajectory_row_id": batch.get("trajectory_row_id", [""])[0],
        "top_ade_norm": float(mode_ade[top_mode]),
        "best_ade_norm": float(mode_ade[best_mode]),
        "top_mode": top_mode,
        "best_mode": best_mode,
        "top_prob": float(probs[top_mode]),
        "top_is_best": int(top_mode == best_mode),
        "gt_maneuver": gt_maneuver,
        "top_pred_maneuver": top_pred_maneuver,
        "top_maneuver_correct": int(top_pred_maneuver == gt_maneuver),
    }


def _score_index(model, dataset, index, device, num_steps=None, sample_seed=None):
    loader = DataLoader(torch.utils.data.Subset(dataset, [index]), batch_size=1, shuffle=False)
    batch = next(iter(loader))
    batch_d = _batch_to_device(batch, device)
    _set_torch_seed(sample_seed)
    with torch.no_grad():
        params = model(batch_d, num_sampling_steps=num_steps, capture_steps=False)
    return _top_prediction_quality(index, params, batch)


def _select_quality_indices(model, dataset, indices, args, device):
    base_seed = args.quality_seed
    print(f"Scoring {len(indices)} scene candidates by top-mode ADE", flush=True)
    scored = [
        _score_index(
            model,
            dataset,
            idx,
            device,
            num_steps=args.num_steps,
            sample_seed=_sample_seed(base_seed, idx),
        )
        for idx in indices
    ]
    scored_by_idx = {row["sample_index"]: row for row in scored}
    selected = []
    selection_by_idx = {}

    def add_rows(label, rows):
        for row in rows:
            idx = row["sample_index"]
            if idx not in selection_by_idx:
                selected.append(idx)
                selection_by_idx[idx] = label
            elif label not in selection_by_idx[idx].split("+"):
                selection_by_idx[idx] = f"{selection_by_idx[idx]}+{label}"

    if int(args.bad or 0) > 0:
        bad_rows = sorted(scored, key=lambda row: (-row["top_ade_norm"], row["sample_index"]))
        add_rows("bad", bad_rows[: int(args.bad)])
    if int(args.good or 0) > 0:
        good_rows = sorted(scored, key=lambda row: (row["top_ade_norm"], row["sample_index"]))
        add_rows("good", good_rows[: int(args.good)])

    print(
        "Selected by quality: "
        + ", ".join(
            f"{idx}:{selection_by_idx[idx]} topADE={scored_by_idx[idx]['top_ade_norm']:.4f}"
            for idx in selected
        ),
        flush=True,
    )
    return selected, selection_by_idx, scored_by_idx


def _tensor_to_pil(tensor):
    arr = (tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGBA")


def _final_valid_future_step(batch):
    mask = batch["future_mask"][0].detach().cpu().numpy().astype(bool)
    if mask.ndim == 1:
        valid = np.flatnonzero(mask)
    else:
        valid = np.flatnonzero(mask.any(axis=0))
    return int(valid[-1]) if len(valid) else 0


def _future_frame_base(dataset, index, batch, future_step=None):
    samples = getattr(dataset, "samples", None)
    if samples is None:
        return _pil_from_history(batch), 0
    sample = samples[index]
    future = sample.get("future", [])
    if not future:
        return _pil_from_history(batch), 0
    if future_step is None:
        future_step = _final_valid_future_step(batch)
    future_step = max(0, min(int(future_step), len(future) - 1))
    try:
        segment_images = dataset._load_segment_images(sample["segment"])
        row = future[future_step]
        image_row = segment_images[str(row["image_id"])]
        return _tensor_to_pil(dataset._decode_image(image_row)), future_step
    except Exception as exc:
        print(f"Falling back to last observed frame for GMM PNG: {exc}", flush=True)
        return _pil_from_history(batch), future_step


def _draw_future_gt_box(image, dataset, index, future_step):
    samples = getattr(dataset, "samples", None)
    if samples is None:
        return image
    sample = samples[index]
    future = sample.get("future", [])
    if not future:
        return image
    future_step = max(0, min(int(future_step), len(future) - 1))
    try:
        segment_images = dataset._load_segment_images(sample["segment"])
        row = future[future_step]
        image_row = segment_images[str(row["image_id"])]
        box = dataset._box_to_unit(row, image_row)
        draw = ImageDraw.Draw(image, "RGBA")
        color = AGENT_COLORS[0]
        rect = _box_unit_to_px(box, image.size[0], image.size[1])
        draw.rectangle(rect, outline=color + (255,), width=3)
        draw.rectangle([rect[0], max(0, rect[1] - 16), rect[0] + 102, rect[1]], fill=(0, 0, 0, 150))
        draw.text((rect[0] + 4, max(0, rect[1] - 15)), "future gt box", fill=color + (255,))
    except Exception as exc:
        print(f"Skipping future GT bbox on GMM PNG: {exc}", flush=True)
    return image


# Catmull-Rom basis matrix. With alpha=0.5 and u ∈ [0, 1] over one segment:
#     [P(u)] = 0.5 · [u^3, u^2, u, 1] · M · [p_{i-1}, p_i, p_{i+1}, p_{i+2}]^T
# Rows of (0.5 · M · CP) directly give cubic coefficients (a, b, c, d) such that
# P(u) = a·u^3 + b·u^2 + c·u + d, matching make_catmull_rom_basis in the model.
_CATMULL_ROM_M = np.array(
    [
        [-1.0,  3.0, -3.0,  1.0],
        [ 2.0, -5.0,  4.0, -1.0],
        [-1.0,  0.0,  1.0,  0.0],
        [ 0.0,  2.0,  0.0,  0.0],
    ],
    dtype=np.float64,
)


def _catmull_rom_segment_coeffs(knot_xy):
    """Cubic coefficients per segment for one mode's knot positions.

    `knot_xy` is `[K, 2]` (x, y) knot positions in order. Returns a list of
    `K-1` segments. Each segment is `{"x": [a, b, c, d], "y": [a, b, c, d]}`
    where p(u) = a·u^3 + b·u^2 + c·u + d with `u ∈ [0, 1]` spanning that
    segment. Endpoint phantoms mirror the model: p_{-1} = 2·p_0 − p_1 and
    p_K = 2·p_{K-1} − p_{K-2}.
    """
    pts = np.asarray(knot_xy, dtype=np.float64)
    K = pts.shape[0]
    if K < 2:
        return []
    phantom_lo = (2.0 * pts[0] - pts[1])[None]
    phantom_hi = (2.0 * pts[-1] - pts[-2])[None]
    extended = np.concatenate([phantom_lo, pts, phantom_hi], axis=0)  # [K+2, 2]
    segments = []
    for i in range(K - 1):
        cp = extended[i:i + 4]  # [4, 2]
        coeffs = 0.5 * _CATMULL_ROM_M @ cp  # rows = [a, b, c, d] for x, y
        segments.append({
            "x": [round(float(coeffs[r, 0]), 6) for r in range(4)],
            "y": [round(float(coeffs[r, 1]), 6) for r in range(4)],
        })
    return segments


def _sample_summary(index, frames, batch, mode_selection="best", knot_indices=None):
    final = frames[-1]["params"]
    mu = final.mu[0].detach().cpu().numpy()
    probs = final.mode_probs[0].detach().cpu().numpy()
    gt = batch["trajectory"][0].detach().cpu().numpy()
    mask = batch["future_mask"][0].detach().cpu().numpy().astype(bool)
    if mu.ndim == 3:
        mu = mu[:, None]
    mode_ade = _mode_ade(mu, gt, mask)
    best_mode = int(np.nanargmin(mode_ade))
    top_mode = int(np.argmax(probs))
    shown_mode = top_mode if mode_selection == "top" else best_mode
    entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0))))
    gt_maneuver = _maneuver(gt[0][mask[0]])
    shown_pred_maneuver = _maneuver(mu[shown_mode, 0])
    top_pred_maneuver = _maneuver(mu[top_mode, 0])
    summary = {
        "sample_index": index,
        "scene_id": batch.get("scene_id", [""])[0],
        "trajectory_row_id": batch.get("trajectory_row_id", [""])[0],
        "shown_mode": shown_mode,
        "shown_prob": float(probs[shown_mode]),
        "top_mode": top_mode,
        "top_prob": float(probs[top_mode]),
        "mode_entropy": entropy,
        "minade_norm": float(mode_ade[shown_mode]),
        "top_ade_norm": float(mode_ade[top_mode]),
        "best_ade_norm": float(mode_ade[best_mode]),
        "top_is_best": int(top_mode == best_mode),
        "gt_maneuver": gt_maneuver,
        "pred_maneuver": shown_pred_maneuver,
        "top_pred_maneuver": top_pred_maneuver,
        "top_maneuver_correct": int(top_pred_maneuver == gt_maneuver),
    }

    horizon = mu.shape[2]
    if knot_indices is not None:
        if hasattr(knot_indices, "detach"):
            indices = knot_indices.detach().cpu().tolist()
        else:
            indices = list(knot_indices)
        indices = [int(i) for i in indices if 0 <= int(i) < int(horizon)]
    else:
        indices = []

    if len(indices) >= 2:
        # Knot positions = mu at knot times (basis pass-through). Agent 0.
        shown_knots = mu[shown_mode, 0, indices]
        shown_segments = _catmull_rom_segment_coeffs(shown_knots)
        # Annotate each segment with the future-step range it spans, so the
        # user can convert u ∈ [0, 1] back to the absolute future step t via
        # t = t_start + u · (t_end − t_start).
        shown_segments_annotated = [
            {"t_start": int(indices[i]), "t_end": int(indices[i + 1]), **seg}
            for i, seg in enumerate(shown_segments)
        ]
        all_modes_segments = [
            _catmull_rom_segment_coeffs(mu[m, 0, indices])
            for m in range(mu.shape[0])
        ]
        summary["poly_coeffs_shown"] = json.dumps(shown_segments_annotated)
        summary["poly_coeffs_all"] = json.dumps(all_modes_segments)
    return summary


def render_one(
    model,
    dataset,
    index,
    device,
    output_path,
    num_steps=None,
    frame_ms=450,
    gt_only_frames=2,
    mode_selection="best",
    gmm_heatmap="all",
    heatmap_alpha=105,
    save_gmm_png=True,
    gmm_png_future_step=None,
    sample_seed=None,
    show_initial_noise=False,
):
    loader = DataLoader(torch.utils.data.Subset(dataset, [index]), batch_size=1, shuffle=False)
    batch = next(iter(loader))
    batch_d = _batch_to_device(batch, device)
    _set_torch_seed(sample_seed)
    with torch.no_grad():
        final_params, frames = model(
            batch_d,
            num_sampling_steps=num_steps,
            capture_steps=True,
            capture_initial_noise=show_initial_noise,
        )
    if not frames:
        frames = [{"sigma": 0.0, "params": final_params}]
    base = _pil_from_history(batch)
    images = [_render_gt_only(base, batch, i, gt_only_frames) for i in range(1, gt_only_frames + 1)]
    knot_indices = getattr(model, "knot_indices", None) if getattr(model, "use_control_points", False) else None
    images += [
        _render_step(
            base,
            frame["params"],
            batch,
            i,
            len(frames),
            frame["sigma"],
            mode_selection=mode_selection,
            gmm_heatmap="off" if frame.get("kind") == "initial_noise" else gmm_heatmap,
            heatmap_alpha=heatmap_alpha,
            knot_indices=knot_indices,
            frame_label="initial fan-out noise" if frame.get("kind") == "initial_noise" else None,
        )
        for i, frame in enumerate(frames, start=1)
    ]
    images += [images[-1]] * 2
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output_path, save_all=True, append_images=images[1:], duration=frame_ms, loop=0, optimize=False)
    summary = _sample_summary(index, frames, batch, mode_selection=mode_selection, knot_indices=knot_indices)
    if save_gmm_png:
        png_path = output_path.with_name(f"{output_path.stem}_gmm.png")
        final_params = frames[-1]["params"]
        gmm_base, resolved_future_step = _future_frame_base(dataset, index, batch, future_step=gmm_png_future_step)
        gmm_image = _render_gmm_png(
            gmm_base,
            final_params,
            batch,
            mode_selection=mode_selection,
            gmm_heatmap=gmm_heatmap,
            heatmap_alpha=heatmap_alpha,
            draw_observed_boxes=False,
            future_step=resolved_future_step,
            knot_indices=knot_indices,
        )
        gmm_image = _draw_future_gt_box(gmm_image.convert("RGBA"), dataset, index, resolved_future_step).convert("RGB")
        gmm_image.save(png_path)
        summary["gmm_png"] = str(png_path)
        summary["gmm_png_future_step"] = int(resolved_future_step)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Render several image-plane diffusion GIFs and a CSV summary.")
    parser.add_argument("--config", default="configs/astra_edm_diffusion_waymo_image_plane.yml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--outs_subdir", "--outs_dir", dest="outs_subdir", default=None, help="Save outputs under visualizations/outs/<value>. Cannot be combined with --output_dir.")
    parser.add_argument("--indices", default=None, help="Comma/range list, e.g. 0,5,9-12")
    parser.add_argument("--start_index", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--random", action=argparse.BooleanOptionalAction, default=None, help="Choose random validation samples instead of sequential indices.")
    parser.add_argument("--random_indices", type=int, default=None, help="Choose exactly N random candidate indices.")
    parser.add_argument("--random_scene", action=argparse.BooleanOptionalAction, default=None, help="Choose one random scene, then sample indices from that scene.")
    parser.add_argument("--bad", type=int, default=None, help="Rank one selected scene by top-mode ADE and render the K worst-predicted agents.")
    parser.add_argument("--good", type=int, default=None, help="Rank one selected scene by top-mode ADE and render the J best-predicted agents.")
    parser.add_argument("--moving_only", action=argparse.BooleanOptionalAction, default=None, help="Restrict candidates to moving vehicles, pedestrians, and cyclists.")
    parser.add_argument("--min_motion_px", type=float, default=None, help="Minimum bbox-center motion in image pixels for --moving_only.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. Omit for a different random scene/indices each run.")
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--show_initial_noise", action=argparse.BooleanOptionalAction, default=None, help="Include the initial fan-out noise as the first refinement frame.")
    parser.add_argument("--frame_ms", type=int, default=None)
    parser.add_argument("--gt_only_frames", type=int, default=None)
    parser.add_argument("--mode_selection", choices=["best", "top"], default=None)
    parser.add_argument("--gmm_heatmap", choices=["all", "shown", "top", "off"], default=None)
    parser.add_argument("--heatmap_alpha", type=int, default=None)
    parser.add_argument("--save_gmm_png", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--gmm_png_future_step", type=int, default=None, help="Future timestep image for the GMM PNG. Defaults to the last valid future frame.")
    parser.add_argument("--device", default=None)
    parsed = parser.parse_args()
    mode_selection_cli = parsed.mode_selection is not None
    output_dir_cli = parsed.output_dir is not None
    args = _apply_config_defaults(parsed)
    if args.outs_subdir is not None:
        if output_dir_cli:
            raise ValueError("Pass either --outs_subdir/--outs_dir or --output_dir, not both.")
        args.output_dir = _outs_output_dir(args.outs_subdir)
    if (args.bad is not None and int(args.bad) < 0) or (args.good is not None and int(args.good) < 0):
        raise ValueError("--bad and --good must be non-negative integers.")
    if _quality_requested(args) and not mode_selection_cli:
        args.mode_selection = "top"
    args.quality_seed = args.seed if args.seed is not None else random.randrange(0, 2**31 - 1)
    quality_seed_note = f" quality_seed={args.quality_seed}" if _quality_requested(args) else ""

    print(
        f"Visualization config: checkpoint={args.checkpoint} output_dir={args.output_dir} "
        f"indices={args.indices} random={args.random} random_scene={args.random_scene} "
        f"bad={args.bad} good={args.good} mode_selection={args.mode_selection} "
        f"num_samples={args.num_samples} num_steps={args.num_steps} show_initial_noise={args.show_initial_noise}{quality_seed_note}",
        flush=True,
    )
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, dataset = load_model(args.config, args.checkpoint, device)

    indices = _select_indices(args, dataset)
    selection_by_idx = {}
    scored_by_idx = {}
    if _quality_requested(args):
        indices, selection_by_idx, scored_by_idx = _select_quality_indices(model, dataset, indices, args, device)
    if not indices:
        raise ValueError("No valid sample indices selected after moving-object filtering.")

    output_dir = Path(args.output_dir)
    rows = []
    for idx in indices:
        label = selection_by_idx.get(idx)
        stem = f"sample_{label}_{idx:06d}" if label else f"sample_{idx:06d}"
        out = output_dir / f"{stem}.gif"
        print(f"Rendering sample {idx} -> {out}", flush=True)
        row = render_one(
            model,
            dataset,
            idx,
            device,
            out,
            num_steps=args.num_steps,
            frame_ms=args.frame_ms,
            gt_only_frames=args.gt_only_frames,
            mode_selection=args.mode_selection,
            gmm_heatmap=args.gmm_heatmap,
            heatmap_alpha=args.heatmap_alpha,
            save_gmm_png=args.save_gmm_png,
            gmm_png_future_step=args.gmm_png_future_step,
            sample_seed=_sample_seed(args.quality_seed, idx) if _quality_requested(args) else None,
            show_initial_noise=args.show_initial_noise,
        )
        if idx in selection_by_idx:
            row["selection_bucket"] = selection_by_idx[idx]
            row["selection_top_ade_norm"] = scored_by_idx[idx]["top_ade_norm"]
        row["gif"] = str(out)
        rows.append(row)

    summary_path = output_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
