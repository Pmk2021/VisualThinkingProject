#!/usr/bin/env python3
"""Generate visualization batches for random RGB frames from the Waymo parquet dataset."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from box import Box

PROJECT_ROOT = next(
    p for p in Path(__file__).resolve().parents
    if (p / "AnytimeTrajectoryPredictor").is_dir()
)
sys.path.insert(0, str(PROJECT_ROOT))

import visualizations.visualize_diffusion as vd


def _segment_candidates(waymo_root: Path, max_segments: int | None, rng: random.Random) -> list[Path]:
    segments = [
        p for p in waymo_root.iterdir()
        if p.is_dir()
        and (p / "images.parquet").exists()
        and (p / "trajectories.parquet").exists()
        and (p / "image_trajectories.parquet").exists()
    ]
    rng.shuffle(segments)
    return segments[:max_segments] if max_segments else segments


def _image_rows(seg_path: Path, camera_name: int | None, rng: random.Random) -> list[dict]:
    table = vd._read_table_existing(seg_path / "images.parquet", vd.IMAGE_COLS)
    data = table.to_pydict()
    rows = [{k: data[k][i] for k in data} for i in range(len(data.get("image_id", [])))]
    if camera_name is not None:
        rows = [r for r in rows if int(r.get("camera_name", -1)) == int(camera_name)]
    rng.shuffle(rows)
    return rows


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _complete_sample_dirs(output_dir: Path) -> list[Path]:
    required = {
        "rgb_refinement.gif",
        "scene_diffusion.gif",
        "gmm_output.png",
        "rgb_heatmap.png",
    }
    if not output_dir.exists():
        return []
    return sorted(
        p for p in output_dir.iterdir()
        if p.is_dir() and required.issubset({f.name for f in p.iterdir() if f.is_file()})
    )


def _sample_index(path: Path) -> int:
    try:
        return int(path.name.split("_", 1)[0])
    except (ValueError, IndexError):
        return -1


def _save_bev_gif(frames: list[dict], agents: list[dict], output_path: Path,
                  frame_ms: int, final_hold_frames: int):
    anchors = np.array([[a["anchor_x"], a["anchor_y"]] for a in agents])
    cx, cy = anchors.mean(axis=0)
    anchor_spread = np.max(np.linalg.norm(anchors - np.array([cx, cy]), axis=1))
    bev_half = max(anchor_spread * 1.5 + 30.0, 40.0)
    total = len(frames) - 1
    images = [
        vd.render_frame(frame, i, total, i == total, cx, cy, bev_half)
        for i, frame in enumerate(frames)
    ]
    images += [images[-1]] * final_hold_frames
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=frame_ms,
        loop=0,
        optimize=False,
    )


def _save_gmm_png(frames: list[dict], agents: list[dict], output_path: Path):
    anchors = np.array([[a["anchor_x"], a["anchor_y"]] for a in agents])
    cx, cy = anchors.mean(axis=0)
    anchor_spread = np.max(np.linalg.norm(anchors - np.array([cx, cy]), axis=1))
    bev_half = max(anchor_spread * 1.5 + 30.0, 40.0)
    vd.render_gmm_png(frames[-1], cx, cy, bev_half, output_path=str(output_path))


def _save_rgb_refinement_gif(frames: list[dict], ctx: dict, output_path: Path,
                             frame_ms: int, final_hold_frames: int):
    images = [
        vd._render_rgb_prediction_frame(frame, ctx, i, len(frames) - 1)
        for i, frame in enumerate(frames)
    ]
    images += [images[-1]] * final_hold_frames
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=frame_ms,
        loop=0,
        optimize=False,
    )


def _save_rgb_heatmap_png(frames: list[dict], seg_path: Path, agents: list[dict], camera: int,
                          output_path: Path, rgb_max_width: int, future_step: int):
    future_timestamp = vd._target_future_timestamp(seg_path, agents, future_step)
    ctx = vd._load_rgb_context(
        seg_path,
        agents,
        camera_name=camera,
        max_width=rgb_max_width,
        target_timestamp=future_timestamp,
        require_selected_visible=False,
    )
    if ctx is None:
        return False
    vd._render_rgb_heatmap_png(frames[-1], ctx, str(output_path), future_step=future_step)
    return True


def generate_batch(args) -> list[Path]:
    rng = random.Random(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    with open(args.config) as f:
        cfg = Box(yaml.safe_load(f))
    waymo_root = Path(args.waymo_root or cfg.feature_extractor.waymo_root)

    print(f"Device: {device}", flush=True)
    print("Loading model once for batch visualization ...", flush=True)
    model, epoch = vd.load_model(args.config, args.checkpoint, device)
    print(f"  epoch: {epoch}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    segments = _segment_candidates(waymo_root, args.max_segments, rng)
    if not segments:
        raise RuntimeError(f"No RGB-capable segments found under {waymo_root}")

    made: list[Path] = _complete_sample_dirs(output_dir)
    next_index = max([_sample_index(p) for p in made], default=-1) + 1
    used_segments = {"_".join(p.name.split("_")[1:-3]) for p in made}
    if made:
        print(f"Resuming: found {len(made)} complete samples in {output_dir}", flush=True)
    attempted = 0
    max_attempts = max(args.count * 100, args.count + 50)
    H, T = model.max_history, model.future_horizon

    for seg_path in segments:
        if len(made) >= args.count or attempted >= max_attempts:
            break
        if seg_path.name in used_segments:
            continue
        accepted_segment = False
        for image_row in _image_rows(seg_path, args.camera_name, rng):
            if accepted_segment or len(made) >= args.count or attempted >= max_attempts:
                break
            attempted += 1
            timestamp = int(image_row["frame_timestamp_micros"])
            camera = int(image_row.get("camera_name", args.camera_name or 1))
            try:
                _, scene_id, agents = vd.find_scene(
                    str(waymo_root),
                    max_segs=0,
                    min_agents=args.min_agents,
                    max_agents=args.max_agents,
                    H=H,
                    T=T,
                    scene_id=seg_path.name,
                    anchor_timestamp=timestamp,
                )
                ctx = vd._load_rgb_context(
                    seg_path,
                    agents,
                    camera_name=camera,
                    max_width=args.rgb_max_width,
                    target_timestamp=timestamp,
                    require_selected_visible=True,
                )
                if ctx is None:
                    continue

                stem = f"{next_index:03d}_{_safe_name(scene_id)}_{timestamp}_camera_{camera}"
                next_index += 1
                sample_dir = output_dir / stem
                sample_dir.mkdir(parents=True, exist_ok=True)

                frames = vd.sample_scene(model, agents, device, num_steps=args.num_steps)
                _save_rgb_refinement_gif(
                    frames, ctx, sample_dir / "rgb_refinement.gif",
                    args.frame_ms, args.final_hold_frames,
                )
                _save_bev_gif(
                    frames, agents, sample_dir / "scene_diffusion.gif",
                    args.frame_ms, args.final_hold_frames,
                )
                _save_gmm_png(frames, agents, sample_dir / "gmm_output.png")
                _save_rgb_heatmap_png(
                    frames, seg_path, agents, camera,
                    sample_dir / "rgb_heatmap.png",
                    args.rgb_max_width,
                    args.rgb_heatmap_future_step,
                )

                made.append(sample_dir)
                used_segments.add(scene_id)
                print(
                    f"[{len(made):02d}/{args.count}] {sample_dir}  "
                    f"agents={len(agents)}  frames={len(frames)}",
                    flush=True,
                )
                accepted_segment = True
            except Exception as exc:
                if args.verbose:
                    print(f"[skip] {seg_path.name} ts={timestamp}: {exc}", flush=True)
                continue

    if len(made) < args.count:
        print(f"Only generated {len(made)}/{args.count} GIFs after {attempted} attempts.", flush=True)
    return made


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/astra_edm_diffusion_waymo.yml")
    p.add_argument("--checkpoint", default="checkpoints/astra_edm_diffusion_waymo_latest.pth")
    p.add_argument("--waymo_root", default=None)
    p.add_argument("--output_dir", default="visualizations/outs/batch_all")
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--num_steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=503)
    p.add_argument("--max_segments", type=int, default=80)
    p.add_argument("--min_agents", type=int, default=1)
    p.add_argument("--max_agents", type=int, default=5)
    p.add_argument("--camera_name", type=int, default=1)
    p.add_argument("--rgb_max_width", type=int, default=640)
    p.add_argument("--frame_ms", type=int, default=160)
    p.add_argument("--final_hold_frames", type=int, default=2)
    p.add_argument("--rgb_heatmap_future_step", type=int, default=5)
    p.add_argument("--device", default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    generate_batch(parse_args())


if __name__ == "__main__":
    main()
