#!/usr/bin/env python3
"""
Visualize ASTRA-EDM diffusion denoising for ALL agents in a scene simultaneously.

All agents in the scene are batched through the same diffusion loop, and every
frame of the GIF shows their predictions in a shared world-coordinate BEV frame.

Usage:
    python visualizations/visualize_diffusion.py \
        --config  configs/astra_edm_diffusion_waymo.yml \
        --checkpoint checkpoints/astra_edm_diffusion_waymo_latest.pth \
        --output  scene_diffusion.gif \
        --min_agents 4
"""

import argparse
import math
import sys
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from box import Box
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from PIL import Image, ImageDraw

PROJECT_ROOT = next(
    p for p in Path(__file__).resolve().parents
    if (p / "AnytimeTrajectoryPredictor").is_dir()
)
sys.path.insert(0, str(PROJECT_ROOT))

from AnytimeTrajectoryPredictor.models.TrajectoryPredictor import TrajectoryPredictor
from AnytimeTrajectoryPredictor.models.architectures.astra_edm_diffusion import make_karras_sigmas

# ── visual constants ─────────────────────────────────────────────────────────
AGENT_COLORS = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
    "#8c564b","#e377c2","#bcbd22","#17becf","#aec7e8",
    "#ffbb78","#98df8a","#ff9896","#c5b0d5","#c49c94",
    "#f7b6d2","#dbdb8d","#9edae5","#7f7f7f","#c7c7c7",
]
RGB_REFINEMENT_COLORS = [
    "#00e5ff", "#ff6d00", "#00ff4c", "#ff1744", "#d500f9",
    "#ffff00", "#00ffb3", "#ff00aa", "#2979ff", "#76ff03",
]
BG_COLOR   = "#0d0d1a"
GRID_COLOR = "#1c1c3a"
ELLIPSE_TS = [9, 19, 39, 59, 79]
OBJ_SIZE   = {1: (4.5, 2.0), 2: (0.6, 0.6), 3: (1.8, 0.8)}
OBJ_NAMES  = {1: "Veh", 2: "Ped", 3: "Cyc"}
DEFAULT_SZ = (2.0, 1.0)
DYNAMIC_OBJECT_TYPES = {1, 2, 3}

TRAJ_COLS = [
    "scene_id", "trajectory_id", "trajectory_row_id", "object_type", "num_steps",
    "timestamps_micros",
    "x", "y", "heading", "velocity_x", "velocity_y",
    "length", "width",
]

IMAGE_COLS = [
    "image_id", "scene_id", "frame_timestamp_micros", "camera_name",
    "camera_name_text", "image_jpeg", "image_width", "image_height",
]

LINK_COLS = [
    "image_id", "frame_timestamp_micros", "camera_name", "camera_name_text",
    "trajectory_row_id", "bbox_center_x", "bbox_center_y",
    "bbox_width", "bbox_height", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
]


# ── parquet / tensor helpers ──────────────────────────────────────────────────

def _rot(x, y, h):
    """Rotate (x,y) into agent-centric frame with heading h."""
    c, s = np.cos(-h), np.sin(-h)
    return c * x - s * y, s * x + c * y


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def _arr(values, start, end):
    """Slice a parquet list-column value to a numpy float32 array, zero-padded to length (end-start)."""
    vals = np.asarray(values if values is not None else [], dtype=np.float32)
    n    = end - start
    out  = np.zeros(n, dtype=np.float32)
    s    = max(start, 0)
    e    = min(end, len(vals))
    if e > s:
        out[s - start : e - start] = vals[s:e]
    return out


def _read_table_existing(path: Path, columns: list[str]):
    """Read only columns present in a parquet file, preserving older dataset compatibility."""
    schema = pq.read_schema(path)
    present = [c for c in columns if c in schema.names]
    return pq.read_table(path, columns=present)


# ── coordinate transforms ─────────────────────────────────────────────────────

def to_world(local_xy: np.ndarray, ax: float, ay: float, ah: float) -> np.ndarray:
    """
    (N, 2) agent-centric coordinates → (N, 2) world coordinates.
    NaN inputs propagate (used for masking invalid timesteps).
    Inverse of _rot: world = R(-ah) @ local + (ax, ay)
    where R(-ah) = [[cos(ah), -sin(ah)], [sin(ah), cos(ah)]].
    """
    c, s  = np.cos(ah), np.sin(ah)
    R_inv = np.array([[c, -s], [s, c]])
    return (R_inv @ local_xy.T).T + np.array([ax, ay])


def chol_to_world(L: np.ndarray, ah: float) -> np.ndarray:
    """
    (2,2) Cholesky factor in agent frame → world frame.
    If Sigma_a = L @ L^T, then Sigma_w = R_inv @ Sigma_a @ R_inv^T = L_w @ L_w^T
    where L_w = R_inv @ L.
    """
    c, s  = np.cos(ah), np.sin(ah)
    R_inv = np.array([[c, -s], [s, c]])
    return R_inv @ L


# ── scene loading ─────────────────────────────────────────────────────────────

def _build_agent_from_traj(data: dict, i: int, H: int, T: int, anchor_index: int | None = None):
    """
    Build agent-centric tensors from one row of trajectories.parquet.
    Uses the requested scene-wide anchor timestep when available; otherwise
    falls back to the latest available window.
    Returns (features, trajectory, obs_mask, fut_mask,
             anchor_x, anchor_y, anchor_heading, box_length, box_width).
    """
    num_steps = int(data["num_steps"][i] or 0)
    if anchor_index is None:
        # Fallback for old datasets without timestamps: use the latest window.
        fut_len = min(T, num_steps - H)
        anchor_index = num_steps - fut_len - 1
    else:
        fut_len = min(T, num_steps - anchor_index - 1)
    if fut_len < 1:
        return None  # not enough data

    hist_end   = anchor_index + 1
    hist_start = max(0, hist_end - H)
    actual_H   = hist_end - hist_start     # may be < H if trajectory is very short

    obs_x  = _arr(data["x"][i],          hist_start, hist_end)
    obs_y  = _arr(data["y"][i],          hist_start, hist_end)
    obs_h  = _arr(data["heading"][i],    hist_start, hist_end)
    obs_vx = _arr(data["velocity_x"][i], hist_start, hist_end)
    obs_vy = _arr(data["velocity_y"][i], hist_start, hist_end)
    fut_x  = _arr(data["x"][i],          hist_end,   hist_end + fut_len)
    fut_y  = _arr(data["y"][i],          hist_end,   hist_end + fut_len)

    # Anchor = last observed position
    ax, ay, ah = float(obs_x[-1]), float(obs_y[-1]), float(obs_h[-1])
    avx, avy = float(obs_vx[-1]), float(obs_vy[-1])
    timestamps = data.get("timestamps_micros", [None] * len(data["x"]))[i]
    anchor_time = None
    if timestamps is not None and anchor_index < len(timestamps):
        anchor_time = int(timestamps[anchor_index])

    lx,  ly  = _rot(obs_x - ax, obs_y - ay, ah)
    lvx, lvy = _rot(obs_vx, obs_vy, ah)
    rel_h    = _wrap(obs_h - ah)
    obs_valid_f = np.ones(actual_H, dtype=np.float32)

    # Pad history to H if shorter
    def _pad(a, target):
        if len(a) >= target:
            return a[:target]
        return np.concatenate([np.zeros(target - len(a), dtype=np.float32), a])

    features = np.stack([
        _pad(lx, H), _pad(ly, H), _pad(rel_h, H),
        _pad(lvx, H), _pad(lvy, H),
        np.concatenate([np.zeros(H - actual_H, np.float32), obs_valid_f]),
    ], axis=-1).astype(np.float32)

    fx, fy = _rot(fut_x - ax, fut_y - ay, ah)
    # Pad future to T
    traj = np.zeros((T, 2), dtype=np.float32)
    traj[:fut_len, 0] = fx
    traj[:fut_len, 1] = fy
    fut_mask = np.zeros(T, dtype=np.float32)
    fut_mask[:fut_len] = 1.0

    # Observed mask (1 for valid steps, 0 for padding at the start)
    obs_mask = np.concatenate([np.zeros(H - actual_H, np.float32), obs_valid_f])

    # Actual bounding box from data
    bl_vals = data.get("length", [None])[i]
    bw_vals = data.get("width",  [None])[i]
    if bl_vals and len(bl_vals):
        bl = float(np.nanmean(np.asarray(bl_vals[:10], dtype=np.float32)))
        bw = float(np.nanmean(np.asarray(bw_vals[:10], dtype=np.float32)))
    else:
        bl, bw = OBJ_SIZE.get(int(data["object_type"][i]), DEFAULT_SZ)

    return features, traj, obs_mask, fut_mask, ax, ay, ah, avx, avy, anchor_time, anchor_index, bl, bw


def _choose_common_anchor_timestamp(
    data: dict,
    H: int,
    min_fut: int,
    object_types: set[int] | None = None,
) -> int | None:
    """Pick the timestamp with the most agents that have enough history/future."""
    timestamps_col = data.get("timestamps_micros")
    if not timestamps_col:
        return None
    counts: dict[int, int] = {}
    for i, timestamps in enumerate(timestamps_col):
        if object_types is not None and int(data["object_type"][i]) not in object_types:
            continue
        if timestamps is None:
            continue
        num_steps = int(data["num_steps"][i] or 0)
        last_anchor = num_steps - min_fut - 1
        if last_anchor < H - 1:
            continue
        for idx in range(H - 1, last_anchor + 1):
            if idx < len(timestamps):
                counts[int(timestamps[idx])] = counts.get(int(timestamps[idx]), 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def _timestamp_index(timestamps, timestamp: int | None) -> int | None:
    if timestamp is None or timestamps is None:
        return None
    for idx, value in enumerate(timestamps):
        if int(value) == int(timestamp):
            return idx
    return None


def load_scene_agents(seg_path: Path, H: int, T: int,
                      min_fut: int = 5, max_agents: int = 24,
                      anchor_timestamp: int | None = None,
                      object_types: set[int] | None = None) -> list:
    """Load all trajectories from a segment as a list of agent dicts (world-frame anchors included)."""
    table = _read_table_existing(seg_path / "trajectories.parquet", TRAJ_COLS)
    data  = table.to_pydict()
    if object_types is None:
        object_types = DYNAMIC_OBJECT_TYPES
    common_anchor_time = (
        int(anchor_timestamp)
        if anchor_timestamp is not None
        else _choose_common_anchor_timestamp(data, H, min_fut, object_types=object_types)
    )
    agents = []
    for i in range(table.num_rows):
        if object_types is not None and int(data["object_type"][i]) not in object_types:
            continue
        anchor_index = _timestamp_index(
            data.get("timestamps_micros", [None] * table.num_rows)[i],
            common_anchor_time,
        )
        if common_anchor_time is not None and anchor_index is None:
            continue
        result = _build_agent_from_traj(data, i, H, T, anchor_index=anchor_index)
        if result is None:
            continue
        feats, traj, omask, fmask, ax, ay, ah, avx, avy, anchor_time, anchor_index, bl, bw = result
        if fmask.sum() < min_fut:
            continue
        agents.append({
            "features":       torch.from_numpy(feats).unsqueeze(0).float(),
            "trajectory":     torch.from_numpy(traj).unsqueeze(0).float(),
            "observed_mask":  torch.from_numpy(omask).unsqueeze(0).float(),
            "future_mask":    torch.from_numpy(fmask).unsqueeze(0).float(),
            "anchor_x": ax, "anchor_y": ay, "anchor_heading": ah,
            "anchor_vx": avx, "anchor_vy": avy,
            "anchor_timestamp": anchor_time,
            "anchor_index": anchor_index,
            "trajectory_id": str(data.get("trajectory_id", [""] * table.num_rows)[i]),
            "trajectory_row_id": str(data.get("trajectory_row_id", [""] * table.num_rows)[i]),
            "scene_id": str(data.get("scene_id", [""] * table.num_rows)[i]),
            "object_type": int(data["object_type"][i]),
            "box_length": bl, "box_width": bw,
        })
        if len(agents) >= max_agents:
            break
    return agents


def find_scene(waymo_root: str, max_segs: int, min_agents: int, max_agents: int,
               H: int, T: int, scene_id: str | None = None,
               anchor_timestamp: int | None = None,
               object_types: set[int] | None = None):
    """
    Scan segments and return (seg_path, scene_id_str, agents_list).
    Uses trajectories.parquet (prediction_targets.parquet is empty for this dataset).
    Picks the segment with the most qualifying agents, up to max_agents.
    """
    root     = Path(waymo_root)
    segments = sorted(p for p in root.iterdir()
                      if p.is_dir() and (p / "trajectories.parquet").exists())
    if not segments:
        raise FileNotFoundError(f"No trajectories.parquet found under {waymo_root}")

    # If a specific scene_id is given, the segment name IS the scene_id for this dataset
    if scene_id is not None:
        for seg in segments:
            if scene_id in seg.name:
                agents = load_scene_agents(
                    seg, H, T, max_agents=max_agents,
                    anchor_timestamp=anchor_timestamp,
                    object_types=object_types,
                )
                if len(agents) >= min_agents:
                    return seg, seg.name, agents
        raise RuntimeError(f"Scene '{scene_id}' not found in {waymo_root}")

    best = (None, None, [])
    for seg in segments[:max_segs]:
        agents = load_scene_agents(
            seg, H, T, max_agents=max_agents,
            anchor_timestamp=anchor_timestamp,
            object_types=object_types,
        )
        if len(agents) >= min_agents and len(agents) > len(best[2]):
            best = (seg, seg.name, agents)

    if best[0] is not None:
        return best
    raise RuntimeError(
        f"No segment with >= {min_agents} qualifying agents found (searched {max_segs} segments)"
    )


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(config_path: str, ckpt_path: str, device: torch.device):
    """Load model from YAML config + checkpoint. Normalizer is restored from the state dict."""
    with open(config_path) as f:
        cfg = Box(yaml.safe_load(f))
    model = TrajectoryPredictor.create_model(cfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    return model, ckpt.get("epoch", "?")


# ── diffusion sampling with per-step world-space capture ─────────────────────

def sample_scene(model, agents: list, device: torch.device, num_steps: int | None = None) -> list:
    """
    Run the Karras-EDM loop for all agents simultaneously (one batch element per agent).

    Returns a list of frame dicts (initial noise + one per denoising step):
        sigma       : float
        agents_data : list of per-agent dicts:
            mu          (K, T, 2)  GMM mean in world coords
            mode_probs  (K,)
            cov_chol_w  (K, T, 2, 2) world-frame Cholesky factors, or None
            history_w   (H, 2) world coords observed path  (NaN = invalid)
            future_w    (T, 2) world coords GT future      (NaN = invalid)
            anchor_x, anchor_y, anchor_heading
            object_type
    """
    N  = len(agents)
    batch = {
        "features":      torch.stack([a["features"]      for a in agents]),
        "trajectory":    torch.stack([a["trajectory"]    for a in agents]),
        "observed_mask": torch.stack([a["observed_mask"] for a in agents]),
        "future_mask":   torch.stack([a["future_mask"]   for a in agents]),
    }
    batch_d = {k: v.to(device) for k, v in batch.items()}

    # Pre-compute world-space history and GT for each agent (constant across frames)
    world_hist, world_fut = [], []
    for a in agents:
        feats  = a["features"][0].numpy()          # (H, 6)
        omask  = a["observed_mask"][0].numpy().astype(bool)
        traj   = a["trajectory"][0].numpy()        # (T, 2)
        fmask  = a["future_mask"][0].numpy().astype(bool)
        ax, ay, ah = a["anchor_x"], a["anchor_y"], a["anchor_heading"]

        h_xy = feats[:, :2].copy()
        h_xy[~omask] = np.nan
        world_hist.append(to_world(h_xy, ax, ay, ah))

        f_xy = traj.copy()
        f_xy[~fmask] = np.nan
        world_fut.append(to_world(f_xy, ax, ay, ah))

    def _build_frame(sigma_val, gmm_mu_cpu, probs_cpu, chol_cpu):
        """Assemble per-agent world-space data for one frame."""
        out = []
        for i, a in enumerate(agents):
            ax, ay, ah = a["anchor_x"], a["anchor_y"], a["anchor_heading"]
            K  = gmm_mu_cpu.shape[1]

            # (K, T, 2) world coords
            mu_loc = gmm_mu_cpu[i, :, 0].numpy()     # (K, T, 2)
            mu_w   = np.stack([to_world(mu_loc[k], ax, ay, ah) for k in range(K)])

            # (K, T, 2, 2) world-frame Cholesky
            if chol_cpu is not None:
                cl = chol_cpu[i, :, 0].numpy()       # (K, T, 2, 2)
                cw = np.stack([
                    np.stack([chol_to_world(cl[k, t], ah) for t in range(cl.shape[1])])
                    for k in range(K)
                ])
            else:
                cw = None

            out.append({
                "mu":          mu_w,
                "mode_probs":  probs_cpu[i].numpy(),
                "cov_chol_w":  cw,
                "history_w":   world_hist[i],
                "future_w":    world_fut[i],
                "anchor_x": ax, "anchor_y": ay, "anchor_heading": ah,
                "anchor_vx": a.get("anchor_vx", 0.0),
                "anchor_vy": a.get("anchor_vy", 0.0),
                "anchor_timestamp": a.get("anchor_timestamp"),
                "anchor_index": a.get("anchor_index"),
                "trajectory_id": a.get("trajectory_id", ""),
                "trajectory_row_id": a.get("trajectory_row_id", ""),
                "scene_id": a.get("scene_id", ""),
                "object_type": a["object_type"],
                "box_length":  a.get("box_length"),
                "box_width":   a.get("box_width"),
            })
        return {"sigma": sigma_val, "agents_data": out}

    with torch.no_grad():
        context = model.encode_context(batch_d)
        T      = model.future_horizon
        steps  = num_steps or model.num_sampling_steps
        sigmas = make_karras_sigmas(steps, model.sigma_min, model.sigma_max, model.rho, device)

        x = torch.randn(N, model.num_modes, 1, T, model.trajectory_dim, device=device) * sigmas[0]

        frames = []
        # Initial noise frame — raw noise as trajectories, uniform mode probs
        x_dn  = model.normalizer.denormalize(x)
        unif  = torch.full((N, model.num_modes), 1.0 / model.num_modes)
        frames.append(_build_frame(float(sigmas[0]), x_dn.cpu(), unif, None))

        use_heun = int(getattr(model, "sampler_order", 2)) >= 2

        for i in range(len(sigmas) - 1):
            sigma_val      = sigmas[i]
            sigma_next_val = sigmas[i + 1]
            sigma_b        = sigma_val.expand(N)

            x_clean, hidden = model.denoise(x, context, sigma_b)

            if sigma_next_val == 0 or not use_heun:
                gmm    = model.gmm_head(x_clean, hidden)
                gmm_mu = model.normalizer.denormalize(gmm.mu)
                frames.append(_build_frame(
                    float(sigma_next_val),
                    gmm_mu.cpu(),
                    gmm.mode_probs.cpu(),
                    gmm.cov_cholesky.cpu(),
                ))
                if sigma_next_val == 0:
                    x = x_clean
                else:
                    d = (x - x_clean) / sigma_val.clamp_min(1e-8)
                    x = x + (sigma_next_val - sigma_val) * d
            else:
                d       = (x - x_clean) / sigma_val.clamp_min(1e-8)
                x_euler = x + (sigma_next_val - sigma_val) * d

                sigma_next_b = sigma_next_val.expand(N)
                x_clean_next, hidden = model.denoise(x_euler, context, sigma_next_b)

                # Render the more refined (corrector) prediction at sigma_{i+1}
                gmm    = model.gmm_head(x_clean_next, hidden)
                gmm_mu = model.normalizer.denormalize(gmm.mu)
                frames.append(_build_frame(
                    float(sigma_next_val),
                    gmm_mu.cpu(),
                    gmm.mode_probs.cpu(),
                    gmm.cov_cholesky.cpu(),
                ))

                d_next = (x_euler - x_clean_next) / sigma_next_val.clamp_min(1e-8)
                x      = x + (sigma_next_val - sigma_val) * 0.5 * (d + d_next)

    return frames


# ── rendering ─────────────────────────────────────────────────────────────────

def _cov_ellipse(ax_mpl, mu_xy, L22, color, alpha):
    """1-σ ellipse from a world-frame (2,2) Cholesky factor."""
    Sigma    = L22 @ L22.T
    vals, V  = np.linalg.eigh(Sigma)
    vals     = np.maximum(vals, 1e-9)
    angle    = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    ell = Ellipse(xy=mu_xy, width=2*np.sqrt(vals[1]), height=2*np.sqrt(vals[0]),
                  angle=angle, facecolor=color, edgecolor=color,
                  alpha=alpha, lw=0.5, zorder=5)
    ax_mpl.add_patch(ell)


def _display_heading(agent: dict) -> float | None:
    """
    Choose a display heading that is visually meaningful.

    Waymo box headings can be ambiguous for pedestrians and sometimes disagree
    with motion direction. For non-stationary agents, prefer velocity when the
    stored heading is unreliable; for stationary pedestrians, suppress arrows.
    """
    heading = float(agent.get("anchor_heading", 0.0))
    vx = float(agent.get("anchor_vx", 0.0) or 0.0)
    vy = float(agent.get("anchor_vy", 0.0) or 0.0)
    speed = math.hypot(vx, vy)
    obj_type = int(agent.get("object_type", -1))
    if speed < 0.25 and obj_type == 2:
        return None
    if speed >= 0.5:
        vel_heading = math.atan2(vy, vx)
        disagreement = abs(_wrap(heading - vel_heading))
        if obj_type in (2, 3) or disagreement > math.radians(100):
            return vel_heading
    return heading


def _agent_box(ax_mpl, cx, cy, heading, box_l, box_w, color, arrow_heading=None):
    """Draw an oriented bounding box as a polygon in world coordinates."""
    # Box corners in agent frame
    corners = np.array([[-box_l/2,-box_w/2],[ box_l/2,-box_w/2],
                        [ box_l/2, box_w/2],[-box_l/2, box_w/2]])
    c, s  = np.cos(heading), np.sin(heading)
    R     = np.array([[c, -s], [s, c]])
    world = (R @ corners.T).T + np.array([cx, cy])
    poly  = plt.Polygon(world, closed=True,
                        facecolor=color, edgecolor="white", lw=1.2, alpha=0.85, zorder=8)
    ax_mpl.add_patch(poly)
    # Direction cue. It may use velocity rather than box yaw for agents whose
    # annotated heading is visually misleading.
    if arrow_heading is not None:
        ac, ass = np.cos(arrow_heading), np.sin(arrow_heading)
        arrow_len = max(box_l, box_w) * 0.65
        ax_mpl.annotate("", xy=(cx + arrow_len*ac, cy + arrow_len*ass), xytext=(cx, cy),
                        arrowprops=dict(arrowstyle="->", color="white", lw=1.0), zorder=9)


def _add_gaussian_to_heatmap(
    density: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    mu_xy: np.ndarray,
    cov: np.ndarray,
    weight: float,
    radius_sigma: float = 3.5,
    min_std: float = 0.0,
):
    """Accumulate one weighted 2D Gaussian PDF into a regular grid."""
    if weight <= 0 or not np.isfinite(mu_xy).all() or not np.isfinite(cov).all():
        return
    cov = cov + np.eye(2) * max(float(min_std) ** 2, 1e-5)
    try:
        vals = np.linalg.eigvalsh(cov)
        if np.any(vals <= 0):
            return
        inv = np.linalg.inv(cov)
        det = float(np.linalg.det(cov))
    except np.linalg.LinAlgError:
        return

    sx = math.sqrt(max(float(cov[0, 0]), 1e-5))
    sy = math.sqrt(max(float(cov[1, 1]), 1e-5))
    ix0 = max(0, int(np.searchsorted(xs, mu_xy[0] - radius_sigma * sx)))
    ix1 = min(len(xs), int(np.searchsorted(xs, mu_xy[0] + radius_sigma * sx, side="right")))
    iy0 = max(0, int(np.searchsorted(ys, mu_xy[1] - radius_sigma * sy)))
    iy1 = min(len(ys), int(np.searchsorted(ys, mu_xy[1] + radius_sigma * sy, side="right")))
    if ix1 <= ix0 or iy1 <= iy0:
        return

    X, Y = np.meshgrid(xs[ix0:ix1], ys[iy0:iy1])
    dx = X - mu_xy[0]
    dy = Y - mu_xy[1]
    expo = inv[0, 0] * dx * dx + 2.0 * inv[0, 1] * dx * dy + inv[1, 1] * dy * dy
    patch = weight * np.exp(-0.5 * expo) / (2.0 * math.pi * math.sqrt(det))
    density[iy0:iy1, ix0:ix1] += patch


def _world_gmm_heatmap(
    adata: list[dict],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    pixels: int = 520,
    timestep_stride: int = 1,
    min_std_m: float = 1.75,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a time-marginal GMM occupancy PDF on a BEV grid."""
    w = int(pixels)
    h = int(pixels)
    xs = np.linspace(xlim[0], xlim[1], w, dtype=np.float64)
    ys = np.linspace(ylim[0], ylim[1], h, dtype=np.float64)
    density = np.zeros((h, w), dtype=np.float64)

    for ad in adata:
        chol = ad.get("cov_chol_w")
        if chol is None:
            continue
        mu_w = ad["mu"]
        probs = np.asarray(ad["mode_probs"], dtype=np.float64)
        timesteps = list(range(0, mu_w.shape[1], max(1, int(timestep_stride))))
        denom = max(1, len(timesteps))
        for k, prob in enumerate(probs):
            for t in timesteps:
                L = chol[k, t]
                cov = L @ L.T
                _add_gaussian_to_heatmap(
                    density, xs, ys, mu_w[k, t], cov,
                    float(prob) / denom, min_std=min_std_m,
                )
    return density, xs, ys


def _overlay_density_on_axis(
    ax_mpl,
    density: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    alpha: float = 0.78,
    zorder: int = 2,
    contour: bool = True,
):
    """Draw a transparent heatmap where alpha follows normalized density."""
    if density.size == 0 or not np.isfinite(density).any() or float(np.nanmax(density)) <= 0:
        return
    scale = np.nanpercentile(density[density > 0], 97.5) if np.any(density > 0) else 0.0
    if scale <= 0:
        scale = float(np.nanmax(density))
    norm = np.clip(density / scale, 0.0, 1.0)
    rgba = matplotlib.colormaps["inferno"](norm)
    rgba[..., 3] = alpha * np.power(norm, 0.36)
    ax_mpl.imshow(rgba, extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
                  origin="lower", interpolation="bilinear", zorder=zorder)
    if contour:
        levels = np.linspace(0.18, 0.92, 5) * scale
        levels = levels[levels < float(np.nanmax(density))]
        if len(levels):
            ax_mpl.contour(
                density,
                levels=levels,
                extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
                origin="lower",
                colors=[(1.0, 0.9, 0.55, 0.36)],
                linewidths=0.75,
                zorder=zorder + 0.2,
            )


def render_frame(frame: dict, step_idx: int, total_steps: int, is_final: bool,
                 world_cx: float, world_cy: float, bev_half: float,
                 figsize=(10, 10)) -> Image.Image:
    """Render all agents in world coordinates for one diffusion step."""
    adata = frame["agents_data"]
    sigma = frame["sigma"]
    N     = len(adata)

    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(world_cx - bev_half, world_cx + bev_half)
    ax.set_ylim(world_cy - bev_half, world_cy + bev_half)
    ax.set_aspect("equal")
    ax.axis("off")

    for d in np.arange(world_cx - bev_half, world_cx + bev_half + 10, 10):
        ax.axvline(d, color=GRID_COLOR, lw=0.4)
    for d in np.arange(world_cy - bev_half, world_cy + bev_half + 10, 10):
        ax.axhline(d, color=GRID_COLOR, lw=0.4)

    legend_handles = []

    for a_idx, ad in enumerate(adata):
        color  = AGENT_COLORS[a_idx % len(AGENT_COLORS)]
        probs  = ad["mode_probs"]        # (K,)
        mu_w   = ad["mu"]               # (K, T, 2) world coords
        hist_w = ad["history_w"]        # (H, 2), NaN = invalid
        fut_w  = ad["future_w"]         # (T, 2), NaN = invalid
        K      = mu_w.shape[0]

        # observed history (dashed)
        vm = ~np.isnan(hist_w[:, 0])
        if vm.any():
            ax.plot(hist_w[vm, 0], hist_w[vm, 1],
                    color=color, lw=1.0, ls="--", alpha=0.55, zorder=3)
            ax.scatter(hist_w[vm, 0], hist_w[vm, 1],
                       color=color, s=10, alpha=0.4, zorder=4)

        # GT future (dotted)
        vf = ~np.isnan(fut_w[:, 0])
        if vf.any():
            ax.plot(fut_w[vf, 0], fut_w[vf, 1],
                    color=color, lw=1.5, ls=":", alpha=0.8, zorder=5)

        # mode predictions
        for k in range(K):
            p = float(probs[k])
            ax.plot(mu_w[k, :, 0], mu_w[k, :, 1],
                    color=color, lw=0.6 + 2.0*p, alpha=0.15 + 0.75*p, zorder=6)
            ax.scatter(mu_w[k, -1, 0], mu_w[k, -1, 1],
                       color=color, s=20, alpha=0.2 + 0.7*p, zorder=7)

        # covariance ellipses (final frame only)
        if is_final and ad["cov_chol_w"] is not None:
            for k in range(K):
                p = float(probs[k])
                for t in ELLIPSE_TS:
                    if t < mu_w.shape[1]:
                        _cov_ellipse(ax, mu_w[k, t], ad["cov_chol_w"][k, t],
                                     color, alpha=0.07 + 0.18*p)

        # bounding box at anchor (use actual measured dimensions from data)
        bl = ad.get("box_length") or OBJ_SIZE.get(ad["object_type"], DEFAULT_SZ)[0]
        bw = ad.get("box_width")  or OBJ_SIZE.get(ad["object_type"], DEFAULT_SZ)[1]
        _agent_box(ax, ad["anchor_x"], ad["anchor_y"], ad["anchor_heading"], bl, bw, color,
                   arrow_heading=_display_heading(ad))

        label = f"A{a_idx+1} {OBJ_NAMES.get(ad['object_type'],'?')}"
        legend_handles.append(Line2D([0], [0], color=color, lw=2.5, label=label))

    legend_handles += [
        Line2D([0], [0], color="#888", lw=1.0, ls="--", label="History"),
        Line2D([0], [0], color="#888", lw=1.5, ls=":",  label="Ground truth"),
    ]

    if step_idx == 0:
        title = f"Initial noise   σ = {sigma:.3f}   |   {N} agents"
    elif is_final:
        title = f"Step {step_idx}/{total_steps}   σ = {sigma:.4f}   Final GMM   |   {N} agents"
    else:
        title = f"Step {step_idx}/{total_steps}   σ = {sigma:.3f}   |   {N} agents"
    ax.set_title(title, color="white", fontsize=11, pad=7, fontweight="bold")

    ncol = max(1, (len(legend_handles) + 9) // 10)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7, ncol=ncol,
              facecolor="#14142a", edgecolor="#444", labelcolor="white", framealpha=0.85)

    fig.tight_layout(pad=0.3)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor=BG_COLOR)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return img


# ── static GMM final-output visualization ────────────────────────────────────

def render_gmm_png(
    frame: dict,
    world_cx: float, world_cy: float, bev_half: float,
    output_path: str = "gmm_output.png",
    dpi: int = 150,
    figsize=(13, 13),
    heatmap_pixels: int = 760,
    heatmap_alpha: float = 0.84,
    heatmap_min_std_m: float = 1.75,
    show_history: bool = True,
    show_ground_truth: bool = True,
):
    """
    Render a static BEV of the time-marginal future occupancy PDF:
      - All future timestep GMM components mixed into one spatial heatmap
      - Agent bounding boxes with heading arrows
      - Optional observed history / ground-truth context
    """
    adata = frame["agents_data"]
    N     = len(adata)
    xlim = (world_cx - bev_half, world_cx + bev_half)
    ylim = (world_cy - bev_half, world_cy + bev_half)

    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")

    for d in np.arange(world_cx - bev_half, world_cx + bev_half + 10, 10):
        ax.axvline(d, color=GRID_COLOR, lw=0.3)
    for d in np.arange(world_cy - bev_half, world_cy + bev_half + 10, 10):
        ax.axhline(d, color=GRID_COLOR, lw=0.3)

    density, _, _ = _world_gmm_heatmap(
        adata, xlim, ylim, pixels=heatmap_pixels, min_std_m=heatmap_min_std_m
    )
    _overlay_density_on_axis(ax, density, xlim, ylim, alpha=heatmap_alpha, zorder=2)

    legend_handles = []

    for a_idx, ad in enumerate(adata):
        color  = AGENT_COLORS[a_idx % len(AGENT_COLORS)]
        probs  = ad["mode_probs"]    # (K,)
        hist_w = ad["history_w"]

        # history
        vm = ~np.isnan(hist_w[:, 0])
        if show_history and vm.any():
            ax.plot(hist_w[vm, 0], hist_w[vm, 1],
                    color=color, lw=1.3, ls="--", alpha=0.85, zorder=6)
            ax.scatter(hist_w[vm, 0], hist_w[vm, 1],
                       color=color, s=14, alpha=0.7, zorder=7)

        # GT future context is useful for qualitative evaluation.
        fut_w  = ad["future_w"]
        vf = ~np.isnan(fut_w[:, 0])
        if show_ground_truth and vf.any():
            ax.plot(fut_w[vf, 0], fut_w[vf, 1],
                    color=color, lw=2.0, ls=":", alpha=0.9, zorder=8)

        # bounding box
        bl = ad.get("box_length") or OBJ_SIZE.get(ad["object_type"], DEFAULT_SZ)[0]
        bw = ad.get("box_width")  or OBJ_SIZE.get(ad["object_type"], DEFAULT_SZ)[1]
        _agent_box(ax, ad["anchor_x"], ad["anchor_y"], ad["anchor_heading"], bl, bw, color,
                   arrow_heading=_display_heading(ad))

        # mode probability annotation next to anchor
        top_k  = int(np.argmax(probs))
        label  = f"A{a_idx+1} {OBJ_NAMES.get(ad['object_type'],'?')}"
        legend_handles.append(
            Line2D([0], [0], color=color, lw=2.5,
                   label=f"{label}  best={probs[top_k]*100:.0f}%")
        )

    legend_handles += [
        mpatches.Patch(facecolor=matplotlib.colormaps["inferno"](0.82), edgecolor="none",
                       alpha=0.82, label="Time-marginal future PDF"),
    ]
    if show_history:
        legend_handles.append(Line2D([0], [0], color="#888", lw=1.2, ls="--", label="History"))
    if show_ground_truth:
        legend_handles.append(Line2D([0], [0], color="#888", lw=2.0, ls=":", label="Ground truth"))

    ax.set_title(
        f"Future occupancy PDF - all modes and future timesteps mixed  |  {N} agents",
        color="white", fontsize=13, pad=9, fontweight="bold",
    )
    ncol = max(1, (len(legend_handles) + 9) // 10)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7, ncol=ncol,
              facecolor="#14142a", edgecolor="#444", labelcolor="white", framealpha=0.88)

    fig.tight_layout(pad=0.4)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"GMM output saved → {output_path}")
    return output_path


# ── RGB overlay visualization ────────────────────────────────────────────────

def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_refinement_color(agent_idx: int) -> tuple[int, int, int]:
    return _hex_to_rgb(RGB_REFINEMENT_COLORS[agent_idx % len(RGB_REFINEMENT_COLORS)])


def _fit_affine(world_xy: np.ndarray, uv: np.ndarray) -> dict | None:
    if len(world_xy) < 3:
        return None
    X = np.column_stack([world_xy[:, 0], world_xy[:, 1], np.ones(len(world_xy))])
    try:
        B, *_ = np.linalg.lstsq(X, uv, rcond=None)
    except np.linalg.LinAlgError:
        return None
    pred = X @ B
    residual = float(np.median(np.linalg.norm(pred - uv, axis=1)))
    return {"type": "affine", "matrix": B, "residual_px": residual}


def _fit_temporal_affine(world_xy: np.ndarray, uv: np.ndarray, dt_s: np.ndarray,
                         target_world_xy: np.ndarray, target_uv: np.ndarray) -> dict | None:
    """Fit a same-camera local projection with a linear time term; evaluate at dt=0."""
    if len(world_xy) < 6:
        return None
    X = np.column_stack([world_xy[:, 0], world_xy[:, 1], dt_s, np.ones(len(world_xy))])
    try:
        B, *_ = np.linalg.lstsq(X, uv, rcond=None)
    except np.linalg.LinAlgError:
        return None
    pred = _project_points({"type": "temporal_affine", "matrix": B}, target_world_xy)
    if not np.isfinite(pred).all():
        return None
    residual = float(np.median(np.linalg.norm(pred - target_uv, axis=1)))
    return {"type": "temporal_affine", "matrix": B, "residual_px": residual}


def _fit_homography(world_xy: np.ndarray, uv: np.ndarray) -> dict | None:
    if len(world_xy) < 4:
        return None
    rows = []
    for (x, y), (u, v) in zip(world_xy, uv):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    A = np.asarray(rows, dtype=np.float64)
    try:
        _, _, vh = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    H = vh[-1].reshape(3, 3)
    if abs(H[2, 2]) > 1e-8:
        H = H / H[2, 2]
    pred = _project_points({"type": "homography", "matrix": H}, world_xy)
    if not np.isfinite(pred).all():
        return None
    residual = float(np.median(np.linalg.norm(pred - uv, axis=1)))
    return {"type": "homography", "matrix": H, "residual_px": residual}


def _project_points(projector: dict, points_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64)
    flat = pts.reshape(-1, 2)
    if projector["type"] == "affine":
        X = np.column_stack([flat[:, 0], flat[:, 1], np.ones(len(flat))])
        out = X @ projector["matrix"]
    elif projector["type"] == "temporal_affine":
        X = np.column_stack([flat[:, 0], flat[:, 1], np.zeros(len(flat)), np.ones(len(flat))])
        out = X @ projector["matrix"]
    else:
        X = np.column_stack([flat[:, 0], flat[:, 1], np.ones(len(flat))])
        uvw = X @ projector["matrix"].T
        denom = uvw[:, 2:3]
        denom = np.where(np.abs(denom) < 1e-8, np.nan, denom)
        out = uvw[:, :2] / denom
    return out.reshape(*pts.shape[:-1], 2)


def _projection_jacobian(projector: dict, xy: np.ndarray, eps: float = 0.25) -> np.ndarray:
    if projector["type"] == "affine":
        B = projector["matrix"]
        return np.array([[B[0, 0], B[1, 0]], [B[0, 1], B[1, 1]]], dtype=np.float64)
    if projector["type"] == "temporal_affine":
        B = projector["matrix"]
        return np.array([[B[0, 0], B[1, 0]], [B[0, 1], B[1, 1]]], dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float64)
    center = _project_points(projector, xy[None])[0]
    dx = _project_points(projector, (xy + np.array([eps, 0.0]))[None])[0]
    dy = _project_points(projector, (xy + np.array([0.0, eps]))[None])[0]
    return np.column_stack([(dx - center) / eps, (dy - center) / eps])


def _load_rgb_context(
    seg_path: Path,
    agents: list[dict],
    camera_name: int | None = 1,
    max_width: int = 1280,
    target_timestamp: int | None = None,
    require_selected_visible: bool = True,
) -> dict | None:
    """Load an RGB frame near the target timestamp and fit world→image projection."""
    image_path = seg_path / "images.parquet"
    link_path = seg_path / "image_trajectories.parquet"
    traj_path = seg_path / "trajectories.parquet"
    if not image_path.exists() or not link_path.exists() or not traj_path.exists():
        return None

    anchor_times = [a.get("anchor_timestamp") for a in agents if a.get("anchor_timestamp") is not None]
    if not anchor_times:
        return None
    target_time = int(target_timestamp) if target_timestamp is not None else max(set(anchor_times), key=anchor_times.count)

    images = _read_table_existing(image_path, IMAGE_COLS).to_pydict()
    links = _read_table_existing(link_path, LINK_COLS).to_pydict()
    trajs = _read_table_existing(traj_path, TRAJ_COLS).to_pydict()

    image_rows = [{k: images[k][i] for k in images} for i in range(len(images.get("image_id", [])))]
    link_rows = [{k: links[k][i] for k in links} for i in range(len(links.get("image_id", [])))]
    traj_rows = [{k: trajs[k][i] for k in trajs} for i in range(len(trajs.get("trajectory_row_id", [])))]
    if not image_rows or not link_rows or not traj_rows:
        return None

    selected_ids = {a.get("trajectory_row_id") for a in agents if a.get("trajectory_row_id")}
    links_by_image: dict[str, list[dict]] = {}
    for row in link_rows:
        links_by_image.setdefault(row["image_id"], []).append(row)

    exact_images = [row for row in image_rows if int(row["frame_timestamp_micros"]) == int(target_time)]
    candidates = exact_images or sorted(
        image_rows,
        key=lambda row: abs(int(row["frame_timestamp_micros"]) - int(target_time)),
    )[:5]
    if not candidates:
        return None

    def _score(row):
        image_links = links_by_image.get(row["image_id"], [])
        visible_selected = sum(1 for l in image_links if l.get("trajectory_row_id") in selected_ids)
        camera_bonus = 0
        if camera_name is not None:
            camera_bonus = 100 if int(row.get("camera_name", -1)) == int(camera_name) else 0
        elif int(row.get("camera_name", -1)) == 1:
            camera_bonus = 25
        return visible_selected * 1000 + len(image_links) * 10 + camera_bonus

    image_row = max(candidates, key=_score)
    image_links = links_by_image.get(image_row["image_id"], [])

    traj_by_id = {row.get("trajectory_row_id"): row for row in traj_rows}
    world_points, image_points = [], []
    target_gt_by_agent: dict[str, np.ndarray] = {}
    for link in image_links:
        traj = traj_by_id.get(link.get("trajectory_row_id"))
        if not traj:
            continue
        idx = _timestamp_index(traj.get("timestamps_micros"), int(image_row["frame_timestamp_micros"]))
        if idx is None or idx >= len(traj.get("x", [])) or idx >= len(traj.get("y", [])):
            continue
        world_xy = [float(traj["x"][idx]), float(traj["y"][idx])]
        image_xy = [float(link["bbox_center_x"]), float(link["bbox_center_y"])]
        world_points.append(world_xy)
        image_points.append(image_xy)
        if link.get("trajectory_row_id") in selected_ids:
            target_gt_by_agent[link["trajectory_row_id"]] = np.asarray(world_xy, dtype=np.float64)

    world_points = np.asarray(world_points, dtype=np.float64)
    image_points = np.asarray(image_points, dtype=np.float64)
    candidate_projectors = [
        p for p in (
            _fit_homography(world_points, image_points),
            _fit_affine(world_points, image_points),
        )
        if p is not None
    ]

    temporal_world, temporal_uv, temporal_dt = [], [], []
    target_camera = int(image_row.get("camera_name", -1))
    target_time_int = int(image_row["frame_timestamp_micros"])
    temporal_window_micros = 1_500_000
    same_camera_images = [
        row for row in image_rows
        if int(row.get("camera_name", -2)) == target_camera
        and abs(int(row["frame_timestamp_micros"]) - target_time_int) <= temporal_window_micros
    ]
    for img in same_camera_images:
        dt_s = (int(img["frame_timestamp_micros"]) - target_time_int) / 1_000_000.0
        for link in links_by_image.get(img["image_id"], []):
            traj = traj_by_id.get(link.get("trajectory_row_id"))
            if not traj:
                continue
            idx = _timestamp_index(traj.get("timestamps_micros"), int(img["frame_timestamp_micros"]))
            if idx is None or idx >= len(traj.get("x", [])) or idx >= len(traj.get("y", [])):
                continue
            temporal_world.append([float(traj["x"][idx]), float(traj["y"][idx])])
            temporal_uv.append([float(link["bbox_center_x"]), float(link["bbox_center_y"])])
            temporal_dt.append(dt_s)
    if temporal_world:
        temporal_projector = _fit_temporal_affine(
            np.asarray(temporal_world, dtype=np.float64),
            np.asarray(temporal_uv, dtype=np.float64),
            np.asarray(temporal_dt, dtype=np.float64),
            world_points,
            image_points,
        )
        if temporal_projector is not None:
            candidate_projectors.append(temporal_projector)

    projector = min(candidate_projectors, key=lambda item: item["residual_px"]) if candidate_projectors else None
    if projector is None:
        return None

    image = Image.open(BytesIO(image_row["image_jpeg"])).convert("RGB")
    pixel_scale = 1.0
    if max_width and image.width > max_width:
        pixel_scale = max_width / float(image.width)
        new_h = int(round(image.height * pixel_scale))
        image = image.resize((max_width, new_h), Image.Resampling.LANCZOS)

    links_by_agent = {
        row["trajectory_row_id"]: row
        for row in image_links
        if row.get("trajectory_row_id") in selected_ids
    }
    if require_selected_visible and not links_by_agent:
        return None
    agent_offsets = {}
    for rid, link in links_by_agent.items():
        gt_world = target_gt_by_agent.get(rid)
        if gt_world is None:
            continue
        projected = _project_points(projector, gt_world[None])[0]
        box_center = np.array([float(link["bbox_center_x"]), float(link["bbox_center_y"])], dtype=np.float64)
        if np.isfinite(projected).all():
            agent_offsets[rid] = box_center - projected

    return {
        "image": image,
        "image_row": image_row,
        "projector": projector,
        "pixel_scale": pixel_scale,
        "links_by_agent": links_by_agent,
        "agent_offsets": agent_offsets,
        "projectable_ids": set(links_by_agent),
    }


def _target_future_timestamp(seg_path: Path, agents: list[dict], future_step: int) -> int | None:
    """Return the most common timestamp at prediction time + future_step."""
    traj_path = seg_path / "trajectories.parquet"
    if not traj_path.exists():
        return None
    try:
        trajs = _read_table_existing(traj_path, TRAJ_COLS).to_pydict()
    except Exception:
        return None
    traj_rows = [{k: trajs[k][i] for k in trajs} for i in range(len(trajs.get("trajectory_row_id", [])))]
    traj_by_id = {row.get("trajectory_row_id"): row for row in traj_rows}
    step = max(1, int(future_step))
    timestamps = []
    for agent in agents:
        row = traj_by_id.get(agent.get("trajectory_row_id"))
        if not row:
            continue
        anchor_index = agent.get("anchor_index")
        times = row.get("timestamps_micros")
        if anchor_index is None or times is None:
            continue
        target_idx = int(anchor_index) + step
        if 0 <= target_idx < len(times):
            timestamps.append(int(times[target_idx]))
    if not timestamps:
        return None
    return max(set(timestamps), key=timestamps.count)


def _scaled_project(projector: dict, points_xy: np.ndarray, scale: float) -> np.ndarray:
    return _project_points(projector, points_xy) * float(scale)


def _scaled_project_agent(ctx: dict, ad: dict, points_xy: np.ndarray) -> np.ndarray:
    pts = _project_points(ctx["projector"], points_xy)
    offset = ctx.get("agent_offsets", {}).get(ad.get("trajectory_row_id"))
    if offset is not None:
        pts = pts + offset
    return pts * float(ctx["pixel_scale"])


def _draw_scaled_box(
    draw: ImageDraw.ImageDraw,
    link: dict,
    scale: float,
    color: tuple[int, int, int],
    width: int = 2,
    alpha: int = 210,
):
    x1 = float(link["bbox_x1"]) * scale
    y1 = float(link["bbox_y1"]) * scale
    x2 = float(link["bbox_x2"]) * scale
    y2 = float(link["bbox_y2"]) * scale
    draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 0, 190), width=width + 2)
    draw.rectangle([x1, y1, x2, y2], outline=color + (alpha,), width=width)


def _inside_image(pt: np.ndarray, width: int, height: int, margin: float = 0.0) -> bool:
    return (
        np.isfinite(pt).all()
        and -margin <= float(pt[0]) <= width + margin
        and -margin <= float(pt[1]) <= height + margin
    )


def _draw_history_dots(draw: ImageDraw.ImageDraw, ad: dict, projector: dict, scale: float,
                       size: tuple[int, int], color: tuple[int, int, int], ctx: dict | None = None):
    """Draw observed positions that project inside the RGB image."""
    hist = ad.get("history_w")
    if hist is None:
        return
    pts = _scaled_project_agent(ctx, ad, hist) if ctx is not None else _scaled_project(projector, hist, scale)
    w, h = size
    for idx, pt in enumerate(pts):
        if not _inside_image(pt, w, h):
            continue
        alpha = int(85 + 140 * (idx + 1) / max(1, len(pts)))
        radius = 3.0 if idx < len(pts) - 1 else 4.5
        x, y = float(pt[0]), float(pt[1])
        draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                     fill=color + (alpha,), outline=(255, 255, 255, min(180, alpha + 35)))


def _projected_polyline_segments(pts: np.ndarray, max_jump_px: float) -> list[np.ndarray]:
    """Split projected trajectories at large image-space jumps."""
    if len(pts) < 2:
        return []
    segments = []
    current = [pts[0]]
    for prev, cur in zip(pts[:-1], pts[1:]):
        if np.linalg.norm(cur - prev) > max_jump_px:
            if len(current) >= 2:
                segments.append(np.asarray(current))
            current = [cur]
        else:
            current.append(cur)
    if len(current) >= 2:
        segments.append(np.asarray(current))
    return segments


def _render_rgb_prediction_frame(frame: dict, ctx: dict, step_idx: int, total_steps: int) -> Image.Image:
    base = ctx["image"].convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    scale = float(ctx["pixel_scale"])
    projector = ctx["projector"]
    w, h = base.size

    for a_idx, ad in enumerate(frame["agents_data"]):
        rid = ad.get("trajectory_row_id")
        if rid not in ctx["projectable_ids"]:
            continue
        color = _rgb_refinement_color(a_idx)
        link = ctx["links_by_agent"].get(rid)
        if link:
            _draw_scaled_box(draw, link, scale, color, width=4, alpha=245)
        if ad.get("cov_chol_w") is None:
            continue
        probs = np.asarray(ad["mode_probs"], dtype=np.float64)
        order = np.argsort(probs)
        for k in order:
            pts = _scaled_project_agent(ctx, ad, ad["mu"][k])
            finite = np.isfinite(pts).all(axis=1)
            pts = pts[finite]
            if len(pts) < 2:
                continue
            pts[:, 0] = np.clip(pts[:, 0], -w, 2 * w)
            pts[:, 1] = np.clip(pts[:, 1], -h, 2 * h)
            alpha = int(np.clip(120 + 135 * float(probs[k]), 120, 255))
            width = max(3, int(round(3 + 8 * float(probs[k]))))
            segments = _projected_polyline_segments(pts, max_jump_px=max(w, h) * 0.45)
            for segment in segments:
                draw.line([tuple(p) for p in segment], fill=(255, 255, 255, 100), width=width + 2, joint="curve")
                draw.line([tuple(p) for p in segment], fill=color + (alpha,), width=width, joint="curve")
            ex, ey = pts[-1]
            r = 3.0 + 8.0 * float(probs[k])
            draw.ellipse([ex - r, ey - r, ex + r, ey + r], fill=color + (alpha,),
                         outline=(255, 255, 255, 220), width=2)

    if step_idx == 0:
        title = f"initial noise hidden  sigma={frame['sigma']:.4f}"
    else:
        title = f"diffusion {step_idx}/{total_steps}  sigma={frame['sigma']:.4f}"
    draw.rectangle([8, 8, 360, 34], fill=(0, 0, 0, 120))
    draw.text((16, 14), title, fill=(255, 255, 255, 230))
    return Image.alpha_composite(base, overlay).convert("RGB")


def _render_rgb_heatmap_png(
    frame: dict,
    ctx: dict,
    output_path: str,
    future_step: int = 5,
    heatmap_alpha: float = 0.62,
    heatmap_min_std_px: float = 8.0,
):
    """
    Render the future occupancy PDF at one target timestep onto the matching RGB frame.

    Dots are observed history positions projected into this RGB image. Boxes are
    projected boxes for target agents visible in the RGB frame. The heatmap is
    the model GMM at prediction time + future_step.
    """
    base = ctx["image"].convert("RGBA")
    w, h = base.size
    scale = float(ctx["pixel_scale"])
    projector = ctx["projector"]
    xs = np.arange(w, dtype=np.float64)
    ys = np.arange(h, dtype=np.float64)
    density = np.zeros((h, w), dtype=np.float64)
    t_idx = max(0, int(future_step) - 1)

    for ad in frame["agents_data"]:
        rid = ad.get("trajectory_row_id")
        if rid not in ctx["links_by_agent"] or ad.get("cov_chol_w") is None or t_idx >= ad["mu"].shape[1]:
            continue
        probs = np.asarray(ad["mode_probs"], dtype=np.float64)
        for k, prob in enumerate(probs):
            mu_world = ad["mu"][k, t_idx]
            mu_px = _scaled_project_agent(ctx, ad, mu_world[None])[0]
            if not _inside_image(mu_px, w, h, margin=max(w, h) * 0.5):
                continue
            J = _projection_jacobian(projector, mu_world) * scale
            cov_world = ad["cov_chol_w"][k, t_idx] @ ad["cov_chol_w"][k, t_idx].T
            cov_px = J @ cov_world @ J.T
            _add_gaussian_to_heatmap(
                density, xs, ys, mu_px, cov_px, float(prob),
                min_std=heatmap_min_std_px,
            )

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    if density.max() > 0:
        scale_den = np.percentile(density[density > 0], 98.5)
        norm = np.clip(density / max(scale_den, 1e-12), 0.0, 1.0)
        rgba = matplotlib.colormaps["inferno"](norm)
        rgba[..., 3] = heatmap_alpha * np.power(norm, 0.42)
        overlay = Image.fromarray((rgba * 255).astype(np.uint8), mode="RGBA")

    out = Image.alpha_composite(base, overlay)
    draw = ImageDraw.Draw(out, "RGBA")
    for a_idx, ad in enumerate(frame["agents_data"]):
        rid = ad.get("trajectory_row_id")
        if rid not in ctx["links_by_agent"]:
            continue
        color = _hex_to_rgb(AGENT_COLORS[a_idx % len(AGENT_COLORS)])
        _draw_history_dots(draw, ad, projector, scale, out.size, color, ctx=ctx)
        link = ctx["links_by_agent"].get(rid)
        if link:
            _draw_scaled_box(draw, link, scale, color)
    label = (
        f"RGB @ prediction+{future_step} | heatmap: predicted occupancy @ +{future_step} | "
        "dots: history | boxes: target objects"
    )
    draw.rectangle([8, 8, min(w - 8, 8 + 760), 38], fill=(0, 0, 0, 135))
    draw.text((16, 16), label, fill=(255, 255, 255, 235))
    out.convert("RGB").save(output_path)
    print(f"RGB heatmap saved → {output_path}")
    return output_path


def make_rgb_overlay(
    config_path: str,
    ckpt_path: str,
    output_path: str = "rgb_refinement.gif",
    output_final_png: str = "rgb_heatmap.png",
    scene_id: str | None = None,
    min_agents: int = 3,
    max_agents: int = 20,
    max_segs: int = 5,
    num_steps: int | None = None,
    frame_ms: int = 450,
    waymo_root: str | None = None,
    device: torch.device | None = None,
    camera_name: int | None = 1,
    rgb_max_width: int = 1280,
    rgb_heatmap_future_step: int = 5,
    render_final_heatmap: bool = True,
    anchor_timestamp: int | None = None,
):
    """Produce RGB GIF of refinement steps plus final transparent heatmap PNG."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model …")
    model, epoch = load_model(config_path, ckpt_path, device)
    print(f"  epoch: {epoch}")

    if waymo_root is None:
        with open(config_path) as f:
            waymo_root = Box(yaml.safe_load(f)).feature_extractor.waymo_root

    H, T = model.max_history, model.future_horizon
    print("Finding RGB-capable scene …")
    seg_path, chosen_sid, agents = find_scene(
        waymo_root, max_segs=max_segs,
        min_agents=min_agents, max_agents=max_agents,
        H=H, T=T, scene_id=str(scene_id) if scene_id else None,
        anchor_timestamp=anchor_timestamp,
    )
    print(f"  scene: {chosen_sid}  ({len(agents)} agents)")

    ctx = _load_rgb_context(seg_path, agents, camera_name=camera_name, max_width=rgb_max_width)
    if ctx is None:
        raise RuntimeError(
            f"No RGB frame/projection could be built for scene {chosen_sid}. "
            "Need images.parquet, image_trajectories.parquet, and enough visible objects."
        )
    row = ctx["image_row"]
    print(
        f"  RGB: {row['image_id']}  camera={row.get('camera_name_text', row.get('camera_name'))}  "
        f"projection={ctx['projector']['type']} median_residual={ctx['projector']['residual_px']:.1f}px"
    )

    print("Running diffusion sampling …")
    frames = sample_scene(model, agents, device, num_steps=num_steps)
    total = len(frames) - 1

    images = []
    for i, frame in enumerate(frames):
        images.append(_render_rgb_prediction_frame(frame, ctx, i, total))
        print(f"  rgb frame {i+1:3d}/{len(frames)}  σ={frame['sigma']:.4f}", end="\r")
    print()
    images += [images[-1]] * 2
    print(f"Saving RGB GIF → {output_path}")
    images[0].save(output_path, save_all=True, append_images=images[1:],
                   duration=frame_ms, loop=0, optimize=False)
    if not render_final_heatmap:
        return output_path, None

    future_timestamp = _target_future_timestamp(seg_path, agents, rgb_heatmap_future_step)
    heatmap_ctx = _load_rgb_context(
        seg_path, agents,
        camera_name=camera_name,
        max_width=rgb_max_width,
        target_timestamp=future_timestamp,
        require_selected_visible=False,
    )
    if heatmap_ctx is None:
        print("  future RGB context unavailable; falling back to prediction-time RGB for heatmap")
        heatmap_ctx = ctx
    else:
        row = heatmap_ctx["image_row"]
        print(
            f"  RGB heatmap frame: {row['image_id']}  "
            f"camera={row.get('camera_name_text', row.get('camera_name'))}  "
            f"projection={heatmap_ctx['projector']['type']} "
            f"median_residual={heatmap_ctx['projector']['residual_px']:.1f}px"
        )
    _render_rgb_heatmap_png(
        frames[-1], heatmap_ctx, output_final_png,
        future_step=rgb_heatmap_future_step,
    )
    return output_path, output_final_png


def make_gmm_png(
    config_path: str,
    ckpt_path: str,
    output_path: str = "gmm_output.png",
    scene_id: str | None = None,
    min_agents: int = 3,
    max_agents: int = 20,
    max_segs: int = 5,
    num_steps: int | None = None,
    heatmap_min_std_m: float = 1.75,
    waymo_root: str | None = None,
    device: torch.device | None = None,
    anchor_timestamp: int | None = None,
):
    """Produce a single static GMM visualization (no diffusion animation)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model …")
    model, epoch = load_model(config_path, ckpt_path, device)
    print(f"  epoch: {epoch}")

    if waymo_root is None:
        with open(config_path) as f:
            waymo_root = Box(yaml.safe_load(f)).feature_extractor.waymo_root

    H, T = model.max_history, model.future_horizon

    print(f"Finding scene …")
    seg_path, chosen_sid, agents = find_scene(
        waymo_root, max_segs=max_segs,
        min_agents=min_agents, max_agents=max_agents,
        H=H, T=T, scene_id=str(scene_id) if scene_id else None,
        anchor_timestamp=anchor_timestamp,
    )
    print(f"  scene: {chosen_sid}  ({len(agents)} agents)")

    print("Running inference (final step only) …")
    # Run full diffusion — only the last frame is used
    frames   = sample_scene(model, agents, device, num_steps=num_steps)
    final_fr = frames[-1]

    anchors  = np.array([[a["anchor_x"], a["anchor_y"]] for a in agents])
    cx, cy   = anchors.mean(axis=0)
    anchor_spread = np.max(np.linalg.norm(anchors - np.array([cx, cy]), axis=1))
    bev_half      = max(anchor_spread * 1.5 + 30.0, 40.0)

    render_gmm_png(
        final_fr, cx, cy, bev_half,
        output_path=output_path,
        heatmap_min_std_m=heatmap_min_std_m,
    )
    return output_path


# ── main ──────────────────────────────────────────────────────────────────────

def make_gif(
    config_path: str,
    ckpt_path: str,
    output_path: str = "scene_diffusion.gif",
    scene_id: str | None = None,
    min_agents: int = 3,
    max_agents: int = 20,
    max_segs: int = 5,
    num_steps: int | None = None,
    frame_ms: int = 450,
    waymo_root: str | None = None,
    device: torch.device | None = None,
    anchor_timestamp: int | None = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model …")
    model, epoch = load_model(config_path, ckpt_path, device)
    print(f"  epoch: {epoch}  |  normalizer mean={model.normalizer.mean.flatten().tolist()}")

    if waymo_root is None:
        with open(config_path) as f:
            waymo_root = Box(yaml.safe_load(f)).feature_extractor.waymo_root

    H = model.max_history
    T = model.future_horizon

    print(f"Searching for scene (min_agents={min_agents}, max_segs={max_segs}) …")
    seg_path, chosen_sid, agents = find_scene(
        waymo_root, max_segs=max_segs,
        min_agents=min_agents, max_agents=max_agents,
        H=H, T=T, scene_id=str(scene_id) if scene_id else None,
        anchor_timestamp=anchor_timestamp,
    )
    print(f"  scene: {chosen_sid}")
    print(f"  {len(agents)} agents  types={[OBJ_NAMES.get(a['object_type'],'?') for a in agents]}")

    print("Running diffusion sampling …")
    frames = sample_scene(model, agents, device, num_steps=num_steps)
    total  = len(frames) - 1
    print(f"  {len(frames)} frames  ({total} denoising steps + initial noise)")

    # World-space view: center on agents' current positions.
    # Extend only far enough to show all boxes + a fixed margin — do NOT extend
    # to cover full GT futures, which can be 200+ m and would shrink boxes to dots.
    anchors  = np.array([[a["anchor_x"], a["anchor_y"]] for a in agents])
    cx, cy   = anchors.mean(axis=0)
    anchor_spread = np.max(np.linalg.norm(anchors - np.array([cx, cy]), axis=1))
    bev_half = max(anchor_spread * 1.5 + 30.0, 40.0)

    images = []
    for i, frame in enumerate(frames):
        is_final = (i == len(frames) - 1)
        img = render_frame(frame, i, total, is_final, cx, cy, bev_half)
        images.append(img)
        print(f"  frame {i+1:3d}/{len(frames)}  σ={frame['sigma']:.4f}", end="\r")
    print()

    images += [images[-1]] * 2  # hold on final frame

    print(f"Saving → {output_path}")
    images[0].save(output_path, save_all=True, append_images=images[1:],
                   duration=frame_ms, loop=0, optimize=False)
    print("Done.")
    return output_path


def main():
    p = argparse.ArgumentParser(
        description="Visualize ASTRA-EDM for all agents in a scene.\n"
                    "  --mode gif   → animated diffusion GIF (default)\n"
                    "  --mode gmm   → static final GMM PDF heatmap PNG\n"
                    "  --mode rgb   → RGB refinement GIF + final RGB heatmap PNG\n"
                    "  --mode rgb_gif → RGB refinement GIF only\n"
                    "  --mode both  → BEV GIF + GMM heatmap PNG\n"
                    "  --mode all   → all visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config",       required=True)
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--output",       default="scene_diffusion.gif",
                                     help="GIF output path (or PNG if --mode gmm)")
    p.add_argument("--output_gmm",   default="gmm_output.png",
                                     help="Static GMM PNG path (used with --mode both or gmm)")
    p.add_argument("--output_rgb",   default="rgb_refinement.gif",
                                     help="RGB refinement GIF path (used with --mode rgb or all)")
    p.add_argument("--output_rgb_final", default="rgb_heatmap.png",
                                     help="Final RGB transparent heatmap PNG path")
    p.add_argument("--mode",         default="gif", choices=["gif", "gmm", "rgb", "rgb_gif", "both", "all"])
    p.add_argument("--scene_id",     default=None)
    p.add_argument("--anchor_timestamp", type=int, default=None,
                                    help="Force prediction anchor to this frame timestamp in micros")
    p.add_argument("--min_agents",   type=int, default=3)
    p.add_argument("--max_agents",   type=int, default=20)
    p.add_argument("--max_segments", type=int, default=5)
    p.add_argument("--num_steps",    type=int, default=None)
    p.add_argument("--frame_ms",     type=int, default=450)
    p.add_argument("--waymo_root",   default=None)
    p.add_argument("--device",       default=None)
    p.add_argument("--rgb_camera",   type=int, default=1,
                                    help="Preferred Waymo camera enum for RGB overlays (default: 1 FRONT)")
    p.add_argument("--rgb_max_width", type=int, default=1280,
                                    help="Downscale RGB overlays to this width; 0 keeps original size")
    p.add_argument("--rgb_heatmap_future_step", type=int, default=5,
                                    help="Future step used for final RGB heatmap and RGB frame selection")
    p.add_argument("--gmm_heatmap_min_std_m", type=float, default=1.75,
                                    help="Visual minimum stddev in meters for GMM heatmaps; prevents dot-like PDFs")
    a = p.parse_args()
    dev = torch.device(a.device) if a.device else None
    shared = dict(config_path=a.config, ckpt_path=a.checkpoint,
                  scene_id=a.scene_id, min_agents=a.min_agents,
                  max_agents=a.max_agents, max_segs=a.max_segments,
                  waymo_root=a.waymo_root, device=dev,
                  anchor_timestamp=a.anchor_timestamp)
    if a.mode in ("gif", "both", "all"):
        make_gif(**shared, output_path=a.output, num_steps=a.num_steps, frame_ms=a.frame_ms)
    if a.mode in ("gmm", "both", "all"):
        out = a.output_gmm if a.mode in ("both", "all") else a.output
        make_gmm_png(**shared, output_path=out, num_steps=a.num_steps,
                     heatmap_min_std_m=a.gmm_heatmap_min_std_m)
    if a.mode in ("rgb", "rgb_gif", "all"):
        make_rgb_overlay(**shared, output_path=a.output_rgb,
                         output_final_png=a.output_rgb_final,
                         num_steps=a.num_steps, frame_ms=a.frame_ms,
                         camera_name=a.rgb_camera,
                         rgb_max_width=a.rgb_max_width,
                         rgb_heatmap_future_step=a.rgb_heatmap_future_step,
                         render_final_heatmap=(a.mode != "rgb_gif"))


if __name__ == "__main__":
    main()
