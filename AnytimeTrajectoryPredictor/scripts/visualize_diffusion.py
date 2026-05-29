#!/usr/bin/env python3
"""
Visualize ASTRA-EDM diffusion denoising for ALL agents in a scene simultaneously.

All agents in the scene are batched through the same diffusion loop, and every
frame of the GIF shows their predictions in a shared world-coordinate BEV frame.

Usage:
    python visualize_diffusion.py \
        --config  configs/astra_edm_diffusion_waymo.yml \
        --checkpoint checkpoints/astra_edm_diffusion_waymo_latest.pth \
        --output  scene_diffusion.gif \
        --min_agents 4
"""

import argparse
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
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
BG_COLOR   = "#0d0d1a"
GRID_COLOR = "#1c1c3a"
ELLIPSE_TS = [9, 19, 39, 59, 79]
OBJ_SIZE   = {1: (4.5, 2.0), 2: (0.6, 0.6), 3: (1.8, 0.8)}
OBJ_NAMES  = {1: "Veh", 2: "Ped", 3: "Cyc"}
DEFAULT_SZ = (2.0, 1.0)

TRAJ_COLS = [
    "scene_id", "trajectory_row_id", "object_type", "num_steps",
    "x", "y", "heading", "velocity_x", "velocity_y",
    "length", "width",
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

def _build_agent_from_traj(data: dict, i: int, H: int, T: int):
    """
    Build agent-centric tensors from one row of trajectories.parquet.
    Uses the latest available window: history = last H steps, future = next ≤T steps.
    Returns (features, trajectory, obs_mask, fut_mask,
             anchor_x, anchor_y, anchor_heading, box_length, box_width).
    """
    num_steps = int(data["num_steps"][i] or 0)
    fut_len   = min(T, num_steps - H)
    if fut_len < 1:
        return None  # not enough data

    hist_end   = num_steps - fut_len
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

    return features, traj, obs_mask, fut_mask, ax, ay, ah, bl, bw


def load_scene_agents(seg_path: Path, H: int, T: int,
                      min_fut: int = 5, max_agents: int = 24) -> list:
    """Load all trajectories from a segment as a list of agent dicts (world-frame anchors included)."""
    table = pq.read_table(seg_path / "trajectories.parquet", columns=TRAJ_COLS)
    data  = table.to_pydict()
    agents = []
    for i in range(table.num_rows):
        result = _build_agent_from_traj(data, i, H, T)
        if result is None:
            continue
        feats, traj, omask, fmask, ax, ay, ah, bl, bw = result
        if fmask.sum() < min_fut:
            continue
        agents.append({
            "features":       torch.from_numpy(feats).unsqueeze(0).float(),
            "trajectory":     torch.from_numpy(traj).unsqueeze(0).float(),
            "observed_mask":  torch.from_numpy(omask).unsqueeze(0).float(),
            "future_mask":    torch.from_numpy(fmask).unsqueeze(0).float(),
            "anchor_x": ax, "anchor_y": ay, "anchor_heading": ah,
            "object_type": int(data["object_type"][i]),
            "box_length": bl, "box_width": bw,
        })
        if len(agents) >= max_agents:
            break
    return agents


def find_scene(waymo_root: str, max_segs: int, min_agents: int, max_agents: int,
               H: int, T: int, scene_id: str | None = None):
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
                agents = load_scene_agents(seg, H, T, max_agents=max_agents)
                if len(agents) >= min_agents:
                    return seg, seg.name, agents
        raise RuntimeError(f"Scene '{scene_id}' not found in {waymo_root}")

    best = (None, None, [])
    for seg in segments[:max_segs]:
        agents = load_scene_agents(seg, H, T, max_agents=max_agents)
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

        for i in range(len(sigmas) - 1):
            sigma      = sigmas[i].expand(N)
            sigma_next = sigmas[i + 1]

            x_clean, hidden = model.denoise(x, context, sigma)
            gmm    = model.gmm_head(x_clean, hidden)
            gmm_mu = model.normalizer.denormalize(gmm.mu)

            frames.append(_build_frame(
                float(sigma_next),
                gmm_mu.cpu(),
                gmm.mode_probs.cpu(),
                gmm.cov_cholesky.cpu(),
            ))

            if sigma_next == 0:
                x = x_clean
            else:
                d = (x - x_clean) / sigmas[i].clamp_min(1e-8)
                x = x + (sigma_next - sigmas[i]) * d

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


def _agent_box(ax_mpl, cx, cy, heading, box_l, box_w, color):
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
    # Heading arrow
    ax_mpl.annotate("", xy=(cx + box_l*0.65*c, cy + box_l*0.65*s), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.0), zorder=9)


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
        _agent_box(ax, ad["anchor_x"], ad["anchor_y"], ad["anchor_heading"], bl, bw, color)

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
):
    """
    Render a high-quality static BEV showing each agent's full GMM output:
      - All K modes as colored trajectories (α and lw ∝ mode probability)
      - 1-σ covariance ellipses at 5 key future timesteps
      - Agent bounding boxes with heading arrows
      - Observed history paths
      - Ground-truth future trajectories
      - Mode probability bar chart inset per agent
    """
    adata = frame["agents_data"]
    N     = len(adata)

    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(world_cx - bev_half, world_cx + bev_half)
    ax.set_ylim(world_cy - bev_half, world_cy + bev_half)
    ax.set_aspect("equal")
    ax.axis("off")

    for d in np.arange(world_cx - bev_half, world_cx + bev_half + 10, 10):
        ax.axvline(d, color=GRID_COLOR, lw=0.3)
    for d in np.arange(world_cy - bev_half, world_cy + bev_half + 10, 10):
        ax.axhline(d, color=GRID_COLOR, lw=0.3)

    legend_handles = []

    for a_idx, ad in enumerate(adata):
        color  = AGENT_COLORS[a_idx % len(AGENT_COLORS)]
        probs  = ad["mode_probs"]    # (K,)
        mu_w   = ad["mu"]           # (K, T, 2)
        hist_w = ad["history_w"]
        fut_w  = ad["future_w"]
        K      = mu_w.shape[0]

        # history
        vm = ~np.isnan(hist_w[:, 0])
        if vm.any():
            ax.plot(hist_w[vm, 0], hist_w[vm, 1],
                    color=color, lw=1.2, ls="--", alpha=0.6, zorder=3)
            ax.scatter(hist_w[vm, 0], hist_w[vm, 1],
                       color=color, s=12, alpha=0.45, zorder=4)

        # GT future
        vf = ~np.isnan(fut_w[:, 0])
        if vf.any():
            ax.plot(fut_w[vf, 0], fut_w[vf, 1],
                    color=color, lw=2.0, ls=":", alpha=0.85, zorder=5)

        # sort modes by probability so best mode is drawn last (on top)
        order = np.argsort(probs)
        for k in order:
            p = float(probs[k])
            ax.plot(mu_w[k, :, 0], mu_w[k, :, 1],
                    color=color, lw=0.5 + 3.0*p, alpha=0.1 + 0.85*p, zorder=6)
            ax.scatter(mu_w[k, -1, 0], mu_w[k, -1, 1],
                       color=color, s=25 + 60*p, alpha=0.15 + 0.8*p, zorder=7)

        # covariance ellipses for all modes
        if ad["cov_chol_w"] is not None:
            for k in order:
                p = float(probs[k])
                for t in ELLIPSE_TS:
                    if t < mu_w.shape[1]:
                        _cov_ellipse(ax, mu_w[k, t], ad["cov_chol_w"][k, t],
                                     color, alpha=0.06 + 0.20*p)

        # bounding box
        bl = ad.get("box_length") or OBJ_SIZE.get(ad["object_type"], DEFAULT_SZ)[0]
        bw = ad.get("box_width")  or OBJ_SIZE.get(ad["object_type"], DEFAULT_SZ)[1]
        _agent_box(ax, ad["anchor_x"], ad["anchor_y"], ad["anchor_heading"], bl, bw, color)

        # mode probability annotation next to anchor
        top_k  = int(np.argmax(probs))
        label  = f"A{a_idx+1} {OBJ_NAMES.get(ad['object_type'],'?')}"
        legend_handles.append(
            Line2D([0], [0], color=color, lw=2.5,
                   label=f"{label}  best={probs[top_k]*100:.0f}%")
        )

    legend_handles += [
        Line2D([0], [0], color="#888", lw=1.2, ls="--", label="History"),
        Line2D([0], [0], color="#888", lw=2.0, ls=":",  label="Ground truth"),
        mpatches.Patch(facecolor="none", edgecolor="#888", lw=0.8, label="1-σ ellipses"),
    ]

    ax.set_title(f"Final GMM output — {N} agents  |  {len(adata[0]['mode_probs'])} modes each",
                 color="white", fontsize=13, pad=9, fontweight="bold")
    ncol = max(1, (len(legend_handles) + 9) // 10)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7, ncol=ncol,
              facecolor="#14142a", edgecolor="#444", labelcolor="white", framealpha=0.88)

    fig.tight_layout(pad=0.4)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"GMM output saved → {output_path}")
    return output_path


def make_gmm_png(
    config_path: str,
    ckpt_path: str,
    output_path: str = "gmm_output.png",
    scene_id: str | None = None,
    min_agents: int = 3,
    max_agents: int = 20,
    max_segs: int = 5,
    waymo_root: str | None = None,
    device: torch.device | None = None,
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
    )
    print(f"  scene: {chosen_sid}  ({len(agents)} agents)")

    print("Running inference (final step only) …")
    # Run full diffusion — only the last frame is used
    frames   = sample_scene(model, agents, device)
    final_fr = frames[-1]

    anchors  = np.array([[a["anchor_x"], a["anchor_y"]] for a in agents])
    cx, cy   = anchors.mean(axis=0)
    anchor_spread = np.max(np.linalg.norm(anchors - np.array([cx, cy]), axis=1))
    bev_half      = max(anchor_spread * 1.5 + 30.0, 40.0)

    render_gmm_png(final_fr, cx, cy, bev_half, output_path=output_path)
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
                    "  --mode gmm   → static final GMM output PNG\n"
                    "  --mode both  → GIF + PNG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config",       required=True)
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--output",       default="scene_diffusion.gif",
                                     help="GIF output path (or PNG if --mode gmm)")
    p.add_argument("--output_gmm",   default="gmm_output.png",
                                     help="Static GMM PNG path (used with --mode both or gmm)")
    p.add_argument("--mode",         default="gif", choices=["gif", "gmm", "both"])
    p.add_argument("--scene_id",     default=None)
    p.add_argument("--min_agents",   type=int, default=3)
    p.add_argument("--max_agents",   type=int, default=20)
    p.add_argument("--max_segments", type=int, default=5)
    p.add_argument("--num_steps",    type=int, default=None)
    p.add_argument("--frame_ms",     type=int, default=450)
    p.add_argument("--waymo_root",   default=None)
    p.add_argument("--device",       default=None)
    a = p.parse_args()
    dev = torch.device(a.device) if a.device else None
    shared = dict(config_path=a.config, ckpt_path=a.checkpoint,
                  scene_id=a.scene_id, min_agents=a.min_agents,
                  max_agents=a.max_agents, max_segs=a.max_segments,
                  waymo_root=a.waymo_root, device=dev)
    if a.mode in ("gif", "both"):
        make_gif(**shared, output_path=a.output, num_steps=a.num_steps, frame_ms=a.frame_ms)
    if a.mode in ("gmm", "both"):
        out = a.output_gmm if a.mode == "both" else a.output
        make_gmm_png(**shared, output_path=out)


if __name__ == "__main__":
    main()
