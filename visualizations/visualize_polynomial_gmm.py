#!/usr/bin/env python3
"""Visualize GRU/GNN polynomial-GMM trajectory outputs.

This tool is for the output contract used by the non-diffusion models and the
diversity evaluator:

    predictions = [
        Tensor(B, N, K * (D + D * D)),
        ...
    ]

where each frame predicts K trajectory modes, each mode stores D polynomial
mean coefficients followed by a DxD covariance parameter matrix. The current
models use D = 3 dims * 4 cubic coefficients. This visualizer plots the x/y
polynomial modes in the same coordinate space as the model inputs, typically
bbox center pixels.

Examples:
    python visualizations/visualize_polynomial_gmm.py \
        --config configs/gru_config.yml \
        --predictions outputs/predictions.pt \
        --image outputs/rgb_frame.jpg \
        --output visualizations/outs/gru_prediction.png

    python visualizations/visualize_polynomial_gmm.py \
        --config configs/gnn.yml \
        --checkpoint checkpoint/gnn.pth \
        --batch outputs/batch.pt \
        --output visualizations/outs/gnn_prediction.png \
        --gif visualizations/outs/gnn_prediction.gif
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from box import Box
from matplotlib.patches import Ellipse
from PIL import Image

PROJECT_ROOT = next(
    p for p in Path(__file__).resolve().parents
    if (p / "AnytimeTrajectoryPredictor").is_dir()
)
sys.path.insert(0, str(PROJECT_ROOT))

from AnytimeTrajectoryPredictor.models.TrajectoryPredictor import TrajectoryPredictor

IMAGE_COLS = [
    "image_id",
    "scene_id",
    "frame_timestamp_micros",
    "camera_name",
    "camera_name_text",
    "image_jpeg",
    "image_width",
    "image_height",
]


@dataclass(frozen=True)
class PolynomialGMMSpec:
    num_modes: int
    num_coeffs: int = 4
    num_dims: int = 3
    covariance_eps: float = 1e-3

    @property
    def coeff_dim(self) -> int:
        return self.num_dims * self.num_coeffs

    @property
    def params_per_mode(self) -> int:
        return self.coeff_dim + self.coeff_dim * self.coeff_dim

    @property
    def output_dim(self) -> int:
        return self.num_modes * self.params_per_mode


def _spec_from_model(model: torch.nn.Module) -> PolynomialGMMSpec:
    return PolynomialGMMSpec(
        num_modes=int(getattr(model, "num_trajectory_possibilities")),
        num_coeffs=int(getattr(model, "num_coeffs", 4)),
        num_dims=int(getattr(model, "num_dims", 3)),
        covariance_eps=float(getattr(model, "COVARIANCE_EPS", 1e-3)),
    )


def _spec_from_config(config_path: str, num_modes: int | None) -> PolynomialGMMSpec:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = Box(yaml.safe_load(f))
    modes = num_modes if num_modes is not None else int(cfg.model.num_trajectory_possibilities)
    return PolynomialGMMSpec(num_modes=modes)


def _load_model(config_path: str, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = Box(yaml.safe_load(f))
    model = TrajectoryPredictor.create_model(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _load_torch_payload(path: str, device: torch.device | None = None) -> Any:
    map_location = device if device is not None else "cpu"
    return torch.load(path, map_location=map_location, weights_only=False)


def _find_payload_value(payload: Any, names: tuple[str, ...]) -> Any | None:
    if not isinstance(payload, dict):
        return None
    for name in names:
        if name in payload:
            return payload[name]
    return None


def _read_table_existing(path: Path, columns: list[str]):
    schema = pq.read_schema(path)
    present = [column for column in columns if column in schema.names]
    return pq.read_table(path, columns=present)


def _config_waymo_root(config_path: str | None) -> str | None:
    if not config_path:
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = Box(yaml.safe_load(f))
    feature_extractor = getattr(cfg, "feature_extractor", {})
    for key in ("dataset_root", "waymo_root"):
        value = getattr(feature_extractor, key, None)
        if value is not None:
            return str(value)
    return None


def _resolve_waymo_segment(waymo_root: str | None, segment: str | None) -> Path | None:
    if segment:
        path = Path(segment)
        if path.exists():
            return path
        if waymo_root:
            candidate = Path(waymo_root) / segment
            if candidate.exists():
                return candidate
    if waymo_root:
        root = Path(waymo_root)
        if (root / "images.parquet").exists():
            return root
        candidates = sorted(
            p for p in root.iterdir()
            if p.is_dir() and (p / "images.parquet").exists()
        )
        if candidates:
            return candidates[0]
    return None


def _row_matches_waymo_request(
    row: dict,
    image_id: str | None,
    timestamp: int | None,
    camera_name: int | None,
) -> bool:
    if image_id is not None and str(row.get("image_id")) != str(image_id):
        return False
    if timestamp is not None and int(row.get("frame_timestamp_micros", -1)) != int(timestamp):
        return False
    if camera_name is not None and int(row.get("camera_name", -1)) != int(camera_name):
        return False
    return True


def _load_waymo_rgb_image(
    waymo_root: str | None,
    segment: str | None = None,
    image_id: str | None = None,
    timestamp: int | None = None,
    camera_name: int | None = 1,
    max_width: int = 1280,
) -> tuple[Image.Image | None, dict | None]:
    """Load an RGB frame from Waymo parquet images by image id or timestamp/camera."""
    seg_path = _resolve_waymo_segment(waymo_root, segment)
    if seg_path is None:
        return None, None
    image_path = seg_path / "images.parquet"
    if not image_path.exists():
        return None, None

    images = _read_table_existing(image_path, IMAGE_COLS).to_pydict()
    rows = [{key: images[key][i] for key in images} for i in range(len(images.get("image_id", [])))]
    if not rows:
        return None, None

    matches = [
        row for row in rows
        if _row_matches_waymo_request(row, image_id, timestamp, camera_name)
    ]
    if not matches and timestamp is not None:
        camera_rows = [
            row for row in rows
            if camera_name is None or int(row.get("camera_name", -1)) == int(camera_name)
        ]
        matches = sorted(
            camera_rows or rows,
            key=lambda row: abs(int(row["frame_timestamp_micros"]) - int(timestamp)),
        )[:1]
    if not matches and image_id is None and timestamp is None:
        matches = [
            row for row in rows
            if camera_name is None or int(row.get("camera_name", -1)) == int(camera_name)
        ] or rows
    if not matches:
        return None, None

    row = matches[0]
    image = Image.open(BytesIO(row["image_jpeg"])).convert("RGB")
    scale = 1.0
    if max_width and image.width > max_width:
        scale = max_width / float(image.width)
        new_h = int(round(image.height * scale))
        image = image.resize((max_width, new_h), Image.Resampling.LANCZOS)
    meta = dict(row)
    meta["segment_path"] = str(seg_path)
    meta["pixel_scale"] = scale
    return image, meta


def _to_rgb_image(value: Any) -> Image.Image | None:
    """Convert common payload image formats to a PIL RGB image."""
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (str, Path)):
        return Image.open(value).convert("RGB")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return Image.open(BytesIO(bytes(value))).convert("RGB")
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 2:
        return Image.fromarray(_image_uint8(array), mode="L").convert("RGB")
    if array.ndim == 3 and array.shape[-1] in (3, 4):
        return Image.fromarray(_image_uint8(array[..., :3])).convert("RGB")
    raise ValueError(f"Unsupported image payload shape {array.shape}")


def _image_uint8(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.dtype == np.uint8:
        return array
    array = array.astype(np.float32)
    if array.size and float(np.nanmax(array)) <= 1.0:
        array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def _payload_image(payload: Any) -> Image.Image | None:
    value = _find_payload_value(
        payload,
        ("image", "rgb_image", "image_rgb", "image_jpeg", "image_bytes", "jpeg"),
    )
    return _to_rgb_image(value) if value is not None else None


def _looks_like_prediction_frame(frame: torch.Tensor, spec: PolynomialGMMSpec) -> bool:
    if frame.dim() == 3 and frame.shape[-1] == spec.output_dim:
        return True
    return frame.dim() == 4 and frame.shape[-2:] == (spec.num_modes, spec.params_per_mode)


def _split_prediction_tensor(
    predictions: torch.Tensor,
    spec: PolynomialGMMSpec,
    layout: str = "auto",
) -> list[torch.Tensor]:
    if _looks_like_prediction_frame(predictions, spec):
        return [predictions]
    if predictions.dim() not in (4, 5):
        raise ValueError(
            "Prediction tensor must be a single frame or include a frame dimension, "
            "for example (T, B, N, output_dim) or (B, T, N, output_dim)."
        )
    if layout == "time_first":
        return [predictions[i] for i in range(predictions.shape[0])]
    if layout == "batch_first":
        return [predictions[:, i] for i in range(predictions.shape[1])]
    if predictions.shape[0] == 1 and predictions.shape[1] > 1:
        return [predictions[:, i] for i in range(predictions.shape[1])]
    if predictions.shape[1] == 1 and predictions.shape[0] > 1:
        return [predictions[i] for i in range(predictions.shape[0])]
    return [predictions[i] for i in range(predictions.shape[0])]


def _as_prediction_list(
    predictions: Any,
    spec: PolynomialGMMSpec,
    layout: str = "auto",
) -> list[torch.Tensor]:
    if isinstance(predictions, (list, tuple)):
        frames = list(predictions)
    elif torch.is_tensor(predictions):
        frames = _split_prediction_tensor(predictions, spec, layout)
    else:
        raise TypeError("predictions must be a list/tuple of tensors or a tensor")
    if not frames:
        raise ValueError("No prediction frames were provided")
    return [frame.detach().cpu() for frame in frames]


def _frame_to_modes(frame: torch.Tensor, spec: PolynomialGMMSpec) -> torch.Tensor:
    """Return one prediction frame as (B, N, K, params_per_mode)."""
    if frame.dim() == 3 and frame.shape[-1] == spec.output_dim:
        return frame.view(frame.shape[0], frame.shape[1], spec.num_modes, spec.params_per_mode)
    if frame.dim() == 4 and frame.shape[-2:] == (spec.num_modes, spec.params_per_mode):
        return frame
    if frame.dim() == 4 and frame.shape[-1] == spec.output_dim:
        if frame.shape[0] != 1:
            raise ValueError(
                "Got a 4D frame with a flattened output dim. Pass predictions with "
                "one explicit frame dimension, not both frame and mode dimensions."
            )
        squeezed = frame.squeeze(0)
        return squeezed.view(squeezed.shape[0], squeezed.shape[1], spec.num_modes, spec.params_per_mode)
    raise ValueError(
        f"Unsupported prediction frame shape {tuple(frame.shape)}. Expected "
        f"(B, N, {spec.output_dim}) or (B, N, {spec.num_modes}, {spec.params_per_mode})."
    )


def extract_polynomial_gmm_params(
    predictions: Any,
    spec: PolynomialGMMSpec,
    prediction_layout: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract means and stabilized covariances from GRU/GNN output tensors.

    Returns:
        means: (B, T, N, K, D)
        covs:  (B, T, N, K, D, D)
    """
    frames = [
        _frame_to_modes(frame, spec)
        for frame in _as_prediction_list(predictions, spec, prediction_layout)
    ]
    means = []
    covs = []
    eye = torch.eye(spec.coeff_dim)
    for frame in frames:
        frame_means = frame[..., : spec.coeff_dim]
        raw_covs = frame[..., spec.coeff_dim :].view(*frame.shape[:3], spec.coeff_dim, spec.coeff_dim)
        raw_covs = 0.5 * (raw_covs + raw_covs.transpose(-1, -2))
        stable_covs = raw_covs @ raw_covs.transpose(-1, -2) + spec.covariance_eps * eye
        means.append(frame_means)
        covs.append(stable_covs)
    return torch.stack(means, dim=1), torch.stack(covs, dim=1)


def _features_to_tbn(
    features: torch.Tensor,
    layout: str = "auto",
    num_frames: int | None = None,
) -> torch.Tensor:
    if features.dim() != 4:
        raise ValueError("features must have shape (T, B, N, F) or (B, T, N, F)")
    if layout == "time_first":
        return features
    if layout == "batch_first":
        return features.transpose(0, 1)
    if num_frames is not None:
        if features.shape[0] == num_frames:
            return features
        if features.shape[1] == num_frames:
            return features.transpose(0, 1)
    # Dataloaders produce (B, T, N, F), while models consume (T, B, N, F).
    if features.shape[0] == 1 and features.shape[1] > 1:
        return features.transpose(0, 1)
    if features.shape[1] == 1 and features.shape[0] > 1:
        return features
    if features.shape[0] > features.shape[1]:
        return features.transpose(0, 1)
    return features


def _trajectory_to_btncc(
    trajectory: torch.Tensor,
    num_coeffs: int,
    layout: str = "auto",
    num_frames: int | None = None,
) -> torch.Tensor:
    if trajectory.dim() != 5:
        raise ValueError("trajectory must have shape (B, T, N, dims, coeffs) or (T, B, N, dims, coeffs)")
    if trajectory.shape[-1] != num_coeffs:
        raise ValueError(f"trajectory last dimension must be num_coeffs={num_coeffs}")
    if layout == "batch_first":
        return trajectory
    if layout == "time_first":
        return trajectory.transpose(0, 1)
    if num_frames is not None:
        if trajectory.shape[1] == num_frames:
            return trajectory
        if trajectory.shape[0] == num_frames:
            return trajectory.transpose(0, 1)
    if trajectory.shape[0] == 1 and trajectory.shape[1] > 1:
        return trajectory
    if trajectory.shape[1] == 1 and trajectory.shape[0] > 1:
        return trajectory.transpose(0, 1)
    if trajectory.shape[0] > trajectory.shape[1]:
        return trajectory
    return trajectory.transpose(0, 1)


def _basis(num_points: int, num_coeffs: int) -> np.ndarray:
    t = np.linspace(-1.0, 1.0, num_points, dtype=np.float64)
    return np.stack([t**i for i in range(num_coeffs)], axis=1)


def evaluate_xy_coeffs(coeffs: np.ndarray, num_points: int) -> np.ndarray:
    """Evaluate flattened polynomial coefficients into an (S, 2) curve."""
    num_coeffs = coeffs.shape[-1] // 3
    basis = _basis(num_points, num_coeffs)
    x = basis @ coeffs[:num_coeffs]
    y = basis @ coeffs[num_coeffs : 2 * num_coeffs]
    return np.stack([x, y], axis=-1)


def _xy_covariances(coeff_cov: np.ndarray, num_points: int, num_coeffs: int) -> np.ndarray:
    basis = _basis(num_points, num_coeffs)
    covs = []
    for b in basis:
        projector = np.zeros((2, coeff_cov.shape[-1]), dtype=np.float64)
        projector[0, :num_coeffs] = b
        projector[1, num_coeffs : 2 * num_coeffs] = b
        covs.append(projector @ coeff_cov @ projector.T)
    return np.stack(covs, axis=0)


def _add_cov_ellipse(ax: plt.Axes, center: np.ndarray, cov: np.ndarray, color: str, alpha: float) -> None:
    cov = 0.5 * (cov + cov.T)
    try:
        vals, vecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return
    vals = np.maximum(vals, 1e-9)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    ellipse = Ellipse(
        xy=center,
        width=2.0 * np.sqrt(vals[1]),
        height=2.0 * np.sqrt(vals[0]),
        angle=angle,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        linewidth=0.8,
    )
    ax.add_patch(ellipse)


def _object_is_visible(features_tbn: torch.Tensor | None, frame_idx: int, batch_idx: int, object_idx: int) -> bool:
    if features_tbn is None:
        return True
    history = features_tbn[: frame_idx + 1, batch_idx, object_idx, :2]
    return bool(torch.isfinite(history).all() and history.abs().sum() > 0)


def _line_effects(on_image: bool):
    if not on_image:
        return None
    return [
        pe.Stroke(linewidth=4.6, foreground=(0, 0, 0, 0.78)),
        pe.Normal(),
    ]


def _plot_styled_line(
    ax: plt.Axes,
    xy: np.ndarray,
    *,
    color: str,
    linestyle: str,
    linewidth: float,
    alpha: float,
    label: str | None,
    on_image: bool,
    zorder: float,
):
    line, = ax.plot(
        xy[:, 0],
        xy[:, 1],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        label=label,
        zorder=zorder,
    )
    effects = _line_effects(on_image)
    if effects is not None:
        line.set_path_effects(effects)
    return line


def plot_prediction_frame(
    means: torch.Tensor,
    covs: torch.Tensor,
    spec: PolynomialGMMSpec,
    output_path: str | None = None,
    features_tbn: torch.Tensor | None = None,
    trajectory_btncc: torch.Tensor | None = None,
    frame_idx: int = -1,
    batch_idx: int = 0,
    object_indices: list[int] | None = None,
    future_points: int = 90,
    max_objects: int = 5,
    show_covariance: bool = True,
    invert_y: bool = True,
    background_image: Image.Image | None = None,
    coordinate_scale: float = 1.0,
    show_axes: bool = False,
) -> Image.Image:
    """Render one prediction frame and optionally save it."""
    num_frames = means.shape[1]
    if frame_idx < 0:
        frame_idx = num_frames + frame_idx
    frame_idx = int(np.clip(frame_idx, 0, num_frames - 1))

    num_objects = means.shape[2]
    if object_indices is None:
        object_indices = [
            i for i in range(num_objects)
            if _object_is_visible(features_tbn, frame_idx, batch_idx, i)
        ][:max_objects]
    if not object_indices:
        object_indices = list(range(min(max_objects, num_objects)))

    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    on_image = background_image is not None
    if on_image:
        image_w, image_h = background_image.size
        fig_w = max(7.0, image_w / 160.0)
        fig_h = max(5.0, image_h / 160.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="black")
        ax.imshow(background_image)
        ax.set_xlim(0, image_w)
        ax.set_ylim(image_h, 0)
        ax.set_title(
            f"Polynomial GMM overlay | frame {frame_idx} | batch {batch_idx}",
            color="white",
            fontsize=10,
        )
        if show_axes:
            ax.tick_params(colors="white")
            ax.set_xlabel("image x", color="white")
            ax.set_ylabel("image y", color="white")
        else:
            ax.axis("off")
    else:
        fig, ax = plt.subplots(figsize=(9, 7), facecolor="white")
        ax.set_title(f"Polynomial GMM predictions | frame {frame_idx} | batch {batch_idx}")
        ax.set_xlabel("x / bbox center x")
        ax.set_ylabel("y / bbox center y")
        ax.grid(True, color="#d7dbe2", linewidth=0.7, alpha=0.8)
    ax.set_aspect("equal", adjustable="box" if on_image else "datalim")

    for object_order, object_idx in enumerate(object_indices):
        color = palette[object_order % len(palette)]
        label_prefix = f"obj {object_idx}"

        if features_tbn is not None:
            history = features_tbn[: frame_idx + 1, batch_idx, object_idx, :2].detach().cpu().numpy()
            valid = np.isfinite(history).all(axis=1) & (np.abs(history).sum(axis=1) > 0)
            if valid.any():
                history_xy = history[valid] * coordinate_scale
                _plot_styled_line(
                    ax,
                    history_xy,
                    color=color,
                    linestyle="--",
                    linewidth=2.2 if on_image else 1.6,
                    alpha=0.95 if on_image else 0.8,
                    label=f"{label_prefix} history",
                    on_image=on_image,
                    zorder=8,
                )
                ax.scatter(
                    history_xy[:, 0],
                    history_xy[:, 1],
                    color=color,
                    edgecolor="black" if on_image else color,
                    linewidth=0.5 if on_image else 0.0,
                    s=22 if on_image else 14,
                    alpha=0.95 if on_image else 0.8,
                    zorder=9,
                )

        if trajectory_btncc is not None and frame_idx < trajectory_btncc.shape[1]:
            gt_coeffs = trajectory_btncc[batch_idx, frame_idx, object_idx].reshape(-1).detach().cpu().numpy()
            gt_curve = evaluate_xy_coeffs(gt_coeffs, future_points) * coordinate_scale
            _plot_styled_line(
                ax,
                gt_curve,
                color=color,
                linestyle=(0, (1.0, 1.6)),
                linewidth=3.0 if on_image else 2.0,
                alpha=0.98 if on_image else 0.9,
                label=f"{label_prefix} future",
                on_image=on_image,
                zorder=7,
            )

        for mode_idx in range(spec.num_modes):
            coeffs = means[batch_idx, frame_idx, object_idx, mode_idx].detach().cpu().numpy()
            curve = evaluate_xy_coeffs(coeffs, future_points) * coordinate_scale
            alpha = 0.30 + 0.45 / max(1, spec.num_modes)
            _plot_styled_line(
                ax,
                curve,
                color=color,
                linestyle="-",
                linewidth=1.4 if on_image else 1.2,
                alpha=alpha,
                label=f"{label_prefix} predicted modes" if mode_idx == 0 else None,
                on_image=on_image,
                zorder=6,
            )
            ax.scatter(
                curve[-1, 0],
                curve[-1, 1],
                color=color,
                edgecolor="black" if on_image else color,
                linewidth=0.4 if on_image else 0.0,
                s=20,
                alpha=alpha,
                zorder=6.5,
            )

            if show_covariance:
                coeff_cov = covs[batch_idx, frame_idx, object_idx, mode_idx].detach().cpu().numpy()
                xy_covs = _xy_covariances(coeff_cov, future_points, spec.num_coeffs) * (coordinate_scale ** 2)
                for step in np.linspace(0, future_points - 1, 4, dtype=int)[1:]:
                    _add_cov_ellipse(ax, curve[step], xy_covs[step], color, alpha=0.04)

    if invert_y and not on_image:
        ax.invert_yaxis()

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend = ax.legend(handles, labels, loc="best", fontsize=8, framealpha=0.88)
        if on_image:
            legend.get_frame().set_facecolor((0, 0, 0, 0.7))
            legend.get_frame().set_edgecolor((1, 1, 1, 0.35))
            for text in legend.get_texts():
                text.set_color("white")

    if on_image:
        fig.tight_layout(pad=0.15)
    else:
        fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    image = Image.open(buf).convert("RGB").copy()
    buf.close()
    if output_path:
        image.save(output_path)
    plt.close(fig)
    return image


def _run_model_on_batch(
    model: torch.nn.Module,
    batch_payload: Any,
    device: torch.device,
    refinement_steps: int,
    feature_layout: str,
) -> tuple[Any, torch.Tensor | None, torch.Tensor | None]:
    if not isinstance(batch_payload, dict):
        raise ValueError("--batch must point to a dict containing at least a 'features' tensor")
    features = _find_payload_value(batch_payload, ("features", "feature", "x"))
    if features is None:
        raise ValueError("--batch payload does not contain 'features', 'feature', or 'x'")
    features_tbn = _features_to_tbn(features.to(device), layout=feature_layout)
    f_ = _find_payload_value(batch_payload, ("f_", "refinement_steps"))
    if f_ is None:
        f_ = [int(refinement_steps)] * int(features_tbn.shape[0])
    predictions = model(features_tbn, f_)
    trajectory = _find_payload_value(batch_payload, ("trajectory", "target", "y"))
    return predictions, features_tbn.detach().cpu(), trajectory.detach().cpu() if torch.is_tensor(trajectory) else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML config used to infer model output dimensions.")
    parser.add_argument("--checkpoint", help="Optional model checkpoint. Requires --config and --batch.")
    parser.add_argument("--batch", help="Optional .pt dict with features and optional trajectory target.")
    parser.add_argument("--predictions", help="Optional .pt predictions or dict containing predictions.")
    parser.add_argument("--image", help="Optional RGB image path to draw the trajectories on.")
    parser.add_argument("--waymo_root", help="Waymo RGB parquet dataset root. Defaults to config feature_extractor.dataset_root.")
    parser.add_argument("--waymo_segment", help="Waymo segment directory path or name under --waymo_root.")
    parser.add_argument("--waymo_image_id", help="Exact image_id to load from the segment images.parquet.")
    parser.add_argument("--waymo_timestamp", type=int, help="Frame timestamp in micros. Nearest frame is used if exact match is absent.")
    parser.add_argument("--waymo_camera", type=int, default=1, help="Waymo camera enum to load, default 1 (FRONT).")
    parser.add_argument("--waymo_max_width", type=int, default=1280, help="Downscale Waymo image overlays to this width; 0 keeps original size.")
    parser.add_argument("--output", default="visualizations/outs/polynomial_gmm.png")
    parser.add_argument("--gif", default=None, help="Optional GIF path animating all prediction frames.")
    parser.add_argument("--num_modes", type=int, default=None, help="Required when --config is omitted.")
    parser.add_argument("--batch_index", type=int, default=0)
    parser.add_argument("--object_index", type=int, nargs="*", default=None)
    parser.add_argument("--frame_index", type=int, default=-1)
    parser.add_argument("--future_points", type=int, default=90)
    parser.add_argument("--max_objects", type=int, default=5)
    parser.add_argument("--refinement_steps", type=int, default=3)
    parser.add_argument(
        "--prediction_layout",
        default="auto",
        choices=["auto", "time_first", "batch_first"],
        help="Layout for tensor predictions with an explicit frame dimension.",
    )
    parser.add_argument(
        "--feature_layout",
        default="auto",
        choices=["auto", "time_first", "batch_first"],
        help="Layout for saved feature tensors. Dataloader batches are batch_first.",
    )
    parser.add_argument(
        "--trajectory_layout",
        default="auto",
        choices=["auto", "time_first", "batch_first"],
        help="Layout for saved target polynomial tensors.",
    )
    parser.add_argument("--frame_ms", type=int, default=350)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no_covariance", action="store_true")
    parser.add_argument("--no_invert_y", action="store_true")
    parser.add_argument("--show_axes", action="store_true", help="Keep image pixel axes visible on RGB overlays.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.checkpoint and not (args.config and args.batch):
        raise SystemExit("--checkpoint requires both --config and --batch")
    if not args.predictions and not args.checkpoint:
        raise SystemExit("Provide either --predictions or --checkpoint with --batch")
    if not args.config and args.num_modes is None:
        raise SystemExit("Provide --config or --num_modes so the output dimensions can be inferred")

    model = None
    features_tbn = None
    trajectory = None
    background_image = _to_rgb_image(args.image) if args.image else None
    coordinate_scale = 1.0
    waymo_meta = None

    if args.checkpoint:
        model = _load_model(args.config, args.checkpoint, device)
        spec = _spec_from_model(model)
        batch_payload = _load_torch_payload(args.batch, device=device)
        if background_image is None:
            background_image = _payload_image(batch_payload)
        predictions, features_tbn, trajectory = _run_model_on_batch(
            model, batch_payload, device, args.refinement_steps, args.feature_layout
        )
    else:
        spec = _spec_from_config(args.config, args.num_modes) if args.config else PolynomialGMMSpec(args.num_modes)
        payload = _load_torch_payload(args.predictions)
        if background_image is None:
            background_image = _payload_image(payload)
        predictions = _find_payload_value(payload, ("predictions", "prediction", "outputs", "output"))
        if predictions is None:
            predictions = payload
        features = _find_payload_value(payload, ("features", "feature", "x"))
        target = _find_payload_value(payload, ("trajectory", "target", "y"))
        features_tbn = None
        if torch.is_tensor(features):
            features_tbn = _features_to_tbn(features, layout=args.feature_layout).cpu()
        trajectory = target.cpu() if torch.is_tensor(target) else None

    should_load_waymo = (
        background_image is None
        and (
            args.waymo_root
            or args.waymo_segment
            or args.waymo_image_id
            or args.waymo_timestamp is not None
        )
    )
    if should_load_waymo:
        waymo_root = args.waymo_root or _config_waymo_root(args.config)
        background_image, waymo_meta = _load_waymo_rgb_image(
            waymo_root,
            segment=args.waymo_segment,
            image_id=args.waymo_image_id,
            timestamp=args.waymo_timestamp,
            camera_name=args.waymo_camera,
            max_width=args.waymo_max_width,
        )
        if background_image is None:
            raise SystemExit(
                "Could not load a Waymo RGB image. Check --waymo_root, "
                "--waymo_segment, --waymo_image_id/--waymo_timestamp, and --waymo_camera."
            )
        coordinate_scale = float(waymo_meta.get("pixel_scale", 1.0))

    means, covs = extract_polynomial_gmm_params(
        predictions,
        spec,
        prediction_layout=args.prediction_layout,
    )
    if features_tbn is not None and features_tbn.shape[0] != means.shape[1]:
        features_tbn = _features_to_tbn(
            features_tbn,
            layout="auto",
            num_frames=means.shape[1],
        ).cpu()
    trajectory_btncc = (
        _trajectory_to_btncc(
            trajectory,
            spec.num_coeffs,
            layout=args.trajectory_layout,
            num_frames=means.shape[1],
        )
        if torch.is_tensor(trajectory)
        else None
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_prediction_frame(
        means,
        covs,
        spec,
        output_path=str(output_path),
        features_tbn=features_tbn,
        trajectory_btncc=trajectory_btncc,
        frame_idx=args.frame_index,
        batch_idx=args.batch_index,
        object_indices=args.object_index,
        future_points=args.future_points,
        max_objects=args.max_objects,
        show_covariance=not args.no_covariance,
        invert_y=not args.no_invert_y,
        background_image=background_image,
        coordinate_scale=coordinate_scale,
        show_axes=args.show_axes,
    )
    print(f"Saved {output_path}")
    if waymo_meta is not None:
        print(
            "Waymo image: "
            f"segment={Path(waymo_meta['segment_path']).name} "
            f"image_id={waymo_meta.get('image_id')} "
            f"timestamp={waymo_meta.get('frame_timestamp_micros')} "
            f"camera={waymo_meta.get('camera_name_text', waymo_meta.get('camera_name'))}"
        )

    if args.gif:
        gif_path = Path(args.gif)
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        images = [
            plot_prediction_frame(
                means,
                covs,
                spec,
                features_tbn=features_tbn,
                trajectory_btncc=trajectory_btncc,
                frame_idx=i,
                batch_idx=args.batch_index,
                object_indices=args.object_index,
                future_points=args.future_points,
                max_objects=args.max_objects,
                show_covariance=not args.no_covariance,
                invert_y=not args.no_invert_y,
                background_image=background_image,
                coordinate_scale=coordinate_scale,
                show_axes=args.show_axes,
            )
            for i in range(means.shape[1])
        ]
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:] + [images[-1], images[-1]],
            duration=args.frame_ms,
            loop=0,
            optimize=False,
        )
        print(f"Saved {gif_path}")


if __name__ == "__main__":
    main()
