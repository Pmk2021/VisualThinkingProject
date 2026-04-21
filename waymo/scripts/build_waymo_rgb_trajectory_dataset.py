#!/usr/bin/env python3
"""Build an RGB-to-trajectory dataset from Waymo v2 component parquet files.

The converter treats each camera as a standalone input and discards lidar point
clouds. It uses label tables as supervision:

  - camera_image: RGB/JPEG bytes for each camera frame.
  - projected_lidar_box: objects visible in each camera image.
  - lidar_box: per-object 3D boxes over time, grouped into trajectories.

Output is three PyTorch-friendly parquet tables:

  - images.parquet: one row per RGB camera image.
  - trajectories.parquet: one row per scene/object trajectory.
  - image_trajectories.parquet: many-to-many visible-object links from image
    rows to trajectory rows, with projected 2D boxes.
  - ego_poses.parquet: one row per vehicle pose timestep.
  - prediction_targets.parquet: tracks_to_predict from WOMD scenario TFRecords,
    including observed history and optional future labels when present.

This script does not use lidar point clouds as model inputs. It only uses
Waymo's box/track labels to define the trajectory targets.
"""

from __future__ import annotations

import argparse
import json
import math
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

from VisualThinkingProject.waymo_dataset.scripts.build_waymo_web_viewer import build_web_viewer
from VisualThinkingProject.waymo_dataset.scripts.extract_waymo_motion import iter_tfrecord_records, parse_scenario


CAMERA_NAMES = {
    0: "UNKNOWN",
    1: "FRONT",
    2: "FRONT_LEFT",
    3: "FRONT_RIGHT",
    4: "SIDE_LEFT",
    5: "SIDE_RIGHT",
    6: "REAR_LEFT",
    7: "REAR",
    8: "REAR_RIGHT",
}

IMAGE_COL = "[CameraImageComponent].image"
VEHICLE_POSE_COL = "[VehiclePoseComponent].world_from_vehicle.transform"

CAMERA_IMAGE_VELOCITY_COLUMNS = {
    "ego_velocity_x": "[CameraImageComponent].velocity.linear_velocity.x",
    "ego_velocity_y": "[CameraImageComponent].velocity.linear_velocity.y",
    "ego_velocity_z": "[CameraImageComponent].velocity.linear_velocity.z",
    "ego_angular_velocity_x": "[CameraImageComponent].velocity.angular_velocity.x",
    "ego_angular_velocity_y": "[CameraImageComponent].velocity.angular_velocity.y",
    "ego_angular_velocity_z": "[CameraImageComponent].velocity.angular_velocity.z",
}

LIDAR_BOX_COLUMNS = {
    "x": "[LiDARBoxComponent].box.center.x",
    "y": "[LiDARBoxComponent].box.center.y",
    "z": "[LiDARBoxComponent].box.center.z",
    "length": "[LiDARBoxComponent].box.size.x",
    "width": "[LiDARBoxComponent].box.size.y",
    "height": "[LiDARBoxComponent].box.size.z",
    "heading": "[LiDARBoxComponent].box.heading",
    "object_type": "[LiDARBoxComponent].type",
    "velocity_x": "[LiDARBoxComponent].speed.x",
    "velocity_y": "[LiDARBoxComponent].speed.y",
    "velocity_z": "[LiDARBoxComponent].speed.z",
}

PROJECTED_BOX_COLUMNS = {
    "bbox_center_x": "[ProjectedLiDARBoxComponent].box.center.x",
    "bbox_center_y": "[ProjectedLiDARBoxComponent].box.center.y",
    "bbox_width": "[ProjectedLiDARBoxComponent].box.size.x",
    "bbox_height": "[ProjectedLiDARBoxComponent].box.size.y",
    "object_type": "[ProjectedLiDARBoxComponent].type",
}


def main() -> int:
    args = parse_args()
    sensory_root = Path(args.sensory_root)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    split_names = [normalize_split_name(s.strip()) for s in args.splits.split(",") if s.strip()]
    all_images: list[dict[str, Any]] = []
    all_ego_poses: list[dict[str, Any]] = []
    all_trajectories: list[dict[str, Any]] = []
    all_links: list[dict[str, Any]] = []
    all_prediction_targets: list[dict[str, Any]] = []

    for split in split_names:
        split_dir = sensory_root / split
        if not split_dir.exists():
            print(f"[skip] split directory not found: {split_dir}")
            continue

        print(f"[split] {split}")
        images = load_component(split_dir, "camera_image")
        vehicle_pose = load_component(split_dir, "vehicle_pose")
        projected = load_component(split_dir, "projected_lidar_box")
        lidar_boxes = load_component(split_dir, "lidar_box")

        if images.empty:
            print(f"  no camera_image rows in {split_dir}")
            continue

        split_images = build_images_table(images, split)
        split_ego_poses = build_ego_poses_table(vehicle_pose, images, split)
        all_ego_poses.extend(split_ego_poses)
        if projected.empty:
            print(f"  no projected_lidar_box rows in {split_dir}; keeping images without GT links")
            all_images.extend(split_images)
            continue
        if lidar_boxes.empty:
            print(f"  no lidar_box rows in {split_dir}; keeping images without GT links")
            all_images.extend(split_images)
            continue

        split_trajectories = build_trajectories_table(lidar_boxes, split)
        split_links = build_image_trajectory_table(projected, split)

        trajectory_keys = {
            (row["scene_id"], row["trajectory_id"]) for row in split_trajectories
        }
        split_links = [
            row
            for row in split_links
            if (row["scene_id"], row["trajectory_id"]) in trajectory_keys
        ]

        image_ids = {row["image_id"] for row in split_images}
        split_links = [row for row in split_links if row["image_id"] in image_ids]

        visible_by_image: dict[str, list[str]] = {}
        for row in split_links:
            visible_by_image.setdefault(row["image_id"], []).append(row["trajectory_id"])
        for row in split_images:
            ids = visible_by_image.get(row["image_id"], [])
            row["visible_trajectory_ids"] = ids
            row["num_visible_trajectories"] = len(ids)

        all_images.extend(split_images)
        all_trajectories.extend(split_trajectories)
        all_links.extend(split_links)

        print(
            f"  images={len(split_images)} "
            f"ego_poses={len(split_ego_poses)} "
            f"trajectories={len(split_trajectories)} "
            f"image_trajectories={len(split_links)}"
        )

    prediction_splits = [
        normalize_split_name(s.strip())
        for s in args.prediction_target_splits.split(",")
        if s.strip()
    ]
    if args.motion_scenario_root:
        all_prediction_targets = build_prediction_targets(
            motion_root=Path(args.motion_scenario_root),
            splits=prediction_splits,
            max_records_per_split=args.motion_max_records_per_split,
        )

    write_parquet(output_root / "images.parquet", all_images)
    write_parquet(output_root / "ego_poses.parquet", all_ego_poses)
    write_parquet(output_root / "trajectories.parquet", all_trajectories)
    write_parquet(output_root / "image_trajectories.parquet", all_links)
    write_parquet(output_root / "prediction_targets.parquet", all_prediction_targets)
    write_manifest(
        output_root,
        all_images,
        all_ego_poses,
        all_trajectories,
        all_links,
        all_prediction_targets,
        split_names,
    )

    if args.visualize:
        visualize_samples(
            output_root=output_root,
            images=all_images,
            links=all_links,
            max_images=args.visualize,
        )

    if args.web_viewer:
        build_web_viewer(
            viewer_dir=output_root / "viewer",
            images=all_images,
            trajectories=all_trajectories,
            links=all_links,
            prediction_targets=all_prediction_targets,
            max_scenes=args.web_max_scenes,
            max_images_per_scene=args.web_max_images_per_scene,
        )

    print("\nWrote:")
    print(f"  {output_root / 'images.parquet'}")
    print(f"  {output_root / 'ego_poses.parquet'}")
    print(f"  {output_root / 'trajectories.parquet'}")
    print(f"  {output_root / 'image_trajectories.parquet'}")
    print(f"  {output_root / 'prediction_targets.parquet'}")
    if args.visualize:
        print(f"  {output_root / 'visualizations'}")
    if args.web_viewer:
        print(f"  {output_root / 'viewer' / 'index.html'}")
    return 0


def load_component(split_dir: Path, component: str) -> pd.DataFrame:
    component_dir = split_dir / component
    if not component_dir.exists():
        return pd.DataFrame()
    frames = []
    for path in sorted(component_dir.glob("*.parquet")):
        frame = pd.read_parquet(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_images_table(images: pd.DataFrame, split: str) -> list[dict[str, Any]]:
    required = {
        "key.segment_context_name",
        "key.frame_timestamp_micros",
        "key.camera_name",
        IMAGE_COL,
    }
    require_columns(images, required, "camera_image")

    rows = []
    for _, row in images.iterrows():
        scene_id = str(row["key.segment_context_name"])
        timestamp = int(row["key.frame_timestamp_micros"])
        camera_name = int(row["key.camera_name"])
        image_bytes = bytes(row[IMAGE_COL])
        image_id = make_image_id(scene_id, timestamp, camera_name)
        width, height = image_size(image_bytes)
        rows.append(
            {
                "split": split,
                "image_id": image_id,
                "scene_id": scene_id,
                "frame_timestamp_micros": timestamp,
                "camera_name": camera_name,
                "camera_name_text": CAMERA_NAMES.get(camera_name, f"UNKNOWN_{camera_name}"),
                "image_jpeg": image_bytes,
                "image_format": "jpeg",
                "image_width": width,
                "image_height": height,
                "visible_trajectory_ids": [],
                "num_visible_trajectories": 0,
            }
        )
    return rows


def build_ego_poses_table(
    vehicle_pose: pd.DataFrame,
    camera_images: pd.DataFrame,
    split: str,
) -> list[dict[str, Any]]:
    if vehicle_pose.empty:
        print(f"  no vehicle_pose rows for {split}; ego_poses will be empty")
        return []

    required = {
        "key.segment_context_name",
        "key.frame_timestamp_micros",
        VEHICLE_POSE_COL,
    }
    require_columns(vehicle_pose, required, "vehicle_pose")

    velocity_by_key = camera_velocity_by_key(camera_images)

    pose_rows = []
    for _, row in vehicle_pose.iterrows():
        scene_id = str(row["key.segment_context_name"])
        timestamp = int(row["key.frame_timestamp_micros"])
        transform = [float(v) for v in row[VEHICLE_POSE_COL]]
        x, y, z = transform_translation(transform)
        roll, pitch, yaw = transform_rpy(transform)
        velocity = velocity_by_key.get((scene_id, timestamp), {})
        pose_rows.append(
            {
                "split": split,
                "scene_id": scene_id,
                "frame_timestamp_micros": timestamp,
                "ego_pose_transform": transform,
                "ego_x": x,
                "ego_y": y,
                "ego_z": z,
                "ego_roll": roll,
                "ego_pitch": pitch,
                "ego_yaw": yaw,
                "ego_velocity_x": velocity.get("ego_velocity_x", math.nan),
                "ego_velocity_y": velocity.get("ego_velocity_y", math.nan),
                "ego_velocity_z": velocity.get("ego_velocity_z", math.nan),
                "ego_angular_velocity_x": velocity.get("ego_angular_velocity_x", math.nan),
                "ego_angular_velocity_y": velocity.get("ego_angular_velocity_y", math.nan),
                "ego_angular_velocity_z": velocity.get("ego_angular_velocity_z", math.nan),
                "ego_velocity_source": "camera_image" if velocity else "finite_difference",
            }
        )

    add_finite_difference_ego_velocity(pose_rows)
    for row in pose_rows:
        if math.isnan(row["ego_velocity_x"]):
            row["ego_velocity_x"] = row["ego_fd_velocity_x"]
            row["ego_velocity_y"] = row["ego_fd_velocity_y"]
            row["ego_velocity_z"] = row["ego_fd_velocity_z"]
    return pose_rows


def camera_velocity_by_key(camera_images: pd.DataFrame) -> dict[tuple[str, int], dict[str, float]]:
    if camera_images.empty:
        return {}
    required = {
        "key.segment_context_name",
        "key.frame_timestamp_micros",
        *CAMERA_IMAGE_VELOCITY_COLUMNS.values(),
    }
    missing = required.difference(camera_images.columns)
    if missing:
        return {}
    grouped = camera_images.groupby(["key.segment_context_name", "key.frame_timestamp_micros"], sort=False)
    result: dict[tuple[str, int], dict[str, float]] = {}
    for (scene_id, timestamp), group in grouped:
        result[(str(scene_id), int(timestamp))] = {
            out_name: float(group[col].astype("float64").mean())
            for out_name, col in CAMERA_IMAGE_VELOCITY_COLUMNS.items()
        }
    return result


def transform_translation(transform: list[float]) -> tuple[float, float, float]:
    if len(transform) < 12:
        return math.nan, math.nan, math.nan
    return transform[3], transform[7], transform[11]


def transform_rpy(transform: list[float]) -> tuple[float, float, float]:
    if len(transform) < 11:
        return math.nan, math.nan, math.nan
    r00, r01, r02 = transform[0], transform[1], transform[2]
    r10, r11, r12 = transform[4], transform[5], transform[6]
    r20, r21, r22 = transform[8], transform[9], transform[10]
    roll = math.atan2(r21, r22)
    pitch = math.atan2(-r20, math.sqrt(r21 * r21 + r22 * r22))
    yaw = math.atan2(r10, r00)
    return roll, pitch, yaw


def add_finite_difference_ego_velocity(rows: list[dict[str, Any]]) -> None:
    rows_by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_scene.setdefault(row["scene_id"], []).append(row)
    for scene_rows in rows_by_scene.values():
        scene_rows.sort(key=lambda row: row["frame_timestamp_micros"])
        for i, row in enumerate(scene_rows):
            prev_row = scene_rows[i - 1] if i > 0 else None
            next_row = scene_rows[i + 1] if i + 1 < len(scene_rows) else None
            a = prev_row if prev_row is not None else row
            b = next_row if next_row is not None else row
            if a is row and b is row:
                vx = vy = vz = math.nan
            else:
                dt = (b["frame_timestamp_micros"] - a["frame_timestamp_micros"]) / 1_000_000.0
                if dt > 0:
                    vx = (b["ego_x"] - a["ego_x"]) / dt
                    vy = (b["ego_y"] - a["ego_y"]) / dt
                    vz = (b["ego_z"] - a["ego_z"]) / dt
                else:
                    vx = vy = vz = math.nan
            row["ego_fd_velocity_x"] = vx
            row["ego_fd_velocity_y"] = vy
            row["ego_fd_velocity_z"] = vz


def build_trajectories_table(lidar_boxes: pd.DataFrame, split: str) -> list[dict[str, Any]]:
    required = {
        "key.segment_context_name",
        "key.frame_timestamp_micros",
        "key.laser_object_id",
        *LIDAR_BOX_COLUMNS.values(),
    }
    require_columns(lidar_boxes, required, "lidar_box")

    sort_cols = [
        "key.segment_context_name",
        "key.laser_object_id",
        "key.frame_timestamp_micros",
    ]
    lidar_boxes = lidar_boxes.sort_values(sort_cols)

    rows = []
    grouped = lidar_boxes.groupby(["key.segment_context_name", "key.laser_object_id"], sort=False)
    for (scene_id, trajectory_id), group in grouped:
        timestamps = group["key.frame_timestamp_micros"].astype("int64").tolist()
        values = {
            out_name: group[col].astype("float64").tolist()
            for out_name, col in LIDAR_BOX_COLUMNS.items()
            if out_name != "object_type"
        }
        object_types = group[LIDAR_BOX_COLUMNS["object_type"]].dropna().astype("int64")
        object_type = int(object_types.iloc[0]) if len(object_types) else -1
        rows.append(
            {
                "split": split,
                "scene_id": str(scene_id),
                "trajectory_id": str(trajectory_id),
                "trajectory_row_id": make_trajectory_row_id(str(scene_id), str(trajectory_id)),
                "object_type": object_type,
                "num_steps": len(group),
                "timestamps_micros": timestamps,
                "x": values["x"],
                "y": values["y"],
                "z": values["z"],
                "length": values["length"],
                "width": values["width"],
                "height": values["height"],
                "heading": values["heading"],
                "velocity_x": values["velocity_x"],
                "velocity_y": values["velocity_y"],
                "velocity_z": values["velocity_z"],
            }
        )
    return rows


def build_image_trajectory_table(projected: pd.DataFrame, split: str) -> list[dict[str, Any]]:
    required = {
        "key.segment_context_name",
        "key.frame_timestamp_micros",
        "key.camera_name",
        "key.laser_object_id",
        *PROJECTED_BOX_COLUMNS.values(),
    }
    require_columns(projected, required, "projected_lidar_box")

    rows = []
    projected = projected.sort_values(
        ["key.segment_context_name", "key.frame_timestamp_micros", "key.camera_name"]
    )
    for _, row in projected.iterrows():
        scene_id = str(row["key.segment_context_name"])
        timestamp = int(row["key.frame_timestamp_micros"])
        camera_name = int(row["key.camera_name"])
        trajectory_id = str(row["key.laser_object_id"])
        bbox_cx = float(row[PROJECTED_BOX_COLUMNS["bbox_center_x"]])
        bbox_cy = float(row[PROJECTED_BOX_COLUMNS["bbox_center_y"]])
        bbox_w = float(row[PROJECTED_BOX_COLUMNS["bbox_width"]])
        bbox_h = float(row[PROJECTED_BOX_COLUMNS["bbox_height"]])
        rows.append(
            {
                "split": split,
                "image_id": make_image_id(scene_id, timestamp, camera_name),
                "scene_id": scene_id,
                "frame_timestamp_micros": timestamp,
                "camera_name": camera_name,
                "camera_name_text": CAMERA_NAMES.get(camera_name, f"UNKNOWN_{camera_name}"),
                "trajectory_id": trajectory_id,
                "trajectory_row_id": make_trajectory_row_id(scene_id, trajectory_id),
                "object_type": int(row[PROJECTED_BOX_COLUMNS["object_type"]]),
                "bbox_center_x": bbox_cx,
                "bbox_center_y": bbox_cy,
                "bbox_width": bbox_w,
                "bbox_height": bbox_h,
                "bbox_x1": bbox_cx - bbox_w / 2.0,
                "bbox_y1": bbox_cy - bbox_h / 2.0,
                "bbox_x2": bbox_cx + bbox_w / 2.0,
                "bbox_y2": bbox_cy + bbox_h / 2.0,
            }
        )
    return rows


def build_prediction_targets(
    motion_root: Path,
    splits: list[str],
    max_records_per_split: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in splits:
        split_dir = motion_root / split
        files = sorted(split_dir.glob("*.tfrecord-*"))
        if not files:
            print(f"[prediction-targets] no TFRecords found for split {split}: {split_dir}")
            continue
        decoded = 0
        before = len(rows)
        for path in files:
            for payload in iter_tfrecord_records(path):
                if max_records_per_split and decoded >= max_records_per_split:
                    break
                scenario = parse_scenario(payload)
                rows.extend(prediction_targets_from_scenario(scenario, split))
                decoded += 1
            if max_records_per_split and decoded >= max_records_per_split:
                break
        print(
            f"[prediction-targets] {split}: scenarios={decoded} "
            f"targets={len(rows) - before}"
        )
    return rows


def prediction_targets_from_scenario(
    scenario: dict[str, Any],
    split: str,
) -> list[dict[str, Any]]:
    scene_id = scenario["scenario_id"]
    timestamps = scenario["timestamps_seconds"]
    current_time_index = int(scenario["current_time_index"])
    rows = []
    for rank, target in enumerate(scenario["tracks_to_predict"]):
        track_index = int(target["track_index"])
        if track_index < 0 or track_index >= len(scenario["tracks"]):
            continue
        track = scenario["tracks"][track_index]
        states = track["states"]
        observed_indices = [i for i in range(len(states)) if i <= current_time_index]
        future_indices = [i for i in range(len(states)) if i > current_time_index]
        rows.append(
            {
                "split": split,
                "scene_id": scene_id,
                "trajectory_id": str(track["id"]),
                "trajectory_row_id": make_trajectory_row_id(scene_id, str(track["id"])),
                "track_index": track_index,
                "track_id": int(track["id"]),
                "rank": rank,
                "difficulty": int(target["difficulty"]),
                "difficulty_name": target["difficulty_name"],
                "object_type": int(track["object_type"]),
                "object_type_name": track["object_type_name"],
                "current_time_index": current_time_index,
                "observed_timestamps_seconds": values_at(timestamps, observed_indices),
                "observed_valid": [bool(states[i]["valid"]) for i in observed_indices],
                "observed_x": [float(states[i]["center_x"]) for i in observed_indices],
                "observed_y": [float(states[i]["center_y"]) for i in observed_indices],
                "observed_z": [float(states[i]["center_z"]) for i in observed_indices],
                "observed_heading": [float(states[i]["heading"]) for i in observed_indices],
                "observed_velocity_x": [float(states[i]["velocity_x"]) for i in observed_indices],
                "observed_velocity_y": [float(states[i]["velocity_y"]) for i in observed_indices],
                "future_timestamps_seconds": values_at(timestamps, future_indices),
                "future_valid": [bool(states[i]["valid"]) for i in future_indices],
                "future_x": [float(states[i]["center_x"]) for i in future_indices],
                "future_y": [float(states[i]["center_y"]) for i in future_indices],
                "future_z": [float(states[i]["center_z"]) for i in future_indices],
                "future_heading": [float(states[i]["heading"]) for i in future_indices],
                "future_velocity_x": [float(states[i]["velocity_x"]) for i in future_indices],
                "future_velocity_y": [float(states[i]["velocity_y"]) for i in future_indices],
                "has_future_gt": len(future_indices) > 0,
            }
        )
    return rows


def values_at(values: list[Any], indices: list[int]) -> list[Any]:
    return [values[i] for i in indices if i < len(values)]


def require_columns(frame: pd.DataFrame, columns: set[str], component: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{component} table is missing columns: {missing}")


def make_image_id(scene_id: str, timestamp_micros: int, camera_name: int) -> str:
    return f"{scene_id};{timestamp_micros};camera_{camera_name}"


def make_trajectory_row_id(scene_id: str, trajectory_id: str) -> str:
    return f"{scene_id};{trajectory_id}"


def image_size(image_bytes: bytes) -> tuple[int | None, int | None]:
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            return image.size
    except Exception:
        return None, None


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows) if rows else pa.table({})
    pq.write_table(table, path, compression="zstd")


def write_manifest(
    output_root: Path,
    images: list[dict[str, Any]],
    ego_poses: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
    links: list[dict[str, Any]],
    prediction_targets: list[dict[str, Any]],
    splits: list[str],
) -> None:
    manifest = {
        "splits": splits,
        "tables": {
            "images": {
                "path": str(output_root / "images.parquet"),
                "rows": len(images),
                "description": "One row per standalone RGB camera image.",
            },
            "ego_poses": {
                "path": str(output_root / "ego_poses.parquet"),
                "rows": len(ego_poses),
                "description": "One row per vehicle pose timestep from v2 vehicle_pose.",
            },
            "trajectories": {
                "path": str(output_root / "trajectories.parquet"),
                "rows": len(trajectories),
                "description": "One row per scene/object 3D trajectory from box labels.",
            },
            "image_trajectories": {
                "path": str(output_root / "image_trajectories.parquet"),
                "rows": len(links),
                "description": "Visible trajectory IDs and 2D projected boxes for each image.",
            },
            "prediction_targets": {
                "path": str(output_root / "prediction_targets.parquet"),
                "rows": len(prediction_targets),
                "description": "WOMD tracks_to_predict with observed history and optional future labels.",
            },
        },
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def visualize_samples(
    output_root: Path,
    images: list[dict[str, Any]],
    links: list[dict[str, Any]],
    max_images: int,
) -> None:
    vis_dir = output_root / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    links_by_image: dict[str, list[dict[str, Any]]] = {}
    for row in links:
        links_by_image.setdefault(row["image_id"], []).append(row)

    drawn = 0
    for image_row in images:
        image_links = links_by_image.get(image_row["image_id"], [])
        if not image_links:
            continue
        image = Image.open(BytesIO(image_row["image_jpeg"])).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        for link in image_links:
            x1 = float(link["bbox_x1"])
            y1 = float(link["bbox_y1"])
            x2 = float(link["bbox_x2"])
            y2 = float(link["bbox_y2"])
            draw.rectangle((x1, y1, x2, y2), outline=(255, 64, 64), width=3)
            label = str(link["trajectory_id"])[:8]
            draw.text((x1 + 3, max(0, y1 - 12)), label, fill=(255, 255, 0), font=font)

        out_path = vis_dir / f"{safe_name(image_row['image_id'])}.jpg"
        image.save(out_path, quality=92)
        print(f"[visualize] {out_path}")
        drawn += 1
        if drawn >= max_images:
            break


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sensory-root",
        default="dataset/waymo_rgb_sample",
        help="Root containing split/component parquet folders.",
    )
    parser.add_argument(
        "--output",
        default="dataset/rgb_trajectory_dataset",
        help="Output directory for the three converted parquet tables.",
    )
    parser.add_argument(
        "--splits",
        default="training,validation,testing",
        help="Comma-separated split directories to convert.",
    )
    parser.add_argument(
        "--visualize",
        type=int,
        default=3,
        help="Number of RGB + visible trajectory overlay images to write. Use 0 to disable.",
    )
    parser.add_argument(
        "--motion-scenario-root",
        default="dataset/trajectory_dataset/uncompressed/scenario",
        help="Root containing WOMD scenario TFRecord split folders for prediction targets.",
    )
    parser.add_argument(
        "--prediction-target-splits",
        default="testing",
        help="Comma-separated WOMD scenario splits to export into prediction_targets.parquet.",
    )
    parser.add_argument(
        "--motion-max-records-per-split",
        type=int,
        default=0,
        help="Limit WOMD scenarios decoded per split for prediction targets. 0 means all.",
    )
    parser.add_argument(
        "--web-viewer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate a static HTML viewer under output/viewer.",
    )
    parser.add_argument(
        "--web-max-scenes",
        type=int,
        default=24,
        help="Maximum scenes exported to the web viewer. 0 means all scenes.",
    )
    parser.add_argument(
        "--web-max-images-per-scene",
        type=int,
        default=12,
        help="Maximum RGB images exported per scene in the web viewer. 0 means all.",
    )
    return parser.parse_args()


def normalize_split_name(split: str) -> str:
    aliases = {
        "train": "training",
        "val": "validation",
        "valid": "validation",
        "test": "testing",
    }
    return aliases.get(split, split)


if __name__ == "__main__":
    raise SystemExit(main())
