import argparse
import json
import hashlib
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import os
import platform
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import torch
import warnings

from AnytimeTrajectoryPredictor.models.ObjectTracker import ObjectTracker

warnings.filterwarnings("ignore", message="Could not initialize NNPACK!")

NODE = platform.node()
NODE_TEMPLATE = r"\b(izar1|i[0-9]{2}|ixl[0-9]{2})\b"
IZAR = re.fullmatch(NODE_TEMPLATE, NODE) is not None

DATA_PATH = (
    "/work/cs-503/santanto/waymo"
    if IZAR
    else "/Users/nathangromb/Documents/MA4/VI/project/data"
)

NUM_WORKERS = min(max(1, (os.cpu_count() or 1) - 1), 4)


def hash_to_int64(s: str) -> np.int64:
    h = hashlib.md5(s.encode("utf-8")).digest()
    # take first 8 bytes as unsigned integer
    val = int.from_bytes(h[:8], byteorder="big", signed=False)
    # map unsigned 64-bit into signed int64 range to avoid OverflowError when casting
    if val >= (1 << 63):
        val -= 1 << 64
    return np.int64(val)


def _resolve_output_path(path: Path) -> Path:
    if IZAR:
        return Path(str(path).replace("/santanto/", "/gromb/"))
    return path


def _build_feature_values(
    feat, feat_len, feature_offset, batch_idx, output, n_objects, obj_id_col
):
    features = output["features"]

    def slice_vals(b_idx, obj_idx, off, length):
        seg = features[b_idx, obj_idx, off : off + length]
        try:
            arr = seg.detach().cpu().numpy()
        except Exception:
            arr = np.array(seg)
        return arr

    if feat == "bboxes":
        return [
            {
                "cx": float(slice_vals(batch_idx, j, feature_offset, 1)[0]),
                "cy": float(slice_vals(batch_idx, j, feature_offset + 1, 1)[0]),
                "w": float(slice_vals(batch_idx, j, feature_offset + 2, 1)[0]),
                "h": float(slice_vals(batch_idx, j, feature_offset + 3, 1)[0]),
                "object_id": int(slice_vals(batch_idx, j, obj_id_col, 1)[0]),
            }
            for j in range(n_objects)
        ]

    elif feat in ["confidences", "class_ids"]:
        key = feat[:-1]
        return [
            {
                key: (
                    float(slice_vals(batch_idx, j, feature_offset, feat_len)[0])
                    if feat_len == 1
                    else list(
                        map(float, slice_vals(batch_idx, j, feature_offset, feat_len))
                    )
                ),
                "object_id": int(slice_vals(batch_idx, j, obj_id_col, 1)[0]),
            }
            for j in range(n_objects)
        ]

    elif feat == "local_latent_features":
        return [
            {
                **{
                    f"{feat}_{k}": float(
                        slice_vals(batch_idx, j, feature_offset + k, 1)[0]
                    )
                    for k in range(feat_len)
                },
                "object_id": int(slice_vals(batch_idx, j, obj_id_col, 1)[0]),
            }
            for j in range(n_objects)
        ]

    elif feat == "latent_features":
        if n_objects == 0:
            return []
        vals = slice_vals(batch_idx, 0, feature_offset, feat_len)
        return [{f"{feat}_{k}": float(vals[k]) for k in range(feat_len)}]

    elif feat == "object_ids":
        return []

    else:
        raise ValueError(f"Unknown feature: {feat}")


def _get_feature_column_index(feat, output):
    col_idx = 0
    for feature_name, length in output["lengths"]:
        if feature_name == "object_ids":
            return col_idx
        col_idx += length
    raise ValueError("object_ids not found in feature lengths")


def _is_all_zero_feature_row(row: dict) -> bool:
    feature_keys = [
        key for key in row.keys() if key.startswith("local_latent_features_")
    ]
    return bool(feature_keys) and all(float(row[key]) == 0.0 for key in feature_keys)


def process_dir(dir_path: Path):
    manifest_path = dir_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json in {dir_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tables = manifest["tables"]

    images_path = Path(tables["images"]["path"]) if "images" in tables else None
    image_traj_path = (
        Path(tables["image_trajectories"]["path"])
        if "image_trajectories" in tables
        else None
    )

    if not IZAR:
        images_path = "images.parquet"
        image_traj_path = "image_trajectories.parquet"

    if images_path is None or image_traj_path is None:
        raise RuntimeError("manifest missing images or image_trajectories paths")

    # Paths in manifest are relative to manifest directory
    base = manifest_path.parent
    images_table = pd.read_parquet(base / images_path)
    # Ensure deterministic processing order: sort by scene, camera, timestamp (same as save_fe_features.py)
    images_table = images_table.sort_values(
        ["scene_id", "camera_name", "frame_timestamp_micros"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    image_traj = pd.read_parquet(base / image_traj_path)

    local_rows = []

    print(
        f"found rows: {len(images_table)} in images table, {len(image_traj)} in image_trajectories table"
    )

    curr_scene, curr_camera = None, None
    tracker = None
    for _, row in tqdm(
        images_table.iterrows(),
        total=len(images_table),
        desc=f"Processing {dir_path.name}",
    ):
        # Recreate tracker when scene or camera changes (keeps behavior consistent with save_fe_features.py)
        if (row["scene_id"], row["camera_name"]) != (curr_scene, curr_camera):
            curr_scene, curr_camera = row["scene_id"], row["camera_name"]
            tracker = ObjectTracker(
                model_name="yolo26n.pt",
                feature_components=(
                    "object_ids",
                    "class_ids",
                    "latent_features",
                    "local_latent_features",
                ),
                imgsz=640,
                verbose=False,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        image_id = row["image_id"]
        img_rows = image_traj[image_traj["image_id"] == image_id]
        if img_rows.shape[0] == 0:
            continue

        # build tracking_override
        boxes = img_rows[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(
            dtype=np.float32
        )
        confidences = np.ones((boxes.shape[0],), dtype=np.float32)
        object_ids_hashed = np.array(
            [hash_to_int64(tid) for tid in img_rows["trajectory_id"].astype(str)],
            dtype=np.int64,
        )
        class_ids = img_rows["object_type"].to_numpy(dtype=np.int64)

        tracking_override = SimpleNamespace(
            xyxy=boxes,
            conf=confidences,
            id=object_ids_hashed,
            cls=class_ids,
        )

        image = Image.open(BytesIO(row["image_jpeg"]))
        output = tracker(image, tracking_override=tracking_override)
        image.close()

        # Normalize feature tensor to shape (batch, n_objects, feat_dim)
        feats = output.get("features")
        if isinstance(feats, torch.Tensor):
            # Handle ObjectTracker returning a 4D tensor (1,1,max_det,feat_dim)
            if feats.dim() == 4 and feats.shape[0] == 1:
                # squeeze the leading batch singleton -> (1, max_det, feat_dim)
                output["features"] = feats.squeeze(0)

        # parse output features
        features = output["features"]
        # features may be padded to max_det; real object count is number of GT rows for this image
        n_objects = int(min(img_rows.shape[0], features.shape[1]))
        obj_id_col = _get_feature_column_index("object_ids", output)

        feature_col_offset = 0
        for feat, feat_len in output["lengths"]:
            values = _build_feature_values(
                feat, feat_len, feature_col_offset, 0, output, n_objects, obj_id_col
            )
            feature_col_offset += feat_len

            if feat == "local_latent_features":
                # The tracker does not preserve a reliable join key for object_ids
                # because they are stored as float32 in the feature tensor.
                # Use the GT row order itself as the working key.
                traj_ids = list(img_rows["trajectory_id"].astype(str))
                traj_row_ids = list(img_rows["trajectory_row_id"].astype(str))
                if len(values) != len(traj_ids):
                    raise ValueError(
                        f"Object count mismatch in image_id {image_id}: tracker returned {len(values)} local rows, GT has {len(traj_ids)} rows"
                    )

                for idx, (feat_dict, traj_id_str, traj_row_id_str) in enumerate(
                    zip(values, traj_ids, traj_row_ids)
                ):
                    out = {
                        "split": row["split"],
                        "image_id": row["image_id"],
                        "scene_id": row["scene_id"],
                        "frame_timestamp_micros": row["frame_timestamp_micros"],
                        "camera_name": row["camera_name"],
                        "camera_name_text": row["camera_name_text"],
                        "trajectory_id": traj_id_str,
                        "trajectory_row_id": traj_row_id_str,
                    }
                    # remove object_id key (we keep trajectory_id instead)
                    fd = {k: v for k, v in feat_dict.items() if k != "object_id"}
                    out.update(fd)
                    if _is_all_zero_feature_row(out):
                        raise ValueError(
                            f"All-zero local feature row for image_id {image_id}, trajectory_id {traj_id_str}, trajectory_row_id {traj_row_id_str}"
                        )
                    local_rows.append(out)

    out_dir = _resolve_output_path(dir_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    if local_rows:
        table = pa.Table.from_pandas(pd.DataFrame(local_rows))
        pq.write_table(table, out_dir / "fe_gt_local_latent_features.parquet")


def main():
    parser = argparse.ArgumentParser(
        description="Save GT-based feature extractor outputs to parquet."
    )
    split_group = parser.add_mutually_exclusive_group(required=True)
    split_group.add_argument(
        "--train", action="store_true", help="Process training split."
    )
    split_group.add_argument(
        "--val", action="store_true", help="Process validation split."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DATA_PATH,
        help="Base directory containing split folders and manifests",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Process directories in parallel using multiprocessing (process_map).",
    )
    args = parser.parse_args()

    split = "training" if args.train else "validation"
    base_path = Path(args.data_path)
    if IZAR:
        selected_dirs = list(
            base_path.glob("training__*" if args.train else "validation__*")
        )
    else:
        selected_dirs = list(base_path.glob("waymo"))

    if not selected_dirs:
        print(f"No {split} directories found under {base_path}.")
        return

    dirs_to_process = []
    for split_dir in sorted(selected_dirs):
        if (split_dir / "fe_gt_local_latent_features.parquet").exists():
            print(f"Skipping {split_dir.name} (already processed)")
            continue
        else:
            print(f"Scheduling {split_dir.name} for processing because {split_dir / 'fe_gt_local_latent_features.parquet'} does not exist")
        dirs_to_process.append(split_dir)

    if not dirs_to_process:
        print(f"No {split} directories need processing.")
        return

    if args.parallel:
        process_map(
            process_dir,
            dirs_to_process,
            max_workers=NUM_WORKERS,
            chunksize=1,
            desc=f"Processing {split} directories",
        )
    else:
        for d in tqdm(dirs_to_process, desc=f"Processing {split} directories"):
            process_dir(d)


if __name__ == "__main__":
    main()
