import pyarrow
import pyarrow.parquet as pq
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from io import BytesIO
import os
import platform
import re
import numpy as np

import warnings

warnings.filterwarnings("ignore", message="Could not initialize NNPACK!")

from AnytimeTrajectoryPredictor.models.ObjectTracker import ObjectTracker

NODE = platform.node()
NODE_TEMPLATE = r"\b(izar1|i[0-9]{2}|ixl[0-9]{2})\b"
IZAR = re.fullmatch(NODE_TEMPLATE, NODE) is not None

DATA_PATH = (
    "/work/cs-503/santanto/waymo"
    if IZAR
    else "/Users/nathangromb/Documents/MA4/VI/project/data"
)
FEATURES = [
    "bboxes",
    "confidences",
    "object_ids",
    "class_ids",
    "latent_features",
    "local_latent_features",
]
OUTPUT_TABLES = [
    "fe_bboxes.parquet",
    "fe_confidences.parquet",
    "fe_class_ids.parquet",
    "fe_latent_features.parquet",
    "fe_local_latent_features.parquet",
]

train_dirs = list(Path(DATA_PATH).glob("training__*" if IZAR else "waymo"))
val_dirs = list(Path(DATA_PATH).glob("validation__*")) if IZAR else []


def _build_feature_values(
    feat, feat_len, feature_offset, batch_idx, output, n_objects, obj_id_col
):
    """Extract feature values based on feature type.

    Args:
        feat: Feature name
        feat_len: Length/dimension of the feature
        feature_offset: Starting column index in the feature tensor
        batch_idx: Batch index (usually 0 for single images)
        output: Tracker output dict
        n_objects: Number of objects detected
        obj_id_col: Column index for object_ids in the feature tensor
    """
    features = output["features"]

    if feat == "bboxes":
        return [
            {
                "cx": features[batch_idx, j, feature_offset].item(),
                "cy": features[batch_idx, j, feature_offset + 1].item(),
                "w": features[batch_idx, j, feature_offset + 2].item(),
                "h": features[batch_idx, j, feature_offset + 3].item(),
                "object_id": features[batch_idx, j, obj_id_col].item(),
            }
            for j in range(n_objects)
        ]

    elif feat in ["confidences", "class_ids"]:
        return [
            {
                feat[:-1]: features[batch_idx, j, feature_offset].item(),
                "object_id": features[batch_idx, j, obj_id_col].item(),
            }
            for j in range(n_objects)
        ]

    elif feat == "local_latent_features":
        return [
            {
                **{
                    f"{feat}_{k}": features[batch_idx, j, feature_offset + k].item()
                    for k in range(feat_len)
                },
                "object_id": features[batch_idx, j, obj_id_col].item(),
            }
            for j in range(n_objects)
        ]

    elif feat == "latent_features":  # Global features (one per image)
        return (
            [
                {
                    f"{feat}_{k}": features[batch_idx, 0, feature_offset + k].item()
                    for k in range(feat_len)
                }
            ]
            if n_objects > 0
            else []
        )

    elif feat == "object_ids":
        return []

    else:
        raise ValueError(f"Unknown feature: {feat}")


def _get_feature_column_index(feat, output):
    """Find the column index for object_ids in the feature tensor."""
    col_idx = 0
    for feature_name, length in output["lengths"]:
        if feature_name == "object_ids":
            return col_idx
        col_idx += length
    raise ValueError("object_ids not found in feature lengths")


def process_dir(dir_path):
    parquet_path = dir_path / "images.parquet"
    table = pq.read_table(parquet_path)

    table = table.sort_by(
        [
            ("scene_id", "ascending"),
            ("camera_name", "ascending"),
            ("frame_timestamp_micros", "ascending"),
        ]
    )

    columns_keep = [
        "split",
        "image_id",
        "scene_id",
        "frame_timestamp_micros",
        "camera_name",
        "camera_name_text",
    ]

    # Check for duplicate images in input data
    df_input = table.to_pandas()
    image_key_cols = ["scene_id", "frame_timestamp_micros", "camera_name"]
    if df_input.duplicated(subset=image_key_cols).any():
        dupes = df_input[df_input.duplicated(subset=image_key_cols, keep=False)]
        raise ValueError(
            f"Duplicate images found in {dir_path.name}/images.parquet for {image_key_cols}:\n{dupes[image_key_cols]}"
        )

    features_out = {feat: [] for feat in FEATURES if feat != "object_ids"}

    curr_scene, curr_camera = None, None
    for row in tqdm(table.to_pylist(), desc=f"Processing {dir_path.name}"):
        # Reset tracker if we encounter a new scene or camera (only continuous image sequences should be fed to the same tracker instance)
        if (row["scene_id"], row["camera_name"]) != (curr_scene, curr_camera):
            curr_scene, curr_camera = row["scene_id"], row["camera_name"]
            tracker = ObjectTracker(  # Default model : yolo26n.pt
                model_name="yolo26n.pt",
                feature_components=FEATURES,
                imgsz=640,
                verbose=False,
            )

        # Run image through tracker
        image = Image.open(BytesIO(row["image_jpeg"]))
        output = tracker(image)

        n_objects = output["features"].shape[1]
        obj_id_col = _get_feature_column_index("object_ids", output)
        feature_col_offset = 0  # Running feature column offset

        # Check for duplicate object IDs in model output
        if n_objects > 0:
            object_ids = output["features"][0, :, obj_id_col].numpy()
            if len(object_ids) != len(set(object_ids)):
                unique, counts = np.unique(object_ids, return_counts=True)
                dupes = unique[counts > 1]
                warnings.warn(
                    f"Duplicate object IDs found in tracker output for {dir_path.name}: {dupes}. Keeping only first occurrence of each ID.",
                    UserWarning,
                )
                # Keep only the first occurrence of each object ID
                seen_ids = set()
                valid_indices = []
                for j in range(n_objects):
                    obj_id = object_ids[j]
                    if obj_id not in seen_ids:
                        seen_ids.add(obj_id)
                        valid_indices.append(j)

                # Filter output features to keep only valid indices
                output["features"] = output["features"][:, valid_indices, :]
                n_objects = len(valid_indices)
                object_ids = object_ids[valid_indices]

        # Build metadata for this row
        base_data = {col: row[col] for col in columns_keep}

        # Process each feature (validating order matches FEATURES)
        for feat_check, (feat, feat_len) in zip(
            FEATURES, output["lengths"], strict=True
        ):
            if feat != feat_check:
                raise ValueError(f"Feature mismatch: expected {feat_check}, got {feat}")

            if feat == "object_ids":
                feature_col_offset += feat_len
                continue

            values = _build_feature_values(
                feat, feat_len, feature_col_offset, 0, output, n_objects, obj_id_col
            )

            for value in values:
                features_out[feat].append({**base_data, **value})

            feature_col_offset += feat_len

    # Save each feature type as a separate parquet file
    for feat in FEATURES:
        if feat == "object_ids":
            continue

        features_table = pyarrow.Table.from_pylist(features_out[feat])
        if feat == "latent_features" and features_table.num_rows > 0:
            latent_df = features_table.to_pandas()
            latent_df = latent_df.drop_duplicates(
                subset=["scene_id", "frame_timestamp_micros", "camera_name"],
                keep="first",
            )
            features_table = pyarrow.Table.from_pandas(latent_df, preserve_index=False)

        output_path = dir_path / f"fe_{feat}.parquet"
        pq.write_table(features_table, output_path)


def test():
    """
    Loads saved parquet files and prints summary stats to make sure results are coherent
    """
    print("\n" + "=" * 80)
    print("FEATURE EXTRACTION VALIDATION SUMMARY")
    print("=" * 80)

    summary_data = []
    checked = 0

    for dir in train_dirs:
        if checked >= 3:
            break

        print(f"\n📁 Directory: {dir.name}")
        print("-" * 80)

        for table_name in OUTPUT_TABLES:
            table_path = dir / table_name
            table = pq.read_table(table_path)
            df = table.to_pandas()

            # Determine key columns based on feature type
            key_cols = [
                "scene_id",
                "frame_timestamp_micros",
                "camera_name",
            ] + (["object_id"] if table_name != "fe_latent_features.parquet" else [])

            # Check for duplicates
            is_valid = not df.duplicated(subset=key_cols).any()
            status = "✓" if is_valid else "✗"

            # Summary stats
            n_rows = table.num_rows
            n_cols = len(table.column_names)
            n_unique_objects = (
                df["object_id"].nunique() if "object_id" in df.columns else 1
            )

            print(
                f"  {status} {table_name:35} | Rows: {n_rows:6d} | Cols: {n_cols:2d}\t| Unique Objects: {n_unique_objects:5d}"
            )

            summary_data.append(
                {
                    "Feature": table_name.replace("fe_", "").replace(".parquet", ""),
                    "Rows": n_rows,
                    "Valid": is_valid,
                }
            )

            if not is_valid:
                dupes = df[df.duplicated(subset=key_cols, keep=False)]
                raise ValueError(
                    f"Duplicate rows found for {key_cols} in {dir.name}/{table_name}:\n{dupes}"
                )

        checked += 1

    print("\n" + "=" * 80)
    print(f"✓ All {len(summary_data)} feature tables validated successfully!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    print("Using data path:", DATA_PATH)

    for train_dir in tqdm(train_dirs, desc="Processing training directories"):
        if all((train_dir / table_name).exists() for table_name in OUTPUT_TABLES):
            print(f"Skipping {train_dir.name} (already processed)")
            continue
        features = process_dir(train_dir)

    val_dirs = []
    for val_dir in tqdm(val_dirs, desc="Processing validation directories"):
        if all((val_dir / table_name).exists() for table_name in OUTPUT_TABLES):
            print(f"Skipping {val_dir.name} (already processed)")
            continue
        features = process_dir(val_dir)

    test()

    print("Feature extraction completed for all directories.")
