from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import platform 
import re
from tqdm import tqdm

NODE = platform.node()
NODE_TEMPLATE = r"\b(izar1|i[0-9]{2}|ixl[0-9]{2})\b"
IZAR = re.fullmatch(NODE_TEMPLATE, NODE) is not None

DEFAULT_SOURCE_ROOT = Path("/work/cs-503/santanto/waymo") if IZAR else Path("/Users/nathangromb/Documents/MA4/VI/project/data")
DEFAULT_OUTPUT_ROOT = Path("/work/cs-503/gromb/waymoyolo") if IZAR else Path("/Users/nathangromb/Documents/MA4/VI/project/data/waymoyolo")
DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

SPLIT_TO_FOLDER = {
	"training": "train",
	"validation": "val",
}

def resolve_table_path(chunk_dir: Path, manifest: dict[str, Any], table_name: str) -> Path:
    tables = manifest.get("tables", {})
    entry = tables.get(table_name, {})
    raw_path = entry.get("path")
    if raw_path:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = chunk_dir / candidate
        if candidate.exists():
            return candidate

    fallback = chunk_dir / f"{table_name}.parquet"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Missing {table_name} table in {chunk_dir}")

def image_stem(image_id: str) -> str:
	return str(image_id).replace("/", "_").replace(";", "_")


def row_to_waymo_label(row: pd.Series, width: int, height: int) -> str | None:
    object_type = row["object_type"]
    bbox_center_x = row["bbox_center_x"]
    bbox_center_y = row["bbox_center_y"]
    bbox_width = row["bbox_width"]
    bbox_height = row["bbox_height"]
    if not all(pd.notna(v) for v in [bbox_center_x, bbox_center_y, bbox_width, bbox_height, object_type]):
        raise ValueError(f"Missing fields in row {row.name} of image_trajectories")

    cx = float(bbox_center_x) / width
    cy = float(bbox_center_y) / height
    w = float(bbox_width) / width
    h = float(bbox_height) / height
    if w <= 0.0 or h <= 0.0:
        return None
    return f"{int(object_type)-1} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

def convert_chunk(chunk_dir: Path, output_root: Path) -> None:
    split = chunk_dir.name.split("__")[0]
    split_dir = SPLIT_TO_FOLDER.get(split, split)
    image_dir = output_root / "images" / split_dir
    label_dir = output_root / "labels" / split_dir
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    images = pd.read_parquet(chunk_dir / "images.parquet")
    print(f"{image_stem(images['image_id'][0])}.jpg")
    print(image_dir)
    if f"{image_stem(images['image_id'][0])}.jpg" in [p.name for p in image_dir.glob("*.jpg")]:
        print(f"Chunk {chunk_dir} already converted, skipping.")
        return

    image_trajectories = pd.read_parquet(chunk_dir / "image_trajectories.parquet")

    images = images.sort_values(
        ["scene_id", "camera_name", "frame_timestamp_micros"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    image_trajectories = image_trajectories.sort_values(
        ["scene_id", "frame_timestamp_micros", "camera_name", "trajectory_row_id"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    labels_by_image_id: dict[str, pd.DataFrame] = {
        str(image_id): group.copy()
        for image_id, group in image_trajectories.groupby("image_id", sort=False)
    }

    for _, row in images.iterrows():
        
        stem = image_stem(str(row["image_id"]))
        image_path = image_dir / f"{stem}.jpg"
        label_path = label_dir / f"{stem}.txt"

        image_path.write_bytes(bytes(row["image_jpeg"]))

        width, height = row["image_width"], row["image_height"]
        label_rows = labels_by_image_id.get(str(row["image_id"]), pd.DataFrame())
        labels = []
        if not label_rows.empty:
            for _, label_row in label_rows.iterrows():
                label = row_to_waymo_label(label_row, width, height)
                if label is not None:
                    labels.append(label)

        label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")

def make_yaml(output_root: Path) -> None:
    yaml_path = output_root / "data.yaml"
    yaml_content = f"""train: {output_root / 'images' / ('train' if IZAR else 'waymo')}
val: {output_root / 'images' / 'val'}
nc: 3
names: ['TYPE_VEHICLE', 'TYPE_CYCLIST', 'TYPE_PEDESTRIAN']
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")

def create_val() -> None:
    # Create val folder
    val_image_dir = DEFAULT_OUTPUT_ROOT / "images" / "val"
    val_label_dir = DEFAULT_OUTPUT_ROOT / "labels" / "val"
    val_image_dir.mkdir(parents=True, exist_ok=True)
    val_label_dir.mkdir(parents=True, exist_ok=True)
    # Copy first 10 images and labels from waymo to val
    waymo_image_dir = DEFAULT_OUTPUT_ROOT / "images" / "waymo"
    waymo_label_dir = DEFAULT_OUTPUT_ROOT / "labels" / "waymo"
    for i, image_path in enumerate(waymo_image_dir.glob("*.jpg")):
        if i >= 160:
            break
        label_path = waymo_label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            (val_image_dir / image_path.name).write_bytes(image_path.read_bytes())
            (val_label_dir / label_path.name).write_text(label_path.read_text(encoding="utf-8"), encoding="utf-8")
            image_path.unlink()
            label_path.unlink()

def main(val=False) -> None:
	if val:
	    chunk_dirs = list(DEFAULT_SOURCE_ROOT.glob("**/validation__*")) if IZAR else []
	else:
        chunk_dirs = (list(DEFAULT_SOURCE_ROOT.glob("**/training__*")) + list(DEFAULT_SOURCE_ROOT.glob("**/validation__*"))) if IZAR else [DEFAULT_SOURCE_ROOT / "waymo"]

    DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)  
	
    make_yaml(DEFAULT_OUTPUT_ROOT)

    for chunk_dir in tqdm(chunk_dirs):  
        convert_chunk(chunk_dir, DEFAULT_OUTPUT_ROOT)

    if not IZAR:
        create_val()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
        description="Prepare the Waymo dataset for YOLO finetuning."
    )
    parser.add_argument(
        "--val",
        action="store_true",
        help="Process validation directories only.",
    )
    args = parser.parse_args()
	
	main(args.val)
