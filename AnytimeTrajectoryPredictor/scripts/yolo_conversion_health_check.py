from __future__ import annotations

import argparse
import json
import random
from io import BytesIO
from pathlib import Path
import platform
import re

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


NODE = platform.node()
NODE_TEMPLATE = r"\b(izar1|i[0-9]{2}|ixl[0-9]{2})\b"
IZAR = re.fullmatch(NODE_TEMPLATE, NODE) is not None


def image_stem(image_id: str) -> str:
    return str(image_id).replace("/", "_").replace(";", "_")


def find_table_path(chunk_dir: Path, manifest: dict | None, table_name: str) -> Path:
    if manifest:
        tables = manifest.get("tables", {})
        entry = tables.get(table_name)
        if entry:
            p = Path(entry.get("path"))
            if not p.is_absolute():
                p = chunk_dir / p
            if p.exists():
                return p

    candidates = [
        chunk_dir / f"{table_name}.parquet",
        chunk_dir / "dataset" / "rgb_trajectory_dataset" / f"{table_name}.parquet",
        chunk_dir / "dataset/rgb_trajectory_dataset" / f"{table_name}.parquet",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not find {table_name}.parquet under {chunk_dir}")


def load_tables(chunk_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_path = chunk_dir / "manifest.json"
    manifest = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    images_path = find_table_path(chunk_dir, manifest, "images")
    links_path = find_table_path(chunk_dir, manifest, "image_trajectories")

    images = pd.read_parquet(images_path)
    links = pd.read_parquet(links_path)
    return images, links


def draw_waymo_boxes(ax, img_row: pd.Series, link_rows: pd.DataFrame) -> None:
    img = Image.open(BytesIO(bytes(img_row["image_jpeg"]))).convert("RGB")
    ax.imshow(img)
    w, h = img.size
    for _, link in link_rows.iterrows():
        cx = float(link["bbox_center_x"])
        cy = float(link["bbox_center_y"])
        bw = float(link["bbox_width"]) 
        bh = float(link["bbox_height"]) 
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        rect = plt.Rectangle((x1, y1), bw, bh, edgecolor="red", facecolor="none", linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, y1, str(int(link.get("object_type", -1))), color="yellow", fontsize=10, backgroundcolor="black")
    ax.set_title("Original (Waymo)")
    ax.axis("off")


def draw_yolo_boxes(ax, yolo_image_path: Path, yolo_label_path: Path) -> None:
    img = Image.open(yolo_image_path).convert("RGB")
    ax.imshow(img)
    w, h = img.size
    if yolo_label_path.exists():
        for line in yolo_label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = parts[0]
            cx, cy, bw_rel, bh_rel = map(float, parts[1:5])
            bw = bw_rel * w
            bh = bh_rel * h
            x1 = cx * w - bw / 2.0
            y1 = cy * h - bh / 2.0
            rect = plt.Rectangle((x1, y1), bw, bh, edgecolor="lime", facecolor="none", linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1, cls, color="white", fontsize=10, backgroundcolor="black")
    ax.set_title("Converted (YOLO)")
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Waymo original boxes vs YOLO-converted boxes side-by-side.")
    parser.add_argument("--chunk-dir", type=Path, default=(Path("/work/cs-503/santanto/waymo") if IZAR else Path("data/waymo")))
    parser.add_argument("--yolo-root", type=Path, default=(Path("/work/cs-503/gromb/waymoyolo") if IZAR else Path("data/waymoyolo")))
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    images, links = load_tables(args.chunk_dir)

    # group links by image_id
    labels_by_image = {k: g for k, g in links.groupby("image_id", sort=False)}

    # candidate image rows that have at least one link and a corresponding YOLO file
    candidates = []
    for _, img_row in images.iterrows():
        image_id = img_row["image_id"]
        if image_id not in labels_by_image:
            continue
        stem = image_stem(image_id)
        split = img_row.get("split", "training")
        split_dir = "train" if str(split).lower().startswith("train") else ("val" if str(split).lower().startswith("val") else str(split).lower())
        yolo_img = args.yolo_root / "images" / split_dir / f"{stem}.jpg"
        yolo_lbl = args.yolo_root / "labels" / split_dir / f"{stem}.txt"
        if yolo_img.exists():
            candidates.append((img_row, labels_by_image[image_id], yolo_img, yolo_lbl))

    if not candidates:
        print("No matching converted images found for comparison.")
        return

    random.seed(42)
    sample = random.sample(candidates, min(args.n, len(candidates)))

    for img_row, link_rows, yolo_img, yolo_lbl in sample:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        draw_waymo_boxes(axes[0], img_row, link_rows)
        draw_yolo_boxes(axes[1], yolo_img, yolo_lbl)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
