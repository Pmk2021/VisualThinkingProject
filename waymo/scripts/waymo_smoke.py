#!/usr/bin/env python3
"""Local smoke checks for the ported Waymo conversion/viewer scripts."""

from __future__ import annotations

import argparse
import json
import py_compile
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd


SCRIPT_NAMES = (
    "build_waymo_rgb_trajectory_dataset.py",
    "build_waymo_web_viewer.py",
    "extract_waymo_motion.py",
    "generate_rgb_trajectory_schema_doc.py",
    "stream_waymo_to_izar.py",
    "waymo_motion_dataset.py",
)

TABLES = (
    "images",
    "ego_poses",
    "trajectories",
    "image_trajectories",
    "prediction_targets",
)


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    waymo_root = repo_root / "waymo"
    scripts_dir = waymo_root / "scripts"
    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = repo_root / dataset_root

    print(f"[smoke] repo: {repo_root}")
    compile_scripts(scripts_dir)
    generate_schema_doc(repo_root, scripts_dir)
    validate_dataset(dataset_root)
    rebuild_small_viewer(repo_root, dataset_root)
    import_torch_dataset(scripts_dir)
    print("[smoke] ok")
    return 0


def compile_scripts(scripts_dir: Path) -> None:
    print("[smoke] compiling Waymo scripts")
    for name in SCRIPT_NAMES:
        path = scripts_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        py_compile.compile(str(path), doraise=True)


def generate_schema_doc(repo_root: Path, scripts_dir: Path) -> None:
    print("[smoke] generating schema interface into a temp file")
    sys.path.insert(0, str(scripts_dir))
    from generate_rgb_trajectory_schema_doc import load_converter_schema, render_markdown

    converter = repo_root / "waymo" / "scripts" / "build_waymo_rgb_trajectory_dataset.py"
    schema = load_converter_schema(converter)
    markdown = render_markdown(schema, Path("waymo/scripts/build_waymo_rgb_trajectory_dataset.py"))
    required = (
        "images.parquet",
        "ego_poses.parquet",
        "trajectories.parquet",
        "image_trajectories.parquet",
        "prediction_targets.parquet",
    )
    missing = [name for name in required if name not in markdown]
    if missing:
        raise AssertionError(f"generated schema doc is missing tables: {missing}")


def validate_dataset(dataset_root: Path) -> None:
    if not dataset_root.exists():
        print(f"[smoke] dataset not found, skipping parquet checks: {dataset_root}")
        return

    print(f"[smoke] validating parquet tables in {dataset_root}")
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_tables = set(manifest.get("tables", {}))
    missing_manifest_tables = sorted(set(TABLES).difference(manifest_tables))
    if missing_manifest_tables:
        raise AssertionError(f"manifest is missing tables: {missing_manifest_tables}")

    frames = {}
    for name in TABLES:
        path = dataset_root / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        frames[name] = frame
        print(f"  {name}: rows={len(frame)} cols={len(frame.columns)}")

    images = frames["images"]
    ego = frames["ego_poses"]
    required_image_cols = {"split", "scene_id", "frame_timestamp_micros"}
    required_ego_cols = {"split", "scene_id", "frame_timestamp_micros", "ego_x"}
    if required_image_cols.issubset(images.columns) and required_ego_cols.issubset(ego.columns):
        joined = images.merge(
            ego.loc[:, sorted(required_ego_cols)],
            on=["split", "scene_id", "frame_timestamp_micros"],
            how="left",
        )
        missing = int(joined["ego_x"].isna().sum())
        if missing:
            raise AssertionError(f"{missing}/{len(joined)} image rows do not join to ego_poses")
        print(f"  image->ego joins: {len(joined) - missing}/{len(joined)}")

    viewer_index = dataset_root / "viewer" / "index.html"
    viewer_data = dataset_root / "viewer" / "viewer_data.js"
    if not viewer_index.exists() or not viewer_data.exists():
        raise FileNotFoundError("viewer/index.html or viewer/viewer_data.js is missing")


def rebuild_small_viewer(repo_root: Path, dataset_root: Path) -> None:
    if not dataset_root.exists():
        print("[smoke] dataset not found, skipping viewer rebuild")
        return

    print("[smoke] rebuilding a tiny viewer in /tmp")
    scripts_dir = repo_root / "waymo" / "scripts"
    sys.path.insert(0, str(scripts_dir))
    from build_waymo_web_viewer import build_web_viewer

    out = Path(tempfile.mkdtemp(prefix="waymo_viewer_smoke_"))
    try:
        images = pd.read_parquet(dataset_root / "images.parquet").head(80).to_dict("records")
        trajectories = pd.read_parquet(dataset_root / "trajectories.parquet").head(80).to_dict("records")
        links = pd.read_parquet(dataset_root / "image_trajectories.parquet").head(500).to_dict("records")
        targets = pd.read_parquet(dataset_root / "prediction_targets.parquet").head(80).to_dict("records")
        build_web_viewer(
            out,
            images,
            trajectories,
            links,
            targets,
            max_scenes=2,
            max_images_per_scene=4,
        )
        if not (out / "index.html").exists() or not (out / "viewer_data.js").exists():
            raise AssertionError("small viewer rebuild did not create expected files")
    finally:
        shutil.rmtree(out, ignore_errors=True)


def import_torch_dataset(scripts_dir: Path) -> None:
    print("[smoke] importing PyTorch dataset helper")
    sys.path.insert(0, str(scripts_dir))
    import torch
    from waymo_motion_dataset import WaymoMotionDataset, pad_motion_batch

    if not callable(pad_motion_batch):
        raise AssertionError("pad_motion_batch is not callable")
    print(f"  torch={torch.__version__}, dataset={WaymoMotionDataset.__name__}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=".",
        help="VisualThinkingProject repository root.",
    )
    parser.add_argument(
        "--dataset-root",
        default="waymo/dataset/rgb_trajectory_dataset",
        help="Converted RGB trajectory dataset root. Missing datasets are skipped.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
