import argparse
import sys
from pathlib import Path

import yaml
from box import Box
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from AnytimeTrajectoryPredictor.Data.feature_extractor import WaymoImagePlaneDataset


def _cfg_get(config, key, default=None):
    return getattr(config, key, default) if config is not None else default


def main():
    parser = argparse.ArgumentParser(
        description="Build the expensive Waymo image-plane sample index once and save it as pickle cache files."
    )
    parser.add_argument("--config", default="configs/astra_edm_diffusion_waymo_image_plane.yml")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--splits", default="train,val", help="Comma-separated splits to build, e.g. train,val")
    args_cli = parser.parse_args()

    with open(args_cli.config, "r") as f:
        cfg = Box(yaml.safe_load(f))

    output_dir = args_cli.output_dir or _cfg_get(
        cfg.feature_extractor,
        "sample_cache_dir",
        "cache/waymo_image_plane",
    )
    cfg.feature_extractor.use_sample_cache = False
    cfg.feature_extractor.use_image_cache = False
    cfg.feature_extractor.sample_cache_dir = output_dir
    cfg.feature_extractor.image_cache_dir = output_dir

    splits = [item.strip() for item in args_cli.splits.split(",") if item.strip()]
    split_bar = tqdm(splits, desc="Building image-plane caches", unit="split")
    for split in split_bar:
        split_bar.set_postfix_str(f"split={split}")
        dataset = WaymoImagePlaneDataset(cfg.feature_extractor, split=split)
        sample_cache_path = dataset.save_sample_cache(output_dir=output_dir)
        tqdm.write(
            f"Wrote {sample_cache_path} ({len(dataset)} samples, {len(dataset.segments)} segments)"
        )
        npy_path, meta_path = dataset.save_image_cache(output_dir=output_dir)
        tqdm.write(f"Wrote {npy_path} and {meta_path}")


if __name__ == "__main__":
    main()
