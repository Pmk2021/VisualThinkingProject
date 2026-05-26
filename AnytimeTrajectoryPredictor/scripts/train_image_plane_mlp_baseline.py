#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import yaml
from box import Box

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from AnytimeTrajectoryPredictor.scripts.train_trajectory_model import main as train_main


def _set_if_not_none(obj, key, value):
    if value is not None:
        setattr(obj, key, value)


def main():
    parser = argparse.ArgumentParser(description='Train the simple image-plane MLP GMM baseline.')
    parser.add_argument('--config', default='configs/image_plane_mlp_gmm_baseline.yml')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--save_to', default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = Box(yaml.safe_load(f))
    cfg.model.type = 'image_plane_mlp_gmm'
    _set_if_not_none(cfg.training, 'num_epochs', args.epochs)
    _set_if_not_none(cfg.training, 'batch_size', args.batch_size)
    _set_if_not_none(cfg.feature_extractor, 'max_samples', args.max_samples)
    _set_if_not_none(cfg.training, 'save_to', args.save_to)
    train_main(cfg)


if __name__ == '__main__':
    main()
