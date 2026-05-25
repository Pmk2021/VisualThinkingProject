import argparse
import sys
from pathlib import Path

import torch
import yaml
from box import Box
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from AnytimeTrajectoryPredictor.Data.feature_extractor import (
    FeatureDataset,
    WaymoImagePlaneDataset,
    WaymoPredictionDataset,
)
from AnytimeTrajectoryPredictor.models.TrajectoryPredictor import TrajectoryPredictor
from AnytimeTrajectoryPredictor.trainer import Trainer


def _cfg_get(config, key, default=None):
    return getattr(config, key, default) if config is not None else default


def _describe_dataset(name, dataset):
    segment_count = len(getattr(dataset, "segments", []))
    split_source = getattr(dataset, "split_source", "n/a")
    segment_names = [segment.name for segment in getattr(dataset, "segments", [])[:3]]
    print(f"{name}: {len(dataset)} samples from {segment_count} segments (split_source={split_source})")
    if segment_names:
        print(f"{name} first segments: {', '.join(segment_names)}")


def make_dataloaders(args):
    dataset_type = _cfg_get(args.feature_extractor, "dataset_type", "feature")
    if dataset_type == "waymo_prediction":
        train_dataset = WaymoPredictionDataset(args.feature_extractor, split="train")
        val_dataset = WaymoPredictionDataset(args.feature_extractor, split="val")
    elif dataset_type == "waymo_image_plane":
        train_dataset = WaymoImagePlaneDataset(args.feature_extractor, split="train")
        val_dataset = WaymoImagePlaneDataset(args.feature_extractor, split="val")
        args.model.use_rgb_context = True
        args.model.input_dim = 4
        args.model.trajectory_mean = [0.0, 0.0]
        args.model.trajectory_std = [1.0, 1.0]
    else:
        train_dataset = FeatureDataset(args.feature_extractor)
        val_dataset = FeatureDataset(args.feature_extractor)

    args.model.trajectory_mean = getattr(train_dataset, "target_mean", _cfg_get(args.model, "trajectory_mean", [0.0, 0.0]))
    args.model.trajectory_std = getattr(train_dataset, "target_std", _cfg_get(args.model, "trajectory_std", [1.0, 1.0]))
    args.model.history_steps_H = getattr(train_dataset, "history_steps", _cfg_get(args.model, "history_steps_H", 11))
    args.model.future_horizon_T = getattr(train_dataset, "future_steps", _cfg_get(args.model, "future_horizon_T", 80))
    _describe_dataset("train", train_dataset)
    _describe_dataset("validation", val_dataset)

    num_workers = int(_cfg_get(args.training, "num_workers", 0))
    pin_memory = bool(_cfg_get(args.training, "pin_memory", False))
    persistent_workers = bool(_cfg_get(args.training, "persistent_workers", False)) and num_workers > 0
    loader_kwargs = {
        "batch_size": args.training.batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(_cfg_get(args.training, "prefetch_factor", 2))

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = make_dataloaders(args)

    model = TrajectoryPredictor.create_model(args).to(device)
    checkpoint_path = _cfg_get(args.training, "from_checkpoint", None)
    if checkpoint_path:
        print("Loading model from:", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
    else:
        print("No pre-trained model specified. Initializing new model.")

    optimizer_name = _cfg_get(args.training, "optimizer", "adamw").lower()
    learning_rate = float(args.training.learning_rate)
    weight_decay = float(_cfg_get(args.training, "weight_decay", 0.0))
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        args=args,
    )
    print("Starting training...")
    trainer.train(num_epochs=args.training.num_epochs)
    print("Training complete! Model saved to:", args.training.save_to)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    cli_args = parser.parse_args()
    with open(cli_args.config) as f:
        main(Box(yaml.safe_load(f)))
