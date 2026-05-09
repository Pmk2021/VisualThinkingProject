from AnytimeTrajectoryPredictor.models.TrajectoryPredictor import (
    TrajectoryPredictor,
)
from AnytimeTrajectoryPredictor.Data.feature_extractor import (
    FeatureDataset,
    WaymoPredictionDataset,
)
from AnytimeTrajectoryPredictor.trainer import Trainer
import argparse
import yaml
from box import Box
import torch
from torch.utils.data import DataLoader


def _cfg_get(config, key, default=None):
    return getattr(config, key, default) if config is not None else default


def make_dataloaders(args):
    """
    Create dataloaders for training and validation datasets.
    """
    dataset_type = _cfg_get(args.feature_extractor, "dataset_type", "feature")
    if dataset_type == "waymo_prediction":
        train_dataset = WaymoPredictionDataset(args.feature_extractor, split="train")
        val_dataset = WaymoPredictionDataset(args.feature_extractor, split="val")
        args.model.trajectory_mean = train_dataset.target_mean
        args.model.trajectory_std = train_dataset.target_std
        args.model.history_steps_H = train_dataset.history_steps
        args.model.future_horizon_T = train_dataset.future_steps
    else:
        train_dataset = FeatureDataset(args.feature_extractor)
        val_dataset = FeatureDataset(args.feature_extractor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.training.batch_size,
        shuffle=True,
        num_workers=int(_cfg_get(args.training, "num_workers", 0)),
        pin_memory=bool(_cfg_get(args.training, "pin_memory", False)),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.training.batch_size,
        shuffle=False,
        num_workers=int(_cfg_get(args.training, "num_workers", 0)),
        pin_memory=bool(_cfg_get(args.training, "pin_memory", False)),
    )

    return train_loader, val_loader


def main(args):
    """Main function to set up data, model, optimizer, and trainer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = make_dataloaders(args)

    # Check if model exists and load it, otherwise create a new one
    if args.training.from_checkpoint:
        print("Loading model from:", args.training.from_checkpoint)
        model = TrajectoryPredictor.create_model(args).to(device)
        checkpoint = torch.load(args.training.from_checkpoint, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
    else:
        print("No pre-trained model specified. Initializing new model.")
        model = TrajectoryPredictor.create_model(args).to(device)

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
        args = Box(yaml.safe_load(f))

    main(args)
