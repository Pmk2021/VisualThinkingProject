from pathlib import Path
import sys
import math

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from AnytimeTrajectoryPredictor.models.TrajectoryPredictor import (
    TrajectoryPredictor,
)
from AnytimeTrajectoryPredictor.Data.feature_extractor import FeatureDataset
from AnytimeTrajectoryPredictor.trainer import Trainer
import argparse
import yaml
from box import Box
import torch
import os
from torch.utils.data import DataLoader


def make_dataloaders(args):
    """
    Create dataloaders for training and validation datasets.
    """
    train_dataset = FeatureDataset(
        args.feature_extractor,
    )

    val_dataset = FeatureDataset(
        args.feature_extractor,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.training.batch_size, shuffle=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.training.batch_size, shuffle=False
    )

    return train_loader, val_loader


def make_lr_scheduler(optimizer, args, steps_per_epoch):
    scheduler_config = getattr(args.training, "lr_scheduler", None)
    if scheduler_config is None or not scheduler_config.get("enabled", False):
        return None

    total_steps = max(1, int(args.training.num_epochs) * steps_per_epoch)
    warmup_epochs = int(scheduler_config.get("warmup_epochs", 1))
    warmup_steps = int(scheduler_config.get("warmup_steps", warmup_epochs * steps_per_epoch))
    if total_steps > 1:
        warmup_steps = max(1, min(warmup_steps, total_steps - 1))
    else:
        warmup_steps = 1

    def lr_multiplier(step):
        if total_steps <= 1:
            return 0.0
        if step < warmup_steps:
            return step / max(1, warmup_steps - 1)

        decay_steps = max(1, total_steps - warmup_steps)
        progress = (step - warmup_steps + 1) / decay_steps
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)


def main(args):
    """Main function to set up data, model, optimizer, and trainer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = make_dataloaders(args)

    # Check if model exists and load it, otherwise create a new one
    if args.training.from_checkpoint:
        print("Loading model from:", args.training.from_checkpoint)
        model = TrajectoryPredictor.create_model(args).to(device)
        model.load_state_dict(torch.load(args.training.load_path))
    else:
        print("No pre-trained model specified. Initializing new model.")
        model = TrajectoryPredictor.create_model(args).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.training.learning_rate
    )
    scheduler = make_lr_scheduler(optimizer, args, len(train_loader))

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        args=args,
    )

    print("Starting training...")
    trainer.train(num_epochs=args.training.num_epochs)

    print("Training complete! Model saved to:", args.training.save_to)
    save_dir = os.path.dirname(args.training.save_to)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(trainer.model.state_dict(), args.training.save_to)

    print("Validating:")
    trainer.validate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    cli_args = parser.parse_args()

    with open(cli_args.config) as f:
        args = Box(yaml.safe_load(f))

    main(args)
