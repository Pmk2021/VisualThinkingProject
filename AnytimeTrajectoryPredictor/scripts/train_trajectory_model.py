from AnytimeTrajectoryPredictor.models.TrajectoryPredictor import (
    TrajectoryPredictor,
)
from AnytimeTrajectoryPredictor.Data.feature_extractor import FeatureDataset
from AnytimeTrajectoryPredictor.trainer import Trainer
import argparse
import yaml
from box import Box
import torch
from torch.utils.data import DataLoader


def make_dataloaders(args):
    """
    Create dataloaders for training and validation datasets.
    """
    train_dataset = FeatureDataset(
        args.feature_extractor,
        split="training",
    )

    val_dataset = FeatureDataset(
        args.feature_extractor,
        split="validation",
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=args.num_workers
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=args.num_workers
    )

    return train_loader, val_loader


def main(args):
    """Main function to set up data, model, optimizer, and trainer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    # Check if model exists and load it, otherwise create a new one
    if args.training.from_checkpoint:
        print("Loading model from:", args.training.from_checkpoint)
        model = TrajectoryPredictor.create_model(args).to(device)
        model.load_state_dict(torch.load(args.training.load_path))
    else:
        print("No pre-trained model specified. Initializing new model.")
        model = TrajectoryPredictor.create_model(args).to(device)
    train_loader, val_loader = make_dataloaders(args)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.training.learning_rate
    )

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
    torch.save(trainer.model.state_dict(), "model.pth")

    print("Validating:")
    trainer.validate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    cli_args = parser.parse_args()

    with open(cli_args.config) as f:
        args = Box(yaml.safe_load(f))

    main(args)
