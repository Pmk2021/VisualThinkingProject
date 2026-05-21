from AnytimeTrajectoryPredictor.models import TrajectoryPredictor
from AnytimeTrajectoryPredictor.Data.feature_extractor import FeatureDataset
from AnytimeTrajectoryPredictor.trainer import Trainer, trainer
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
        args.feature_extractor.train_feature_path,
        args.feature_extractor,
        data_path=args.data.train_path,
    )

    val_dataset = FeatureDataset(
        args.feature_extractor.val_feature_path,
        args.feature_extractor,
        data_path=args.data.val_path,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.training.batch_size, shuffle=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.training.batch_size, shuffle=False
    )

    return train_loader, val_loader


def main(args):
    """Main function to set up data, model, optimizer, and trainer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = make_dataloaders(args)

    # Check if model exists and load it, otherwise create a new one
    if args.training.load_path:
        print("Loading model from:", args.training.load_path)
        model = TrajectoryPredictor(args.model).to(device)
        model.load_state_dict(torch.load(args.training.load_path))
    else:
        print("No pre-trained model specified. Initializing new model.")
        model = TrajectoryPredictor(args.model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.lr)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
    )

    print("Starting training...")
    trainer.train(num_epochs=args.training.num_epochs)

    print("Training complete! Model saved to:", args.training.save_path)
    trainer.save_model(args.training.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    cli_args = parser.parse_args()

    with open(cli_args.config) as f:
        args = Box(yaml.safe_load(f))

    main(args)
