from AnytimeTrajectoryPredictor.models import TrajectoryPredictor
from AnytimeTrajectoryPredictor.data import BuildTrajectoryDataset
from AnytimeTrajectoryPredictor.trainer import Trainer, trainer
import argparse
import yaml
from box import Box
import torch


def make_dataloaders(args):
    """
    Create dataloaders for training and validation datasets.
    """
    train_dataset = BuildTrajectoryDataset(
        data_path=args.data.train_path, features=args.data.features
    )
    val_dataset = BuildTrajectoryDataset(
        data_path=args.data.val_path, features=args.data.features
    )

    return train_dataset, val_dataset


def main():
    """Main function to set up data, model, optimizer, and trainer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = make_dataloaders(args)
    model = TrajectoryPredictor(args.model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.lr)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
    )

    trainer.train(num_epochs=args.training.num_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    cli_args = parser.parse_args()

    with open(cli_args.config) as f:
        args = Box(yaml.safe_load(f))

    main(args)
