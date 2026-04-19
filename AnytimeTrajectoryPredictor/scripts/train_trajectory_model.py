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
