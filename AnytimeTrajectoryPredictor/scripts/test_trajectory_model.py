import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from box import Box

from AnytimeTrajectoryPredictor.Data.feature_extractor import FeatureDataset


def make_dataloader(args):
    dataset = FeatureDataset(
        args.feature_extractor,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,  # IMPORTANT for debugging
    )

    return dataset, loader


def test_dataloader(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, loader = make_dataloader(args)

    print(f"\nDataset size: {len(dataset)}")

    for i, batch in enumerate(loader):
        print(f"\n================ Batch {i} ================")

        x = batch["features"]
        y = batch["trajectory"]

        print("x shape:", x.shape)  # (B, T, F)
        print("y shape:", y.shape)  # (B, T, 3)

        print("\nStats:")
        print("x min/max:", x.min().item(), x.max().item())
        print("y min/max:", y.min().item(), y.max().item())

        if i >= 3:  # only a few batches
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    cli_args = parser.parse_args()

    with open(cli_args.config, "r") as f:
        args = Box(yaml.safe_load(f))

    test_dataloader(args)
