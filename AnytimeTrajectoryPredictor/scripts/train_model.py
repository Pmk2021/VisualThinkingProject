from AnytimeTrajectoryPredictor.models import TrajectoryPredictor
from AnytimeTrajectoryPredictor.data import BuildTrajectoryDataset
from AnytimeTrajectoryPredictor.trainer import trainer
import argparse
import yaml
from box import Box


def make_datasets():
    BuildTrajectoryDataset(args)


def main():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    cli_args = parser.parse_args()

    with open(cli_args.config) as f:
        args = Box(yaml.safe_load(f))

    main(args)
