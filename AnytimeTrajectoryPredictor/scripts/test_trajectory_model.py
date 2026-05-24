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
import random
import matplotlib.pyplot as plt


def plot_inputs_and_traj(inputs, traj, num_points=100, device="cpu"):
    """
    inputs: (3, N) flattened observations per mode
    traj: (3, 4) polynomial coefficients per mode
    """

    inputs = inputs.to(device)
    traj = traj.to(device)

    num_modes = traj.shape[0]

    # shared x-axis for polynomials
    x = torch.linspace(-10, 10, num_points, device=device)

    plt.figure()

    for i in range(num_modes):
        # ----- plot polynomial -----
        print(traj[i])
        a, b, c, d = traj[i]
        y_poly = a + b * x + c * x**2 + d * x**3

        plt.plot(
            x.cpu(),
            y_poly.detach().cpu(),
            label=f"traj poly {i}"
        )

        # ----- plot inputs (assume sampled y-values) -----
        inp = inputs[i]

        # map input to same x-grid (resample)
        inp_x = torch.linspace(-10, 10, inp.numel(), device=device)

        plt.scatter(
            inp_x.cpu(),
            inp.detach().cpu(),
            s=20,
            label=f"inputs {i}"
        )

    plt.title("Inputs vs Trajectory Polynomials")
    plt.legend()
    plt.grid(True)
    print("DD")
    plt.savefig("/home/muralikr/VisualThinkingProject/AnytimeTrajectoryPredictor/scripts/a.png", dpi=300, bbox_inches="tight")
    plt.show()

def make_dataloaders(args):
    """
    Create dataloaders for training and validation datasets.
    """

    val_dataset = FeatureDataset(
        args.feature_extractor,
        split="validation",
    )



    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers
    )

    return None, val_loader


def main(args):
    """Main function to set up data, model, optimizer, and trainer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    # Check if model exists and load it, otherwise create a new one

    print("Loading model from:", "/home/muralikr/VisualThinkingProject/AnytimeTrajectoryPredictor/checkpoints/model_epoch_3.pt")
    model = TrajectoryPredictor.create_model(args).to(device)
    print("Q")
    ckpt = torch.load("/home/muralikr/VisualThinkingProject/AnytimeTrajectoryPredictor/checkpoints/model_epoch_3.pt", map_location="cpu")

    model.load_state_dict(ckpt["model_state_dict"])

    train_loader, val_loader = make_dataloaders(args)
    print("B")
    for batch in val_loader:
        with torch.no_grad():
            f_ =  [random.randint(1, 3) for _ in range(len(batch["features"]))]
            predictions = model(batch["features"].to(device), f_)[-1][-1]
            means = predictions[...,:model.coeff_dim][0]
        print(means)
        print("traj:", batch["trajectory"][-1][-1][0])
        plot_inputs_and_traj(means, batch["trajectory"][-1][-1][0])
        break  # Remove this break to evaluate on the entire validation set 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    cli_args = parser.parse_args()

    with open(cli_args.config) as f:
        args = Box(yaml.safe_load(f))

    main(args)
