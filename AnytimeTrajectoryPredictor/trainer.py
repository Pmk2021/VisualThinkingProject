import torch
import wandb
import random
from tqdm import tqdm


class Trainer:
    def __init__(
        self, model, optimizer, train_loader, val_loader, device, args
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.args = args
        wandb.init(project=args.training.wandb_project, config=args)

    def train_single_epoch(self):
        for batch in self.train_loader:
            feature, trajectory = (
                batch["feature"].to(self.device),
                batch["trajectory"].to(self.device),
            )
            feature = feature.transpose(0, 1)
            trajectory = trajectory.transpose(0, 1)
            f_ = [random.randint(1, 10) for _ in range(len(feature))]
            # Compute Loss
            loss = self.model.compute_loss(feature, trajectory, f_)

            # Apply Gradient Step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss

    def validate(self):
        total_loss = 0
        for batch in self.val_loader:
            feature, trajectory = batch["feature"], batch["trajectory"]

            feature = feature.transpose(0, 1)
            trajectory = trajectory.transpose(0, 1)

            f_ = [
                random.randint(1, 10)
                for _ in range(self.train_loader.future_frames)
            ]
            # Compute Loss
            loss = self.model.compute_loss(feature, trajectory, f_)
            total_loss += loss.item()

        wandb.log({"validation_loss": total_loss / len(self.val_loader)})

    def train(self, num_epochs):
        for epoch in tqdm(range(num_epochs)):
            loss = self.train_single_epoch()
            # Log metrics to Weights & Biases
            wandb.log({"epoch": epoch})
            print(f"EPOCH {epoch} Done!!! with loss {loss}")
