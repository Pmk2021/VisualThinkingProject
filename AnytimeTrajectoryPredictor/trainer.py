import torch
import wandb


class Trainer:
    def __init__(
        self, model, optimizer, train_loader, val_loader, device, args
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader.to(device)
        self.val_loader = val_loader.to(device)
        self.device = device
        self.args = args
        wandb.init(project=args.training.wandb_project, config=args)

    def train_single_epoch(self):
        for batch in self.train_loader:
            feature, trajectory = batch["features"], batch["trajectory"]

            # Compute Loss
            loss = self.model.compute_loss(feature, trajectory)

            # Apply Gradient Step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def validate(self):
        total_loss = 0
        for batch in self.val_loader:
            feature, trajectory = batch["features"], batch["trajectory"]

            # Compute Loss
            loss = self.model.compute_loss(feature, trajectory)
            total_loss += loss.item()

        wandb.log({"validation_loss": total_loss / len(self.val_loader)})

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.train_single_epoch()
            # Log metrics to Weights & Biases
            wandb.log({"epoch": epoch})
