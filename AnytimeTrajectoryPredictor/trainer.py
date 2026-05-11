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
        loss_total = 0
        for batch in self.train_loader:
            feature, trajectory = batch["features"], batch["trajectory"]
            feature = feature.transpose(0, 1).to(self.device)
            trajectory = trajectory.transpose(0, 1).to(self.device)
            f_ = [random.randint(1, 10) for i in range(len(feature))]
            # Compute Loss
            loss = self.model.compute_loss(feature, trajectory, f_)
            loss_total += loss
            # Apply Gradient Step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss_total

    def validate(self):
        total_loss = 0
        for batch in self.val_loader:
            feature, trajectory = (
                batch["features"].to(self.device),
                batch["trajectory"].to(self.device),
            )
            f_ = [random.randint(1, 10) for i in range(len(feature))]
            # Compute Loss
            loss = self.model.compute_loss(feature, trajectory, f_)
            total_loss += loss.item()
        print(total_loss / len(self.val_loader))
        wandb.log({"validation_loss": total_loss / len(self.val_loader)})

    def train(self, num_epochs):
        pbar = tqdm(range(num_epochs), desc="Training")

        for epoch in pbar:
            loss = self.train_single_epoch()

            pbar.set_postfix({"loss": loss})
            wandb.log({"epoch": epoch, "loss": loss})
            if epoch % 10 == 0:
                self.validate()
