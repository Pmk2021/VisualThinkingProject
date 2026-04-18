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

    def train_single_epoch(self):
        for batch in self.train_loader:
            feature, trajectory = batch["features"], batch["trajectory"]

            predicted_distribution = self.model.get_trajectory_dist(feature)

            # Compute Loss
            loss = self.compute_loss(predicted_distribution, trajectory)

            # Apply Gradient Step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def compute_loss(self, distribution, trajectory):
        """Compute the negative log-likelihood loss.
        Args:
            distribution: The predicted distribution over trajectories.
                distribution should have model.numpeaks components, each with a mean and covariance.
            trajectory: The ground truth trajectory.

        Returns:
            loss: The computed loss value.
        """

        ##TODO: Implement the negative log-likelihood loss computation here.
        pass

    def validate(self):
        pass

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.train_single_epoch()
            # Log metrics to Weights & Biases
            wandb.log({"epoch": epoch})
