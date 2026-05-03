import torch
import wandb
import random
from tqdm import tqdm


class Trainer:
    def __init__(
        self, model, optimizer, train_loader, val_loader, device, args, scheduler=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.args = args
        wandb.init(project=args.training.wandb_project, config=args)

    def _prepare_batch(self, batch):
        feature = batch["features"].transpose(0, 1).to(self.device)
        trajectory = batch["trajectory"].transpose(0, 1).to(self.device)
        return feature, trajectory

    def _sample_refinement_steps(self, num_frames):
        return [random.randint(1, 10) for _ in range(num_frames)]

    def train_single_epoch(self):
        loss_total = 0.0
        for batch in self.train_loader:
            feature, trajectory = self._prepare_batch(batch)
            f_ = self._sample_refinement_steps(len(feature))
            # Compute Loss
            loss = self.model.compute_loss(feature, trajectory, f_)
            loss_total += loss.item()
            # Apply Gradient Step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
        return loss_total / len(self.train_loader)

    def validate(self):
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                feature, trajectory = self._prepare_batch(batch)
                f_ = self._sample_refinement_steps(len(feature))

                # Compute Loss
                loss = self.model.compute_loss(feature, trajectory, f_)
                total_loss += loss.item()

        validation_loss = total_loss / len(self.val_loader)
        return validation_loss

    def train(self, num_epochs):
        pbar = tqdm(range(num_epochs), desc="Training")

        for epoch in pbar:
            loss = self.train_single_epoch()
            validation_loss = self.validate()
            learning_rate = self.optimizer.param_groups[0]["lr"]

            pbar.set_postfix(
                {"loss": loss, "val_loss": validation_loss, "lr": learning_rate}
            )
            wandb.log(
                {
                    "epoch": epoch,
                    "loss": loss,
                    "validation_loss": validation_loss,
                    "learning_rate": learning_rate,
                }
            )
