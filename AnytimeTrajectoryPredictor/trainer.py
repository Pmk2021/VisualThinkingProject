import os
import random

import torch
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None


def _cfg_get(config, key, default=None):
    return getattr(config, key, default) if config is not None else default


def _box_to_dict(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {k: _box_to_dict(v) for k, v in value.items()}
    return value


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
        self.gradient_clip_norm = _cfg_get(args.training, "gradient_clip_norm", None)
        self.use_amp = bool(_cfg_get(args.training, "mixed_precision", False)) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.best_validation = float("inf")
        self.save_to = _cfg_get(args.training, "save_to", "checkpoints/model.pth")
        wandb_mode = _cfg_get(args.training, "wandb_mode", "offline")
        self.use_wandb = wandb is not None and wandb_mode != "disabled"
        if self.use_wandb:
            wandb.init(
                project=_cfg_get(args.training, "wandb_project", "astra_edm_diffusion"),
                config=_box_to_dict(args),
                mode=wandb_mode,
            )

    def _move_batch(self, batch):
        moved = {}
        for key, value in batch.items():
            moved[key] = value.to(self.device) if torch.is_tensor(value) else value
        return moved

    def _compute_loss(self, batch):
        if getattr(self.model, "expects_batch", False):
            return self.model.compute_loss(batch)

        feature, trajectory = batch["features"], batch["trajectory"]
        feature = feature.transpose(0, 1).to(self.device)
        trajectory = trajectory.transpose(0, 1).to(self.device)
        f_ = [random.randint(1, 10) for _ in range(len(feature))]
        return {"loss": self.model.compute_loss(feature, trajectory, f_)}

    def _scalar_logs(self, output, prefix):
        logs = {}
        for key, value in output.items():
            if key == "loss":
                metric_key = f"{prefix}/loss"
            else:
                metric_key = f"{prefix}/{key}"
            if torch.is_tensor(value):
                logs[metric_key] = float(value.detach().mean().cpu())
            elif isinstance(value, (int, float)):
                logs[metric_key] = float(value)
        return logs

    def train_single_epoch(self):
        self.model.train()
        totals = {}
        for batch in self.train_loader:
            batch = self._move_batch(batch)
            self.optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                output = self._compute_loss(batch)
                loss = output["loss"]
            self.scaler.scale(loss).backward()
            if self.gradient_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(self.gradient_clip_norm)
                )
                output["grad_norm"] = grad_norm.detach()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for key, value in self._scalar_logs(output, "train").items():
                totals[key] = totals.get(key, 0.0) + value

        return {key: value / max(len(self.train_loader), 1) for key, value in totals.items()}

    def validate(self):
        self.model.eval()
        totals = {}
        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._move_batch(batch)
                output = self._compute_loss(batch)
                for key, value in self._scalar_logs(output, "val").items():
                    totals[key] = totals.get(key, 0.0) + value
        logs = {key: value / max(len(self.val_loader), 1) for key, value in totals.items()}
        if self.use_wandb:
            wandb.log(logs)
        return logs

    def save_checkpoint(self, epoch, metrics, is_best=False):
        path = self.save_to
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": _box_to_dict(self.args),
        }
        torch.save(checkpoint, path)
        if is_best:
            base, ext = os.path.splitext(path)
            torch.save(checkpoint, f"{base}.best{ext or '.pth'}")

    def train(self, num_epochs):
        pbar = tqdm(range(num_epochs), desc="Training")

        for epoch in pbar:
            train_logs = self.train_single_epoch()
            val_logs = self.validate()
            epoch_logs = {"epoch": epoch, **train_logs, **val_logs}
            validation_score = val_logs.get("val/loss", float("inf"))
            is_best = validation_score < self.best_validation
            if is_best:
                self.best_validation = validation_score
            self.save_checkpoint(epoch, epoch_logs, is_best=is_best)
            pbar.set_postfix({"train_loss": train_logs.get("train/loss"), "val_loss": validation_score})
            if self.use_wandb:
                wandb.log(epoch_logs)
