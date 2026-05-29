import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from box import Box
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    import wandb
except ImportError:
    wandb = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from AnytimeTrajectoryPredictor.Data.feature_extractor import WaymoKeypointDataset
from AnytimeTrajectoryPredictor.models.unet_keypoint import UNetKeypointModel, load_unet_keypoint_state


def _cfg_get(config, key, default=None):
    return getattr(config, key, default) if config is not None else default


def _box_to_dict(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {k: _box_to_dict(v) for k, v in value.items()}
    return value


def make_dataloaders(args):
    train_dataset = WaymoKeypointDataset(args.feature_extractor, split="train")
    val_dataset = WaymoKeypointDataset(args.feature_extractor, split="val")
    print(f"train: {len(train_dataset)} images from {len(train_dataset.segments)} segments")
    print(f"val: {len(val_dataset)} images from {len(val_dataset.segments)} segments")
    num_workers = int(_cfg_get(args.training, "num_workers", 0))
    pin_memory = bool(_cfg_get(args.training, "pin_memory", False))
    persistent_workers = bool(_cfg_get(args.training, "persistent_workers", False)) and num_workers > 0
    loader_kwargs = {
        "batch_size": args.training.batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(_cfg_get(args.training, "prefetch_factor", 2))
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def heatmap_loss(pred, target, positive_weight=1.0):
    """MSE with extra weight on positive (peak) regions to counter sparsity."""
    weight = 1.0 + positive_weight * target
    return (weight * (pred - target).pow(2)).mean()


def save_checkpoint(save_to, model, optimizer, epoch, metrics, args, is_best=False):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": _box_to_dict(args),
    }
    torch.save(checkpoint, save_to)
    if is_best:
        base, ext = os.path.splitext(save_to)
        torch.save(checkpoint, f"{base}.best{ext or '.pth'}")


def run_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    args,
    train=True,
    checkpoint_callback=None,
    checkpoint_interval_batches=None,
):
    model.train(train)
    totals = {"loss": 0.0, "pos_mse": 0.0, "neg_mse": 0.0}
    n_batches = 0
    use_amp = bool(_cfg_get(args.training, "mixed_precision", False)) and device.type == "cuda"
    positive_weight = float(_cfg_get(args.training, "positive_weight", 50.0))
    grad_clip = _cfg_get(args.training, "gradient_clip_norm", None)
    bar = tqdm(loader, desc="train" if train else "val", unit="batch", leave=False)
    total_batches = len(loader)
    for batch in bar:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["heatmap"].to(device, non_blocking=True)
        with torch.set_grad_enabled(train), torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(image)
            loss = heatmap_loss(pred, target, positive_weight=positive_weight)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            scaler.step(optimizer)
            scaler.update()
        with torch.no_grad():
            pos_mask = (target > 0.05).float()
            neg_mask = 1.0 - pos_mask
            pos_count = pos_mask.sum().clamp_min(1.0)
            neg_count = neg_mask.sum().clamp_min(1.0)
            pos_mse = ((pred - target).pow(2) * pos_mask).sum() / pos_count
            neg_mse = ((pred - target).pow(2) * neg_mask).sum() / neg_count
        totals["loss"] += float(loss.detach())
        totals["pos_mse"] += float(pos_mse)
        totals["neg_mse"] += float(neg_mse)
        n_batches += 1
        bar.set_postfix({"loss": f"{float(loss):.4f}"})
        if (
            train
            and checkpoint_callback is not None
            and checkpoint_interval_batches is not None
            and checkpoint_interval_batches > 0
            and n_batches % checkpoint_interval_batches == 0
            and n_batches < total_batches
        ):
            running_logs = {k: v / max(n_batches, 1) for k, v in totals.items()}
            checkpoint_callback(n_batches, total_batches, running_logs)
    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = make_dataloaders(args)

    model = UNetKeypointModel(args.model).to(device)
    pretrained_path = _cfg_get(args.model, "pretrained_weights", None)
    if pretrained_path:
        print(f"Loading pretrained weights from {pretrained_path}")
        missing, unexpected = load_unet_keypoint_state(model, pretrained_path, strict=False)
        print(f"Loaded with missing={len(missing)} unexpected={len(unexpected)} keys")

    lr = float(_cfg_get(args.training, "learning_rate", 1e-3))
    weight_decay = float(_cfg_get(args.training, "weight_decay", 5e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    use_amp = bool(_cfg_get(args.training, "mixed_precision", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    save_to = _cfg_get(args.training, "save_to", "checkpoints/unet_keypoints_latest.pth")
    os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
    save_every_epoch_fraction = _cfg_get(args.training, "save_every_epoch_fraction", None)
    checkpoint_interval_batches = None
    if save_every_epoch_fraction is not None:
        fraction = float(save_every_epoch_fraction)
        if fraction <= 0.0 or fraction > 1.0:
            raise ValueError("training.save_every_epoch_fraction must be in (0, 1].")
        checkpoint_interval_batches = max(1, int(len(train_loader) * fraction + 0.999999))

    use_wandb = wandb is not None and _cfg_get(args.training, "wandb_mode", "offline") != "disabled"
    if use_wandb:
        try:
            wandb.init(
                project=_cfg_get(args.training, "wandb_project", "astra-unet-keypoints"),
                config=_box_to_dict(args),
                mode=_cfg_get(args.training, "wandb_mode", "offline"),
            )
        except Exception as exc:
            print(f"W&B init failed: {exc}")
            use_wandb = False

    best_val = float("inf")
    num_epochs = int(args.training.num_epochs)
    epoch_bar = tqdm(range(num_epochs), desc="Pretraining U-Net", unit="epoch")
    for epoch in epoch_bar:
        def save_partial_checkpoint(batch_idx, total_batches, running_train_logs):
            progress_epoch = epoch + batch_idx / max(total_batches, 1)
            logs = {"epoch": progress_epoch, "checkpoint_stage": "train"}
            for k, v in running_train_logs.items():
                logs[f"train/{k}"] = v
            save_checkpoint(save_to, model, optimizer, progress_epoch, logs, args, is_best=False)
            print(f"Saved checkpoint at epoch {progress_epoch:.3f}: {save_to}", flush=True)

        train_logs = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            args,
            train=True,
            checkpoint_callback=save_partial_checkpoint,
            checkpoint_interval_batches=checkpoint_interval_batches,
        )
        val_logs = run_epoch(model, val_loader, optimizer, scaler, device, args, train=False)
        logs = {"epoch": epoch}
        for k, v in train_logs.items():
            logs[f"train/{k}"] = v
        for k, v in val_logs.items():
            logs[f"val/{k}"] = v
        epoch_bar.set_postfix({"train_loss": train_logs["loss"], "val_loss": val_logs["loss"]})
        is_best = val_logs["loss"] < best_val
        if is_best:
            best_val = val_logs["loss"]
        save_checkpoint(save_to, model, optimizer, epoch, logs, args, is_best=is_best)
        if use_wandb:
            try:
                wandb.log(logs)
            except Exception as exc:
                print(f"W&B log failed: {exc}")
                use_wandb = False
    print(f"Done. Best val loss: {best_val:.6f}. Checkpoint: {save_to}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    cli = parser.parse_args()
    with open(cli.config) as f:
        main(Box(yaml.safe_load(f)))
