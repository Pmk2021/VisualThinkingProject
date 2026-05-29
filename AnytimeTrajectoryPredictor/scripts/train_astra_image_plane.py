#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

import torch
import yaml
from box import Box
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASTRA_ROOT = PROJECT_ROOT / "ASTRA"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ASTRA_ROOT))

try:
    import icecream  # noqa: F401
except ImportError:
    class _NoOpIceCream:
        def __call__(self, *args, **kwargs):
            return args[0] if len(args) == 1 else args

        def disable(self):
            return None

    sys.modules["icecream"] = type(sys)("icecream")
    sys.modules["icecream"].ic = _NoOpIceCream()

try:
    from sklearn.utils.extmath import cartesian as _sklearn_cartesian  # noqa: F401
except ImportError:
    import itertools
    import importlib.machinery
    import types
    import numpy as np

    sklearn_mod = types.ModuleType("sklearn")
    sklearn_utils_mod = types.ModuleType("sklearn.utils")
    sklearn_extmath_mod = types.ModuleType("sklearn.utils.extmath")
    sklearn_mod.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None)
    sklearn_utils_mod.__spec__ = importlib.machinery.ModuleSpec("sklearn.utils", loader=None)
    sklearn_extmath_mod.__spec__ = importlib.machinery.ModuleSpec("sklearn.utils.extmath", loader=None)

    def _cartesian(arrays):
        return np.asarray(list(itertools.product(*arrays)))

    sklearn_extmath_mod.cartesian = _cartesian
    sklearn_utils_mod.extmath = sklearn_extmath_mod
    sklearn_mod.utils = sklearn_utils_mod
    sys.modules["sklearn"] = sklearn_mod
    sys.modules["sklearn.utils"] = sklearn_utils_mod
    sys.modules["sklearn.utils.extmath"] = sklearn_extmath_mod

try:
    from einops import rearrange as _einops_rearrange  # noqa: F401
except ImportError:
    import types
    import importlib.machinery

    einops_mod = types.ModuleType("einops")
    einops_mod.__spec__ = importlib.machinery.ModuleSpec("einops", loader=None)

    def _rearrange(tensor, pattern):
        if pattern == "b a k l -> (b a k) l":
            return tensor.reshape(tensor.shape[0] * tensor.shape[1] * tensor.shape[2], tensor.shape[3])
        if pattern == "b a k f c -> (b a) k f c":
            return tensor.reshape(tensor.shape[0] * tensor.shape[1], tensor.shape[2], tensor.shape[3], tensor.shape[4])
        raise NotImplementedError(f"Fallback einops.rearrange does not support pattern: {pattern}")

    einops_mod.rearrange = _rearrange
    sys.modules["einops"] = einops_mod

from AnytimeTrajectoryPredictor.Data.feature_extractor import WaymoImagePlaneDataset
from models import astra_model as astra_model_module
from models.astra_model import ASTRA_model
from models.keypoint_model import UNETEmbeddingExtractor
from utils.losses import Loss as ASTRALoss
from utils.losses import DiversityLoss, GaussianKLDivergenceLoss, KLDivergenceLoss


def _cfg_get(config, key, default=None):
    return getattr(config, key, default) if config is not None else default


def make_astra_cfg(args, device, history_steps, future_steps):
    astra_cfg_path = Path(_cfg_get(args.astra, "config", ASTRA_ROOT / "configs" / "pie.yaml"))
    with open(astra_cfg_path) as f:
        cfg = Box(yaml.safe_load(f))

    cfg.DATASET = _cfg_get(args.astra, "dataset_name", "ETH_UCY")
    cfg.SUBSET = _cfg_get(args.astra, "subset", "waymo_image_plane")
    cfg.DATA.FREQUENCY = 1
    cfg.PREDICTION.OBS_TIME = int(history_steps)
    cfg.PREDICTION.PRED_TIME = int(future_steps)
    cfg.TRAIN.BATCH_SIZE = int(args.training.batch_size)
    cfg.TRAIN.LR = float(args.training.learning_rate)
    cfg.MODEL.USE_PRETRAINED_UNET = bool(_cfg_get(args.astra, "use_pretrained_unet", True))
    cfg.MODEL.USE_SOCIAL = bool(_cfg_get(args.astra, "use_social", False))
    cfg.MODEL.USE_VAE = bool(_cfg_get(args.astra, "use_vae", False))
    cfg.MODEL.INC_VELO = bool(_cfg_get(args.astra, "include_velocity", False))
    cfg.MODEL.TRANS_MASK = bool(_cfg_get(args.astra, "transformer_mask", True))
    cfg.MODEL.FEATURE_EXTRACTOR = _cfg_get(args.astra, "feature_extractor", _cfg_get(cfg.MODEL, "FEATURE_EXTRACTOR", "resnet18"))
    cfg.MODEL.FEATURE_DIM = int(_cfg_get(args.astra, "feature_dim", _cfg_get(cfg.MODEL, "FEATURE_DIM", 512)))
    cfg.MODEL.UNET_DIM = int(_cfg_get(args.astra, "unet_dim", _cfg_get(cfg.MODEL, "UNET_DIM", 16)))
    cfg.UNET_MODE = "testing"
    loss_name = str(_cfg_get(args.training, "loss", _cfg_get(cfg.LOSS, "FUN", "SmoothL1"))).lower()
    cfg.LOSS.FUN = {"smooth_l1": "SmoothL1", "smoothl1": "SmoothL1", "mse": "MSE", "rmse": "RMSE"}.get(loss_name, loss_name)
    if hasattr(args.training, "weighted_penalty"):
        cfg.LOSS.WEIGHTED_PENALTY = args.training.weighted_penalty
    cfg.LOSS.KL_WEIGHT = float(_cfg_get(args.training, "kl_weight", _cfg_get(cfg.LOSS, "KL_WEIGHT", 0.0)))
    cfg.LOSS.GAUSSIAN_KL_WEIGHT = float(_cfg_get(args.training, "gaussian_kl_weight", _cfg_get(cfg.LOSS, "GAUSSIAN_KL_WEIGHT", 0.0)))
    cfg.LOSS.DIVERSITY_WEIGHT = float(_cfg_get(args.training, "diversity_weight", _cfg_get(cfg.LOSS, "DIVERSITY_WEIGHT", 0.0)))
    cfg.LOSS.CVAE_RECON_WEIGHT = float(_cfg_get(args.training, "cvae_recon_weight", _cfg_get(cfg.LOSS, "CVAE_RECON_WEIGHT", 0.0)))
    cfg.device = device
    cfg.device_list = [device.index or 0] if device.type == "cuda" else []
    return cfg


class ASTRAImagePlaneAdapter(torch.nn.Module):
    """Batch adapter around ASTRA_model; ASTRA_model outputs are left unchanged."""

    expects_batch = True

    def __init__(self, astra_cfg, unet_weights=None, freeze_unet=True):
        super().__init__()
        astra_model_module.ic.disable()
        self.cfg = astra_cfg
        self.model = ASTRA_model(astra_cfg)
        self.trajectory_loss = ASTRALoss(astra_cfg)
        self.kl_loss = KLDivergenceLoss()
        self.gaussian_kl_loss = GaussianKLDivergenceLoss()
        self.diversity_loss = DiversityLoss()
        self.embedding_extractor = None
        self.freeze_unet = bool(freeze_unet)
        mean = torch.tensor(astra_cfg.DATA.MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(astra_cfg.DATA.STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("rgb_mean", mean, persistent=False)
        self.register_buffer("rgb_std", std, persistent=False)
        if astra_cfg.MODEL.USE_PRETRAINED_UNET:
            self.embedding_extractor = UNETEmbeddingExtractor(astra_cfg)
            if unet_weights:
                self._load_unet_weights(unet_weights)
            if self.freeze_unet:
                self.embedding_extractor.eval()
                for param in self.embedding_extractor.parameters():
                    param.requires_grad = False

    def _load_unet_weights(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
        result = self.embedding_extractor.load_state_dict(state, strict=False)
        missing = getattr(result, "missing_keys", [])
        unexpected = getattr(result, "unexpected_keys", [])
        print(f"Loaded ASTRA U-Net weights from {path} (missing={len(missing)}, unexpected={len(unexpected)})")
        if missing[:5]:
            print(f"  first missing keys: {missing[:5]}")
        if unexpected[:5]:
            print(f"  first unexpected keys: {unexpected[:5]}")

    def train(self, mode=True):
        super().train(mode)
        if self.embedding_extractor is not None and self.freeze_unet:
            self.embedding_extractor.eval()
        return self

    def _extract_unet_features(self, batch, device):
        if self.embedding_extractor is None:
            return None
        rgb = batch.get("rgb_history")
        if rgb is None:
            raise KeyError("ASTRA U-Net baseline requires rgb_history in the batch.")
        batch_size, history, channels, _, _ = rgb.shape
        images = rgb.to(device=device, dtype=torch.float32).reshape(batch_size * history, channels, rgb.shape[-2], rgb.shape[-1])
        images = torch.nn.functional.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        images = (images - self.rgb_mean.to(device=images.device, dtype=images.dtype)) / self.rgb_std.to(device=images.device, dtype=images.dtype).clamp_min(1e-6)
        context = torch.no_grad() if self.freeze_unet else torch.enable_grad()
        with context:
            _, _, features = self.embedding_extractor(images)
        features = features.view(batch_size, history, -1).unsqueeze(1)  # [B H F] -> [B A H F]
        return features

    def _batch_to_astra_inputs(self, batch):
        boxes = batch.get("box_history", batch.get("features"))
        if boxes is None:
            raise KeyError("ASTRA image-plane adapter requires box_history or features.")
        # ASTRA ETH_UCY mode expects [B, A, H, 2] center coordinates.
        past_loc = boxes[..., :2].to(dtype=torch.float32)
        fut_loc = batch["trajectory"].to(dtype=torch.float32)
        unet_features = self._extract_unet_features(batch, past_loc.device)
        return past_loc, fut_loc, unet_features

    def _raw_prediction_to_modes(self, raw_output):
        # ASTRA no-VAE: [B, A, T, D]. ASTRA VAE: [B, A, K, T, D].
        pred = raw_output[..., :2]
        if pred.dim() == 4:
            pred = pred[:, None]
        elif pred.dim() == 5:
            pred = pred.permute(0, 2, 1, 3, 4)
        else:
            raise ValueError(f"Unexpected ASTRA output shape: {tuple(raw_output.shape)}")
        return pred

    def forward(self, batch, mode="train"):
        past_loc, fut_loc, unet_features = self._batch_to_astra_inputs(batch)
        return self.model(past_loc, fut_loc, unet_features, mode=mode)

    def predict_modes(self, batch):
        _, _, model_output, _, _, _ = self.forward(batch, mode="test")
        return self._raw_prediction_to_modes(model_output)

    def _masked_astra_trajectory_loss(self, pred_modes, target, mask):
        # ASTRA loss expects [B A K T 2]; adapter metrics use [B K A T 2].
        pred = pred_modes.permute(0, 2, 1, 3, 4)  # [B K A T 2] -> [B A K T 2]
        target = target.unsqueeze(-3)  # [B A T 2] -> [B A 1 T 2]
        loss = self.trajectory_loss.criterion(pred, target)  # [B A K T 2]
        loss = loss[..., 0] + loss[..., 1]  # [B A K T]
        if self.trajectory_loss.loss_func == "RMSE":
            loss = torch.sqrt(loss + 1e-6)
        if self.cfg.LOSS.WEIGHTED_PENALTY:
            loss = loss * self.trajectory_loss.weights.view(1, 1, 1, -1)

        mask_modes = mask[:, :, None, :]  # [B A T] -> [B A 1 T]
        loss_per_mode = (loss * mask_modes).sum(dim=-1) / mask_modes.sum(dim=-1).clamp_min(1.0)  # [B A K]
        loss_per_agent = loss_per_mode.min(dim=2).values if loss_per_mode.shape[2] > 1 else loss_per_mode.squeeze(2)
        valid = (mask.sum(dim=-1) > 0).to(loss.dtype)  # [B A]
        return (loss_per_agent * valid).sum() / valid.sum().clamp_min(1.0)

    def compute_loss(self, batch):
        mean, log_var, model_output, mean_c, log_var_c, c_model_output = self.forward(batch, mode="train")
        pred_modes = self._raw_prediction_to_modes(model_output)
        target = batch["trajectory"].to(device=pred_modes.device, dtype=pred_modes.dtype)
        mask = batch.get("future_mask")
        if mask is None:
            mask = torch.ones(target.shape[:-1], device=target.device, dtype=target.dtype)
        else:
            mask = mask.to(device=target.device, dtype=target.dtype)

        loss_traj = self._masked_astra_trajectory_loss(pred_modes, target, mask)
        loss = loss_traj

        loss_cvae_recon = pred_modes.new_tensor(0.0)
        if c_model_output is not None:
            c_pred_modes = self._raw_prediction_to_modes(c_model_output)
            loss_cvae_recon = self._masked_astra_trajectory_loss(c_pred_modes, target, mask)
            loss = loss + float(_cfg_get(self.cfg.LOSS, "CVAE_RECON_WEIGHT", 0.0)) * loss_cvae_recon

        loss_kl = pred_modes.new_tensor(0.0)
        if mean is not None and log_var is not None:
            loss_kl = self.kl_loss(mean, log_var) / target.numel()
            loss = loss + float(_cfg_get(self.cfg.LOSS, "KL_WEIGHT", 0.0)) * loss_kl

        loss_gaussian_kl = pred_modes.new_tensor(0.0)
        if mean is not None and log_var is not None and mean_c is not None and log_var_c is not None:
            loss_gaussian_kl = self.gaussian_kl_loss(mean, log_var, mean_c, log_var_c)
            loss = loss + float(_cfg_get(self.cfg.LOSS, "GAUSSIAN_KL_WEIGHT", 0.0)) * loss_gaussian_kl

        loss_diversity = pred_modes.new_tensor(0.0)
        diversity_weight = float(_cfg_get(self.cfg.LOSS, "DIVERSITY_WEIGHT", 0.0))
        if diversity_weight > 0 and pred_modes.shape[1] > 1:
            loss_diversity = self.diversity_loss(pred_modes.permute(0, 2, 1, 3, 4), pred_modes.device)
            loss = loss + diversity_weight * loss_diversity

        metrics = trajectory_metrics(pred_modes, target, mask)
        metrics = {key: value.detach() for key, value in metrics.items()}
        metrics["loss_traj"] = loss_traj.detach()
        metrics["loss_cvae_recon"] = loss_cvae_recon.detach()
        metrics["loss_kl"] = loss_kl.detach()
        metrics["loss_gaussian_kl"] = loss_gaussian_kl.detach()
        metrics["loss_diversity"] = loss_diversity.detach()
        metrics["loss"] = loss
        return metrics


def trajectory_metrics(pred_modes, target, mask):
    # pred_modes: [B K A T 2], target: [B A T 2], mask: [B A T]
    distances = torch.linalg.norm(pred_modes - target[:, None], dim=-1)  # [B K A T]
    mask_modes = mask[:, None]  # [B 1 A T]
    valid = mask.sum(dim=-1) > 0  # [B A]
    valid_f = valid.to(distances.dtype)

    ade = (distances * mask_modes).sum(dim=-1) / mask_modes.sum(dim=-1).clamp_min(1.0)  # [B K A]
    minade_per_agent = ade.min(dim=1).values  # [B A]
    maxade_per_agent = ade.max(dim=1).values  # [B A]

    valid_counts = mask.long().sum(dim=-1).clamp_min(1)  # [B A]
    last_idx = (valid_counts - 1).view(mask.shape[0], 1, mask.shape[1], 1).expand(-1, pred_modes.shape[1], -1, 1)
    fde = distances.gather(dim=3, index=last_idx).squeeze(-1)  # [B K A]
    minfde_per_agent = fde.min(dim=1).values  # [B A]

    denom = valid_f.sum().clamp_min(1.0)
    return {
        "minADE": (minade_per_agent * valid_f).sum() / denom,
        "maxADE": (maxade_per_agent * valid_f).sum() / denom,
        "minFDE": (minfde_per_agent * valid_f).sum() / denom,
    }


def move_batch(batch, device):
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def make_dataloaders(args):
    train_dataset = WaymoImagePlaneDataset(args.feature_extractor, split="train")
    val_dataset = WaymoImagePlaneDataset(args.feature_extractor, split="val")
    num_workers = int(_cfg_get(args.training, "num_workers", 0))
    pin_memory = bool(_cfg_get(args.training, "pin_memory", False))
    loader_kwargs = {
        "batch_size": int(args.training.batch_size),
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": bool(_cfg_get(args.training, "persistent_workers", False)) and num_workers > 0,
        "drop_last": bool(_cfg_get(args.training, "drop_last", True)),
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(_cfg_get(args.training, "prefetch_factor", 2))
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    print(f"train: {len(train_dataset)} samples; val: {len(val_dataset)} samples")
    return train_loader, val_loader, train_dataset


def run_epoch(model, loader, device, optimizer=None, max_batches=None):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "minADE": 0.0, "maxADE": 0.0, "minFDE": 0.0}
    n = 0
    latency_total = 0.0
    latency_count = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        batch = move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
            output = model.compute_loss(batch)
            output["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(getattr(model, "gradient_clip_norm", 1.0)))
            optimizer.step()
        else:
            with torch.no_grad():
                output = model.compute_loss(batch)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model.predict_modes(batch)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                latency_total += time.perf_counter() - start
                latency_count += batch["trajectory"].shape[0]

        batch_size = batch["trajectory"].shape[0]
        for key in totals:
            totals[key] += float(output[key].detach().cpu()) * batch_size
        n += batch_size

    denom = max(n, 1)
    logs = {key: value / denom for key, value in totals.items()}
    logs["latency_ms_per_sample"] = 1000.0 * latency_total / max(latency_count, 1)
    return logs


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, train_dataset = make_dataloaders(args)
    astra_cfg = make_astra_cfg(args, device, train_dataset.history_steps, train_dataset.future_steps)
    model = ASTRAImagePlaneAdapter(
        astra_cfg,
        unet_weights=_cfg_get(args.astra, "unet_weights", "checkpoints/unet_keypoints_waymo_latest.pth"),
        freeze_unet=bool(_cfg_get(args.astra, "freeze_unet", True)),
    ).to(device)
    model.gradient_clip_norm = float(_cfg_get(args.training, "gradient_clip_norm", 1.0))

    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=float(args.training.learning_rate),
        weight_decay=float(_cfg_get(args.training, "weight_decay", 0.0)),
    )
    save_to = Path(_cfg_get(args.training, "save_to", "checkpoints/astra_image_plane_latest.pth"))
    save_to.parent.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_val = float("inf")
    checkpoint_path = _cfg_get(args.training, "from_checkpoint", None)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val = float(checkpoint.get("best_val", best_val))
        print(f"Resumed ASTRA baseline from {checkpoint_path} at epoch {start_epoch}")

    patience = int(_cfg_get(args.training, "early_stopping_patience", 0) or 0)
    min_delta = float(_cfg_get(args.training, "early_stopping_min_delta", 0.0))
    bad_epochs = 0

    for epoch in range(start_epoch, int(args.training.num_epochs) + 1):
        train_logs = run_epoch(model, train_loader, device, optimizer=optimizer, max_batches=_cfg_get(args.training, "max_train_batches", None))
        val_logs = run_epoch(model, val_loader, device, optimizer=None, max_batches=_cfg_get(args.training, "max_val_batches", None))
        print(
            f"epoch {epoch:03d} "
            f"train/loss={train_logs['loss']:.6f} train/minADE={train_logs['minADE']:.6f} "
            f"val/loss={val_logs['loss']:.6f} val/minADE={val_logs['minADE']:.6f} "
            f"val/maxADE={val_logs['maxADE']:.6f} val/minFDE={val_logs['minFDE']:.6f} "
            f"latency_ms={val_logs['latency_ms_per_sample']:.3f}"
        )
        improved = val_logs["minADE"] < best_val - min_delta
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "train_logs": train_logs,
            "val_logs": val_logs,
        }
        torch.save(checkpoint, save_to.with_suffix(".last.pth"))
        if improved:
            best_val = val_logs["minADE"]
            checkpoint["best_val"] = best_val
            torch.save(checkpoint, save_to)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if patience > 0 and bad_epochs >= patience:
                print(f"early stopping at epoch {epoch}: best val/minADE={best_val:.6f}")
                break

    print(f"best val/minADE={best_val:.6f}; saved to {save_to}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cli = parser.parse_args()
    with open(cli.config) as f:
        main(Box(yaml.safe_load(f)))
