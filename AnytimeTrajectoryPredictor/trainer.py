import csv
import os
import random
from itertools import islice
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from tqdm import tqdm
import os

from AnytimeTrajectoryPredictor.evaluation.diversity import compute_diversity_metrics
from AnytimeTrajectoryPredictor.evaluation.latency import LatencyProfiler


def _get_training_arg(args, key, default):
    return getattr(args.training, key, default) if key in args.training else default



def _get_log_config(args, key, fallback_metric, fallback_steps):
    max_number_of_batches = None
    if key in args:
        log_config = getattr(args, key)
    elif key in args.training:
        log_config = getattr(args.training, key)
    else:
        log_config = None

    if log_config is None:
        metric = fallback_metric
        steps = fallback_steps
    else:
        metric = getattr(log_config, "metric", fallback_metric)
        steps = getattr(log_config, "steps", fallback_steps)
        max_number_of_batches = getattr(
            log_config, "max_number_of_batches", None
        )

    if metric not in {"batch", "epoch"}:
        raise ValueError(f"{key}.metric must be either 'batch' or 'epoch'")

    if max_number_of_batches is not None:
        max_number_of_batches = int(max_number_of_batches)
        if max_number_of_batches <= 0:
            raise ValueError(f"{key}.max_number_of_batches must be positive")

    return {
        "metric": metric,
        "steps": int(steps),
        "max_number_of_batches": max_number_of_batches,
    }


def _should_log(count, log_config):
    steps = log_config["steps"]
    return steps > 0 and count % steps == 0


def _epoch_progress(epoch, batch_idx, num_batches):
    return epoch + batch_idx / max(num_batches, 1)

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


def _as_int_list(value, default):
    if value is None:
        value = default
    if value is None:
        return []
    return [int(item) for item in value]


@contextmanager
def _preserve_torch_rng(seed=None):
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        if seed is not None:
            torch.manual_seed(int(seed))
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


class Trainer:
    def __init__(self, model, optimizer, train_loader, val_loader, device, args):
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
        self.early_stopping_patience = _cfg_get(args.training, "early_stopping_patience", None)
        if self.early_stopping_patience is not None:
            self.early_stopping_patience = max(int(self.early_stopping_patience), 1)
        self.early_stopping_min_delta = float(_cfg_get(args.training, "early_stopping_min_delta", 0.0))
        self.early_stopping_monitor = _cfg_get(args.training, "early_stopping_monitor", "val/loss")
        self.diagnostics_enabled = bool(_cfg_get(args.training, "diagnostics_enabled", True))
        self.diagnostics_interval = max(int(_cfg_get(args.training, "diagnostics_interval", 1)), 1)
        self.diagnostics_batches = max(int(_cfg_get(args.training, "diagnostics_batches", 1)), 0)
        default_steps = getattr(model, "monotonicity_eval_steps", [1, 2, 4, 8, 16])
        self.monotonicity_eval_steps = _as_int_list(
            _cfg_get(args.training, "monotonicity_eval_steps", default_steps),
            default_steps,
        )
        self.monotonicity_repeats = max(int(_cfg_get(args.training, "monotonicity_repeats", 1)), 1)
        self.monotonicity_seed = _cfg_get(args.training, "monotonicity_seed", None)
        self.mode_diagnostics_sampling_steps = _cfg_get(args.training, "mode_diagnostics_sampling_steps", None)
        self.mode_collapse_prob_threshold = float(_cfg_get(args.training, "mode_collapse_prob_threshold", 0.90))
        self.mode_collapse_diversity_threshold = float(
            _cfg_get(args.training, "mode_collapse_diversity_threshold", 0.05)
        )
        self.visualization_enabled = bool(_cfg_get(args.training, "visualization_enabled", False))
        self.visualization_interval = max(int(_cfg_get(args.training, "visualization_interval", 1)), 1)
        self.visualization_num_samples = max(int(_cfg_get(args.training, "visualization_num_samples", 4)), 0)
        self.visualization_indices = _cfg_get(args.training, "visualization_indices", None)
        self.visualization_random = bool(_cfg_get(args.training, "visualization_random", False))
        self.visualization_seed = _cfg_get(args.training, "visualization_seed", None)
        if self.visualization_seed is not None:
            self.visualization_seed = int(self.visualization_seed)
        self.visualization_num_steps = _cfg_get(args.training, "visualization_num_steps", None)
        self.visualization_frame_ms = int(_cfg_get(args.training, "visualization_frame_ms", 450))
        self.visualization_gt_only_frames = max(int(_cfg_get(args.training, "visualization_gt_only_frames", 2)), 0)
        self.visualization_mode_selection = _cfg_get(args.training, "visualization_mode_selection", "best")
        self.visualization_gmm_heatmap = _cfg_get(args.training, "visualization_gmm_heatmap", "all")
        self.visualization_heatmap_alpha = int(_cfg_get(args.training, "visualization_heatmap_alpha", 105))
        self.visualization_save_gmm_png = bool(_cfg_get(args.training, "visualization_save_gmm_png", True))
        self.visualization_output_dir = _cfg_get(
            args.training,
            "visualization_output_dir",
            "visualizations/image_plane_training",
        )
        wandb_mode = _cfg_get(args.training, "wandb_mode", "offline")
        self.use_wandb = wandb is not None and wandb_mode != "disabled"
        if self.use_wandb:
            try:
                wandb.init(
                    project=_cfg_get(args.training, "wandb_project", "astra_edm_diffusion"),
                    config=_box_to_dict(args),
                    mode=wandb_mode,
                )
            except Exception as exc:
                self.use_wandb = False
                print(f"W&B initialization failed; continuing without W&B logging. Error: {exc}")

    def _move_batch(self, value):
        if torch.is_tensor(value):
            return value.to(self.device, non_blocking=True)
        if isinstance(value, dict):
            return {key: self._move_batch(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self._move_batch(item) for item in value)
        return value

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
            metric_key = f"{prefix}/loss" if key == "loss" else f"{prefix}/{key}"
            if torch.is_tensor(value):
                logs[metric_key] = float(value.detach().mean().cpu())
            elif isinstance(value, (int, float)):
                logs[metric_key] = float(value)
            if key == "loss" and metric_key in logs:
                logs[f"{prefix}/total_loss"] = logs[metric_key]
        return logs

    def _monotonicity_scalar_logs(self, metrics, prefix):
        logs = {
            f"{prefix}/monotonicity/repeats": float(metrics["repeats"]),
            f"{prefix}/monotonicity/minade_violations": float(metrics["monotonic_minade_violations"]),
            f"{prefix}/monotonicity/minfde_violations": float(metrics["monotonic_minfde_violations"]),
            f"{prefix}/monotonicity/best_minade_step": float(metrics["best_minade_step"]),
            f"{prefix}/monotonicity/best_minfde_step": float(metrics["best_minfde_step"]),
        }
        for index, step in enumerate(metrics["steps"]):
            tag = f"step_{int(step)}"
            logs[f"{prefix}/monotonicity/nfe_{tag}"] = float(metrics["nfe"][index])
            logs[f"{prefix}/monotonicity/minade_mean_{tag}"] = float(metrics["minade_mean"][index])
            logs[f"{prefix}/monotonicity/minfde_mean_{tag}"] = float(metrics["minfde_mean"][index])
            logs[f"{prefix}/monotonicity/nll_mean_{tag}"] = float(metrics["nll_mean"][index])
        return logs

    def _mode_diversity_logs(self, params, batch, prefix):
        mu = params.mu.detach()
        probs = params.mode_probs.detach()
        logs = {}
        if mu.ndim != 5 or probs.ndim != 2:
            return logs

        batch_size, modes, agents, horizon, _ = mu.shape
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1)
        logs[f"{prefix}/mode_diversity/prob_entropy_mean"] = float(entropy.mean().cpu())
        logs[f"{prefix}/mode_diversity/effective_modes_mean"] = float(torch.exp(entropy).mean().cpu())
        logs[f"{prefix}/mode_diversity/max_mode_prob_mean"] = float(probs.max(dim=1).values.mean().cpu())
        if modes > 1:
            logs[f"{prefix}/mode_diversity/prob_entropy_normalized_mean"] = float(
                (entropy / torch.log(torch.tensor(float(modes), device=entropy.device))).mean().cpu()
            )
        else:
            logs[f"{prefix}/mode_diversity/prob_entropy_normalized_mean"] = 0.0

        max_prob = probs.max(dim=1).values
        logs[f"{prefix}/mode_diversity/prob_collapse_rate"] = float(
            (max_prob >= self.mode_collapse_prob_threshold).float().mean().cpu()
        )
        if modes < 2:
            return logs

        upper = torch.triu_indices(modes, modes, offset=1, device=mu.device)
        pair_dist = torch.linalg.norm(mu[:, upper[0]] - mu[:, upper[1]], dim=-1)
        mask = batch.get("future_mask")
        if mask is None:
            mask = torch.ones(batch_size, agents, horizon, device=mu.device, dtype=mu.dtype)
        else:
            mask = mask.to(device=mu.device, dtype=mu.dtype)
        pair_ade = (pair_dist * mask[:, None]).sum(dim=(2, 3)) / mask[:, None].sum(dim=(2, 3)).clamp_min(1.0)
        min_pair_ade = pair_ade.min(dim=1).values
        logs[f"{prefix}/mode_diversity/pairwise_ade_mean"] = float(pair_ade.mean().cpu())
        logs[f"{prefix}/mode_diversity/pairwise_ade_min_mean"] = float(min_pair_ade.mean().cpu())
        logs[f"{prefix}/mode_diversity/low_diversity_rate"] = float(
            (min_pair_ade <= self.mode_collapse_diversity_threshold).float().mean().cpu()
        )
        return logs

    def _validation_diagnostics(self, epoch):
        if (
            not self.diagnostics_enabled
            or self.diagnostics_batches <= 0
            or epoch % self.diagnostics_interval != 0
            or not getattr(self.model, "expects_batch", False)
        ):
            return {}

        was_training = self.model.training
        self.model.eval()
        totals = {}
        count = 0
        with torch.no_grad(), _preserve_torch_rng(self.monotonicity_seed):
            for batch in self.val_loader:
                batch = self._move_batch(batch)
                if hasattr(self.model, "evaluate_monotonicity") and self.monotonicity_eval_steps:
                    metrics = self.model.evaluate_monotonicity(
                        batch,
                        step_counts=self.monotonicity_eval_steps,
                        repeats=self.monotonicity_repeats,
                        seed=self.monotonicity_seed,
                    )
                    for key, value in self._monotonicity_scalar_logs(metrics, "val").items():
                        totals[key] = totals.get(key, 0.0) + value
                params = self.model(batch, num_sampling_steps=self.mode_diagnostics_sampling_steps)
                for key, value in self._mode_diversity_logs(params, batch, "val").items():
                    totals[key] = totals.get(key, 0.0) + value
                count += 1
                if count >= self.diagnostics_batches:
                    break
        if was_training:
            self.model.train()
        return {key: value / max(count, 1) for key, value in totals.items()}

    def _parse_visualization_indices(self, dataset_len):
        value = self.visualization_indices
        if value is None:
            if self.visualization_random:
                rng = random.Random(self.visualization_seed)
                return rng.sample(range(dataset_len), k=min(self.visualization_num_samples, dataset_len))
            return list(range(min(self.visualization_num_samples, dataset_len)))
        if isinstance(value, str):
            raw_parts = [part.strip() for part in value.split(",") if part.strip()]
        else:
            raw_parts = [str(part).strip() for part in value]
        indices = []
        for part in raw_parts:
            if "-" in part:
                lo, hi = part.split("-", 1)
                indices.extend(range(int(lo), int(hi) + 1))
            else:
                indices.append(int(part))
        return [idx for idx in indices if 0 <= idx < dataset_len]

    def _render_epoch_visualizations(self, epoch):
        if not self.visualization_enabled:
            return []
        if self.visualization_num_samples <= 0:
            return []
        if epoch % self.visualization_interval != 0:
            return []
        if not getattr(self.model, "use_rgb_context", False):
            return []

        dataset = getattr(self.val_loader, "dataset", None)
        if dataset is None:
            print("Skipping visualization: validation loader has no dataset attribute.")
            return []
        indices = self._parse_visualization_indices(len(dataset))
        if not indices:
            print("Skipping visualization: no valid validation sample indices selected.")
            return []

        visualizations_dir = Path(__file__).resolve().parents[1] / "visualizations"
        if str(visualizations_dir) not in sys.path:
            sys.path.insert(0, str(visualizations_dir))
        try:
            from visualize_image_plane_batch import render_one
        except Exception as exc:
            print(f"Skipping visualization: failed to import renderer. Error: {exc}")
            return []

        was_training = self.model.training
        self.model.eval()
        output_dir = Path(self.visualization_output_dir) / f"epoch_{epoch:04d}"
        rows = []
        try:
            for idx in indices:
                gif_path = output_dir / f"sample_{idx:06d}.gif"
                row = render_one(
                    self.model,
                    dataset,
                    idx,
                    self.device,
                    gif_path,
                    num_steps=self.visualization_num_steps,
                    frame_ms=self.visualization_frame_ms,
                    gt_only_frames=self.visualization_gt_only_frames,
                    mode_selection=self.visualization_mode_selection,
                    gmm_heatmap=self.visualization_gmm_heatmap,
                    heatmap_alpha=self.visualization_heatmap_alpha,
                    save_gmm_png=self.visualization_save_gmm_png,
                )
                row["gif"] = str(gif_path)
                rows.append(row)
            if rows:
                output_dir.mkdir(parents=True, exist_ok=True)
                summary_path = output_dir / "summary.csv"
                with open(summary_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"Wrote epoch visualizations to {output_dir}")
        finally:
            if was_training:
                self.model.train()
        return rows

    def train_single_epoch(self):
        self.model.train()
        totals = {}
        for batch in self.train_loader:
            batch = self._move_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                output = self._compute_loss(batch)
                loss = output["loss"]
            self.scaler.scale(loss).backward()
            if self.gradient_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.gradient_clip_norm))
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
            try:
                wandb.log(logs)
            except Exception as exc:
                self.use_wandb = False
                print(f"W&B logging failed; disabling W&B for this run. Error: {exc}")
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
        epochs_without_improvement = 0
        for epoch in pbar:
            train_logs = self.train_single_epoch()
            val_logs = self.validate()
            diagnostic_logs = self._validation_diagnostics(epoch)
            visualization_rows = self._render_epoch_visualizations(epoch)
            visualization_logs = {"visualization/num_samples": float(len(visualization_rows))} if visualization_rows else {}
            epoch_logs = {"epoch": epoch, **train_logs, **val_logs, **diagnostic_logs, **visualization_logs}
            validation_score = epoch_logs.get(self.early_stopping_monitor, epoch_logs.get("val/loss", float("inf")))
            is_best = validation_score < self.best_validation - self.early_stopping_min_delta
            if is_best:
                self.best_validation = validation_score
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            self.save_checkpoint(epoch, epoch_logs, is_best=is_best)
            pbar.set_postfix({"train_loss": train_logs.get("train/loss"), "val_loss": validation_score})
            if self.use_wandb:
                try:
                    wandb.log(epoch_logs)
                except Exception as exc:
                    self.use_wandb = False
                    print(f"W&B logging failed; disabling W&B for this run. Error: {exc}")
            if self.early_stopping_patience is not None and epochs_without_improvement >= self.early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch}: {self.early_stopping_monitor} "
                    f"did not improve by {self.early_stopping_min_delta} for "
                    f"{self.early_stopping_patience} epochs. Best={self.best_validation:.6f}."
                )
                break
