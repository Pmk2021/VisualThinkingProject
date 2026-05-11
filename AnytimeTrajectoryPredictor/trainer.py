import torch
import wandb
import random
from tqdm import tqdm

from AnytimeTrajectoryPredictor.evaluation import (
    LatencyProfiler,
    compute_diversity_metrics,
)


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
        min_steps = getattr(self.args.model, "min_refinement_steps", 1)
        max_steps = getattr(self.args.model, "max_refinement_steps", 10)
        return [random.randint(min_steps, max_steps) for _ in range(num_frames)]

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

    def _diversity_enabled(self):
        eval_cfg = getattr(self.args, "evaluation", None)
        if eval_cfg is None:
            return False
        return bool(eval_cfg.get("measure_diversity", False))

    def validate(self):
        total_loss = 0
        apd_sum = 0.0
        w2_sum = 0.0
        measure_diversity = self._diversity_enabled()
        num_modes = self.args.model.num_trajectory_possibilities
        with torch.no_grad():
            for batch in self.val_loader:
                feature, trajectory = self._prepare_batch(batch)
                f_ = self._sample_refinement_steps(len(feature))
                # Compute Loss
                loss = self.model.compute_loss(feature, trajectory, f_)
                total_loss += loss.item()
                if measure_diversity:
                    predictions = self.model(feature, f_)
                    metrics = compute_diversity_metrics(predictions, num_modes)
                    apd_sum += metrics["apd"]
                    w2_sum += metrics["mean_pairwise_w2"]

        n_batches = len(self.val_loader)
        validation_loss = total_loss / n_batches
        diversity = (
            {
                "apd": apd_sum / n_batches,
                "mean_pairwise_w2": w2_sum / n_batches,
            }
            if measure_diversity
            else None
        )
        return validation_loss, diversity

    def train(self, num_epochs):
        pbar = tqdm(range(num_epochs), desc="Training")

        for epoch in pbar:
            loss = self.train_single_epoch()
            validation_loss, diversity = self.validate()
            learning_rate = self.optimizer.param_groups[0]["lr"]

            postfix = {
                "loss": loss,
                "val_loss": validation_loss,
                "lr": learning_rate,
            }
            log_payload = {
                "epoch": epoch,
                "loss": loss,
                "validation_loss": validation_loss,
                "learning_rate": learning_rate,
            }
            if diversity is not None:
                postfix["apd"] = diversity["apd"]
                postfix["w2"] = diversity["mean_pairwise_w2"]
                log_payload["diversity/apd"] = diversity["apd"]
                log_payload["diversity/mean_pairwise_w2"] = diversity["mean_pairwise_w2"]

            pbar.set_postfix(postfix)
            wandb.log(log_payload)

        self.profile_latency()

    # ------------------------------------------------------------------ #
    # Latency profiling
    # ------------------------------------------------------------------ #

    def _get_latency_sample_features(self):
        """Return a single batch of features for latency measurement.

        Uses the validation loader so the input shape matches inference
        conditions. Cached after the first call to keep timings
        reproducible across invocations.
        """
        if getattr(self, "_latency_sample", None) is not None:
            return self._latency_sample
        batch = next(iter(self.val_loader))
        feature, _ = self._prepare_batch(batch)
        self._latency_sample = feature
        return feature

    def profile_latency(self):
        """Measure inference latency once and log to wandb.

        Logs:
            * ``wandb.summary["latency/pass_*"]``: single refinement-pass stats.
            * ``wandb.summary["latency/full_k{K}_*"]``: full-forward stats for
              each ``k`` in the anytime sweep.
            * ``wandb.log``: a line plot of median full-forward latency vs.
              number of refinement steps.

        Controlled by the ``evaluation`` config section. Skipped silently
        if ``evaluation.measure_latency`` is false or unset.
        """
        eval_cfg = getattr(self.args, "evaluation", None)
        if eval_cfg is None or not eval_cfg.get("measure_latency", False):
            return None

        try:
            features = self._get_latency_sample_features()
        except StopIteration:
            print("[latency] val_loader is empty; skipping latency profiling.")
            return None

        n_warmup = int(eval_cfg.get("latency_n_warmup", 10))
        n_runs = int(eval_cfg.get("latency_n_runs", 100))
        batch_size = eval_cfg.get("latency_batch_size", 1)
        max_steps = int(eval_cfg.get("latency_anytime_max_steps", 10))

        # Slice features for reproducible timing input.
        features = features.to(self.device)
        if batch_size is not None and features.shape[1] > batch_size:
            features = features[:, :batch_size].contiguous()
        num_frames = features.shape[0]

        profiler = LatencyProfiler(
            device=self.device, n_warmup=n_warmup, n_runs=n_runs
        )

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                # Single refinement-pass latency (1 frame, 1 step).
                single_frame = features[:1].contiguous()
                pass_stats = profiler.measure(
                    lambda: self.model(single_frame, f_=[1])
                )

                # Anytime curve: full-forward latency for k = 1..max_steps.
                curve = []  # list of (k, median_ms, mean_ms, p95_ms)
                for k in range(1, max_steps + 1):
                    schedule = [k] * num_frames
                    stats = profiler.measure(
                        lambda s=schedule: self.model(features, f_=s)
                    )
                    curve.append(
                        (
                            k,
                            stats["median_ms"],
                            stats["mean_ms"],
                            stats["p95_ms"],
                        )
                    )
        finally:
            if was_training:
                self.model.train()

        # ---- wandb.summary: scalar metrics for cross-run comparison ----
        summary = {}
        for key, value in pass_stats.items():
            summary[f"latency/pass_{key}"] = value
        for k, median_ms, mean_ms, p95_ms in curve:
            summary[f"latency/full_k{k}_median_ms"] = median_ms
            summary[f"latency/full_k{k}_mean_ms"] = mean_ms
            summary[f"latency/full_k{k}_p95_ms"] = p95_ms
        summary["latency/meta/num_frames"] = int(num_frames)
        summary["latency/meta/batch_size"] = int(features.shape[1])
        summary["latency/meta/num_objects"] = int(features.shape[2])
        summary["latency/meta/state_dim"] = int(features.shape[3])
        summary["latency/meta/max_refinement_steps"] = int(max_steps)
        summary["latency/meta/device"] = self.device.type
        wandb.summary.update(summary)

        # ---- wandb.log: anytime-curve line plot ----
        table = wandb.Table(
            columns=["refinement_steps", "median_ms", "mean_ms", "p95_ms"],
            data=[list(row) for row in curve],
        )
        wandb.log(
            {
                "latency/anytime_curve": wandb.plot.line(
                    table,
                    x="refinement_steps",
                    y="median_ms",
                    title="Full-forward latency vs. refinement steps",
                ),
            }
        )

        print(
            "[latency] pass median = {:.3f} ms | full@k=1 = {:.3f} ms | "
            "full@k={} = {:.3f} ms (device={}, batch={})".format(
                pass_stats["median_ms"],
                curve[0][1],
                max_steps,
                curve[-1][1],
                self.device.type,
                features.shape[1],
            )
        )

        return summary
