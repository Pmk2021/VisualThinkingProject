import torch
import wandb
import random
from tqdm import tqdm

from AnytimeTrajectoryPredictor.evaluation.diversity import compute_diversity_metrics
from AnytimeTrajectoryPredictor.evaluation.latency import LatencyProfiler


def _get_training_arg(args, key, default):
    return getattr(args.training, key, default) if key in args.training else default


def _get_log_config(args, key, fallback_metric, fallback_steps):
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

    if metric not in {"batch", "epoch"}:
        raise ValueError(f"{key}.metric must be either 'batch' or 'epoch'")

    return {
        "metric": metric,
        "steps": int(steps),
    }


def _should_log(count, log_config):
    steps = log_config["steps"]
    return steps > 0 and count % steps == 0


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
        self.measure_diversity = _get_training_arg(args, "measure_diversity", False)
        self.measure_latency = _get_training_arg(args, "measure_latency", False)
        self.training_log = _get_log_config(
            args,
            key="training_log",
            fallback_metric="batch",
            fallback_steps=_get_training_arg(args, "batch_logging_steps", 1),
        )
        self.validation_log = _get_log_config(
            args,
            key="validation_log",
            fallback_metric="epoch",
            fallback_steps=_get_training_arg(args, "logging_steps", 1),
        )
        self.global_step = 0
        self.validation_iterator = iter(self.val_loader)
        wandb.init(project=args.training.wandb_project, config=args)
        wandb.define_metric("global_step")
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("validation/*", step_metric="global_step")
        wandb.define_metric("diversity/*", step_metric="global_step")

    def train_single_epoch(self, epoch):
        self.model.train()
        loss_total = 0
        num_batches = len(self.train_loader)
        for batch_idx, batch in enumerate(self.train_loader, start=1):
            feature, trajectory = batch["features"], batch["trajectory"]
            feature = feature.transpose(0, 1).to(self.device)
            trajectory = trajectory.transpose(0, 1).to(self.device)
            refinement_steps = [random.randint(1, 10) for _ in range(len(feature))]
            # Compute Loss
            loss = self.model.compute_loss(feature, trajectory, refinement_steps)
            batch_loss = loss.item()
            loss_total += batch_loss
            # Apply Gradient Step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.global_step += 1
            if (
                self.training_log["metric"] == "batch"
                and _should_log(self.global_step, self.training_log)
            ):
                wandb.log(
                    {
                        "global_step": self.global_step,
                        "epoch": epoch,
                        "train/batch": batch_idx,
                        "train/batch_loss": batch_loss,
                        "train/epoch_loss_so_far": loss_total / batch_idx,
                        "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                    }
                )

            if (
                self.validation_log["metric"] == "batch"
                and _should_log(self.global_step, self.validation_log)
            ):
                validation_loss, diversity = self.validate_single_batch()
                log_payload: dict[str, object] = {
                    "global_step": self.global_step,
                    "epoch": epoch,
                    "validation/batch": batch_idx,
                    "validation/batch_loss": validation_loss,
                }
                if diversity is not None:
                    log_payload["diversity/batch_apd"] = diversity["apd"]
                    log_payload["diversity/batch_mean_pairwise_w2"] = diversity[
                        "mean_pairwise_w2"
                    ]
                wandb.log(log_payload)

        return loss_total / max(num_batches, 1)

    def _next_validation_batch(self):
        try:
            return next(self.validation_iterator)
        except StopIteration:
            self.validation_iterator = iter(self.val_loader)
            return next(self.validation_iterator)

    def _compute_validation_batch_metrics(self, batch):
        feature, trajectory = (
            batch["features"].to(self.device),
            batch["trajectory"].to(self.device),
        )
        refinement_steps = [random.randint(1, 10) for _ in range(len(feature))]
        loss = self.model.compute_loss(feature, trajectory, refinement_steps)

        diversity = None
        if self.measure_diversity:
            predictions = self.model(feature, refinement_steps)
            diversity = compute_diversity_metrics(predictions, self.model)

        return loss.item(), diversity

    def validate_single_batch(self):
        batch = self._next_validation_batch()
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                return self._compute_validation_batch_metrics(batch)
        finally:
            if was_training:
                self.model.train()

    def validate(self):
        total_loss = 0
        apd_sum = 0.0
        w2_sum = 0.0
        num_batches = 0
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for batch in self.val_loader:
                    batch_loss, diversity = self._compute_validation_batch_metrics(
                        batch
                    )
                    total_loss += batch_loss
                    num_batches += 1
                    if diversity is not None:
                        apd_sum += diversity["apd"]
                        w2_sum += diversity["mean_pairwise_w2"]
        finally:
            if was_training:
                self.model.train()

        loss = total_loss / max(num_batches, 1)
        print(f"Validation Loss: {loss:.4f}")

        diversity = (
            {
                "apd": apd_sum,
                "mean_pairwise_w2": w2_sum,
            }
            if self.measure_diversity
            else None
        )

        return loss, diversity

    def train(self, num_epochs):
        pbar = tqdm(range(num_epochs), desc="Training")

        for epoch in pbar:
            loss = self.train_single_epoch(epoch)
            learning_rate = self.optimizer.param_groups[0]["lr"]

            postfix = {
                "loss": loss,
                "lr": learning_rate,
            }
            log_payload: dict[str, object] = {
                "global_step": self.global_step,
                "epoch": epoch,
            }

            if self.training_log["metric"] == "epoch" and _should_log(
                epoch + 1, self.training_log
            ):
                log_payload["loss"] = loss
                log_payload["train/epoch_loss"] = loss
                log_payload["learning_rate"] = learning_rate
                log_payload["train/learning_rate"] = learning_rate

            if self.validation_log["metric"] == "epoch" and _should_log(
                epoch + 1, self.validation_log
            ):
                validation_loss, diversity = self.validate()

                postfix["validation_loss"] = validation_loss
                log_payload["validation_loss"] = validation_loss
                log_payload["validation/loss"] = validation_loss
                if diversity is not None:
                    postfix["apd"] = diversity["apd"]
                    postfix["w2"] = diversity["mean_pairwise_w2"]
                    log_payload["diversity/apd"] = diversity["apd"]
                    log_payload["diversity/mean_pairwise_w2"] = diversity["mean_pairwise_w2"]

            pbar.set_postfix(postfix)
            if len(log_payload) > 2:
                wandb.log(log_payload)

        if self.measure_latency:
            self.profile_latency()

    # ------------------------------------------------------------------ #
    # Latency profiling
    # ------------------------------------------------------------------ #

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

        # Use a single batch from the validation set for timing.
        batch = next(iter(self.val_loader))
        feature, trajectory = (
            batch["features"].to(self.device),
            batch["trajectory"].to(self.device),
        )

        # Timing parameters
        n_warmup = 10
        n_runs = 100
        batch_size = feature.shape[1]  # Use the entire batch for timing by default
        max_steps = 10 # Maximum refinement steps

        # Slice features for reproducible timing input.
        num_frames = 5  # Use only the first 5 frames for timing to reduce runtime and GPU memory usage
        features = feature[:num_frames, :batch_size].contiguous()

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

