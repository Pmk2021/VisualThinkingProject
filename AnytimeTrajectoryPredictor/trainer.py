import torch
import wandb
import random
from tqdm import tqdm

from AnytimeTrajectoryPredictor.evaluation.diversity import compute_diversity_metrics
from AnytimeTrajectoryPredictor.evaluation.latency import LatencyProfiler


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
        self.measure_diversity = args.training.measure_diversity if "measure_diversity" in args.training else False
        self.measure_latency = args.training.measure_latency if "measure_latency" in args.training else False
        self.logging_steps = args.training.logging_steps if self.measure_diversity else 1 # If measuring diversity, log every epoch. Otherwise, log every logging_steps epochs.
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
            loss_total += loss.item()
            # Apply Gradient Step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss_total

    def validate(self):
        total_loss = 0
        apd_sum = 0.0
        w2_sum = 0.0
        for batch in self.val_loader:
            feature, trajectory = (
                batch["features"].to(self.device),
                batch["trajectory"].to(self.device),
            )
            f_ = [random.randint(1, 10) for i in range(len(feature))]
            # Compute Loss
            loss = self.model.compute_loss(feature, trajectory, f_)
            total_loss += loss.item()
            if self.measure_diversity:
                predictions = self.model(feature, f_) # List of length num_frames, each element is (B, num_objects, output_dim)
                metrics = compute_diversity_metrics(predictions, self.model)
                apd_sum += metrics["apd"]
                w2_sum += metrics["mean_pairwise_w2"]

        loss = total_loss / len(self.val_loader)
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
            loss = self.train_single_epoch()
            learning_rate = self.optimizer.param_groups[0]["lr"]

            postfix = {
                "loss": loss,
                "lr": learning_rate,
            }
            log_payload = {
                "epoch": epoch,
                "loss": loss,
                "learning_rate": learning_rate,
            }

            if epoch % self.logging_steps == 0:
                validation_loss, diversity = self.validate()

                postfix["validation_loss"] = validation_loss
                log_payload["validation_loss"] = validation_loss
                if diversity is not None:
                    postfix["apd"] = diversity["apd"]
                    postfix["w2"] = diversity["mean_pairwise_w2"]
                    log_payload["diversity/apd"] = diversity["apd"]
                    log_payload["diversity/mean_pairwise_w2"] = diversity["mean_pairwise_w2"]

                pbar.set_postfix(postfix)
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

