#!/usr/bin/env python3
import argparse
import csv
import math
import sys
from pathlib import Path

import torch
import yaml
from box import Box
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from AnytimeTrajectoryPredictor.Data.feature_extractor import WaymoImagePlaneDataset
from AnytimeTrajectoryPredictor.models.TrajectoryPredictor import TrajectoryPredictor


def _cfg_get(config, key, default=None):
    return getattr(config, key, default) if config is not None else default


def _best_checkpoint_from_save_to(save_to):
    if not save_to:
        return None
    path = Path(save_to)
    suffix = path.suffix or ".pth"
    best = path.with_name(f"{path.stem}.best{suffix}")
    if best.exists():
        return best
    return path if path.exists() else None


def _move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def _load_model_and_data(config_path, checkpoint_path, split, batch_size, num_workers, device, max_samples=None):
    print(f"Loading config: {config_path}", flush=True)
    with open(config_path, "r") as f:
        args = Box(yaml.safe_load(f))
    args.feature_extractor.max_samples = max_samples or _cfg_get(args.feature_extractor, "max_samples", None)
    print(f"Loading dataset split={split} max_samples={args.feature_extractor.max_samples}", flush=True)
    dataset = WaymoImagePlaneDataset(args.feature_extractor, split=split)
    print(f"Loaded dataset split={split} samples={len(dataset)}", flush=True)
    args.model.use_rgb_context = True
    args.model.input_dim = 4
    args.model.trajectory_mean = getattr(dataset, "target_mean", _cfg_get(args.model, "trajectory_mean", [0.0, 0.0]))
    args.model.trajectory_std = getattr(dataset, "target_std", _cfg_get(args.model, "trajectory_std", [1.0, 1.0]))
    args.model.history_steps_H = getattr(dataset, "history_steps", _cfg_get(args.model, "history_steps_H", 10))
    args.model.future_horizon_T = getattr(dataset, "future_steps", _cfg_get(args.model, "future_horizon_T", 80))
    print("Creating model", flush=True)
    model = TrajectoryPredictor.create_model(args).to(device)
    print("Created model", flush=True)
    if checkpoint_path is None:
        checkpoint_path = _best_checkpoint_from_save_to(_cfg_get(args.training, "save_to", None))
    if checkpoint_path:
        print(f"Loading checkpoint bytes: {checkpoint_path}", flush=True)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        print("Loaded checkpoint bytes", flush=True)
        missing, unexpected = model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        print(f"Loaded checkpoint: {checkpoint_path}")
        if missing:
            print(f"Missing keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")
    else:
        print("No checkpoint supplied/found; using randomly initialized model.")
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    return args, model, dataset, loader


def _minade_minfde(mu, y, mask):
    distances = torch.linalg.norm(mu - y[:, None], dim=-1)
    mask_modes = mask[:, None]
    denom = mask_modes.sum(dim=(2, 3)).clamp_min(1.0)
    ade = (distances * mask_modes).sum(dim=(2, 3)) / denom
    minade, best_mode = ade.min(dim=1)
    valid_counts = mask.long().sum(dim=-1).clamp_min(1)
    last_idx = (valid_counts - 1).view(mask.shape[0], 1, mask.shape[1], 1).expand(-1, mu.shape[1], -1, 1)
    final_dist = distances.gather(dim=3, index=last_idx).squeeze(-1)
    agent_valid = (mask.sum(dim=-1) > 0).to(distances.dtype)
    fde = (final_dist * agent_valid[:, None]).sum(dim=2) / agent_valid[:, None].sum(dim=2).clamp_min(1.0)
    minfde = fde.min(dim=1).values
    return minade, minfde, best_mode, ade


def _nll(model, params, y, mask):
    return model._gmm_nll(params, y, mask)


def _covariance_stats(params, best_mode=None):
    chol = params.cov_cholesky.detach()
    cov = chol @ chol.transpose(-1, -2)
    eig = torch.linalg.eigvalsh(cov).clamp_min(1e-12)
    std_x = chol[..., 0, 0].abs()
    std_y = chol[..., 1, 1].abs()
    corr = cov[..., 0, 1] / (cov[..., 0, 0].clamp_min(1e-12).sqrt() * cov[..., 1, 1].clamp_min(1e-12).sqrt())
    logdet = torch.logdet(cov.clamp_min(1e-12))
    trace = cov[..., 0, 0] + cov[..., 1, 1]
    cond = eig[..., -1] / eig[..., 0]
    out = {
        "std_x_mean": std_x.mean(),
        "std_y_mean": std_y.mean(),
        "std_x_p05": torch.quantile(std_x.flatten(), 0.05),
        "std_y_p05": torch.quantile(std_y.flatten(), 0.05),
        "std_x_p95": torch.quantile(std_x.flatten(), 0.95),
        "std_y_p95": torch.quantile(std_y.flatten(), 0.95),
        "corr_abs_mean": corr.abs().mean(),
        "corr_abs_p95": torch.quantile(corr.abs().flatten(), 0.95),
        "logdet_mean": logdet.mean(),
        "logdet_p05": torch.quantile(logdet.flatten(), 0.05),
        "trace_mean": trace.mean(),
        "trace_p95": torch.quantile(trace.flatten(), 0.95),
        "cond_mean": cond.mean(),
        "cond_p95": torch.quantile(cond.flatten(), 0.95),
        "tiny_std_rate_lt_0p02": ((std_x < 0.02) | (std_y < 0.02)).float().mean(),
        "huge_std_rate_gt_0p5": ((std_x > 0.5) | (std_y > 0.5)).float().mean(),
    }
    if best_mode is not None:
        b = torch.arange(chol.shape[0], device=chol.device)
        chol_b = chol[b, best_mode]
        cov_b = chol_b @ chol_b.transpose(-1, -2)
        eig_b = torch.linalg.eigvalsh(cov_b).clamp_min(1e-12)
        std_x_b = chol_b[..., 0, 0].abs()
        std_y_b = chol_b[..., 1, 1].abs()
        trace_b = cov_b[..., 0, 0] + cov_b[..., 1, 1]
        out.update({
            "best_std_x_mean": std_x_b.mean(),
            "best_std_y_mean": std_y_b.mean(),
            "best_trace_mean": trace_b.mean(),
            "best_cond_p95": torch.quantile((eig_b[..., -1] / eig_b[..., 0]).flatten(), 0.95),
        })
    return out


def _mode_stats(params, best_mode):
    probs = params.mode_probs.detach()
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1)
    top = probs.argmax(dim=1)
    best_prob = probs.gather(1, best_mode[:, None]).squeeze(1)
    return {
        "mode_entropy_mean": entropy.mean(),
        "top_prob_mean": probs.max(dim=1).values.mean(),
        "best_mode_prob_mean": best_prob.mean(),
        "top_is_best_rate": (top == best_mode).float().mean(),
    }


def _mode_spread_stats(params, mask):
    modes = params.mu.shape[1]
    if modes < 2:
        zero = params.mu.new_tensor(0.0)
        return {"mode_pair_ade_mean": zero, "mode_pair_fde_mean": zero}
    upper = torch.triu_indices(modes, modes, offset=1, device=params.mu.device)
    pair_dist = torch.linalg.norm(params.mu[:, upper[0]] - params.mu[:, upper[1]], dim=-1)
    pair_ade = (pair_dist * mask[:, None]).sum(dim=(2, 3)) / mask[:, None].sum(dim=(2, 3)).clamp_min(1.0)
    valid_counts = mask.long().sum(dim=-1).clamp_min(1)
    last_idx = (valid_counts - 1).view(mask.shape[0], 1, mask.shape[1], 1).expand(-1, pair_dist.shape[1], -1, 1)
    pair_fde = pair_dist.gather(dim=3, index=last_idx).squeeze(-1)
    agent_valid = (mask.sum(dim=-1) > 0).to(pair_dist.dtype)
    pair_fde = (pair_fde * agent_valid[:, None]).sum(dim=2) / agent_valid[:, None].sum(dim=2).clamp_min(1.0)
    return {"mode_pair_ade_mean": pair_ade.mean(), "mode_pair_fde_mean": pair_fde.mean()}


def _finite_scalar(x):
    if torch.is_tensor(x):
        x = float(x.detach().cpu())
    return float(x) if math.isfinite(float(x)) else float("nan")


def _parse_steps_list(value, fallback):
    if value is None or str(value).strip() == "":
        return [fallback]
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def main():
    parser = argparse.ArgumentParser(description="Inference diagnostics for image-plane ASTRA-EDM GMM covariance/calibration.")
    parser.add_argument("--config", default="configs/astra_edm_diffusion_waymo_image_plane.yml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--sampling_steps", type=int, default=None)
    parser.add_argument("--sampling_steps_list", default=None, help="Comma-separated sampler step counts, e.g. 4,8,16. Overrides --sampling_steps.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_csv", default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, model, dataset, loader = _load_model_and_data(
        args.config,
        args.checkpoint,
        args.split,
        args.batch_size,
        args.num_workers,
        device,
        max_samples=args.max_samples,
    )
    print(f"Dataset split={args.split} samples={len(dataset)} device={device}")
    steps_list = _parse_steps_list(args.sampling_steps_list, args.sampling_steps)

    totals_by_step = {str(step if step is not None else "default"): {} for step in steps_list}
    counts_by_step = {str(step if step is not None else "default"): 0 for step in steps_list}
    rows = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            batch = _move_batch(batch, device)
            y = model.normalizer.normalize(batch["trajectory"])
            mask = batch.get("future_mask")
            mask = mask.to(device=device, dtype=y.dtype) if mask is not None else torch.ones(y.shape[:-1], device=device, dtype=y.dtype)
            batch_n = y.shape[0]
            for step_count in steps_list:
                step_label = str(step_count if step_count is not None else "default")
                params = model._sample_params(batch, num_sampling_steps=step_count, denormalize=False)
                minade, minfde, best_mode, ade = _minade_minfde(params.mu, y, mask)
                nll = _nll(model, params, y, mask)
                metrics = {
                    "minADE_mean": minade.mean(),
                    "minFDE_mean": minfde.mean(),
                    "NLL": nll,
                }
                metrics.update(_covariance_stats(params, best_mode=best_mode))
                metrics.update(_mode_stats(params, best_mode))
                metrics.update(_mode_spread_stats(params, mask))
                counts_by_step[step_label] += batch_n
                totals = totals_by_step[step_label]
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + _finite_scalar(value) * batch_n
                for i in range(batch_n):
                    rows.append({
                        "sampling_steps": step_label,
                        "batch": batch_idx,
                        "sample_in_batch": i,
                        "scene_id": batch.get("scene_id", [""] * batch_n)[i],
                        "trajectory_row_id": batch.get("trajectory_row_id", [""] * batch_n)[i],
                        "minADE": _finite_scalar(minade[i]),
                        "minFDE": _finite_scalar(minfde[i]),
                        "best_mode": int(best_mode[i].detach().cpu()),
                        "best_mode_prob": _finite_scalar(params.mode_probs[i, best_mode[i]]),
                        "top_prob": _finite_scalar(params.mode_probs[i].max()),
                    })
                std_x_mean = _finite_scalar(metrics["std_x_mean"])
                std_y_mean = _finite_scalar(metrics["std_y_mean"])
                top_is_best = _finite_scalar(metrics["top_is_best_rate"])
                spread_fde = _finite_scalar(metrics["mode_pair_fde_mean"])
                print(f"batch={batch_idx} steps={step_label} n={counts_by_step[step_label]} minADE={_finite_scalar(minade.mean()):.4f} minFDE={_finite_scalar(minfde.mean()):.4f} NLL={_finite_scalar(nll):.4f} std=({std_x_mean:.4f},{std_y_mean:.4f}) top_is_best={top_is_best:.3f} spread_fde={spread_fde:.4f}", flush=True)

    print("\nSUMMARY")
    for step_label in totals_by_step:
        count = max(counts_by_step[step_label], 1)
        summary = {key: value / count for key, value in totals_by_step[step_label].items()}
        print(f"sampling_steps: {step_label}")
        for key in sorted(summary):
            print(f"  {key}: {summary[key]:.6f}")
        print(f"  samples_evaluated: {counts_by_step[step_label]}")

    if args.output_csv:
        path = Path(args.output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
