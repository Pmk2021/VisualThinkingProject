#!/usr/bin/env python3
import argparse
import csv
import math
import time
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
from AnytimeTrajectoryPredictor.models.architectures.astra_edm_diffusion import make_karras_sigmas


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


def _load_model_and_data(config_path, checkpoint_path, split, batch_size, num_workers, device, max_samples=None, image_width=None, image_height=None):
    print(f"Loading config: {config_path}", flush=True)
    with open(config_path, "r") as f:
        args = Box(yaml.safe_load(f))
    args.feature_extractor.max_samples = max_samples or _cfg_get(args.feature_extractor, "max_samples", None)
    if image_width is not None:
        old_width = int(_cfg_get(args.feature_extractor, "image_width", 384))
        old_height = int(_cfg_get(args.feature_extractor, "image_height", 256))
        args.feature_extractor.image_width = int(image_width)
        if image_height is None:
            args.feature_extractor.image_height = max(1, int(round(float(old_height) * float(image_width) / max(float(old_width), 1.0))))
        else:
            args.feature_extractor.image_height = int(image_height)
    elif image_height is not None:
        args.feature_extractor.image_height = int(image_height)
    print(
        f"Loading dataset split={split} max_samples={args.feature_extractor.max_samples} "
        f"image_size={args.feature_extractor.image_width}x{args.feature_extractor.image_height}",
        flush=True,
    )
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


def _parse_samplers(value, default_sampler):
    if value is None or str(value).strip() == "":
        return [default_sampler]
    samplers = [part.strip().lower() for part in str(value).split(",") if part.strip()]
    invalid = [sampler for sampler in samplers if sampler not in {"euler", "heun"}]
    if invalid:
        raise ValueError(f"Invalid samplers: {invalid}. Expected euler, heun, or a comma-separated list.")
    return samplers


def _model_uses_sampler(model):
    return bool(getattr(model, "uses_sampler", hasattr(model, "sampler_type") and hasattr(model, "encode_context")))


def _maybe_compile_model(model, enabled, mode):
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("--compile_model requires torch.compile, but this PyTorch build does not expose it.")
    if torch.cuda.is_available():
        import torch._inductor.config as inductor_config
        inductor_config.triton.cudagraphs = False
    if hasattr(model, "transformer"):
        print(f"Compiling transformer with torch.compile(mode={mode!r}, cudagraphs=False)", flush=True)
        model.transformer = torch.compile(model.transformer, mode=mode)
        model._diagnose_compiled_transformer = True
    else:
        print(f"Compiling full model with torch.compile(mode={mode!r}, cudagraphs=False)", flush=True)
        model = torch.compile(model, mode=mode)
    return model


def _denoise_for_diagnose(model, x, context, sigma, self_cond=None):
    x_clean, hidden = model.denoise(x, context, sigma, self_cond=self_cond)
    if getattr(model, "_diagnose_compiled_transformer", False):
        x_clean = x_clean.clone()
        hidden = hidden.clone()
    return x_clean, hidden


def _sampler_step_with_context(model, x, context, sigma, sigma_next, self_cond=None):
    sigma_batch = sigma.expand(x.shape[0])
    x_clean, hidden = _denoise_for_diagnose(model, x, context, sigma_batch, self_cond=self_cond)
    if sigma_next == 0 or model.sampler_type == "euler":
        if sigma_next == 0:
            return x_clean, hidden, x_clean.detach()
        d = (x - x_clean) / sigma.clamp_min(1e-8)
        return x + (sigma_next - sigma) * d, hidden, x_clean.detach()

    d = (x - x_clean) / sigma.clamp_min(1e-8)
    x_euler = x + (sigma_next - sigma) * d
    x_clean_next, hidden_next = _denoise_for_diagnose(
        model,
        x_euler,
        context,
        sigma_next.expand(x.shape[0]),
        self_cond=x_clean.detach() if model.use_self_conditioning else None,
    )
    d_next = (x_euler - x_clean_next) / sigma_next.clamp_min(1e-8)
    x_next = x + (sigma_next - sigma) * 0.5 * (d + d_next)
    return x_next, hidden_next, x_clean_next.detach()


def _sample_params_with_context(model, batch, context, num_sampling_steps=None, denormalize=False):
    trajectory = batch.get("trajectory")
    if trajectory is not None:
        batch_size, agents, _, _ = trajectory.shape
    else:
        batch_size = batch["features"].shape[0]
        agents = batch["features"].shape[1]
    num_points = model.num_points
    steps = max(int(num_sampling_steps or model.num_sampling_steps), 1)
    sigmas = make_karras_sigmas(steps, model.sigma_min, model.sigma_max, model.rho, context.device)
    prior = model._prediction_prior(batch, agents, num_points, context.device, context.dtype)
    x = model._initial_noise(batch, batch_size, agents, num_points, context.device, sigmas[0], context.dtype, prior=prior)
    hidden = None
    self_cond = None
    for i in range(len(sigmas) - 1):
        x, hidden, self_cond = _sampler_step_with_context(model, x, context, sigmas[i], sigmas[i + 1], self_cond=self_cond)
    params = model._apply_prediction_prior(model.gmm_head(x, hidden), prior)
    params = model._expand_params(params)
    if denormalize:
        params.mu = model.normalizer.denormalize(params.mu)
    return params


def _derive_output_path(base_path, suffix, extension=None):
    base = Path(base_path)
    ext = extension if extension is not None else base.suffix
    return base.with_name(f"{base.stem}{suffix}{ext}")


def _summary_fieldnames(summary_rows):
    keys = set()
    for row in summary_rows:
        keys.update(row.keys())
    first = [key for key in ["sampler_type", "sampling_steps", "samples_evaluated"] if key in keys]
    rest = sorted(key for key in keys if key not in first)
    return first + rest


def _write_summary_csv(summary_rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_summary_fieldnames(summary_rows))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {path}")


TABLE_METRICS = [
    ("minADE", "minADE_mean"),
    ("minFDE", "minFDE_mean"),
    ("maxADE", "maxADE"),
    ("NLL", "NLL"),
    ("mode_entropy_mean", "mode_entropy_mean"),
    ("top_is_best_rate", "top_is_best_rate"),
    ("top_prob_mean", "top_prob_mean"),
    ("latency_ms_mean", "latency_ms_mean"),
    ("total_latency_ms_mean", "total_latency_ms_mean"),
]


TABLE_ROW_ALIASES = {
    "latency": "latency_ms_mean",
    "sample_latency": "latency_ms_mean",
    "total_latency": "total_latency_ms_mean",
    "entropy": "mode_entropy_mean",
    "mode_entropy": "mode_entropy_mean",
    "top_is_best": "top_is_best_rate",
    "top_prob": "top_prob_mean",
}


def _parse_table_rows(value):
    if value is None or str(value).strip().lower() in {"", "default"}:
        return [(label, key) for label, key in TABLE_METRICS]
    if str(value).strip().lower() == "all":
        return [(label, key) for label, key in TABLE_METRICS]

    by_name = {}
    for label, key in TABLE_METRICS:
        by_name[label.lower()] = (label, key)
        by_name[key.lower()] = (label, key)
    for alias, key in TABLE_ROW_ALIASES.items():
        by_name[alias] = next((item for item in TABLE_METRICS if item[1] == key), (key, key))

    rows = []
    unknown = []
    for raw in str(value).split(","):
        name = raw.strip()
        if not name:
            continue
        metric = by_name.get(name.lower())
        if metric is None:
            unknown.append(name)
        else:
            rows.append(metric)
    if unknown:
        known = sorted({label for label, _ in TABLE_METRICS} | {key for _, key in TABLE_METRICS} | set(TABLE_ROW_ALIASES))
        raise ValueError(f"Unknown --rows values: {unknown}. Known rows: {known}")
    return rows


def _write_table_png(summary_rows, path, rows=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = "#ffce91ff"
    primary_light = "#fff4e6"
    body_alt = "#fffaf2"
    text_dark = "#221f1b"
    grid = "#d8cbb8"

    metrics = rows if rows is not None else _parse_table_rows(None)
    columns = [
        (str(row.get("sampler_type", "")) + "\n" + str(row["sampling_steps"])) if row.get("sampler_type") else str(row["sampling_steps"])
        for row in summary_rows
    ]
    cell_text = []
    for _, key in metrics:
        values = []
        for row in summary_rows:
            value = row.get(key, float("nan"))
            values.append("nan" if not math.isfinite(float(value)) else f"{float(value):.4f}")
        cell_text.append(values)

    width = max(7.5, 1.15 * max(len(columns), 1) + 2.6)
    height = max(5.2, 0.72 * len(metrics) + 1.8)
    fig, ax = plt.subplots(figsize=(width, height), facecolor="white")
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=[label for label, _ in metrics],
        colLabels=columns,
        loc="center",
        cellLoc="center",
        rowLoc="right",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.05)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(grid)
        cell.set_linewidth(0.8)
        cell.PAD = 0.16
        cell.get_text().set_color(text_dark)
        if row == 0:
            cell.set_facecolor(primary)
            cell.set_linewidth(1.0)
            cell.get_text().set_weight("bold")
        elif col == -1:
            cell.set_facecolor(primary_light)
            cell.get_text().set_weight("bold")
        elif row % 2 == 0:
            cell.set_facecolor(body_alt)
        else:
            cell.set_facecolor("white")
    ax.set_title("Image-Plane GMM Diagnostics", pad=22, fontsize=16, color=text_dark, fontweight="bold")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


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
    parser.add_argument("--samplers", default="euler,heun", help="Comma-separated samplers to benchmark: euler,heun. Use an empty string to use the config sampler only.")
    parser.add_argument("--compile_model", action=argparse.BooleanOptionalAction, default=False, help="Compile the hot transformer inference module with torch.compile.")
    parser.add_argument("--compile_mode", default="default", help="torch.compile mode, e.g. default or reduce-overhead.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_csv", default=None, help="Per-sample CSV. Also derives *_summary.csv and *_table.png unless overridden.")
    parser.add_argument("--summary_csv", default=None, help="Per-sampler/step summary CSV with all aggregate metrics.")
    parser.add_argument("--table_png", default=None, help="PNG table for key metrics by sampler and sampling step.")
    parser.add_argument("--rows", default=None, help="Comma-separated PNG table rows. Use all/default, labels like minADE,NLL, or metric keys like minADE_mean.")
    parser.add_argument("--image_width", type=int, default=None, help="Override image-plane dataset resize width. If height is omitted, preserves config aspect ratio.")
    parser.add_argument("--image_height", type=int, default=None, help="Override image-plane dataset resize height.")
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
        image_width=args.image_width,
        image_height=args.image_height,
    )
    model = _maybe_compile_model(model, args.compile_model, args.compile_mode)
    model.eval()
    uses_sampler = _model_uses_sampler(model)
    print(f"Dataset split={args.split} samples={len(dataset)} device={device} uses_sampler={uses_sampler}")
    if uses_sampler:
        steps_list = _parse_steps_list(args.sampling_steps_list, args.sampling_steps)
        samplers = _parse_samplers(args.samplers, model.sampler_type)
        eval_keys = [(sampler, str(step if step is not None else "default"), step) for sampler in samplers for step in steps_list]
    else:
        eval_keys = [("", "baseline", None)]

    totals_by_key = {(sampler, step_label): {} for sampler, step_label, _ in eval_keys}
    counts_by_key = {(sampler, step_label): 0 for sampler, step_label, _ in eval_keys}
    rows = []
    original_sampler = getattr(model, "sampler_type", None)
    warmed = False
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            batch = _move_batch(batch, device)
            y = model.normalizer.normalize(batch["trajectory"])
            mask = batch.get("future_mask")
            mask = mask.to(device=device, dtype=y.dtype) if mask is not None else torch.ones(y.shape[:-1], device=device, dtype=y.dtype)
            batch_n = y.shape[0]

            if uses_sampler:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                context_start = time.perf_counter()
                context = model.encode_context(batch)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                context_latency_ms = (time.perf_counter() - context_start) * 1000.0

                if args.compile_model and not warmed:
                    for sampler, _, step_count in eval_keys:
                        model.sampler_type = sampler
                        _ = _sample_params_with_context(model, batch, context, num_sampling_steps=step_count, denormalize=False)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    warmed = True
            else:
                context = None
                context_latency_ms = 0.0

            for sampler, step_label, step_count in eval_keys:
                if uses_sampler:
                    model.sampler_type = sampler
                key = (sampler, step_label)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start_time = time.perf_counter()
                if uses_sampler:
                    params = _sample_params_with_context(model, batch, context, num_sampling_steps=step_count, denormalize=False)
                else:
                    params = model._sample_params(batch, denormalize=False)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                total_latency_ms = context_latency_ms + latency_ms
                minade, minfde, best_mode, ade = _minade_minfde(params.mu, y, mask)
                nll = _nll(model, params, y, mask)
                metrics = {
                    "minADE_mean": minade.mean(),
                    "minFDE_mean": minfde.mean(),
                    "maxADE": minade.max(),
                    "NLL": nll,
                    "context_latency_ms_mean": y.new_tensor(context_latency_ms),
                    "latency_ms_mean": y.new_tensor(latency_ms),
                    "total_latency_ms_mean": y.new_tensor(total_latency_ms),
                }
                metrics.update(_covariance_stats(params, best_mode=best_mode))
                metrics.update(_mode_stats(params, best_mode))
                metrics.update(_mode_spread_stats(params, mask))
                counts_by_key[key] += batch_n
                totals = totals_by_key[key]
                for metric_key, value in metrics.items():
                    scalar = _finite_scalar(value)
                    if metric_key == "maxADE":
                        totals[metric_key] = max(totals.get(metric_key, float("-inf")), scalar)
                    else:
                        totals[metric_key] = totals.get(metric_key, 0.0) + scalar * batch_n
                for i in range(batch_n):
                    sample_row = {
                        "batch": batch_idx,
                        "sample_in_batch": i,
                        "scene_id": batch.get("scene_id", [""] * batch_n)[i],
                        "trajectory_row_id": batch.get("trajectory_row_id", [""] * batch_n)[i],
                        "minADE": _finite_scalar(minade[i]),
                        "minFDE": _finite_scalar(minfde[i]),
                        "best_mode": int(best_mode[i].detach().cpu()),
                        "best_mode_prob": _finite_scalar(params.mode_probs[i, best_mode[i]]),
                        "top_prob": _finite_scalar(params.mode_probs[i].max()),
                        "context_latency_ms": context_latency_ms,
                        "latency_ms": latency_ms,
                        "total_latency_ms": total_latency_ms,
                    }
                    if uses_sampler:
                        sample_row = {"sampler_type": sampler, "sampling_steps": step_label, **sample_row}
                    rows.append(sample_row)
                std_x_mean = _finite_scalar(metrics["std_x_mean"])
                std_y_mean = _finite_scalar(metrics["std_y_mean"])
                top_is_best = _finite_scalar(metrics["top_is_best_rate"])
                spread_fde = _finite_scalar(metrics["mode_pair_fde_mean"])
                print(
                    (f"batch={batch_idx} sampler={sampler} steps={step_label} n={counts_by_key[key]} " if uses_sampler else f"batch={batch_idx} baseline n={counts_by_key[key]} ") +
                    f"minADE={_finite_scalar(minade.mean()):.4f} maxADE={_finite_scalar(minade.max()):.4f} "
                    f"minFDE={_finite_scalar(minfde.mean()):.4f} NLL={_finite_scalar(nll):.4f} "
                    f"context_ms={context_latency_ms:.2f} sample_ms={latency_ms:.2f} total_ms={total_latency_ms:.2f} "
                    f"std=({std_x_mean:.4f},{std_y_mean:.4f}) top_is_best={top_is_best:.3f} spread_fde={spread_fde:.4f}",
                    flush=True,
                )
    if uses_sampler:
        model.sampler_type = original_sampler

    summary_rows = []
    print("\nSUMMARY")
    for sampler, step_label, _ in eval_keys:
        key = (sampler, step_label)
        count = max(counts_by_key[key], 1)
        summary = {
            metric_key: (value if metric_key == "maxADE" else value / count)
            for metric_key, value in totals_by_key[key].items()
        }
        row = {"sampling_steps": step_label, "samples_evaluated": counts_by_key[key]}
        if uses_sampler:
            row = {"sampler_type": sampler, **row}
        row.update(summary)
        summary_rows.append(row)
        if uses_sampler:
            print(f"sampler_type: {sampler} sampling_steps: {step_label}")
        else:
            print("baseline")
        for metric_key in sorted(summary):
            print(f"  {metric_key}: {summary[metric_key]:.6f}")
        print(f"  samples_evaluated: {counts_by_key[key]}")

    summary_csv = Path(args.summary_csv) if args.summary_csv else (_derive_output_path(args.output_csv, "_summary") if args.output_csv else None)
    table_png = Path(args.table_png) if args.table_png else (_derive_output_path(args.output_csv, "_table", ".png") if args.output_csv else None)

    if args.output_csv:
        path = Path(args.output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)
        print(f"Wrote {path}")
    if summary_csv is not None:
        _write_summary_csv(summary_rows, summary_csv)
    if table_png is not None:
        _write_table_png(summary_rows, table_png, rows=_parse_table_rows(args.rows))


if __name__ == "__main__":
    main()
