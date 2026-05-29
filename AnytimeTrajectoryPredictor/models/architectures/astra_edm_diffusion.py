import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision import models
except Exception:
    models = None

try:
    import segmentation_models_pytorch as smp
except Exception:
    smp = None


def _get(config, key, default=None):
    return getattr(config, key, default) if config is not None else default


def _as_float_tensor(value, default, size):
    if value is None:
        value = default
    tensor = torch.tensor(value, dtype=torch.float32)
    if tensor.numel() == 1:
        tensor = tensor.repeat(size)
    return tensor.view(1, 1, 1, size)


def make_karras_sigmas(num_steps, sigma_min, sigma_max, rho, device):
    ramp = torch.linspace(0, 1, max(int(num_steps), 1), device=device)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat([sigmas, torch.zeros(1, device=device)])


def sample_training_sigmas(batch_size, p_mean, p_std, sigma_min, sigma_max, device):
    sigmas = torch.exp(p_mean + p_std * torch.randn(batch_size, device=device))
    return sigmas.clamp(min=sigma_min, max=sigma_max)


def fourier_embedding(x, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=x.device) / max(half - 1, 1))
    args = x[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, 1))
    return emb


def make_knot_indices(num_knots, future_horizon, placement="uniform"):
    if num_knots < 2:
        raise ValueError("num_knots must be >= 2")
    if num_knots > future_horizon:
        raise ValueError("num_knots must be <= future_horizon")
    if placement == "uniform":
        positions = torch.linspace(0.0, float(future_horizon - 1), steps=num_knots)
    elif placement == "quadratic":
        u = torch.linspace(0.0, 1.0, steps=num_knots)
        positions = (u ** 2) * float(future_horizon - 1)
    else:
        raise ValueError(f"Unknown knot_placement: {placement}")
    indices = positions.round().long()
    # enforce strictly increasing knot times
    for i in range(1, num_knots):
        if indices[i] <= indices[i - 1]:
            indices[i] = indices[i - 1] + 1
    indices = indices.clamp(0, future_horizon - 1)
    if indices[-1] != future_horizon - 1:
        indices[-1] = future_horizon - 1
    return indices


def make_catmull_rom_basis(knot_indices, future_horizon):
    """Return a [T, K] linear basis so that traj[t] = sum_k B[t, k] * knot[k].

    Uses uniform Catmull-Rom with reflected phantom endpoints so the curve
    passes through every knot and remains C^1 across segments.
    """
    knots = [int(k) for k in knot_indices]
    K = len(knots)
    T = int(future_horizon)
    B = torch.zeros(T, K)
    for t in range(T):
        seg = 0
        for j in range(K - 1):
            if knots[j] <= t:
                seg = j
        seg = min(seg, K - 2)
        denom = max(knots[seg + 1] - knots[seg], 1)
        u = float(max(0.0, min(1.0, (t - knots[seg]) / denom)))
        u2, u3 = u * u, u * u * u
        #TODO: What is this
        w0 = -0.5 * u + u2 - 0.5 * u3
        w1 = 1.0 - 2.5 * u2 + 1.5 * u3
        w2 = 0.5 * u + 2.0 * u2 - 1.5 * u3
        w3 = -0.5 * u2 + 0.5 * u3

        def _accumulate(idx, weight):
            if idx < 0:
                B[t, 0] += 2.0 * weight
                B[t, 1] += -weight
            elif idx >= K:
                B[t, K - 1] += 2.0 * weight
                B[t, K - 2] += -weight
            else:
                B[t, idx] += weight

        _accumulate(seg - 1, w0)
        _accumulate(seg, w1)
        _accumulate(seg + 1, w2)
        _accumulate(seg + 2, w3)
    return B


@dataclass
class GMMParams:
    mode_logits: torch.Tensor
    mode_probs: torch.Tensor
    mu: torch.Tensor
    cov_cholesky: torch.Tensor
    cov_raw: torch.Tensor


class TrajectoryNormalizer(nn.Module):
    def __init__(self, mean=None, std=None, trajectory_dim=2):
        super().__init__()
        self.register_buffer("mean", _as_float_tensor(mean, [0.0] * trajectory_dim, trajectory_dim))
        self.register_buffer("std", _as_float_tensor(std, [1.0] * trajectory_dim, trajectory_dim))

    def normalize(self, trajectory):
        return (trajectory - self.mean) / self.std.clamp_min(1e-6)

    def denormalize(self, trajectory):
        return trajectory * self.std.clamp_min(1e-6) + self.mean


class ASTRAStyleContextEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_layers, num_heads, dropout, max_history, max_agents):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.time_embedding = nn.Embedding(max_history, d_model)
        self.agent_embedding = nn.Embedding(max_agents, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, observed, observed_mask=None):
        batch, agents, history, _ = observed.shape
        tokens = self.input_projection(observed)
        time_ids = torch.arange(history, device=observed.device).view(1, 1, history)
        agent_ids = torch.arange(agents, device=observed.device).view(1, agents, 1)
        tokens = tokens + self.time_embedding(time_ids) + self.agent_embedding(agent_ids)
        tokens = tokens.reshape(batch, agents * history, -1)
        key_padding_mask = observed_mask.reshape(batch, agents * history) <= 0 if observed_mask is not None else None
        tokens = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        return self.norm(tokens)


class SmallCNNBackbone(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


def _make_resnet18_feature_extractor(weights_path=None):
    if models is None:
        return SmallCNNBackbone(out_dim=256), 256
    try:
        backbone = models.resnet18(weights=None)
        if weights_path:
            state = torch.load(weights_path, map_location="cpu", weights_only=False)
            backbone.load_state_dict(state.get("state_dict", state), strict=False)
        modules = list(backbone.children())[:-3]
        return nn.Sequential(*modules), 256
    except Exception:
        return SmallCNNBackbone(out_dim=256), 256


class _UNetEncoderStage(nn.Module):
    """Picks one stage from an smp.Unet encoder so we get a [B, C, H', W'] feature map."""

    def __init__(self, encoder, stage_index):
        super().__init__()
        self.encoder = encoder
        self.stage_index = stage_index

    def forward(self, x):
        features = self.encoder(x)
        return features[self.stage_index]


def _make_unet_feature_extractor(encoder_name="resnet18", weights_path=None, stage_index=-2):
    """Build an smp.Unet, optionally load pretrained UNetKeypointModel/ASTRA weights, return the chosen
    encoder stage as a feature-map module + its channel count."""
    if smp is None:
        return _make_resnet18_feature_extractor(weights_path)
    unet = smp.Unet(encoder_name=encoder_name, encoder_weights=None, in_channels=3, classes=1, activation=None)
    if weights_path:
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            elif "state_dict" in state:
                state = state["state_dict"]
        encoder_state = {}
        for key, value in state.items():
            new_key = key
            for prefix in ("module.", "model."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            if new_key.startswith("unet.encoder."):
                encoder_state[new_key[len("unet.encoder."):]] = value
            elif new_key.startswith("encoder."):
                encoder_state[new_key[len("encoder."):]] = value
        if encoder_state:
            load_result = unet.encoder.load_state_dict(encoder_state, strict=False)
            missing = getattr(load_result, "missing_keys", []) if load_result is not None else []
            unexpected = getattr(load_result, "unexpected_keys", []) if load_result is not None else []
            print(
                f"Loaded U-Net encoder weights from {weights_path}: "
                f"{len(encoder_state)} keys (missing={len(missing)}, unexpected={len(unexpected)})"
            )
        else:
            print(f"Warning: no encoder.* keys found in {weights_path}; encoder is randomly initialized")
    out_channels_list = unet.encoder.out_channels
    stage_index = int(stage_index)
    if stage_index < 0:
        stage_index += len(out_channels_list)
    stage_index = max(0, min(stage_index, len(out_channels_list) - 1))
    out_channels = out_channels_list[stage_index]
    return _UNetEncoderStage(unet.encoder, stage_index), out_channels


class RGBBoxContextEncoder(nn.Module):
    def __init__(self, config, d_model, num_layers, num_heads, dropout, max_history, max_agents):
        super().__init__()
        self.max_history = max_history
        self.max_agents = max_agents
        self.roi_size = int(_get(config, "roi_feature_size", 3))
        weights_path = _get(config, "rgb_backbone_weights", None)
        backbone_type = str(_get(config, "rgb_backbone", "resnet18"))
        stage_index = int(_get(config, "rgb_backbone_stage", -2))
        if backbone_type.startswith("unet_"):
            encoder_name = backbone_type[len("unet_"):]
            self.backbone, feature_dim = _make_unet_feature_extractor(
                encoder_name=encoder_name,
                weights_path=weights_path,
                stage_index=stage_index,
            )
        else:
            self.backbone, feature_dim = _make_resnet18_feature_extractor(weights_path)
        self.freeze_rgb_backbone = bool(_get(config, "freeze_rgb_backbone", False))
        if self.freeze_rgb_backbone:
            self.backbone.eval()
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
        self.rgb_norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        self.rgb_norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        self.visual_projection = nn.Linear(feature_dim, d_model)
        self.box_projection = nn.Linear(4, d_model)
        self.fusion_projection = nn.Linear(2 * d_model, d_model)
        self.time_embedding = nn.Embedding(max_history, d_model)
        self.agent_embedding = nn.Embedding(max_agents, d_model)
        self.camera_embedding = nn.Embedding(8, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def _roi_pool(self, feature_map, boxes):
        # feature_map: [B, H, C, fh, fw], boxes: [B, A, H, 4] with center in [-1,1] and size in [0,1].
        batch, history, channels, _, _ = feature_map.shape
        agents = boxes.shape[1]
        device = feature_map.device
        dtype = feature_map.dtype
        fmap = feature_map[:, None].expand(batch, agents, history, channels, feature_map.shape[-2], feature_map.shape[-1])
        fmap = fmap.reshape(batch * agents * history, channels, feature_map.shape[-2], feature_map.shape[-1])
        b = boxes.permute(0, 1, 2, 3).reshape(batch * agents * history, 4).to(device=device, dtype=dtype)
        local = torch.linspace(-1.0, 1.0, self.roi_size, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(local, local, indexing="ij")
        cx = b[:, 0].view(-1, 1, 1)
        cy = b[:, 1].view(-1, 1, 1)
        half_w = b[:, 2].clamp_min(1e-3).view(-1, 1, 1)
        half_h = b[:, 3].clamp_min(1e-3).view(-1, 1, 1)
        grid = torch.stack([cx + xx * half_w, cy + yy * half_h], dim=-1).clamp(-1.2, 1.2)
        pooled = F.grid_sample(fmap, grid, mode="bilinear", padding_mode="border", align_corners=False)
        pooled = pooled.mean(dim=(-1, -2))
        return pooled.view(batch, agents, history, channels)

    def forward(self, batch):
        rgb = batch["rgb_history"]
        boxes = batch.get("box_history", batch["features"])
        observed_mask = batch.get("observed_mask")
        camera_name = batch.get("camera_name")
        batch_size, history, channels, height, width = rgb.shape
        agents = boxes.shape[1]
        mean = self.rgb_norm_mean.to(device=rgb.device, dtype=rgb.dtype)
        std = self.rgb_norm_std.to(device=rgb.device, dtype=rgb.dtype)
        rgb = (rgb - mean) / std
        backbone_input = rgb.reshape(batch_size * history, channels, height, width)
        if self.freeze_rgb_backbone:
            self.backbone.eval()
            with torch.no_grad():
                fmap = self.backbone(backbone_input)
        else:
            fmap = self.backbone(backbone_input)
        fmap = fmap.reshape(batch_size, history, fmap.shape[1], fmap.shape[2], fmap.shape[3])
        visual = self._roi_pool(fmap, boxes)
        visual_tokens = self.visual_projection(visual)
        box_tokens = self.box_projection(boxes)
        tokens = self.fusion_projection(torch.cat([visual_tokens, box_tokens], dim=-1))
        time_ids = torch.arange(history, device=rgb.device).view(1, 1, history)
        agent_ids = torch.arange(agents, device=rgb.device).view(1, agents, 1)
        tokens = tokens + self.time_embedding(time_ids) + self.agent_embedding(agent_ids)
        if camera_name is not None:
            camera_ids = camera_name.clamp(0, 7).view(batch_size, 1, 1)
            tokens = tokens + self.camera_embedding(camera_ids)
        tokens = tokens.reshape(batch_size, agents * history, -1)
        key_padding_mask = observed_mask.reshape(batch_size, agents * history) <= 0 if observed_mask is not None else None
        tokens = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        return self.norm(tokens)


class DiffusionTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.self_norm = nn.LayerNorm(d_model)
        self.cross_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, context):
        q = self.self_norm(x)
        x = x + self.self_attn(q, q, q, need_weights=False)[0]
        x = x + self.cross_attn(self.cross_norm(x), context, context, need_weights=False)[0]
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DiffusionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_modes = _get(config, "num_modes_K", _get(config, "num_trajectory_possibilities", 6))
        self.trajectory_dim = _get(config, "trajectory_dim_dy", 2)
        self.use_self_conditioning = bool(_get(config, "use_self_conditioning", False))
        self.d_model = _get(config, "d_model", 256)
        self.max_agents = _get(config, "max_agents", 1)
        self.future_horizon = _get(config, "future_horizon_T", 80)
        num_layers = _get(config, "num_layers", 4)
        num_heads = _get(config, "num_heads", 8)
        dropout = _get(config, "dropout", 0.1)
        input_dim = self.trajectory_dim * 2 if self.use_self_conditioning else self.trajectory_dim
        self.trajectory_projection = nn.Linear(input_dim, self.d_model)
        self.noise_mlp = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.SiLU(), nn.Linear(self.d_model, self.d_model))
        self.mode_embedding = nn.Embedding(self.num_modes, self.d_model)
        self.agent_embedding = nn.Embedding(self.max_agents, self.d_model)
        self.future_time_embedding = nn.Embedding(self.future_horizon, self.d_model)
        self.blocks = nn.ModuleList([DiffusionTransformerBlock(self.d_model, num_heads, dropout) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(self.d_model)
        self.denoise_head = nn.Linear(self.d_model, self.trajectory_dim)

    def forward(self, x, context, noise_level, time_indices=None, self_cond=None):
        batch, modes, agents, num_points, _ = x.shape
        if self.use_self_conditioning:
            if self_cond is None:
                self_cond = torch.zeros_like(x)
            x_input = torch.cat([x, self_cond], dim=-1)
        else:
            x_input = x
        tokens = self.trajectory_projection(x_input)
        mode_ids = torch.arange(modes, device=x.device).view(1, modes, 1, 1)
        agent_ids = torch.arange(agents, device=x.device).view(1, 1, agents, 1)
        if time_indices is None:
            time_indices = torch.arange(num_points, device=x.device)
        time_ids = time_indices.to(device=x.device, dtype=torch.long).view(1, 1, 1, num_points)
        noise_emb = self.noise_mlp(fourier_embedding(noise_level, self.d_model))
        tokens = (
            tokens
            + self.mode_embedding(mode_ids)
            + self.agent_embedding(agent_ids)
            + self.future_time_embedding(time_ids)
            + noise_emb.view(batch, 1, 1, 1, self.d_model)
        )
        tokens = tokens.reshape(batch, modes * agents * num_points, self.d_model)
        for block in self.blocks:
            tokens = block(tokens, context)
        hidden = self.final_norm(tokens)
        denoise = self.denoise_head(hidden).reshape(batch, modes, agents, num_points, -1)
        return denoise, hidden.reshape(batch, modes, agents, num_points, self.d_model)


class GMMHead(nn.Module):
    def __init__(self, d_model, trajectory_dim, log_std_min, log_std_max, offdiag_clip):
        super().__init__()
        if trajectory_dim != 2:
            raise ValueError("ASTRA-EDM currently supports 2D trajectory outputs.")
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.offdiag_clip = offdiag_clip
        self.mode_logits_head = nn.Linear(d_model, 1)
        self.mean_delta_head = nn.Linear(d_model, trajectory_dim)
        self.cov_head = nn.Linear(d_model, 3)

    def forward(self, denoised, hidden):
        mode_hidden = hidden.mean(dim=(2, 3))
        mode_logits = self.mode_logits_head(mode_hidden).squeeze(-1)
        mu = denoised + self.mean_delta_head(hidden)
        cov_raw = self.cov_head(hidden)
        log_l11 = cov_raw[..., 0].clamp(self.log_std_min, self.log_std_max)
        l21 = cov_raw[..., 1].clamp(-self.offdiag_clip, self.offdiag_clip)
        log_l22 = cov_raw[..., 2].clamp(self.log_std_min, self.log_std_max)
        zeros = torch.zeros_like(log_l11)
        row1 = torch.stack([torch.exp(log_l11), zeros], dim=-1)
        row2 = torch.stack([l21, torch.exp(log_l22)], dim=-1)
        chol = torch.stack([row1, row2], dim=-2)
        return GMMParams(
            mode_logits=mode_logits,
            mode_probs=F.softmax(mode_logits, dim=1),
            mu=mu,
            cov_cholesky=chol,
            cov_raw=cov_raw,
        )


class ASTRAEDMDiffusionModel(nn.Module):
    expects_batch = True

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_modes = _get(config, "num_modes_K", _get(config, "num_trajectory_possibilities", 6))
        self.trajectory_dim = _get(config, "trajectory_dim_dy", 2)
        self.future_horizon = _get(config, "future_horizon_T", 80)
        self.max_history = _get(config, "history_steps_H", 11)
        self.max_agents = _get(config, "max_agents", 1)
        self.d_model = _get(config, "d_model", 256)
        self.sigma_data = _get(config, "sigma_data", 1.0)
        self.sigma_min = _get(config, "sigma_min", 0.01)
        self.sigma_max = _get(config, "sigma_max", 5.0)
        self.rho = _get(config, "rho", 7.0)
        self.num_sampling_steps = _get(config, "num_sampling_steps", 12)
        self.p_mean = _get(config, "P_mean", -0.5)
        self.p_std = _get(config, "P_std", 1.0)
        self.sigma_min_train = _get(config, "sigma_min_train", self.sigma_min)
        self.sigma_max_train = _get(config, "sigma_max_train", self.sigma_max)
        self.use_edm_loss_weighting = bool(_get(config, "use_edm_loss_weighting", False))
        self.use_rgb_context = bool(_get(config, "use_rgb_context", False))
        self.use_self_conditioning = bool(_get(config, "use_self_conditioning", False))
        self.self_condition_prob = float(_get(config, "self_condition_prob", 0.5))
        self.sampler_type = _get(config, "sampler_type", "heun")
        if self.sampler_type not in {"euler", "heun"}:
            raise ValueError(f"Unknown sampler_type: {self.sampler_type}")
        self.rollout_loss_prob = float(_get(config, "rollout_loss_prob", 0.10))
        self.rollout_min_steps = int(_get(config, "rollout_min_steps", 1))
        self.rollout_max_steps = int(_get(config, "rollout_max_steps", 2))
        self.lambda_rollout_diff = float(_get(config, "lambda_rollout_diff", 0.5))
        self.gmm_sampled_state_prob = float(_get(config, "gmm_sampled_state_prob", 0.0))
        self.gmm_rollout_steps = int(_get(config, "gmm_rollout_steps", 2))
        self.prior_noise_prob = float(_get(config, "prior_noise_prob", 0.5 if self.use_rgb_context else 0.0))
        self.use_prior_initialization = bool(_get(config, "use_prior_initialization", self.use_rgb_context))
        self.use_prior_correction = bool(_get(config, "use_prior_correction", False))
        self.prior_fan_magnitude = float(_get(config, "prior_fan_magnitude", 0.20))
        self.prior_residual_scale = float(_get(config, "prior_residual_scale", 0.05))
        self.initial_noise_type = _get(config, "initial_noise_type", "gaussian")
        self.radial_mode_radius = float(_get(config, "radial_mode_radius", 0.20))
        self.radial_jitter_scale = float(_get(config, "radial_jitter_scale", self.prior_residual_scale))
        self.radial_gaussian_scale = float(_get(config, "radial_gaussian_scale", 0.5))
        self.radial_train_noise_prob = float(_get(config, "radial_train_noise_prob", 0.0))
        self.radial_include_center_mode = bool(_get(config, "radial_include_center_mode", True))
        self.radial_radius_growth = _get(config, "radial_radius_growth", "linear")
        if self.initial_noise_type not in {"gaussian", "radial_control_points"}:
            raise ValueError(f"Unknown initial_noise_type: {self.initial_noise_type}")
        if self.radial_radius_growth not in {"linear", "sqrt"}:
            raise ValueError(f"Unknown radial_radius_growth: {self.radial_radius_growth}")
        self.monotonicity_eval_steps = _get(config, "monotonicity_eval_steps", [1, 2, 4, 8, 16])
        self.monotonicity_tolerance = float(_get(config, "monotonicity_tolerance", 1e-4))

        num_knots = _get(config, "num_knots", None)
        self.use_control_points = bool(_get(config, "use_control_points", num_knots is not None))
        knot_placement = _get(config, "knot_placement", "uniform")
        interpolation = _get(config, "interpolation", "catmull_rom")
        if self.use_control_points:
            if num_knots is None:
                num_knots = 6
            num_knots = int(num_knots)
            if num_knots >= self.future_horizon:
                # equivalent to old per-timestep behaviour; turn the feature off
                self.use_control_points = False
        if self.use_control_points:
            if interpolation != "catmull_rom":
                raise ValueError(f"Unsupported interpolation: {interpolation}")
            self.num_knots = num_knots
            self.num_points = num_knots
            knot_indices = make_knot_indices(num_knots, self.future_horizon, knot_placement)
            basis = make_catmull_rom_basis(knot_indices, self.future_horizon)
            self.register_buffer("knot_indices", knot_indices, persistent=False)
            self.register_buffer("knot_basis", basis, persistent=False)
        else:
            self.num_knots = self.future_horizon
            self.num_points = self.future_horizon
            self.knot_indices = None
            self.knot_basis = None

        self.lambda_diff = float(_get(config, "lambda_diff", 1.0))
        self.lambda_wta_diff = float(_get(config, "lambda_wta_diff", 0.0))
        self.lambda_nll = float(_get(config, "lambda_nll", 0.1))
        self.lambda_ade = float(_get(config, "lambda_ade", 1.0))
        self.lambda_fde = float(_get(config, "lambda_fde", 1.0))
        self.lambda_mode = float(_get(config, "lambda_mode", 0.1))
        self.lambda_mode_margin = float(_get(config, "lambda_mode_margin", 0.0))
        self.mode_margin = float(_get(config, "mode_margin", 0.5))
        self.lambda_cov = float(_get(config, "lambda_cov", 0.01))
        self.lambda_bounds = float(_get(config, "lambda_bounds", 0.0))
        self.lambda_smooth = float(_get(config, "lambda_smooth", 0.0))
        self.lambda_accel = float(_get(config, "lambda_accel", 0.0))
        self.lambda_jerk = float(_get(config, "lambda_jerk", 0.0))
        self.lambda_prior_residual = float(_get(config, "lambda_prior_residual", 0.0))
        self.lambda_speed = float(_get(config, "lambda_speed", 0.0))
        self.lambda_entropy = float(_get(config, "lambda_entropy", 0.0))
        self.lambda_diversity = float(_get(config, "lambda_diversity", 0.0))
        self.diversity_margin = float(_get(config, "diversity_margin", 0.15))
        self.diversity_reference_box_size = float(_get(config, "diversity_reference_box_size", 0.08))
        self.diversity_min_scale = float(_get(config, "diversity_min_scale", 0.35))
        self.diversity_max_scale = float(_get(config, "diversity_max_scale", 2.50))
        self.diversity_y_top_scale = float(_get(config, "diversity_y_top_scale", 0.75))
        self.diversity_y_bottom_scale = float(_get(config, "diversity_y_bottom_scale", 1.25))
        self.diversity_size_exponent = float(_get(config, "diversity_size_exponent", 1.0))
        self.diversity_y_exponent = float(_get(config, "diversity_y_exponent", 1.0))
        self.speed_cap = float(_get(config, "speed_cap", 0.40))

        context_layers = _get(config, "context_layers", 2)
        num_heads = _get(config, "num_heads", 8)
        dropout = _get(config, "dropout", 0.1)
        input_dim = _get(config, "input_dim", 6)
        self.normalizer = TrajectoryNormalizer(
            mean=_get(config, "trajectory_mean", [0.0, 0.0]),
            std=_get(config, "trajectory_std", [1.0, 1.0]),
            trajectory_dim=self.trajectory_dim,
        )
        if self.use_rgb_context:
            self.context_encoder = RGBBoxContextEncoder(
                config,
                d_model=self.d_model,
                num_layers=context_layers,
                num_heads=num_heads,
                dropout=dropout,
                max_history=self.max_history,
                max_agents=self.max_agents,
            )
        else:
            self.context_encoder = ASTRAStyleContextEncoder(
                input_dim=input_dim,
                d_model=self.d_model,
                num_layers=context_layers,
                num_heads=num_heads,
                dropout=dropout,
                max_history=self.max_history,
                max_agents=self.max_agents,
            )
        self.transformer = DiffusionTransformer(config)
        self.gmm_head = GMMHead(
            d_model=self.d_model,
            trajectory_dim=self.trajectory_dim,
            log_std_min=_get(config, "log_std_min", -5.0),
            log_std_max=_get(config, "log_std_max", 3.0),
            offdiag_clip=_get(config, "offdiag_clip", 5.0),
        )

    def _broadcast_sigma(self, sigma):
        return sigma.view(-1, 1, 1, 1, 1)

    def _edm_coefficients(self, sigma):
        sigma_b = self._broadcast_sigma(sigma)
        sigma_data = torch.as_tensor(self.sigma_data, device=sigma.device, dtype=sigma.dtype)
        denom = sigma_b.square() + sigma_data.square()
        c_skip = sigma_data.square() / denom
        c_out = sigma_b * sigma_data / torch.sqrt(denom)
        c_in = 1.0 / torch.sqrt(denom)
        c_noise = 0.25 * torch.log(sigma.clamp_min(1e-8))
        return c_skip, c_out, c_in, c_noise

    def encode_context(self, batch):
        if self.use_rgb_context:
            return self.context_encoder(batch)
        return self.context_encoder(batch["features"], batch.get("observed_mask"))

    def denoise(self, x, context, sigma, self_cond=None):
        c_skip, c_out, c_in, c_noise = self._edm_coefficients(sigma)
        time_indices = self.knot_indices if self.use_control_points else None
        f_out, hidden = self.transformer(c_in * x, context, c_noise, time_indices=time_indices, self_cond=self_cond)
        return c_skip * x + c_out * f_out, hidden

    def _expand_mu(self, knot_values):
        if not self.use_control_points:
            return knot_values
        return torch.einsum("tk,...kc->...tc", self.knot_basis.to(dtype=knot_values.dtype), knot_values)

    def _expand_cov_cholesky(self, knot_chol):
        if not self.use_control_points:
            return knot_chol
        orig_dtype = knot_chol.dtype
        # torch.linalg.cholesky has no Half kernel; force fp32 even when wrapped in
        # autocast so the matmul/einsum below don't get re-cast back to half.
        device_type = knot_chol.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            chol32 = knot_chol.to(dtype=torch.float32)
            basis32 = self.knot_basis.to(dtype=torch.float32)
            knot_cov = chol32 @ chol32.transpose(-1, -2)
            weights = basis32.pow(2)
            full_cov = torch.einsum("tk,...kij->...tij", weights, knot_cov)
            eye = torch.eye(2, device=full_cov.device, dtype=full_cov.dtype) * 1e-6
            full_cov = full_cov + eye
            full_chol = torch.linalg.cholesky(full_cov)
        return full_chol.to(dtype=orig_dtype)

    def _expand_params(self, params):
        if not self.use_control_points:
            return params
        return GMMParams(
            mode_logits=params.mode_logits,
            mode_probs=params.mode_probs,
            mu=self._expand_mu(params.mu),
            cov_cholesky=self._expand_cov_cholesky(params.cov_cholesky),
            cov_raw=params.cov_raw,
        )

    def _future_time_array(self, num_points, device, dtype):
        """Return the float future-step offsets (1-indexed) used by the CV prior."""
        if self.use_control_points and num_points == self.num_knots:
            return (self.knot_indices.to(device=device, dtype=dtype) + 1.0).view(1, 1, num_points, 1)
        return torch.arange(1, num_points + 1, device=device, dtype=dtype).view(1, 1, num_points, 1)

    def _constant_velocity_prior(self, batch, agents, num_points, modes, device, dtype, add_mode_fan=True):
        boxes = batch.get("box_history", batch.get("features"))
        batch_size = boxes.shape[0] if boxes is not None else batch.get("trajectory").shape[0]
        if boxes is None or boxes.shape[-1] < 2:
            return torch.zeros((batch_size, modes, agents, num_points, 2), device=device, dtype=dtype)
        centers = boxes[..., :2].to(device=device, dtype=dtype)
        observed_mask = batch.get("observed_mask")
        if observed_mask is None:
            observed_mask = torch.ones(centers.shape[:-1], device=device, dtype=dtype)
        else:
            observed_mask = observed_mask.to(device=device, dtype=dtype)
        valid_counts = observed_mask.sum(dim=-1).long().clamp_min(1)
        last_idx = (valid_counts - 1).view(centers.shape[0], agents, 1, 1).expand(-1, -1, 1, 2)
        last = centers.gather(dim=2, index=last_idx).squeeze(2)

        history = centers.shape[2]
        t_obs = torch.arange(history, device=device, dtype=dtype).view(1, 1, history, 1)
        w = observed_mask.unsqueeze(-1)
        w_sum = w.sum(dim=2, keepdim=True).clamp_min(1.0)
        t_mean = (t_obs * w).sum(dim=2, keepdim=True) / w_sum
        x_mean = (centers * w).sum(dim=2, keepdim=True) / w_sum
        centered_t = (t_obs - t_mean) * w
        denom = (centered_t.square()).sum(dim=2).clamp_min(1e-6)
        slope = (centered_t * (centers - x_mean)).sum(dim=2) / denom
        slope = slope * (valid_counts > 1).to(dtype).unsqueeze(-1)

        t_future = self._future_time_array(num_points, device, dtype)
        base = last[:, :, None, :] + t_future * slope[:, :, None, :]
        base = base[:, None].expand(-1, modes, -1, -1, -1).clone()
        if add_mode_fan and modes > 1:
            angles = torch.linspace(-math.pi / 3.0, math.pi / 3.0, modes, device=device, dtype=dtype)
            fan = torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1).view(1, modes, 1, 1, 2)
            # growth is proportional to absolute future offset, so it matches whether we evaluate
            # at every step or only at knot times.
            growth = (t_future / max(self.future_horizon, 1)).view(1, 1, 1, num_points, 1)
            base = base + self.prior_fan_magnitude * growth * fan
        base = base.clamp(-1.5, 1.5)
        return self.normalizer.normalize(base)

    def _prediction_prior(self, batch, agents, num_points, device, dtype):
        if self.use_prior_correction and self.trajectory_dim == 2 and "box_history" in batch:
            return self._constant_velocity_prior(batch, agents, num_points, self.num_modes, device, dtype, add_mode_fan=False)
        return None

    def _radial_residual_offsets(self, batch_size, agents, num_points, device, dtype):
        shape = (batch_size, self.num_modes, agents, num_points, self.trajectory_dim)
        if self.trajectory_dim != 2 or self.num_modes < 1:
            return torch.zeros(shape, device=device, dtype=dtype)

        if self.radial_include_center_mode and self.num_modes > 1:
            ring_modes = self.num_modes - 1
            angles = 2.0 * math.pi * torch.arange(ring_modes, device=device, dtype=dtype) / max(ring_modes, 1)
            ring = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
            directions = torch.cat([torch.zeros(1, 2, device=device, dtype=dtype), ring], dim=0)
        else:
            angles = 2.0 * math.pi * torch.arange(self.num_modes, device=device, dtype=dtype) / max(self.num_modes, 1)
            directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

        growth = self._future_time_array(num_points, device, dtype) / max(float(self.future_horizon), 1.0)
        if self.radial_radius_growth == "sqrt":
            growth = growth.clamp_min(0.0).sqrt()
        growth = growth.view(1, 1, 1, num_points, 1)
        offsets = self.radial_mode_radius * growth * directions.view(1, self.num_modes, 1, 1, 2)
        return offsets.expand(shape)

    def _apply_prediction_prior(self, params, prior):
        if prior is not None:
            params.mu = params.mu + prior
        return params

    def _initial_noise(self, batch, batch_size, agents, num_points, device, sigma0, dtype, prior=None):
        shape = (batch_size, self.num_modes, agents, num_points, self.trajectory_dim)
        if prior is not None:
            if self.initial_noise_type == "radial_control_points":
                offsets = self._radial_residual_offsets(batch_size, agents, num_points, device, dtype)
                jitter = self.radial_jitter_scale * torch.randn(shape, device=device, dtype=dtype)
                gaussian = sigma0 * self.radial_gaussian_scale * torch.randn(shape, device=device, dtype=dtype)
                return offsets + jitter + gaussian
            return self.prior_residual_scale * torch.randn(shape, device=device, dtype=dtype) + sigma0 * torch.randn(shape, device=device, dtype=dtype)
        if self.use_prior_initialization and self.trajectory_dim == 2 and "box_history" in batch:
            init_prior = self._constant_velocity_prior(batch, agents, num_points, self.num_modes, device, dtype, add_mode_fan=True)
            if self.initial_noise_type == "radial_control_points":
                offsets = self._radial_residual_offsets(batch_size, agents, num_points, device, dtype)
                jitter = self.radial_jitter_scale * torch.randn(shape, device=device, dtype=dtype)
                gaussian = sigma0 * self.radial_gaussian_scale * torch.randn(shape, device=device, dtype=dtype)
                return init_prior + offsets + jitter + gaussian
            return init_prior + self.prior_residual_scale * torch.randn(shape, device=device, dtype=dtype) + sigma0 * torch.randn(shape, device=device, dtype=dtype)
        return torch.randn(shape, device=device, dtype=dtype) * sigma0

    def _noise_frame_params(self, x, denormalize, prior=None):
        mu = self._expand_mu(x)
        if prior is not None:
            mu = mu + self._expand_mu(prior)
        if denormalize:
            mu = self.normalizer.denormalize(mu)
        mode_logits = x.new_zeros((x.shape[0], x.shape[1]))
        mode_probs = F.softmax(mode_logits, dim=1)
        chol = mu.new_zeros((*mu.shape[:-1], 2, 2))
        chol[..., 0, 0] = 0.01
        chol[..., 1, 1] = 0.01
        cov_raw = mu.new_zeros((*mu.shape[:-1], 3))
        return GMMParams(
            mode_logits=mode_logits,
            mode_probs=mode_probs,
            mu=mu,
            cov_cholesky=chol,
            cov_raw=cov_raw,
        )

    def _sampler_step(self, x, context, sigma, sigma_next, self_cond=None):
        sigma_batch = sigma.expand(x.shape[0])
        x_clean, hidden = self.denoise(x, context, sigma_batch, self_cond=self_cond)
        if sigma_next == 0 or self.sampler_type == "euler":
            if sigma_next == 0:
                return x_clean, hidden, x_clean.detach()
            d = (x - x_clean) / sigma.clamp_min(1e-8)
            return x + (sigma_next - sigma) * d, hidden, x_clean.detach()

        d = (x - x_clean) / sigma.clamp_min(1e-8)
        x_euler = x + (sigma_next - sigma) * d
        x_clean_next, hidden_next = self.denoise(
            x_euler,
            context,
            sigma_next.expand(x.shape[0]),
            self_cond=x_clean.detach() if self.use_self_conditioning else None,
        )
        d_next = (x_euler - x_clean_next) / sigma_next.clamp_min(1e-8)
        x_next = x + (sigma_next - sigma) * 0.5 * (d + d_next)
        return x_next, hidden_next, x_clean_next.detach()

    def _sample_params(self, batch, num_sampling_steps=None, denormalize=True, capture_steps=False, capture_initial_noise=False):
        context = self.encode_context(batch)
        trajectory = batch.get("trajectory")
        if trajectory is not None:
            batch_size, agents, _, _ = trajectory.shape
        else:
            batch_size = batch["features"].shape[0]
            agents = batch["features"].shape[1]
        num_points = self.num_points
        steps = max(int(num_sampling_steps or self.num_sampling_steps), 1)
        sigmas = make_karras_sigmas(steps, self.sigma_min, self.sigma_max, self.rho, context.device)
        prior = self._prediction_prior(batch, agents, num_points, context.device, context.dtype)
        x = self._initial_noise(batch, batch_size, agents, num_points, context.device, sigmas[0], context.dtype, prior=prior)
        hidden = None
        self_cond = None
        frames = []
        if capture_steps and capture_initial_noise:
            frames.append({
                "sigma": float(sigmas[0].detach().cpu()),
                "params": self._noise_frame_params(x, denormalize, prior=prior),
                "kind": "initial_noise",
            })
        for i in range(len(sigmas) - 1):
            x, hidden, self_cond = self._sampler_step(x, context, sigmas[i], sigmas[i + 1], self_cond=self_cond)
            if capture_steps and hidden is not None:
                params_i = self._apply_prediction_prior(self.gmm_head(self_cond, hidden), prior)
                params_i = self._expand_params(params_i)
                if denormalize:
                    params_i.mu = self.normalizer.denormalize(params_i.mu)
                frames.append({"sigma": float(sigmas[i + 1].detach().cpu()), "params": params_i, "kind": "denoised"})
        params = self._apply_prediction_prior(self.gmm_head(x, hidden), prior)
        params = self._expand_params(params)
        if denormalize:
            params.mu = self.normalizer.denormalize(params.mu)
        return (params, frames) if capture_steps else params

    def forward(self, batch, num_sampling_steps=None, capture_steps=False, capture_initial_noise=False):
        return self._sample_params(
            batch,
            num_sampling_steps=num_sampling_steps,
            denormalize=True,
            capture_steps=capture_steps,
            capture_initial_noise=capture_initial_noise,
        )

    @torch.no_grad()
    def _make_rollout_state(self, y_modes, context, num_steps=None):
        batch_size = y_modes.shape[0]
        sigma_start = sample_training_sigmas(
            batch_size,
            self.p_mean,
            self.p_std,
            self.sigma_min_train,
            self.sigma_max_train,
            y_modes.device,
        )
        x = y_modes + self._broadcast_sigma(sigma_start) * torch.randn_like(y_modes)
        steps = int(num_steps or torch.randint(self.rollout_min_steps, self.rollout_max_steps + 1, (), device=y_modes.device).item())
        sigma_hi = float(sigma_start.max().item())
        sigmas = make_karras_sigmas(max(steps, 1) + 1, self.sigma_min, max(sigma_hi, self.sigma_min), self.rho, y_modes.device)
        self_cond = None
        for i in range(len(sigmas) - 1):
            x, _, self_cond = self._sampler_step(x, context, sigmas[i], sigmas[i + 1], self_cond=self_cond)
        return x.detach(), sigmas[-1].expand(batch_size)

    def compute_loss(self, batch):
        context = self.encode_context(batch)
        y = self.normalizer.normalize(batch["trajectory"])
        mask = batch.get("future_mask")
        if mask is None:
            mask = torch.ones(y.shape[:-1], device=y.device, dtype=y.dtype)
        mask = mask.to(device=y.device, dtype=y.dtype)
        batch_size, agents, horizon, _ = y.shape
        num_points = self.num_points
        if self.use_control_points:
            y_points = y.index_select(-2, self.knot_indices)
            mask_points = mask.index_select(-1, self.knot_indices)
        else:
            y_points = y
            mask_points = mask
        sigma = sample_training_sigmas(batch_size, self.p_mean, self.p_std, self.sigma_min_train, self.sigma_max_train, y.device)
        y_modes = y_points[:, None].expand(batch_size, self.num_modes, agents, num_points, self.trajectory_dim)
        prior = self._prediction_prior(batch, agents, num_points, y.device, y.dtype)
        denoise_target = y_modes - prior if prior is not None else y_modes
        if prior is not None:
            if self.initial_noise_type == "radial_control_points" and self.radial_train_noise_prob > 0 and torch.rand((), device=y.device) < self.radial_train_noise_prob:
                radial = self._radial_residual_offsets(batch_size, agents, num_points, y.device, y.dtype)
                jitter = self.radial_jitter_scale * torch.randn_like(y_modes)
                x_noisy = radial + jitter + self._broadcast_sigma(sigma) * self.radial_gaussian_scale * torch.randn_like(y_modes)
            else:
                x_noisy = denoise_target + self._broadcast_sigma(sigma) * torch.randn_like(y_modes)
        elif self.prior_noise_prob > 0 and torch.rand((), device=y.device) < self.prior_noise_prob and "box_history" in batch:
            init_prior = self._constant_velocity_prior(batch, agents, num_points, self.num_modes, y.device, y.dtype, add_mode_fan=True)
            x_noisy = init_prior + self._broadcast_sigma(sigma) * torch.randn_like(y_modes)
        else:
            x_noisy = y_modes + self._broadcast_sigma(sigma) * torch.randn_like(y_modes)

        self_cond = None
        if self.use_self_conditioning and torch.rand((), device=y.device) < self.self_condition_prob:
            with torch.no_grad():
                self_cond, _ = self.denoise(x_noisy, context, sigma, self_cond=None)
                self_cond = self_cond.detach()
        denoised, hidden = self.denoise(x_noisy, context, sigma, self_cond=self_cond)

        mask_points_modes = mask_points[:, None, :, :, None]
        diff = F.smooth_l1_loss(denoised, denoise_target, reduction="none")
        if self.use_edm_loss_weighting:
            weight = (sigma.square() + self.sigma_data**2) / ((sigma * self.sigma_data).square().clamp_min(1e-8))
            diff = diff * weight.view(batch_size, 1, 1, 1, 1)
        loss_diff = (diff * mask_points_modes).sum() / mask_points_modes.sum().clamp_min(1.0) / self.trajectory_dim

        loss_rollout_diff = y.new_tensor(0.0)
        if self.rollout_loss_prob > 0 and torch.rand((), device=y.device) < self.rollout_loss_prob:
            x_rollout, sigma_rollout = self._make_rollout_state(denoise_target, context)
            denoised_rollout, _ = self.denoise(x_rollout, context, sigma_rollout)
            rollout_diff = F.smooth_l1_loss(denoised_rollout, denoise_target, reduction="none")
            loss_rollout_diff = (rollout_diff * mask_points_modes).sum() / mask_points_modes.sum().clamp_min(1.0) / self.trajectory_dim
        loss_diff_total = loss_diff + self.lambda_rollout_diff * loss_rollout_diff

        params_knots = self._apply_prediction_prior(self.gmm_head(denoised, hidden), prior)
        params = self._expand_params(params_knots)
        prior_full = self._expand_mu(prior) if prior is not None else None
        loss_nll = self._gmm_nll(params, y, mask)
        loss_minade, loss_minfde, best_mode = self._minade_minfde(params.mu, y, mask)
        loss_wta_diff = self._winner_take_all_diff(denoised, denoise_target, mask_points, best_mode, sigma=sigma)
        loss_mode = F.cross_entropy(params.mode_logits, best_mode)
        loss_mode_margin = self._mode_margin_loss(params.mode_logits, best_mode)
        loss_cov = self._covariance_regularization(params_knots.cov_raw)
        loss_bounds = self._image_bounds_loss(params.mu, mask)
        loss_smooth = self._smoothness_loss(params.mu, mask)
        loss_accel = self._acceleration_loss(params.mu, mask)
        loss_jerk = self._jerk_loss(params.mu, mask)
        loss_prior_residual = self._prior_residual_loss(params.mu, prior_full, mask)
        loss_speed = self._speed_cap_loss(params.mu, mask)
        loss_entropy = self._entropy_loss(params.mode_probs)
        loss_diversity = self._diversity_loss(params.mu, mask, batch)

        loss = (
            self.lambda_diff * loss_diff_total
            + self.lambda_wta_diff * loss_wta_diff
            + self.lambda_nll * loss_nll
            + self.lambda_ade * loss_minade
            + self.lambda_fde * loss_minfde
            + self.lambda_mode * loss_mode
            + self.lambda_mode_margin * loss_mode_margin
            + self.lambda_cov * loss_cov
            + self.lambda_bounds * loss_bounds
            + self.lambda_smooth * loss_smooth
            + self.lambda_accel * loss_accel
            + self.lambda_jerk * loss_jerk
            + self.lambda_prior_residual * loss_prior_residual
            + self.lambda_speed * loss_speed
            + self.lambda_entropy * loss_entropy
            + self.lambda_diversity * loss_diversity
        )
        entropy = -(params.mode_probs * params.mode_probs.clamp_min(1e-8).log()).sum(dim=1)
        return {
            "loss": loss,
            "loss_diff": loss_diff.detach(),
            "loss_wta_diff": loss_wta_diff.detach(),
            "loss_rollout_diff": loss_rollout_diff.detach(),
            "loss_nll": loss_nll.detach(),
            "loss_minade": loss_minade.detach(),
            "loss_minfde": loss_minfde.detach(),
            "loss_mode": loss_mode.detach(),
            "loss_mode_margin": loss_mode_margin.detach(),
            "loss_cov": loss_cov.detach(),
            "loss_bounds": loss_bounds.detach(),
            "loss_smooth": loss_smooth.detach(),
            "loss_accel": loss_accel.detach(),
            "loss_jerk": loss_jerk.detach(),
            "loss_prior_residual": loss_prior_residual.detach(),
            "loss_speed": loss_speed.detach(),
            "loss_entropy": loss_entropy.detach(),
            "loss_diversity": loss_diversity.detach(),
            "mode_entropy": entropy.mean().detach(),
            "sigma_mean": sigma.mean().detach(),
            "sigma_max": sigma.max().detach(),
        }

    @torch.no_grad()
    def evaluate_monotonicity(self, batch, step_counts=None, repeats=1, seed=None):
        if seed is not None:
            torch.manual_seed(int(seed))
        step_counts = [int(step) for step in (step_counts or self.monotonicity_eval_steps)]
        y = self.normalizer.normalize(batch["trajectory"])
        mask = batch.get("future_mask")
        if mask is None:
            mask = torch.ones(y.shape[:-1], device=y.device, dtype=y.dtype)
        mask = mask.to(device=y.device, dtype=y.dtype)
        minade_means, minfde_means, nll_means = [], [], []
        for steps in step_counts:
            ade_vals, fde_vals, nll_vals = [], [], []
            for _ in range(max(int(repeats), 1)):
                params = self._sample_params(batch, num_sampling_steps=steps, denormalize=False)
                minade, minfde, _ = self._minade_minfde(params.mu, y, mask)
                ade_vals.append(float(minade.cpu()))
                fde_vals.append(float(minfde.cpu()))
                nll_vals.append(float(self._gmm_nll(params, y, mask).cpu()))
            minade_means.append(float(torch.tensor(ade_vals).mean()))
            minfde_means.append(float(torch.tensor(fde_vals).mean()))
            nll_means.append(float(torch.tensor(nll_vals).mean()))
        tol = self.monotonicity_tolerance
        minade_violations = sum(curr > prev + tol for prev, curr in zip(minade_means[:-1], minade_means[1:]))
        minfde_violations = sum(curr > prev + tol for prev, curr in zip(minfde_means[:-1], minfde_means[1:]))
        return {
            "steps": step_counts,
            "nfe": [2 * step - 1 if self.sampler_type == "heun" else step for step in step_counts],
            "minade_mean": minade_means,
            "minfde_mean": minfde_means,
            "nll_mean": nll_means,
            "monotonic_minade_violations": int(minade_violations),
            "monotonic_minfde_violations": int(minfde_violations),
            "best_minade_step": int(step_counts[min(range(len(minade_means)), key=minade_means.__getitem__)]),
            "best_minfde_step": int(step_counts[min(range(len(minfde_means)), key=minfde_means.__getitem__)]),
            "repeats": int(repeats),
        }

    def _gmm_nll(self, params, y, mask):
        y = y[:, None]
        residual = y - params.mu
        l11 = params.cov_cholesky[..., 0, 0].clamp_min(1e-8)
        l21 = params.cov_cholesky[..., 1, 0]
        l22 = params.cov_cholesky[..., 1, 1].clamp_min(1e-8)
        z0 = residual[..., 0] / l11
        z1 = (residual[..., 1] - l21 * z0) / l22
        mahalanobis = z0.square() + z1.square()
        logdet = 2.0 * (torch.log(l11) + torch.log(l22))
        log_prob = -0.5 * (mahalanobis + logdet + 2.0 * math.log(2.0 * math.pi))
        log_prob = (log_prob * mask[:, None]).sum(dim=(2, 3))
        log_mix = F.log_softmax(params.mode_logits, dim=1) + log_prob
        return -torch.logsumexp(log_mix, dim=1).mean()

    def _minade_minfde(self, mu, y, mask):
        # calculate squared error for trajectory
        _, modes, agents, timesteps, _ = mu.shape
        distances = torch.linalg.norm(mu - y[:, None], dim=-1)**2 # [B K A T 2] -> [B K A T]
        mask_modes = mask[:, None]
        # calculate average distance per mode
        avg_ade = (distances * mask_modes).mean(dim=-1) # [B K A]
        
        # find best mode per agent
        minade, best_mode = avg_ade.min(dim=1) #  [B, A]
        
        # final distance
        fde = distances[:, :, :, -1] # [B K A]
        minfde = fde.min(dim=1)[0] # [B A]
        
        #FIXME: squeeze is because we only have one agent
        return minade, minfde, best_mode.squeeze(1)

    def _winner_take_all_diff(self, denoised, denoise_target, mask_points, best_mode, sigma=None):
        batch_size, _, agents, num_points, dim = denoised.shape
        gather_idx = best_mode.view(batch_size, 1, agents, 1, 1).expand(-1, 1, agents, num_points, dim)
        pred = denoised.gather(dim=1, index=gather_idx).squeeze(1)
        target = denoise_target.gather(dim=1, index=gather_idx).squeeze(1)
        diff = F.smooth_l1_loss(pred, target, reduction="none")
        if self.use_edm_loss_weighting and sigma is not None:
            weight = (sigma.square() + self.sigma_data**2) / ((sigma * self.sigma_data).square().clamp_min(1e-8))
            diff = diff * weight.view(batch_size, 1, 1, 1)
        mask_expanded = mask_points[..., None]
        return (diff * mask_expanded).sum() / mask_expanded.sum().clamp_min(1.0) / dim

    def _mode_margin_loss(self, mode_logits, best_mode):
        if mode_logits.shape[1] < 2:
            return mode_logits.new_tensor(0.0)
        best_logits = mode_logits.gather(1, best_mode[:, None]).squeeze(1)
        other_logits = mode_logits.masked_fill(F.one_hot(best_mode, mode_logits.shape[1]).bool(), -torch.inf)
        max_other = other_logits.max(dim=1).values
        return F.relu(self.mode_margin - (best_logits - max_other)).mean()

    def _covariance_regularization(self, cov_raw):
        log_diag = torch.stack([cov_raw[..., 0], cov_raw[..., 2]], dim=-1)
        low = F.relu(_get(self.config, "log_std_min", -5.0) - log_diag).square()
        high = F.relu(log_diag - _get(self.config, "log_std_max", 3.0)).square()
        return (low + high).mean()

    def _image_bounds_loss(self, mu, mask):
        excess = F.relu(mu.abs() - 1.0).square().sum(dim=-1)
        return (excess * mask[:, None]).sum() / mask[:, None].sum().clamp_min(1.0)

    def _smoothness_loss(self, mu, mask):
        return self._acceleration_loss(mu, mask)

    def _acceleration_loss(self, mu, mask):
        if mu.shape[3] < 3:
            return mu.new_tensor(0.0)
        accel = mu[:, :, :, 2:] - 2 * mu[:, :, :, 1:-1] + mu[:, :, :, :-2]
        valid = mask[:, None, :, 2:] * mask[:, None, :, 1:-1] * mask[:, None, :, :-2]
        return (accel.square().sum(dim=-1) * valid).sum() / valid.sum().clamp_min(1.0)

    def _jerk_loss(self, mu, mask):
        if mu.shape[3] < 4:
            return mu.new_tensor(0.0)
        jerk = mu[:, :, :, 3:] - 3 * mu[:, :, :, 2:-1] + 3 * mu[:, :, :, 1:-2] - mu[:, :, :, :-3]
        valid = mask[:, None, :, 3:] * mask[:, None, :, 2:-1] * mask[:, None, :, 1:-2] * mask[:, None, :, :-3]
        return (jerk.square().sum(dim=-1) * valid).sum() / valid.sum().clamp_min(1.0)

    def _prior_residual_loss(self, mu, prior, mask):
        if prior is None:
            return mu.new_tensor(0.0)
        residual = mu - prior
        return (residual.square().sum(dim=-1) * mask[:, None]).sum() / mask[:, None].sum().clamp_min(1.0)

    def _speed_cap_loss(self, mu, mask):
        if mu.shape[3] < 2:
            return mu.new_tensor(0.0)
        speed = torch.linalg.norm(mu[:, :, :, 1:] - mu[:, :, :, :-1], dim=-1)
        valid = mask[:, None, :, 1:] * mask[:, None, :, :-1]
        return (F.relu(speed - self.speed_cap).square() * valid).sum() / valid.sum().clamp_min(1.0)

    def _entropy_loss(self, mode_probs):
        entropy = -(mode_probs * mode_probs.clamp_min(1e-8).log()).sum(dim=1)
        return -entropy.mean()

    def _diversity_effective_margin(self, batch, mu, mask):
        base_margin = mu.new_full(mask.shape[:2], self.diversity_margin)
        boxes = batch.get("box_history", batch.get("features")) if batch is not None else None
        if boxes is None or boxes.shape[-1] < 4:
            return base_margin

        boxes = boxes.to(device=mu.device, dtype=mu.dtype)
        observed_mask = batch.get("observed_mask") if batch is not None else None
        if observed_mask is None:
            observed_mask = torch.ones(boxes.shape[:-1], device=mu.device, dtype=mu.dtype)
        else:
            observed_mask = observed_mask.to(device=mu.device, dtype=mu.dtype)

        valid_counts = observed_mask.sum(dim=-1).long().clamp_min(1)
        last_idx = (valid_counts - 1).view(boxes.shape[0], boxes.shape[1], 1, 1).expand(-1, -1, 1, boxes.shape[-1])
        last_box = boxes.gather(dim=2, index=last_idx).squeeze(2)

        box_size = torch.sqrt((last_box[..., 2].clamp_min(1e-4) * last_box[..., 3].clamp_min(1e-4)).clamp_min(1e-8))
        size_scale = box_size / max(self.diversity_reference_box_size, 1e-6)

        y01 = ((last_box[..., 1].clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)
        y_scale = self.diversity_y_top_scale + (self.diversity_y_bottom_scale - self.diversity_y_top_scale) * y01

        scale = size_scale.clamp_min(1e-6).pow(self.diversity_size_exponent) * y_scale.clamp_min(1e-6).pow(self.diversity_y_exponent)
        scale = scale.clamp(self.diversity_min_scale, self.diversity_max_scale)
        return base_margin * scale

    def _diversity_loss(self, mu, mask, batch=None):
        modes = mu.shape[1]
        if modes < 2:
            return mu.new_tensor(0.0)
        upper = torch.triu_indices(modes, modes, offset=1, device=mu.device)
        pair_dist = torch.linalg.norm(mu[:, upper[0]] - mu[:, upper[1]], dim=-1)
        valid = mask[:, None]
        pair_ade = (pair_dist * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0)
        margin = self._diversity_effective_margin(batch, mu, mask)[:, None]
        agent_valid = (mask.sum(dim=-1) > 0).to(dtype=mu.dtype)[:, None]
        loss = F.relu(margin - pair_ade) * agent_valid
        return loss.sum() / agent_valid.expand_as(loss).sum().clamp_min(1.0)
