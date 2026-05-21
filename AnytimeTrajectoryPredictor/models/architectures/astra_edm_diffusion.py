import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get(config, key, default=None):
    return getattr(config, key, default) if config is not None else default


def _as_float_tensor(value, default, size):
    if value is None:
        value = default
    tensor = torch.tensor(value, dtype=torch.float32)
    if tensor.numel() == 1:
        tensor = tensor.repeat(size)
    return tensor.view(1, 1, 1, size)


def wrap_angle(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def make_karras_sigmas(num_steps, sigma_min, sigma_max, rho, device):
    ramp = torch.linspace(0, 1, num_steps, device=device)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat([sigmas, torch.zeros(1, device=device)])


def sample_training_sigmas(batch_size, p_mean, p_std, sigma_min, sigma_max, device):
    sigmas = torch.exp(p_mean + p_std * torch.randn(batch_size, device=device))
    return sigmas.clamp(min=sigma_min, max=sigma_max)


def fourier_embedding(x, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=x.device) / max(half - 1, 1)
    )
    args = x[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, 1))
    return emb


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

        key_padding_mask = None
        if observed_mask is not None:
            key_padding_mask = observed_mask.reshape(batch, agents * history) <= 0

        tokens = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        return self.norm(tokens)


class DiffusionTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.self_norm = nn.LayerNorm(d_model)
        self.cross_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, context):
        x = x + self.self_attn(self.self_norm(x), self.self_norm(x), self.self_norm(x), need_weights=False)[0]
        # we care only about the output, not the attention weights
        x = x + self.cross_attn(self.cross_norm(x), context, context, need_weights=False)[0]
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DiffusionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_modes = _get(config, "num_modes_K", _get(config, "num_trajectory_possibilities", 6))
        self.trajectory_dim = _get(config, "trajectory_dim_dy", 2)
        self.d_model = _get(config, "d_model", 256)
        self.max_agents = _get(config, "max_agents", 1)
        self.future_horizon = _get(config, "future_horizon_T", 80)

        num_layers = _get(config, "num_layers", 4)
        num_heads = _get(config, "num_heads", 8)
        dropout = _get(config, "dropout", 0.1)

        self.trajectory_projection = nn.Linear(self.trajectory_dim, self.d_model)
        self.noise_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.mode_embedding = nn.Embedding(self.num_modes, self.d_model)
        self.agent_embedding = nn.Embedding(self.max_agents, self.d_model)
        self.future_time_embedding = nn.Embedding(self.future_horizon, self.d_model)
        self.blocks = nn.ModuleList(
            [DiffusionTransformerBlock(self.d_model, num_heads, dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(self.d_model)
        self.denoise_head = nn.Linear(self.d_model, self.trajectory_dim)

    def forward(self, x, context, noise_level):
        batch, modes, agents, horizon, _ = x.shape
        tokens = self.trajectory_projection(x)

        mode_ids = torch.arange(modes, device=x.device).view(1, modes, 1, 1)
        agent_ids = torch.arange(agents, device=x.device).view(1, 1, agents, 1)
        time_ids = torch.arange(horizon, device=x.device).view(1, 1, 1, horizon)
        noise_emb = self.noise_mlp(fourier_embedding(noise_level, self.d_model))

        tokens = (
            tokens
            + self.mode_embedding(mode_ids)
            + self.agent_embedding(agent_ids)
            + self.future_time_embedding(time_ids)
            + noise_emb.view(batch, 1, 1, 1, self.d_model)
        )
        tokens = tokens.reshape(batch, modes * agents * horizon, self.d_model)

        for block in self.blocks:
            tokens = block(tokens, context)

        hidden = self.final_norm(tokens)
        denoise = self.denoise_head(hidden).reshape(batch, modes, agents, horizon, -1)
        return denoise, hidden.reshape(batch, modes, agents, horizon, self.d_model)


class GMMHead(nn.Module):
    def __init__(self, d_model, trajectory_dim, log_std_min, log_std_max, offdiag_clip):
        super().__init__()
        if trajectory_dim != 2:
            raise ValueError("The first ASTRA-EDM implementation supports 2D trajectories.")
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.offdiag_clip = offdiag_clip
        self.mode_logits_head = nn.Linear(d_model, 1)
        self.mean_delta_head = nn.Linear(d_model, trajectory_dim)
        self.cov_head = nn.Linear(d_model, 3)

    def forward(self, denoised, hidden):
        mode_hidden = hidden.mean(dim=(2, 3))
        mode_logits = self.mode_logits_head(mode_hidden).squeeze(-1)
        
        # modes are denoised output trajectory plus a delta predicted from the hidden state
        delta_mu = self.mean_delta_head(hidden)
        mu = denoised + delta_mu

        # clip the covariance matrix so it does not explode, construct cholesky
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
        self.use_edm_loss_weighting = _get(config, "use_edm_loss_weighting", False)

        self.lambda_diff = _get(config, "lambda_diff", 1.0)
        self.lambda_nll = _get(config, "lambda_nll", 0.1)
        self.lambda_ade = _get(config, "lambda_ade", 1.0)
        self.lambda_fde = _get(config, "lambda_fde", 1.0)
        self.lambda_mode = _get(config, "lambda_mode", 0.1)
        self.lambda_cov = _get(config, "lambda_cov", 0.01)

        context_layers = _get(config, "context_layers", 2)
        num_heads = _get(config, "num_heads", 8)
        dropout = _get(config, "dropout", 0.1)
        input_dim = _get(config, "input_dim", 6)

        #TODO: get actual trajectory normalization data
        self.normalizer = TrajectoryNormalizer(
            mean=_get(config, "trajectory_mean", [0.0, 0.0]),
            std=_get(config, "trajectory_std", [1.0, 1.0]),
            trajectory_dim=self.trajectory_dim,
        )
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
        observed = batch["features"]
        observed_mask = batch.get("observed_mask")
        return self.context_encoder(observed, observed_mask)

    def denoise(self, x, context, sigma):
        c_skip, c_out, c_in, c_noise = self._edm_coefficients(sigma)
        f_out, hidden = self.transformer(c_in * x, context, c_noise)
        return c_skip * x + c_out * f_out, hidden

    def forward(self, batch, num_sampling_steps=None):
        context = self.encode_context(batch)
        trajectory = batch.get("trajectory")
        if trajectory is not None:
            batch_size, agents, horizon, _ = trajectory.shape
        else:
            batch_size = batch["features"].shape[0]
            agents = batch["features"].shape[1]
            horizon = self.future_horizon

        steps = num_sampling_steps or self.num_sampling_steps
        sigmas = make_karras_sigmas(steps, self.sigma_min, self.sigma_max, self.rho, context.device)
        x = torch.randn(
            batch_size,
            self.num_modes,
            agents,
            horizon,
            self.trajectory_dim,
            device=context.device,
        ) * sigmas[0]

        hidden = None
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i].expand(batch_size)
            sigma_next = sigmas[i + 1]
            x_clean, hidden = self.denoise(x, context, sigma)
            if sigma_next == 0:
                x = x_clean
            else:
                derivative = (x - x_clean) / sigmas[i].clamp_min(1e-8)
                x = x + (sigma_next - sigmas[i]) * derivative

        params = self.gmm_head(x, hidden)
        params.mu = self.normalizer.denormalize(params.mu)
        return params

    def compute_loss(self, batch):
        context = self.encode_context(batch)
        y = self.normalizer.normalize(batch["trajectory"])
        mask = batch.get("future_mask")
        if mask is None:
            mask = torch.ones(y.shape[:-1], device=y.device, dtype=y.dtype)
        mask = mask.to(dtype=y.dtype)

        batch_size, agents, horizon, _ = y.shape
        sigma = sample_training_sigmas(
            batch_size,
            self.p_mean,
            self.p_std,
            self.sigma_min_train,
            self.sigma_max_train,
            y.device,
        )
        y_modes = y[:, None].expand(batch_size, self.num_modes, agents, horizon, self.trajectory_dim)
        x_noisy = y_modes + self._broadcast_sigma(sigma) * torch.randn_like(y_modes)
        denoised, hidden = self.denoise(x_noisy, context, sigma)
        params = self.gmm_head(denoised, hidden)

        mask_modes = mask[:, None, :, :, None]
        diff = F.smooth_l1_loss(denoised, y_modes, reduction="none")
        if self.use_edm_loss_weighting:
            weight = (sigma.square() + self.sigma_data**2) / (
                (sigma * self.sigma_data).square().clamp_min(1e-8)
            )
            diff = diff * weight.view(batch_size, 1, 1, 1, 1)
        loss_diff = (diff * mask_modes).sum() / mask_modes.sum().clamp_min(1.0) / self.trajectory_dim

        loss_nll = self._gmm_nll(params, y, mask)
        loss_minade, loss_minfde, best_mode = self._minade_minfde(params.mu, y, mask)
        loss_mode = F.cross_entropy(params.mode_logits, best_mode)
        loss_cov = self._covariance_regularization(params.cov_raw)

        loss = (
            self.lambda_diff * loss_diff
            + self.lambda_nll * loss_nll
            + self.lambda_ade * loss_minade
            + self.lambda_fde * loss_minfde
            + self.lambda_mode * loss_mode
            + self.lambda_cov * loss_cov
        )

        return {
            "loss": loss,
            "loss_diff": loss_diff.detach(),
            "loss_nll": loss_nll.detach(),
            "loss_minade": loss_minade.detach(),
            "loss_minfde": loss_minfde.detach(),
            "loss_mode": loss_mode.detach(),
            "loss_cov": loss_cov.detach(),
            "sigma_mean": sigma.mean().detach(),
            "sigma_max": sigma.max().detach(),
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
        distances = torch.linalg.norm(mu - y[:, None], dim=-1)
        mask_modes = mask[:, None]
        ade = (distances * mask_modes).sum(dim=(2, 3)) / mask_modes.sum(dim=(2, 3)).clamp_min(1.0)
        minade, best_mode = ade.min(dim=1)

        valid_counts = mask.long().sum(dim=-1).clamp_min(1)
        last_idx = (valid_counts - 1).view(mask.shape[0], 1, mask.shape[1], 1).expand(-1, mu.shape[1], -1, 1)
        final_dist = distances.gather(dim=3, index=last_idx).squeeze(-1)
        agent_valid = (mask.sum(dim=-1) > 0).to(distances.dtype)
        fde = (final_dist * agent_valid[:, None]).sum(dim=2) / agent_valid[:, None].sum(dim=2).clamp_min(1.0)
        minfde = fde.min(dim=1)[0]
        return minade.mean(), minfde.mean(), best_mode

    def _covariance_regularization(self, cov_raw):
        log_diag = torch.stack([cov_raw[..., 0], cov_raw[..., 2]], dim=-1)
        low = F.relu(_get(self.config, "log_std_min", -5.0) - log_diag).square()
        high = F.relu(log_diag - _get(self.config, "log_std_max", 3.0)).square()
        return (low + high).mean()
