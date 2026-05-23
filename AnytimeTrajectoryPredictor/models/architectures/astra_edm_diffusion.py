import torch
import torch.nn as nn

from AnytimeTrajectoryPredictor.models.architectures.DiT import DiT as DiffusionTransformer
from AnytimeTrajectoryPredictor.models.architectures.base_model import base_model


def _get(config, key, default=None):
    return getattr(config, key, default) if config is not None else default


def make_karras_sigmas(num_steps, sigma_min, sigma_max, rho, device):
    ramp = torch.linspace(0, 1, num_steps, device=device)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat([sigmas, torch.zeros(1, device=device)])


# to be used only if feature extractor is not available
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


class PolynomialGMMHead(nn.Module):
    """Emit the same raw polynomial GMM parameters as the other architectures."""

    def __init__(self, d_model, coeff_dim):
        super().__init__()
        self.coeff_dim = coeff_dim
        self.mean_delta_head = nn.Linear(d_model, coeff_dim)
        self.cov_head = nn.Linear(d_model, coeff_dim * coeff_dim)

    def forward(self, denoised, hidden):
        denoised = denoised.squeeze(dim=3)
        hidden = hidden.squeeze(dim=3)
        means = denoised + self.mean_delta_head(hidden)
        covs = self.cov_head(hidden)
        return torch.cat([means, covs], dim=-1)


class ASTRAEDMDiffusionModel(base_model):
    """ASTRA-style EDM predictor using the project's polynomial GMM contract."""

    def __init__(
        self,
        config=None,
        state_dim=None,
        num_trajectory_possibilities=None,
    ):
        input_dim = _get(config, "input_dim", state_dim if state_dim is not None else 196)
        num_modes = num_trajectory_possibilities or _get(
            config,
            "num_modes_K",
            _get(config, "num_trajectory_possibilities", 6),
        )
        super().__init__(
            state_dim=input_dim,
            num_trajectory_possibilities=num_modes,
            polynomial_degree=_get(config, "polynomial_degree", 3),
        )
        # The base class creates a linear baseline head. ASTRA owns its output head.
        del self.A

        self.config = config
        self.state_dim = input_dim
        self.num_modes = self.num_trajectory_possibilities
        self.trajectory_dim = self.coeff_dim
        self.max_history = _get(config, "history_steps_H", 128)
        self.max_agents = _get(config, "max_agents", 128)
        self.d_model = _get(config, "d_model", 256)
        self.future_horizon = 1
        self.sigma_data = _get(config, "sigma_data", 1.0)
        self.sigma_min = _get(config, "sigma_min", 0.01)
        self.sigma_max = _get(config, "sigma_max", 5.0)
        self.rho = _get(config, "rho", 7.0)
        self.num_sampling_steps = _get(config, "num_sampling_steps", 12)

        context_layers = _get(config, "context_layers", 2)
        self.num_layers = _get(config, "num_layers", 4)
        self.num_heads = _get(config, "num_heads", 8)
        self.dropout = _get(config, "dropout", 0.1)

        self.context_encoder = ASTRAStyleContextEncoder(
            input_dim=input_dim,
            d_model=self.d_model,
            num_layers=context_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            max_history=self.max_history,
            max_agents=self.max_agents,
        )
        self.transformer = DiffusionTransformer(self)
        self.gmm_head = PolynomialGMMHead(
            d_model=self.d_model,
            coeff_dim=self.coeff_dim,
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

    def _history_mask(self, object_mask, frame):
        if object_mask is None:
            return None

        mask = object_mask[: frame + 1]
        if mask.dim() == 4:
            if mask.shape[-1] != 1:
                raise ValueError("object_mask last dimension must be 1 when it is 4D")
            mask = mask.squeeze(-1)
        if mask.dim() != 3:
            raise ValueError("object_mask must have shape (T, B, N) or (T, B, N, 1)")

        mask = mask[-self.max_history :]
        return mask.permute(1, 2, 0)

    def encode_context(self, frames, frame, object_mask=None):
        history = frames[: frame + 1][-self.max_history :]
        observed = history.permute(1, 2, 0, 3)
        observed_mask = self._history_mask(object_mask, frame)
        return self.context_encoder(observed, observed_mask)

    def denoise(self, x, context, sigma):
        c_skip, c_out, c_in, c_noise = self._edm_coefficients(sigma)
        f_out, hidden = self.transformer(c_in * x, context, c_noise)
        return c_skip * x + c_out * f_out, hidden

    def _sample_frame(self, context, batch_size, num_objects, sampling_steps):
        steps = max(int(sampling_steps), 1)
        sigmas = make_karras_sigmas(
            steps,
            self.sigma_min,
            self.sigma_max,
            self.rho,
            context.device,
        )
        x = torch.randn(
            batch_size,
            self.num_modes,
            num_objects,
            1,
            self.coeff_dim,
            device=context.device,
            dtype=context.dtype,
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
        params = params.permute(0, 2, 1, 3)
        return params.reshape(batch_size, num_objects, self.output_dim)

    def forward(self, frames, f_, object_mask=None, hidden_state=None):
        """Return one polynomial GMM parameter tensor for each input frame."""
        del hidden_state
        if frames.dim() != 4:
            raise ValueError("frames must have shape (T, B, N, F)")

        num_frames, batch_size, num_objects, feature_dim = frames.shape
        if feature_dim != self.state_dim:
            raise ValueError(
                f"frames feature dimension must be {self.state_dim}, got {feature_dim}"
            )
        if len(f_) < num_frames:
            raise ValueError("f_ must contain a refinement budget for each frame")

        predictions = []
        for frame in range(num_frames):
            context = self.encode_context(frames, frame, object_mask)
            predictions.append(
                self._sample_frame(context, batch_size, num_objects, f_[frame])
            )
        return predictions


ASTRAEDM = ASTRAEDMDiffusionModel
