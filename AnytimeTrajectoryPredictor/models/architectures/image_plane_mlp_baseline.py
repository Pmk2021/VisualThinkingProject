import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from AnytimeTrajectoryPredictor.models.architectures.astra_edm_diffusion import GMMParams, TrajectoryNormalizer


def _get(config, key, default=None):
    return getattr(config, key, default) if config is not None else default


class ImagePlaneMLPGMMBaseline(nn.Module):
    """Small non-sampling GMM baseline for image-plane trajectory prediction.

    The model consumes the same batch dictionary as the image-plane diffusion model,
    but keeps the input deliberately cheap: box history, observed mask, camera/object
    IDs, and per-frame RGB mean/std summaries instead of a CNN feature map.
    """

    expects_batch = True
    uses_sampler = False

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_modes = int(_get(config, 'num_modes_K', _get(config, 'num_trajectory_possibilities', 6)))
        self.trajectory_dim = int(_get(config, 'trajectory_dim_dy', 2))
        if self.trajectory_dim != 2:
            raise ValueError('ImagePlaneMLPGMMBaseline currently supports 2D trajectories only.')
        self.future_horizon = int(_get(config, 'future_horizon_T', 80))
        self.max_history = int(_get(config, 'history_steps_H', 10))
        self.max_agents = int(_get(config, 'max_agents', 1))
        self.log_std_min = float(_get(config, 'log_std_min', -5.0))
        self.log_std_max = float(_get(config, 'log_std_max', 2.0))
        self.offdiag_clip = float(_get(config, 'offdiag_clip', 2.0))
        self.lambda_nll = float(_get(config, 'lambda_nll', 0.05))
        self.lambda_ade = float(_get(config, 'lambda_ade', 1.0))
        self.lambda_fde = float(_get(config, 'lambda_fde', 1.0))
        self.lambda_mode = float(_get(config, 'lambda_mode', 0.2))
        self.lambda_cov = float(_get(config, 'lambda_cov', 0.001))
        self.lambda_diversity = float(_get(config, 'lambda_diversity', 0.05))
        self.diversity_margin = float(_get(config, 'diversity_margin', 0.05))
        hidden_dim = int(_get(config, 'hidden_dim', _get(config, 'd_model', 256)))
        num_layers = max(int(_get(config, 'num_layers', 2)), 1)
        dropout = float(_get(config, 'dropout', 0.1))

        self.normalizer = TrajectoryNormalizer(
            mean=_get(config, 'trajectory_mean', [0.0, 0.0]),
            std=_get(config, 'trajectory_std', [1.0, 1.0]),
            trajectory_dim=self.trajectory_dim,
        )
        input_dim = self.max_agents * self.max_history * 5 + self.max_history * 6 + 2
        output_dim = self.num_modes + self.num_modes * self.max_agents * self.future_horizon * 5
        layers = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def _fixed_length(self, tensor, expected):
        if tensor.shape[1] == expected:
            return tensor
        if tensor.shape[1] > expected:
            return tensor[:, :expected]
        pad = tensor.new_zeros(tensor.shape[0], expected - tensor.shape[1])
        return torch.cat([tensor, pad], dim=1)

    def _build_input(self, batch):
        boxes = batch.get('box_history', batch.get('features'))
        if boxes is None:
            raise KeyError('ImagePlaneMLPGMMBaseline requires box_history or features in the batch.')
        boxes = boxes.to(dtype=torch.float32)
        batch_size = boxes.shape[0]
        observed_mask = batch.get('observed_mask')
        if observed_mask is None:
            observed_mask = torch.ones(boxes.shape[:-1], device=boxes.device, dtype=boxes.dtype)
        else:
            observed_mask = observed_mask.to(device=boxes.device, dtype=boxes.dtype)
        boxes = boxes * observed_mask.unsqueeze(-1)
        box_features = torch.cat([boxes, observed_mask.unsqueeze(-1)], dim=-1).reshape(batch_size, -1)
        box_features = self._fixed_length(box_features, self.max_agents * self.max_history * 5)

        rgb = batch.get('rgb_history')
        if rgb is not None:
            rgb = rgb.to(device=boxes.device, dtype=torch.float32)
            rgb_mean = rgb.mean(dim=(-1, -2))
            rgb_std = rgb.std(dim=(-1, -2), unbiased=False)
            rgb_features = torch.cat([rgb_mean, rgb_std], dim=-1).reshape(batch_size, -1)
        else:
            rgb_features = boxes.new_zeros(batch_size, 0)
        rgb_features = self._fixed_length(rgb_features, self.max_history * 6)

        camera = batch.get('camera_name')
        if camera is None:
            camera_feature = boxes.new_zeros(batch_size, 1)
        else:
            camera_feature = camera.to(device=boxes.device, dtype=boxes.dtype).view(batch_size, 1) / 7.0
        object_type = batch.get('object_type')
        if object_type is None:
            object_feature = boxes.new_zeros(batch_size, 1)
        else:
            object_feature = object_type.to(device=boxes.device, dtype=boxes.dtype).view(batch_size, 1) / 4.0
        return torch.cat([box_features, rgb_features, camera_feature, object_feature], dim=1)

    def _raw_to_params(self, raw):
        batch_size = raw.shape[0]
        mode_logits = raw[:, :self.num_modes]
        rest = raw[:, self.num_modes:].view(batch_size, self.num_modes, self.max_agents, self.future_horizon, 5)
        mu = rest[..., :2]
        cov_raw = rest[..., 2:]
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

    def _sample_params(self, batch, num_sampling_steps=None, denormalize=True, capture_steps=False, capture_initial_noise=False):
        params = self._raw_to_params(self.net(self._build_input(batch)))
        if denormalize:
            params.mu = self.normalizer.denormalize(params.mu)
        if capture_steps:
            return params, []
        return params

    def forward(self, batch, num_sampling_steps=None, denormalize=True, capture_steps=False, capture_initial_noise=False):
        return self._sample_params(
            batch,
            num_sampling_steps=num_sampling_steps,
            denormalize=denormalize,
            capture_steps=capture_steps,
            capture_initial_noise=capture_initial_noise,
        )

    def compute_loss(self, batch):
        y = self.normalizer.normalize(batch['trajectory'])
        mask = batch.get('future_mask')
        if mask is None:
            mask = torch.ones(y.shape[:-1], device=y.device, dtype=y.dtype)
        else:
            mask = mask.to(device=y.device, dtype=y.dtype)
        params = self._sample_params(batch, denormalize=False)
        loss_nll = self._gmm_nll(params, y, mask)
        loss_minade, loss_minfde, best_mode = self._minade_minfde(params.mu, y, mask)
        loss_mode = F.cross_entropy(params.mode_logits, best_mode)
        loss_cov = self._covariance_regularization(params.cov_raw)
        loss_diversity = self._diversity_loss(params.mu, mask)
        loss = (
            self.lambda_nll * loss_nll
            + self.lambda_ade * loss_minade
            + self.lambda_fde * loss_minfde
            + self.lambda_mode * loss_mode
            + self.lambda_cov * loss_cov
            + self.lambda_diversity * loss_diversity
        )
        entropy = -(params.mode_probs * params.mode_probs.clamp_min(1e-8).log()).sum(dim=1)
        return {
            'loss': loss,
            'loss_nll': loss_nll.detach(),
            'loss_minade': loss_minade.detach(),
            'loss_minfde': loss_minfde.detach(),
            'loss_mode': loss_mode.detach(),
            'loss_cov': loss_cov.detach(),
            'loss_diversity': loss_diversity.detach(),
            'mode_entropy': entropy.mean().detach(),
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
        minfde = fde.min(dim=1).values
        return minade.mean(), minfde.mean(), best_mode

    def _covariance_regularization(self, cov_raw):
        log_diag = torch.stack([cov_raw[..., 0], cov_raw[..., 2]], dim=-1)
        low = F.relu(self.log_std_min - log_diag).square()
        high = F.relu(log_diag - self.log_std_max).square()
        return (low + high).mean()

    def _diversity_loss(self, mu, mask):
        modes = mu.shape[1]
        if modes < 2:
            return mu.new_tensor(0.0)
        upper = torch.triu_indices(modes, modes, offset=1, device=mu.device)
        pair_dist = torch.linalg.norm(mu[:, upper[0]] - mu[:, upper[1]], dim=-1)
        pair_ade = (pair_dist * mask[:, None]).sum(dim=(2, 3)) / mask[:, None].sum(dim=(2, 3)).clamp_min(1.0)
        return F.relu(self.diversity_margin - pair_ade).mean()
