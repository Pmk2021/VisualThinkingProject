"""Mode diversity metrics for polynomial-coefficient GMM predictors."""

from __future__ import annotations

import torch


MIN_PAIRWISE_MODES = 2


def _extract_gmm_params(predictions, model):
    stacked = torch.stack(predictions, dim=0)
    return model.unpack_gmm_params(stacked)


def _spatial_coefficient_indices(model):
    indices = []
    for trajectory_dim in model.spatial_dims:
        start = trajectory_dim * model.num_coeffs
        stop = start + model.num_coeffs
        indices.extend(range(start, stop))
    return indices


def _psd_sqrt(mat, eps):
    """Symmetric PSD matrix square root via eigendecomposition."""
    mat = 0.5 * (mat + mat.transpose(-1, -2))
    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals = eigvals.clamp(min=eps)
    return eigvecs @ (eigvals.sqrt().unsqueeze(-1) * eigvecs.transpose(-1, -2))


def average_pairwise_distance(means, model):
    """Average pairwise distance between spatial polynomial mode coefficients."""
    num_modes = means.shape[-2]
    if num_modes < MIN_PAIRWISE_MODES:
        return torch.zeros((), device=means.device, dtype=means.dtype)

    spatial_indices = _spatial_coefficient_indices(model)
    selected_means = means[..., spatial_indices]
    diff = selected_means.unsqueeze(-2) - selected_means.unsqueeze(-3)
    pair_dist = torch.linalg.norm(diff, dim=-1)
    traj_dist = pair_dist.mean(dim=0)

    mode_i, mode_j = torch.triu_indices(
        num_modes,
        num_modes,
        offset=MIN_PAIRWISE_MODES - 1,
        device=means.device,
    )
    return traj_dist[..., mode_i, mode_j].mean()


def mean_pairwise_w2(means, covs, eps):
    """Mean pairwise 2-Wasserstein distance between coefficient Gaussians."""
    num_modes = means.shape[-2]
    if num_modes < MIN_PAIRWISE_MODES:
        return torch.zeros((), device=means.device, dtype=means.dtype)

    mode_i, mode_j = torch.triu_indices(
        num_modes,
        num_modes,
        offset=MIN_PAIRWISE_MODES - 1,
        device=means.device,
    )
    mu_i = means[..., mode_i, :]
    mu_j = means[..., mode_j, :]
    cov_i = covs[..., mode_i, :, :]
    cov_j = covs[..., mode_j, :, :]

    mean_sq = ((mu_i - mu_j) ** 2).sum(dim=-1)

    cov_i_half = _psd_sqrt(cov_i, eps)
    inner = cov_i_half @ cov_j @ cov_i_half
    inner_half = _psd_sqrt(inner, eps)
    tr_term = (
        torch.diagonal(cov_i, dim1=-2, dim2=-1).sum(-1)
        + torch.diagonal(cov_j, dim1=-2, dim2=-1).sum(-1)
        - MIN_PAIRWISE_MODES
        * torch.diagonal(inner_half, dim1=-2, dim2=-1).sum(-1)
    )

    w2_sq = (mean_sq + tr_term).clamp(min=0)
    return w2_sq.sqrt().mean()


def compute_diversity_metrics(predictions, model):
    """Compute APD and mean pairwise W2 for one batch of predictions."""
    if model.num_trajectory_possibilities < MIN_PAIRWISE_MODES or not predictions:
        return {"apd": 0.0, "mean_pairwise_w2": 0.0}

    means, covs = _extract_gmm_params(predictions, model)
    apd = average_pairwise_distance(means, model)
    w2 = mean_pairwise_w2(means, covs, model.COVARIANCE_EPS)
    return {"apd": apd.item(), "mean_pairwise_w2": w2.item()}
