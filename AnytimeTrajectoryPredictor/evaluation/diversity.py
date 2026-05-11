"""Mode diversity metrics for the GMM trajectory predictor.

The model emits, for every frame, ``num_trajectory_possibilities`` (= K) mixture
components, each parameterized by a 3D mean ``(dx, dy, d_log_area)`` and a 3x3
covariance factor. This module computes two diversity metrics:

* ``apd`` — Average Pairwise Distance between mode mean *trajectories* (xy by
  default). High APD means the modes propose geometrically different paths.
* ``mean_pairwise_w2`` — Average pairwise 2-Wasserstein distance between
  component Gaussians, summed per frame and averaged over time. Unlike APD, it
  is sensitive to covariance inflation: two modes whose means differ but whose
  covariances are huge and overlapping receive a smaller W2 value.

Parameter extraction is delegated to ``base_model.unpack_gmm_params`` so the
metrics and the training loss share the exact same parameterization.
"""

from __future__ import annotations

import torch

from AnytimeTrajectoryPredictor.models.architectures.base_model import base_model


_EPS_PSD_SQRT = 1e-6


def _extract_gmm_params(predictions, num_modes):
    """Stack the per-frame predictions and return (means, covs).

    Args:
        predictions: list of ``(B, N, K * 12)`` tensors, one per frame.
        num_modes: ``K``.

    Returns:
        means: ``(T, B, N, K, 3)`` tensor.
        covs:  ``(T, B, N, K, 3, 3)`` SPD covariance tensor.
    """
    stacked = torch.stack(predictions, dim=0)  # (T, B, N, K * 12)
    return base_model.unpack_gmm_params(stacked, num_modes)


def _psd_sqrt(mat):
    """Symmetric PSD matrix square root via eigendecomposition."""
    mat = 0.5 * (mat + mat.transpose(-1, -2))
    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals = eigvals.clamp(min=_EPS_PSD_SQRT)
    return eigvecs @ (eigvals.sqrt().unsqueeze(-1) * eigvecs.transpose(-1, -2))


def average_pairwise_distance(means, dims=(0, 1)):
    """Average pairwise distance between mode mean trajectories.

    Args:
        means: ``(T, B, N, K, 3)`` tensor of mode means.
        dims: which mean components to use (default ``(0, 1)`` = xy).

    Returns:
        Scalar tensor.
    """
    K = means.shape[-2]
    if K < 2:
        return torch.zeros((), device=means.device, dtype=means.dtype)

    m = means[..., list(dims)]  # (T, B, N, K, d)
    # Per-frame mode-to-mode Euclidean distance, then averaged over time gives a
    # trajectory-level distance between modes.
    diff = m.unsqueeze(-2) - m.unsqueeze(-3)  # (T, B, N, K, K, d)
    pair_dist = torch.linalg.norm(diff, dim=-1)  # (T, B, N, K, K)
    traj_dist = pair_dist.mean(dim=0)  # (B, N, K, K)

    iu, ju = torch.triu_indices(K, K, offset=1, device=means.device)
    return traj_dist[..., iu, ju].mean()


def mean_pairwise_w2(means, covs):
    """Mean pairwise 2-Wasserstein distance between mixture components.

    Computed per frame on the full 3D Gaussians, then averaged over time and
    batch.

    Args:
        means: ``(T, B, N, K, 3)`` tensor.
        covs:  ``(T, B, N, K, 3, 3)`` SPD covariance tensor.

    Returns:
        Scalar tensor.
    """
    K = means.shape[-2]
    if K < 2:
        return torch.zeros((), device=means.device, dtype=means.dtype)

    iu, ju = torch.triu_indices(K, K, offset=1, device=means.device)
    mu_i = means[..., iu, :]
    mu_j = means[..., ju, :]
    S_i = covs[..., iu, :, :]
    S_j = covs[..., ju, :, :]

    mean_sq = ((mu_i - mu_j) ** 2).sum(dim=-1)  # (T, B, N, P)

    Si_half = _psd_sqrt(S_i)
    inner = Si_half @ S_j @ Si_half
    inner_half = _psd_sqrt(inner)
    tr_term = (
        torch.diagonal(S_i, dim1=-2, dim2=-1).sum(-1)
        + torch.diagonal(S_j, dim1=-2, dim2=-1).sum(-1)
        - 2 * torch.diagonal(inner_half, dim1=-2, dim2=-1).sum(-1)
    )

    w2_sq = (mean_sq + tr_term).clamp(min=0)
    return w2_sq.sqrt().mean()


def compute_diversity_metrics(predictions, num_modes, apd_dims=(0, 1)):
    """Compute APD and mean pairwise W2 for one batch of predictions.

    Args:
        predictions: list of ``(B, N, K * 12)`` tensors (one per frame).
        num_modes: ``K``.
        apd_dims: components of the mean used for APD (default xy).

    Returns:
        ``{"apd": float, "mean_pairwise_w2": float}``.
    """
    if num_modes < 2 or not predictions:
        return {"apd": 0.0, "mean_pairwise_w2": 0.0}

    means, covs = _extract_gmm_params(predictions, num_modes)
    apd = average_pairwise_distance(means, dims=apd_dims)
    w2 = mean_pairwise_w2(means, covs)
    return {"apd": apd.item(), "mean_pairwise_w2": w2.item()}
