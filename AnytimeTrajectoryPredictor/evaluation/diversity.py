"""Mode diversity metrics for polynomial-coefficient GMM predictors."""

from __future__ import annotations

import torch

MIN_PAIRWISE_MODES = 2

def extract_gmm_params(model, predictions):
    """
    Extract mean and covariance parameters from model predictions.
    Params: 
    - model: The trajectory prediction model, used to access the output dimension and covariance stabilization method.
    - predictions: The raw output from the model, expected to be a list of length num_frames, each element of shape (B, num_objects, model.output_dim) representing the predicted parameters for each frame.
    Returns:
        - means: Tensor of shape (B, num_frames, num_objects, model.num_trajectory_possibilities, model.coeff_dim) containing the mean coefficients for each mode.
        - covs: Tensor of shape (B, num_frames, num_objects, model.num_trajectory_possibilities, model.coeff_dim, model.coeff_dim) containing the covariance matrices for each mode.
    """

    num_frames = len(predictions)
    B, num_objects, output_dim = predictions[0].shape
    K = model.num_trajectory_possibilities
    D = model.coeff_dim # Coefficient dimension (e.g., 12 for 3 dimensions with 4 coefficients each)

    # Initialize tensors to hold means and covariances
    means = torch.zeros(B, num_frames, num_objects, K, D, device=predictions[0].device)
    covs = torch.zeros(
        B,
        num_frames,
        num_objects,
        K,
        D,
        D,
        device=predictions[0].device,
    )

    for i in range(num_frames):
        frame_pred = predictions[i] # (B, num_objects, output_dim)
        frame_pred = frame_pred.view(
            B,
            num_objects,
            K,
            model.single_trajectory_param_size,
        ) # (B, num_objects, K, output_dim_per_mode)
        frame_means = frame_pred[..., :D] # (B, num_objects, K, D)
        frame_covs = frame_pred[..., D:] # (B, num_objects, K, D*D)
        frame_covs = frame_covs.view(
            B,
            num_objects,
            K,
            D,
            D,
        ) # (B, num_objects, K, D, D)
        means[:, i] = frame_means
        covs[:, i] = model.stabilize_covariance(frame_covs) # Ensure covariance matrices are well-conditioned

    return means, covs

def _spatial_coefficient_indices(model):
    """
    Get the indices of the spatial coefficients (e.g., x and y) in the output vector.
    """
    indices = []
    for trajectory_dim in model.spatial_indices:
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
    """
    Average pairwise distance between spatial polynomial mode coefficients.
        Params:
        - means: Tensor of shape (B, num_frames, num_objects, model.num_trajectory_possibilities, model.coeff_dim) containing the mean coefficients for each mode.
        - model: The trajectory prediction model, used to access the spatial coefficient indices.
        Returns:
        - A scalar tensor representing the average pairwise distance between the spatial coefficients of the modes.
    """
    num_modes = means.shape[-2]
    
    spatial_indices = _spatial_coefficient_indices(model)
    selected_means = means[..., spatial_indices] # (B, num_frames, num_objects, K, len(spatial_indices))
    diff = selected_means.unsqueeze(-2) - selected_means.unsqueeze(-3) # (B, num_frames, num_objects, K, K, len(spatial_indices))
    pair_dist = torch.linalg.norm(diff, dim=-1) # (B, num_frames, num_objects, K, K)
    
    # Only consider upper triangle of pairwise distance matrix to avoid double counting and self-distances
    mode_i, mode_j = torch.triu_indices(
        num_modes,
        num_modes,
        offset=1,
        device=means.device,
    )

    # Return the mean pairwise distance across all mode pairs, frames, objects, and batch elements
    return pair_dist[..., mode_i, mode_j].mean()


def mean_pairwise_w2(means, covs, eps):
    """
    Mean pairwise 2-Wasserstein distance between Gaussian trajectory modes.
        Params:
        - means: Tensor of shape (B, num_frames, num_objects, model.num_trajectory_possibilities, model.coeff_dim) containing mean coefficients for each mode.
        - covs: Tensor of shape (B, num_frames, num_objects, model.num_trajectory_possibilities, model.coeff_dim, model.coeff_dim) containing covariance matrices for each mode.
        - eps: Small value used to stabilize matrix square roots.
        Returns:
        - A scalar tensor representing the average W2 distance between Gaussian modes across all mode pairs, frames, objects, and batch elements.
    """
    num_modes = means.shape[-2]

    # Only consider upper triangle of pairwise mode matrix to avoid double counting and self-distances
    mode_i, mode_j = torch.triu_indices(
        num_modes,
        num_modes,
        offset=1,
        device=means.device,
    )

    mu_i = means[..., mode_i, :] # (B, num_frames, num_objects, num_pairs, D)
    mu_j = means[..., mode_j, :] # (B, num_frames, num_objects, num_pairs, D)
    cov_i = covs[..., mode_i, :, :] # (B, num_frames, num_objects, num_pairs, D, D)
    cov_j = covs[..., mode_j, :, :] # (B, num_frames, num_objects, num_pairs, D, D)

    mean_term = ((mu_i - mu_j) ** 2).sum(dim=-1) # (B, num_frames, num_objects, num_pairs)

    cov_i_half = _psd_sqrt(cov_i, eps)
    inner = cov_i_half @ cov_j @ cov_i_half
    inner_half = _psd_sqrt(inner, eps)

    trace_term = (
        torch.diagonal(cov_i, dim1=-2, dim2=-1).sum(-1)
        + torch.diagonal(cov_j, dim1=-2, dim2=-1).sum(-1)
        - 2
        * torch.diagonal(inner_half, dim1=-2, dim2=-1).sum(-1)
    )

    pairwise_w2 = (mean_term + trace_term).clamp(min=0).sqrt()
    return pairwise_w2.mean()


def compute_diversity_metrics(predictions, model):
    """Compute APD and mean pairwise W2 for one batch of predictions."""
    no_predictions = (
        predictions is None
        or (isinstance(predictions, (list, tuple)) and len(predictions) == 0)
        or (torch.is_tensor(predictions) and predictions.numel() == 0)
    )
    if model.num_trajectory_possibilities < MIN_PAIRWISE_MODES or no_predictions:
        return {"apd": 0.0, "mean_pairwise_w2": 0.0}

    means, covs = extract_gmm_params(model, predictions)
    apd = average_pairwise_distance(means, model)
    w2 = mean_pairwise_w2(means, covs, model.COVARIANCE_EPS)
    return {"apd": apd.item(), "mean_pairwise_w2": w2.item()}