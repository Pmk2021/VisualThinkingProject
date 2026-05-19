import torch
import torch.nn as nn


class base_model(nn.Module):
    """Linear baseline that predicts a GMM over polynomial coefficients."""

    COVARIANCE_EPS = 1e-2
    COVARIANCE_CLAMP = 20.0

    def __init__(
        self,
        state_dim,
        num_trajectory_possibilities,
        polynomial_degree=None,
        trajectory_dims=None,
        spatial_dims=None,
    ):
        super(base_model, self).__init__()

        self.state_dim = state_dim
        self.num_trajectory_possibilities = num_trajectory_possibilities
        self.polynomial_degree = self._require_config_value(
            polynomial_degree, "model.polynomial_degree"
        )
        self.trajectory_dims = self._require_config_value(
            trajectory_dims, "feature_extractor.trajectory_dims"
        )
        self.spatial_dims = self._normalize_spatial_dims(spatial_dims)

        self.num_coeffs = self.polynomial_degree + 1
        self.mean_dim = self.trajectory_dims * self.num_coeffs
        self.cov_params = self.mean_dim * self.mean_dim
        self.params_per_mode = self.mean_dim + self.cov_params
        self.output_dim = self.num_trajectory_possibilities * self.params_per_mode

        self.A = nn.Linear(self.state_dim, self.output_dim)

    @staticmethod
    def _require_config_value(value, name):
        if value is None:
            raise ValueError(f"{name} must be defined in the configuration file")
        return value

    def _normalize_spatial_dims(self, spatial_dims):
        if spatial_dims is None:
            raise ValueError("model.spatial_dims must be defined in the configuration file")

        normalized_dims = tuple(int(dim) for dim in spatial_dims)
        invalid_dims = [
            dim for dim in normalized_dims if dim < 0 or dim >= self.trajectory_dims
        ]
        if invalid_dims:
            raise ValueError(
                "model.spatial_dims contains values outside the configured "
                f"trajectory dimension range: {invalid_dims}"
            )
        return normalized_dims

    def unpack_gmm_params(self, output, num_modes=None):
        """Split raw model output into GMM means and SPD covariances."""
        if num_modes is None:
            num_modes = self.num_trajectory_possibilities

        expected_output_dim = num_modes * self.params_per_mode
        if output.shape[-1] != expected_output_dim:
            raise ValueError(
                "Last output dimension does not match the configured polynomial "
                f"GMM layout: got {output.shape[-1]}, expected {expected_output_dim}"
            )

        components = output.reshape(
            *output.shape[:-1],
            num_modes,
            self.params_per_mode,
        )
        means = components[..., : self.mean_dim]
        cov_factors = components[..., self.mean_dim :].reshape(
            *output.shape[:-1],
            num_modes,
            self.mean_dim,
            self.mean_dim,
        )

        cov_factors = torch.nan_to_num(
            cov_factors,
            nan=0.0,
            posinf=self.COVARIANCE_CLAMP,
            neginf=-self.COVARIANCE_CLAMP,
        )
        cov_factors = torch.clamp(
            cov_factors,
            min=-self.COVARIANCE_CLAMP,
            max=self.COVARIANCE_CLAMP,
        )
        cov_factors = 0.5 * (
            cov_factors + cov_factors.transpose(-1, -2)
        )
        eye = torch.eye(self.mean_dim, device=output.device, dtype=output.dtype)
        covs = (
            cov_factors @ cov_factors.transpose(-1, -2)
            + self.COVARIANCE_EPS * eye
        )

        return means, covs

    def forward(self, frames, f_, object_mask=None, hidden_state=None):
        """
        Given a sequence of frames, return one raw GMM output tensor per frame.

        Parameters:
            frames: input state features shaped
                ``(num_frames, batch_size, num_objects, feature_dim)``.
            f_: length-``num_frames`` list with refinement iterations per frame.
            object_mask: optional visibility mask shaped
                ``(num_frames, batch_size, num_objects)``.
            hidden_state: optional hidden state for recurrent subclasses.
        """
        num_frames, _, _, _ = frames.shape

        predicted_trajectory_list = []
        for frame_idx in range(num_frames):
            for iteration in range(f_[frame_idx]):
                predicted_trajectory = self.A(frames[frame_idx])
                if iteration == f_[frame_idx] - 1:
                    predicted_trajectory_list.append(predicted_trajectory)

        return predicted_trajectory_list

    def compute_loss(self, frames_features, trajectories, f_):
        """Average negative log-likelihood over all frame prefixes."""
        num_frames, _, _, _ = frames_features.shape

        total_loss = frames_features.new_zeros(())
        for frame_idx in range(num_frames):
            prefix_end = frame_idx + 1
            loss = self.get_single_loss(
                frames_features[:prefix_end],
                trajectories[:prefix_end],
                f_,
            )
            total_loss += loss

        return total_loss / num_frames

    def _validate_trajectory_shape(self, trajectory):
        expected_shape = (self.trajectory_dims, self.num_coeffs)
        if trajectory.shape[-len(expected_shape) :] != expected_shape:
            raise ValueError(
                "Trajectory targets must end with the configured polynomial "
                f"shape {expected_shape}, got {tuple(trajectory.shape)}"
            )

    def get_single_loss(self, x, trajectory, f_):
        """NLL for a frame-prefix, choosing the closest polynomial mode."""
        self._validate_trajectory_shape(trajectory)
        frames, batch_size, num_objects, _ = x.shape

        output_entire = self.forward(x, f_)
        loss = x.new_zeros(())

        for frame_idx in range(frames):
            means, covs = self.unpack_gmm_params(output_entire[frame_idx])

            for obj_idx in range(num_objects):
                for batch_idx in range(batch_size):
                    true = trajectory[frame_idx, batch_idx, obj_idx].reshape(
                        self.mean_dim
                    )
                    mode_means = means[batch_idx, obj_idx]
                    mode_covs = covs[batch_idx, obj_idx]

                    selected_mode = torch.argmin(
                        torch.linalg.norm(mode_means - true.unsqueeze(0), dim=-1)
                    )
                    mean = mode_means[selected_mode]
                    cov = mode_covs[selected_mode]
                    diff = true - mean

                    cholesky_factor = torch.linalg.cholesky(cov)
                    log_det = 2 * torch.sum(
                        torch.log(torch.diagonal(cholesky_factor))
                    )
                    solve_term = diff @ torch.cholesky_solve(
                        diff.unsqueeze(-1),
                        cholesky_factor,
                    ).squeeze(-1)

                    loss += 0.5 * (log_det + solve_term)

        return loss / (frames * batch_size * num_objects)
