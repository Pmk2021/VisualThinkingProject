import torch
import torch.nn as nn


class base_model(nn.Module):
    ### A simple linear model that predicts the mean and covariance of the velocity for each trajectory possibility.

    params_per_mode = 12  # mean (3) + flattened 3x3 covariance factor (9)
    mean_dim = 3
    cov_eps = 1e-2

    def __init__(self, state_dim, num_trajectory_possibilities):
        super(base_model, self).__init__()

        self.state_dim = state_dim
        self.num_trajectory_possibilities = num_trajectory_possibilities

        # Calculating Mean and covariacne for each trajectory possibility
        self.output_dim = (
            self.num_trajectory_possibilities * self.params_per_mode
        )

        # State transition matrix
        self.A = nn.Linear(self.state_dim, self.output_dim)

    @classmethod
    def unpack_gmm_params(cls, output, num_modes):
        """Split a raw output tensor into GMM means and SPD covariances.

        Works for any leading shape so the same helper is used from the loss
        (per-frame ``(B, N, K * 12)`` tensors) and from the diversity metrics
        (frame-stacked ``(T, B, N, K * 12)`` tensors).

        Args:
            output: tensor with last dim ``num_modes * params_per_mode``.
            num_modes: ``K``.

        Returns:
            means: tensor shaped ``(..., K, mean_dim)``.
            covs:  symmetric positive-definite tensor shaped
                ``(..., K, mean_dim, mean_dim)``.
        """
        K = num_modes
        d = cls.mean_dim

        x_mu = output[..., 0:K]
        y_mu = output[..., K : 2 * K]
        s_mu = output[..., 2 * K : 3 * K]
        means = torch.stack([x_mu, y_mu, s_mu], dim=-1)  # (..., K, 3)

        cov_raw = output[..., 3 * K : cls.params_per_mode * K]
        cov_raw = cov_raw.reshape(*output.shape[:-1], K, d, d)

        cov_raw = torch.nan_to_num(cov_raw, nan=0.0, posinf=20.0, neginf=-20.0)
        cov_raw = torch.clamp(cov_raw, min=-20.0, max=20.0)
        cov_sym = 0.5 * (cov_raw + cov_raw.transpose(-1, -2))
        eye = torch.eye(d, device=output.device, dtype=output.dtype)
        covs = cov_sym @ cov_sym.transpose(-1, -2) + cls.cov_eps * eye

        return means, covs

    def forward(self, frames, f_, object_mask=None, hidden_state=None):
        """
        Given a sequence of frames, return a batch X num_frames X self.output_dim tensor representing the trajectory
        Parameters:
            frames_features: The input state features for all frames. Shape: (num_frames, batch_size, num_objects, feature_dim)
            f_: number of frames length list containing number of iterations to spend on each frame
            object_mask: 1 if an object in a given batch is on screen at a certain frame, 0 otherwise (num_frames, batch_size, num_objects)
            hidden_state: hidden state(optional, depends on model)
        Returns:
            predicted_trajectories: batch X num_frames X num_objects X self.output_dim tensor representing trajectory
            hidden_state: hidden_state(optional, depends on model)
        """
        num_frames, b, _, f = frames.shape

        predicted_trajectory_list = []
        for i in range(num_frames):
            for iteration in range(f_[i]):
                predicted_trajectory = self.A(frames[i])
                # Add only if last iteration
                if iteration == f_[i] - 1:
                    predicted_trajectory_list.append(predicted_trajectory)

        return predicted_trajectory_list

    def compute_loss(self, frames_features, trajectories, f_):
        """Calculate the average negative log-likelihood loss for the predicted trajectory distribution across all frames in the batch.
        Args:
            frames_features: The input state features for all frames. Shape: (batch_size, num_frames, num_objects, feature_dim)
            trajectories: The ground truth trajectories for all frames. Shape: (batch_size, num_frames, num_objects, 2)

        Returns:
            loss: The computed average negative log-likelihood loss across the batch.
        """

        num_frames, b, num_objects, feature_dim = frames_features.shape

        total_loss = 0
        for i in range(0, num_frames):
            trajectory = trajectories[: i + 1]

            loss = self.get_single_loss(
                frames_features[: i + 1], trajectory, f_
            )
            total_loss += loss

        return total_loss / len(frames_features)

    def get_single_loss(self, x, trajectory, f_):
        """Calculate the negative log-likelihood loss for the predicted trajectory distribution for a single frame.
        Only calculates loss using the "best" predicted trajectory (the one with the closest mean to the true trajectory).
        Args:
            x: The input state features. Shape: (num_frames, batch_size, num_objects, feature_dim)
            trajectory: The ground truth trajectory. Shape: (num_frames, batch_size, num_objects, 3)
            #delta_x, delta_y, dleta_area
        Returns:
            loss: The computed negative log-likelihood loss.
        """
        frames, b, n_objects, _ = x.shape

        loss = 0

        output_entire = self.forward(
            x, f_
        )  # (num_frames, batch_size, num_objects, 3)

        K = self.num_trajectory_possibilities

        for frame in range(frames):
            # means: (B, N, K, 3), covs: (B, N, K, 3, 3) — same SPD recipe as
            # ``unpack_gmm_params`` (clamp + LL^T + eps*I), shared with the
            # diversity metrics so both stay in sync.
            means, covs = self.unpack_gmm_params(output_entire[frame], K)

            for obj_idx in range(n_objects):
                for batch_idx in range(b):
                    true = trajectory[frame, batch_idx, obj_idx]  # (3,)

                    mu_k = means[batch_idx, obj_idx]   # (K, 3)
                    cov_k = covs[batch_idx, obj_idx]   # (K, 3, 3)

                    # mixture selection: closest mean wins
                    k = torch.argmin(torch.linalg.norm(mu_k - true, dim=-1))

                    mean = mu_k[k]
                    cov = cov_k[k]
                    diff = true - mean

                    # stable Cholesky on the already-SPD covariance
                    L = torch.linalg.cholesky(cov)

                    # stable log-det
                    log_det = 2 * torch.sum(torch.log(torch.diagonal(L)))

                    # stable quadratic form (DON'T invert explicitly if possible)
                    solve_term = diff @ torch.cholesky_solve(
                        diff.unsqueeze(-1), L
                    ).squeeze(-1)
                    nll = 0.5 * (log_det + solve_term)

                    loss += torch.mean(nll)

        return loss / (b * frames * n_objects)
