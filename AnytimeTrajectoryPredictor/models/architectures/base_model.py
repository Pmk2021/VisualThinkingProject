import torch
import torch.nn as nn


class base_model(nn.Module):
    COVARIANCE_EPS = 1e-3

    ### A simple linear model that predicts the mean and covariance of the velocity for each trajectory possibility.
    def __init__(
        self, state_dim, num_trajectory_possibilities, polynomial_degree=3
    ):
        super(base_model, self).__init__()

        self.state_dim = 4
        self.num_trajectory_possibilities = num_trajectory_possibilities

        # cubic polynomial -> 4 coefficients
        self.num_coeffs = polynomial_degree + 1

        # x, y, scale
        self.num_dims = 3
        self.spatial_indices = [0, 1]  # Indices for x and y coefficients

        # total coeffs per trajectory hypothesis
        self.coeff_dim = self.num_dims * self.num_coeffs  # 3 * 4 = 12

        # --------------------------------------------------
        # For each trajectory hypothesis:
        #
        # mean:
        #   12 params
        #
        # covariance:
        #   12 x 12 = 144 params
        # --------------------------------------------------

        self.mean_params = self.coeff_dim
        self.cov_params = self.coeff_dim * self.coeff_dim

        self.single_trajectory_param_size = self.mean_params + self.cov_params

        self.output_dim = (
            self.num_trajectory_possibilities
            * self.single_trajectory_param_size
        )

        self.A = nn.Linear(
            self.state_dim,
            self.output_dim,
        )

    def stabilize_covariance(self, cov):
        """
        Stabilize covariance matrices by ensuring symmetry and adding a small value to the diagonal.
        This helps prevent numerical issues during Cholesky decomposition.
        Modularized to be reusable to compute the diversity merics in evaluation/diversity.py.
        """
        cov = 0.5 * (cov + cov.transpose(-1, -2)) # Ensure symmetry
        eye = torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype)
        return cov @ cov.transpose(-1, -2) + self.COVARIANCE_EPS * eye

    def stable_cholesky(self, cov, max_attempts=5):
        eye = torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype)
        jitter = self.COVARIANCE_EPS

        for _ in range(max_attempts):
            L, info = torch.linalg.cholesky_ex(cov)
            if torch.all(info == 0):
                return L
            cov = cov + jitter * eye
            jitter *= 10

        return torch.linalg.cholesky(cov)

    def forward(self, frames, f_, object_mask=None, hidden_state=None):
        """
        Given a sequence of frames, return a batch X num_frames X self.output_dim tensor representing the trajectory
        Parameters:
            frames_features: The input state features for all frames. Shape: (num_frames, batch_size, num_objects, feature_dim=state_dim)
            f_: number of frames length list containing number of iterations to spend on each frame
            object_mask: 1 if an object in a given batch is on screen at a certain frame, 0 otherwise (num_frames, batch_size, num_objects)
            hidden_state: hidden state(optional, depends on model)
        Returns:
            predicted_trajectories: List of length num_frames, each element of size (batch_size, num_objects, self.output_dim) representing trajectory predictions for each frame
            hidden_state: hidden_state(optional, depends on model)
        """
        num_frames, b, _, f = frames.shape

        predicted_trajectory_list = []
        for i in range(num_frames):
            predicted_trajectory = self.A(frames[i]) # (B, num_objects, output_dim)
            for iteration in range(1, f_[i]):
                predicted_trajectory = self.A(frames[i])
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

    def get_single_loss(
        self,
        x,
        trajectory,
        f_,
    ):
        """
        trajectory shape:
            (frames, batch, objects, 3, 4)

        where:
            trajectory[..., 0, :] -> x coeffs
            trajectory[..., 1, :] -> y coeffs
            trajectory[..., 2, :] -> scale coeffs
        """

        frames, b, n_objects, _ = x.shape

        K = self.num_trajectory_possibilities
        D = self.coeff_dim  # 12

        output_entire = self.forward(x, f_)

        loss = 0.0

        for frame in range(frames):
            output = output_entire[frame]

            # ------------------------------------------
            # reshape:
            #
            # (B, O, K, params_per_mode)
            # ------------------------------------------

            output = output.view(
                b,
                n_objects,
                K,
                self.single_trajectory_param_size,
            )

            means = output[..., :D]

            covs = output[..., D:]

            # reshape covariance
            covs = covs.view(
                b,
                n_objects,
                K,
                D,
                D,
            )

            for obj_idx in range(n_objects):
                for batch_idx in range(b):
                    # ----------------------------------
                    # GT polynomial coefficients
                    # shape: (3, 4)
                    # flatten -> (12,)
                    # ----------------------------------

                    true = trajectory[
                        frame,
                        batch_idx,
                        obj_idx,
                    ].reshape(-1)

                    means_b = means[
                        batch_idx,
                        obj_idx,
                    ]  # (K, 12)

                    covs_b = covs[
                        batch_idx,
                        obj_idx,
                    ]  # (K, 12, 12)

                    # ----------------------------------
                    # choose best trajectory mode
                    # ----------------------------------

                    d = torch.norm(
                        means_b - true.unsqueeze(0),
                        dim=-1,
                    )

                    k = torch.argmin(d)

                    mean = means_b[k]

                    diff = true - mean

                    cov = self.stabilize_covariance(covs_b[k])

                    # ----------------------------------
                    # Cholesky decomposition
                    # ----------------------------------

                    L = self.stable_cholesky(cov)

                    # log determinant
                    log_det = 2 * torch.sum(torch.log(torch.diagonal(L)))

                    # Mahalanobis term
                    solve_term = diff @ torch.cholesky_solve(
                        diff.unsqueeze(-1),
                        L,
                    ).squeeze(-1)

                    nll = 0.5 * (log_det + solve_term)

                    loss += nll

        return loss / (frames * b * n_objects)

