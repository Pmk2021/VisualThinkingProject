import torch
import torch.nn as nn


class GNN(nn.Module):
    ### A simple linear model that predicts the mean and covariance of the velocity for each trajectory possibility.
    def __init__(self, state_dim, num_trajectory_possibilities):
        super(GNN, self).__init__()

        self.state_dim = 4
        self.num_trajectory_possibilities = num_trajectory_possibilities

        # Calculating Mean and covariacne for each trajectory possibility
        cov_mat_size = 9  # Assuming 2D trajectories
        single_trajectory_param_size = (
            cov_mat_size + 3
        )  # mean (2) + covariance (4)
        self.output_dim = (
            self.num_trajectory_possibilities * single_trajectory_param_size
        )

        # State transition matrix
        self.A = nn.Linear(self.state_dim, self.output_dim)

    def forward(self, frames, f_, object_mask=None, hidden_state=None):
        """
        frames: (T, B, N, F)
        f_: list[int] length T, refinement steps per frame
        """

        T, B, N, F = frames.shape

        # initialize hidden state per object
        if hidden_state is None:
            h = torch.zeros(B, N, self.state_dim, device=frames.device)
        else:
            h = hidden_state

        predicted_trajectory_list = []

        for t in range(T):
            # ----------------------------
            # 1. Temporal / state update
            # ----------------------------
            x_t = frames[t]  # (B, N, F)


            # ----------------------------
            # 2. Graph construction step
            # ----------------------------


            # ----------------------------
            # 3. Initial trajectory prediction
            # ----------------------------
            

            # ----------------------------
            # 4. Iterative refinement with graph convolution and residuals
            # ----------------------------
            for k in range(f_[t]):
                
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
        eps = 1e-6

        output_entire = self.forward(
            x, f_
        )  # (num_frames, batch_siez, num_objects, 3)

        for frame in range(frames):
            output = output_entire[frame]

            K = self.num_trajectory_possibilities

            x_mu = output[:, :, 0:K]
            y_mu = output[:, :, K : 2 * K]
            s_mu = output[:, :, 2 * K : 3 * K]
            covs = output[:, :, 3 * K : 12 * K]

            for obj_idx in range(n_objects):
                for batch_idx in range(b):
                    true = trajectory[frame, batch_idx, obj_idx]  # (3,)

                    # mixture selection
                    x_mu_b = x_mu[batch_idx, 0]
                    y_mu_b = y_mu[batch_idx, 0]
                    s_mu_b = s_mu[batch_idx, 0]
                    covs_b = covs[batch_idx, 0]
                    covs_k = covs_b.view(K, 3, 3)
                    d = torch.sqrt(
                        (x_mu_b - true[0]) ** 2
                        + (y_mu_b - true[1]) ** 2
                        + (s_mu_b - true[2]) ** 2
                    )

                    k = torch.argmin(d)

                    mean = torch.stack(
                        [
                            x_mu_b,
                            y_mu_b,
                            s_mu_b,
                        ]
                    )

                    diff = true - mean

                    cov = covs_k[k]

                    # 1. make symmetric
                    cov = 0.5 * (cov + cov.T)

                    # 2. build guaranteed positive-definite matrix
                    cov = cov @ cov.T + 1e-3 * torch.eye(3, device=x.device)

                    # 3. stable Cholesky
                    L = torch.linalg.cholesky(cov)

                    # 4. stable log-det
                    log_det = 2 * torch.sum(torch.log(torch.diagonal(L)))

                    # 5. stable quadratic form (DON'T invert explicitly if possible)
                    solve_term = diff @ torch.cholesky_solve(
                        diff.unsqueeze(-1), L
                    ).squeeze(-1)
                    nll = 0.5 * (log_det + solve_term)

                    loss += torch.mean(nll)

        return loss / (b * frames * n_objects)
