import torch
import torch.nn as nn


class base_model(nn.Module):
    ### A simple linear model that predicts the mean and covariance of the velocity for each trajectory possibility.
    def __init__(self, state_dim, num_trajectory_possibilities):
        super(base_model, self).__init__()
        self.state_dim = state_dim
        self.num_trajectory_possibilities = num_trajectory_possibilities

        # Calculating Mean and covariacne for each trajectory possibility
        cov_mat_size = 4  # Assuming 2D trajectories
        single_trajectory_param_size = (
            cov_mat_size + 2
        )  # mean (2) + covariance (4)
        self.output_dim = (
            self.num_trajectory_possibilities * single_trajectory_param_size
        )

        # State transition matrix
        self.A = nn.Linear(state_dim, self.output_dim)

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
        num_frames, b, _, _ = frames.shape

        predicted_trajectory_list = []
        for i in range(frames):
            for iteration in f_[i]:
                predicted_trajectory = (self.A(frames[i]),)
            predicted_trajectory_list.append(predicted_trajectory)

        return predicted_trajectory_list

    def get_loss(self, frames_features, trajectories, f_):
        """Calculate the average negative log-likelihood loss for the predicted trajectory distribution across all frames in the batch.
        Args:
            frames_features: The input state features for all frames. Shape: (batch_size, num_frames, num_objects, feature_dim)
            trajectories: The ground truth trajectories for all frames. Shape: (batch_size, num_frames, num_objects, 2)

        Returns:
            loss: The computed average negative log-likelihood loss across the batch.
        """

        b, num_frames, num_objects, feature_dim = frames_features.shape

        total_loss = 0
        for i in range(0, num_frames):
            trajectory = trajectories[:, i]

            loss = self.get_single_loss(frames_features[:i], trajectory, f_)
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
        b, frames, n_objects, _ = x.shape

        loss = 0
        eps = 1e-6

        output_entire = self.forward(
            x, f_
        )  # (num_frames, batch_siez, num_objects, 3)
        for frame in range(frames):
            output = output_entire[frame]
            K = self.num_trajectory_possibilities

            x_mu = output[:, 0:K]
            y_mu = output[:, K : 2 * K]
            s_mu = output[:, 2 * K : 3 * K]

            covs = output[:, 3 * K :]  # (b, K * 9)

            for obj_idx in range(n_objects):
                for batch_idx in range(b):
                    true = trajectory[batch_idx, frame, obj_idx]  # (3,)

                    # mixture selection
                    d = torch.sqrt(
                        (x_mu[batch_idx] - true[0]) ** 2
                        + (y_mu[batch_idx] - true[1]) ** 2
                        + (s_mu[batch_idx] - true[2]) ** 2
                    )

                    k = torch.argmin(d)

                    cov = covs[batch_idx, k * 9 : (k + 1) * 9].view(3, 3)
                    cov = cov + eps * torch.eye(3, device=x.device)

                    mean = torch.stack(
                        [
                            x_mu[batch_idx, k],
                            y_mu[batch_idx, k],
                            s_mu[batch_idx, k],
                        ]
                    )

                    diff = true - mean

                    log_det = torch.logdet(cov)
                    solve_term = diff @ torch.linalg.solve(cov, diff)

                    nll = 0.5 * (log_det + solve_term)

                    loss += nll

        return loss / (b * frames * n_objects)
