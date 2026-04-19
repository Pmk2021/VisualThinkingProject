import torch
import torch.nn as nn


class linear_model(nn.Module):
    ### A simple linear model that predicts the mean and covariance of the velocity for each trajectory possibility.
    def __init__(self, state_dim, num_trajectory_possibilities):
        super(linear_model, self).__init__()
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

    def forward(self, x):
        b, _ = x.shape

        return self.A(x)

    def get_loss(self, frames_features, trajectories):
        """Calculate the average negative log-likelihood loss for the predicted trajectory distribution across all frames in the batch.
        Args:
            frames_features: The input state features for all frames. Shape: (batch_size, num_frames, num_objects, feature_dim)
            trajectories: The ground truth trajectories for all frames. Shape: (batch_size, num_frames, num_objects, 2)

        Returns:
            loss: The computed average negative log-likelihood loss across the batch.
        """

        b, num_frames, num_objects, feature_dim = frames_features.shape

        total_loss = 0
        for i in range(1, len(frames_features) + 1):
            trajectory = trajectories[i]

            loss = self.get_single_loss(frames_features[:i], trajectory)
            total_loss += loss

        return total_loss / len(frames_features)

    def get_single_loss(self, x, trajectory):
        """Calculate the negative log-likelihood loss for the predicted trajectory distribution for a single frame.
        Only calculates loss using the "best" predicted trajectory (the one with the closest mean to the true trajectory).
        Args:
            x: The input state features. Shape: (batch_size, num_frames, num_objects, feature_dim)
            trajectory: The ground truth trajectory. Shape: (batch_size, num_frames, num_objects, 2)

        Returns:
            loss: The computed negative log-likelihood loss.
        """

        # Get the features for the last frame in the sequence
        x = x[:, -1, :, :]

        b, n_objects, _ = x.shape

        loss = 0
        for obj_idx in range(n_objects):
            x_ = x[:, obj_idx, :]  # Shape: (b, feature_dim)
            output = self.forward(x_)  # Shape: (b, output_dim)

            x_velocity_mean = output[:, 0 : self.num_trajectory_possibilities]
            y_velocity_mean = output[
                :,
                self.num_trajectory_possibilities : 2
                * self.num_trajectory_possibilities,
            ]
            covariances = output[:, 2 * self.num_trajectory_possibilities :]

            for batch_idx in range(b):
                # Calculate the closest mean to the true trajectory
                x_velocity_mean_batch = x_velocity_mean[batch_idx]
                y_velocity_mean_batch = y_velocity_mean[batch_idx]
                distances = torch.sqrt(
                    (x_velocity_mean_batch - trajectory[batch_idx, 0]) ** 2
                    + (y_velocity_mean_batch - trajectory[batch_idx, 1]) ** 2
                )

                closest_mean_idx = torch.argmin(distances)
                closest_covariance = covariances[
                    batch_idx,
                    closest_mean_idx * 4 : (closest_mean_idx + 1) * 4,
                ]

                # Reshape the covariance parameters into a 2x2 matrix
                covariance_matrix = closest_covariance.view(2, 2)

                mean_vector = torch.tensor(
                    [
                        x_velocity_mean_batch[closest_mean_idx],
                        y_velocity_mean_batch[closest_mean_idx],
                    ]
                )

                # Calculate the negative log-likelihood loss for the closest mean and covariance
                trajectory_vector = trajectory[batch_idx]
                diff = trajectory_vector - mean_vector

                # torch.linalg.solve(A, b) is used to solve the linear system Ax = b,
                # where A is the covariance matrix and b is the difference between
                # the trajectory and the mean vector.
                # This is more numerically stable than directly computing the inverse of A.
                nll_loss = 0.5 * (
                    torch.logdet(covariance_matrix)
                    + diff.T @ torch.linalg.solve(covariance_matrix, diff)
                )
                loss += nll_loss

        return loss / b / n_objects
