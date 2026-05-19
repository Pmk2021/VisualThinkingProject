from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import MessagePassing
import torch

from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
import torch.nn as nn


def fully_connected_edge_index(N, device):
    adj = torch.ones(N, N, device=device)
    edge_index, _ = dense_to_sparse(adj)
    return edge_index


class InteractionGNN(MessagePassing):
    def __init__(self, state_dim):
        super().__init__(aggr="add")  # sum messages

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * state_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim),
        )

        self.node_update = nn.Linear(state_dim, state_dim)

    def forward(self, x, edge_index):
        # x: (B*N, D) OR (N, D) per graph
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i = receiver, x_j = sender
        edge_feat = torch.cat([x_i, x_j], dim=-1)
        return self.edge_mlp(edge_feat)

    def update(self, aggr_out, x):
        return x + self.node_update(aggr_out)


class GNN(nn.Module):
    ### A simple linear model that predicts the mean and covariance of the velocity for each trajectory possibility.
    def __init__(self, state_dim, num_trajectory_possibilities):
        super(GNN, self).__init__()
        self.hidden_dim = 64
        self.state_dim = state_dim
        self.num_trajectory_possibilities = num_trajectory_possibilities

        self.node_encoder = nn.Linear(self.state_dim, self.hidden_dim)
        self.gnn = InteractionGNN(
            self.hidden_dim
        )  # from earlier PyG-style layer
        self.traj_head = nn.Linear(
            self.hidden_dim, self.num_trajectory_possibilities * 11
        )
        self.refine_mlp = nn.Sequential(
            nn.Linear(
                self.hidden_dim + self.num_trajectory_possibilities * 11,
                self.hidden_dim,
            ),
            nn.ReLU(),
        )
        self.delta_head = nn.Linear(
            self.hidden_dim, self.num_trajectory_possibilities * 11
        )

    def forward(self, frames, f_, object_mask=None, hidden_state=None):
        """
        frames: (T, B, N, F)
        f_: list[int] length T, refinement steps per frame
        """

        T, B, N, F = frames.shape

        # initialize hidden state per object
        if hidden_state is None:
            h = torch.zeros(B, N, self.hidden_dim, device=frames.device)
        else:
            h = hidden_state

        edge_index = fully_connected_edge_index(N, frames.device)

        predicted_trajectory_list = []

        for t in range(T):
            # 1. Temporal / state update

            x_t = frames[t]  # (B, N, F)
            x_emb = self.node_encoder(x_t)  # (B, N, H)
            h = h + x_emb

            # ----------------------------
            # 2. Encode Node Features
            # ----------------------------
            x_emb = self.node_encoder(x_t)  # (B, N, hidden_state)

            h = h + x_emb  # residual temporal update

            # 3. Message passing in graph

            h_flat = h.view(B * N, self.hidden_dim)

            h_flat = self.gnn(h_flat, edge_index)

            h = h_flat.view(B, N, self.hidden_dim)

            # 4. Initial trajectory prediction

            y = self.traj_head(h)  # (B, N, P * params)

            y = y.view(B, N, self.num_trajectory_possibilities, -1)

            # 4. Iterative refinement with graph convolution and residuals

            for k in range(f_[t]):
                """
                We repeate everything k times
                """

                # 1. Message passing over nodes

                h_flat = h.view(B * N, -1)

                h_flat = self.gnn(h_flat, edge_index)

                h = h_flat.view(B, N, -1)

                # 2. trajectory-aware interaction

                y_flat = y.view(B, N, -1)

                inp = torch.cat([h, y_flat], dim=-1)

                h_ref = self.refine_mlp(inp)

                # 3. Trajectory update

                delta = self.delta_head(h_ref)

                delta = delta.view_as(y)

                y = y + delta

            predicted_trajectory_list.append(y)

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
