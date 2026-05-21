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
    COVARIANCE_EPS = 1e-3
    ### A simple linear model that predicts the mean and covariance of the velocity for each trajectory possibility.
    def __init__(self, state_dim, num_trajectory_possibilities):
        super(GNN, self).__init__()
        self.hidden_dim = 64
        self.state_dim = 196
        self.num_trajectory_possibilities = num_trajectory_possibilities
        polynomial_degree = 3
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

        self.node_encoder = nn.Linear(self.state_dim, self.hidden_dim)
        self.gnn = InteractionGNN(
            self.hidden_dim
        )  # from earlier PyG-style layer
        self.traj_head = nn.Linear(
            self.hidden_dim, self.num_trajectory_possibilities * self.single_trajectory_param_size,
        )
        self.refine_mlp = nn.Sequential(
            nn.Linear(
                self.hidden_dim + self.num_trajectory_possibilities * self.single_trajectory_param_size,
                self.hidden_dim,
            ),
            nn.ReLU(),
        )
        self.delta_head = nn.Linear(
            self.hidden_dim, self.num_trajectory_possibilities * self.single_trajectory_param_size
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
        
    def stabilize_covariance(self, cov):
        # 1. force symmetry
        cov = 0.5 * (cov + cov.transpose(-1, -2))

        # 2. remove NaN/Inf (CRITICAL)
        cov = torch.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

        # 3. eigen decomposition
        eigvals, eigvecs = torch.linalg.eigh(cov)

        # 4. clamp spectrum
        eigvals = torch.clamp(eigvals, min=self.COVARIANCE_EPS)

        cov_psd = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)

        # 5. final jitter (important in float32)
        eye = torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype)
        cov_psd = cov_psd + self.COVARIANCE_EPS * eye

        return cov_psd

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