import torch
import torch.nn as nn

from AnytimeTrajectoryPredictor.models.architectures.base_model import base_model


class gru_model(base_model):
    """Small GRU-MDN trajectory predictor with iterative refinement.

    Each object keeps one hidden state. For each frame, the same ``GRUCell`` can
    be applied multiple times before emitting MDN parameters, making the
    model queryable after a configurable computation budget.
    """

    def __init__(
        self,
        state_dim,
        num_trajectory_possibilities,
        hidden_dim=None,
    ):
        """
        Initialize the GRU-MDN model.

        Parameters:
            state_dim: Dimension of the input state.
            num_trajectory_possibilities: Number of possible trajectories.
        """
        super(gru_model, self).__init__(
            state_dim,
            num_trajectory_possibilities
        )
        # If hidden_dim is not provided, default to 64.
        self.hidden_dim = hidden_dim if hidden_dim is not None else 64

        # Single GRUCell that is applied iteratively for each frame.
        self.gru_cell = nn.GRUCell(
            input_size=self.state_dim, hidden_size=self.hidden_dim
        )

        # Small MLP to project from GRU hidden state to MDN parameters.
        self.output_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def _initial_hidden(self, hidden_state, batch_size, num_objects, device, dtype):
        """
        Make sure the initial hidden state has the right shape (B*N, H) and dtype, and move it to the right device.
        If no hidden state is provided, return a zero tensor.
        """

        # If no hidden state is provided, return a zero tensor of shape (B*N, H).
        if hidden_state is None:
            return torch.zeros(
                batch_size * num_objects,
                self.hidden_dim,
                device=device,
                dtype=dtype,
            )

        # If a hidden state is provided, ensure it has the right shape and dtype, and move it to the right device.
        if hidden_state.dim() == 3:
            if hidden_state.shape[:2] != (batch_size, num_objects):
                raise ValueError(
                    "hidden_state must have shape "
                    f"({batch_size}, {num_objects}, {self.hidden_dim})"
                )
            hidden_state = hidden_state.reshape(batch_size * num_objects, self.hidden_dim)
        elif hidden_state.dim() != 2:
            raise ValueError("hidden_state must have shape (B, N, H) or (B*N, H)")

        # Ensure hidden_state has the right dtype and device.
        if hidden_state.shape != (batch_size * num_objects, self.hidden_dim):
            raise ValueError(
                "hidden_state must have shape "
                f"({batch_size * num_objects}, {self.hidden_dim})"
            )

        return hidden_state.to(device=device, dtype=dtype)

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
        num_frames, batch_size, num_objects, feature_dim = frames.shape
        if feature_dim != self.state_dim:
            raise ValueError(
                f"Expected frame feature dimension {self.state_dim}, got {feature_dim}"
            )

        # Initialize hidden state, ensuring it has the right shape and is on the right device.
        hidden = self._initial_hidden(
            hidden_state,
            batch_size,
            num_objects,
            frames.device,
            frames.dtype,
        )

        predictions = []
        for i in range(num_frames):  
            frame_features = frames[i].reshape(
                batch_size * num_objects, self.state_dim
            )
            for iteration in range(f_[i]):
                hidden = self.gru_cell(frame_features, hidden)

            frame_prediction = self.output_head(hidden).view(
                batch_size, num_objects, self.output_dim
            )
            predictions.append(frame_prediction)

        hidden = hidden.view(batch_size, num_objects, self.hidden_dim) # convert back to (batch, objects, hidden_dim) for output
        return predictions

GRUModel = gru_model