import torch
import torch.nn as nn

from AnytimeTrajectoryPredictor.models.architectures.base_model import base_model


class lstm_model(base_model):
    """Small LSTM-MDN trajectory predictor with iterative refinement."""

    def __init__(
        self,
        state_dim,
        num_trajectory_possibilities,
        hidden_dim=None,
        polynomial_degree=None,
        trajectory_dims=None,
        spatial_dims=None,
    ):
        super(lstm_model, self).__init__(
            state_dim,
            num_trajectory_possibilities,
            polynomial_degree=polynomial_degree,
            trajectory_dims=trajectory_dims,
            spatial_dims=spatial_dims,
        )
        self.hidden_dim = self._require_config_value(hidden_dim, "model.hidden_dim")

        self.lstm_cell = nn.LSTMCell(
            input_size=self.state_dim, hidden_size=self.hidden_dim
        )
        self.output_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def _initial_state(self, hidden_state, batch_size, num_objects, device, dtype):
        state_shape = (batch_size * num_objects, self.hidden_dim)

        if hidden_state is None:
            hidden = torch.zeros(state_shape, device=device, dtype=dtype)
            cell = torch.zeros(state_shape, device=device, dtype=dtype)
            return hidden, cell

        if isinstance(hidden_state, tuple):
            hidden, cell = hidden_state
        else:
            hidden = hidden_state
            cell = torch.zeros_like(hidden)

        hidden = self._reshape_state(hidden, state_shape, batch_size, num_objects)
        cell = self._reshape_state(cell, state_shape, batch_size, num_objects)
        return (
            hidden.to(device=device, dtype=dtype),
            cell.to(device=device, dtype=dtype),
        )

    def _reshape_state(self, state, state_shape, batch_size, num_objects):
        if state.dim() == 3:
            if state.shape[:2] != (batch_size, num_objects):
                raise ValueError(
                    "LSTM state must have shape "
                    f"({batch_size}, {num_objects}, {self.hidden_dim})"
                )
            state = state.reshape(*state_shape)
        elif state.dim() != 2:
            raise ValueError("LSTM state must have shape (B, N, H) or (B*N, H)")

        if state.shape != state_shape:
            raise ValueError(f"LSTM state must have shape {state_shape}")
        return state

    def forward(self, frames, f_, object_mask=None, hidden_state=None):
        num_frames, batch_size, num_objects, _ = frames.shape
        refinement_steps = self.normalize_refinement_steps(f_, num_frames)
        hidden, cell = self._initial_state(
            hidden_state,
            batch_size,
            num_objects,
            frames.device,
            frames.dtype,
        )

        predictions = []
        for frame_idx, steps in enumerate(refinement_steps):
            frame_features = frames[frame_idx].reshape(
                batch_size * num_objects, self.state_dim
            )
            for _ in range(steps):
                hidden, cell = self.lstm_cell(frame_features, (hidden, cell))

            frame_prediction = self.output_head(hidden).view(
                batch_size, num_objects, self.output_dim
            )
            predictions.append(frame_prediction)

        return predictions

    def normalize_refinement_steps(self, f_, num_frames):
        if isinstance(f_, int):
            return [f_] * num_frames
        if isinstance(f_, list) and len(f_) >= num_frames:
            return f_[:num_frames]
        raise ValueError(
            "f_ must be either an int or a list with at least num_frames entries"
        )


LSTMModel = lstm_model