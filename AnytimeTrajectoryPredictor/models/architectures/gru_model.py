import torch
import torch.nn as nn

from AnytimeTrajectoryPredictor.models.architectures.base_model import base_model


class gru_model(base_model):
    """Small GRU-MDN trajectory predictor with iterative refinement.

    Each object keeps one hidden state. For each frame, the same ``GRUCell`` can
    be applied multiple times before emitting the MDN parameters, making the
    model queryable after a configurable computation budget.
    """

    params_per_mode = 6 # logit, mean_x, mean_y, log_std_x, log_std_y, correlation rho between x and y

    def __init__(
        self,
        state_dim,
        num_trajectory_possibilities,
        hidden_dim=64,
        refinement_steps=3,
    ):
        """
        Initialize the GRU-MDN model.

        Parameters:
            state_dim: Dimension of the input state.
            num_trajectory_possibilities: Number of possible trajectories.
            hidden_dim: size of the GRU memory vector for each object.
            refinement_steps: Number of refinement steps per frame.
        """
        super(gru_model, self).__init__(state_dim, num_trajectory_possibilities)
        self.hidden_dim = hidden_dim
        self.refinement_steps = refinement_steps
        self.output_dim = self.num_trajectory_possibilities * self.params_per_mode

        # Single GRUCell that is applied iteratively for each frame.
        self.gru_cell = nn.GRUCell(input_size=state_dim, hidden_size=hidden_dim)

        # Small MLP to project from GRU hidden state to MDN parameters.
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def _initial_hidden(self, hidden_state, batch_size, num_objects, device, dtype):
        """
        Make sure the initial hidden state has the right shape (B*N, H) and dtype, and move it to the right device.
        If no hidden state is provided, return a zero tensor.
        """
        if hidden_state is None:
            return torch.zeros(
                batch_size * num_objects,
                self.hidden_dim,
                device=device,
                dtype=dtype,
            )

        if hidden_state.dim() == 3:
            if hidden_state.shape[:2] != (batch_size, num_objects):
                raise ValueError(
                    "hidden_state must have shape "
                    f"({batch_size}, {num_objects}, {self.hidden_dim})"
                )
            hidden_state = hidden_state.reshape(batch_size * num_objects, self.hidden_dim)
        elif hidden_state.dim() != 2:
            raise ValueError("hidden_state must have shape (B, N, H) or (B*N, H)")

        if hidden_state.shape != (batch_size * num_objects, self.hidden_dim):
            raise ValueError(
                "hidden_state must have shape "
                f"({batch_size * num_objects}, {self.hidden_dim})"
            )

        return hidden_state.to(device=device, dtype=dtype)

    def forward(self, frames, f_, object_mask=None, hidden_state=None):
        """
        Predict trajectory distribution parameters for each frame/object.

        Parameters:
            frames: Tensor shaped ``(batch, frames, objects, state_dim)``.
            f_: Optional int or length-``num_frames`` list controlling how many
                GRU refinement iterations are spent on each frame.
            hidden_state: Optional previous hidden state shaped
                ``(batch, objects, hidden_dim)`` or ``(batch * objects, hidden_dim)``.

        Returns:
            predicted_trajectories: Tensor shaped
                ``(batch, frames, objects, output_dim)``.
            hidden_state: Final hidden state shaped ``(batch, objects, hidden_dim)``.
        """
        batch_size, num_frames, num_objects, _ = frames.shape
        refinement_steps = self.normalize_refinement_steps(
            f_, num_frames, self.refinement_steps
        )
        hidden = self._initial_hidden(
            hidden_state,
            batch_size,
            num_objects,
            frames.device,
            frames.dtype,
        )

        predictions = []
        for frame_idx, steps in enumerate(refinement_steps):
            frame_features = frames[:, frame_idx, :, :].reshape(
                batch_size * num_objects, self.state_dim
            )
            for _ in range(steps):
                hidden = self.gru_cell(frame_features, hidden)
                if _ == steps - 1: # only emit predictions on the last iteration for this frame
                    frame_prediction = self.output_head(hidden).view(
                        batch_size, num_objects, self.output_dim
                    )
                    predictions.append(frame_prediction)

        predictions = torch.stack(predictions, dim=1) # convert from list to (batch, frames, objects, output_dim)
        hidden = hidden.view(batch_size, num_objects, self.hidden_dim) # convert back to (batch, objects, hidden_dim) for output
        return predictions, hidden
    
    def normalize_refinement_steps(self, f_, num_frames, default_steps):
        """
        Normalize the refinement steps input to a list of length num_frames.

        If f_ is an int, return a list of length num_frames where each element is f_.
        If f_ is already a list of length num_frames, return it as is.
        Otherwise, raise a ValueError.
        """
        if isinstance(f_, int):
            return [f_] * num_frames
        elif isinstance(f_, list) and len(f_) == num_frames:
            return f_
        else:
            raise ValueError(
                "f_ must be either an int or a list of length num_frames"
            )

GRUModel = gru_model