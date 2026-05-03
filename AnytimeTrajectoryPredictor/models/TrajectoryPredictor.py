import torch.nn as nn
from AnytimeTrajectoryPredictor.models.architectures.base_model import (
    base_model,
)
from AnytimeTrajectoryPredictor.models.architectures.gru_model import gru_model


class TrajectoryPredictor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = self.create_model(args)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def compute_loss(self, *args, **kwargs):
        return self.model.compute_loss(*args, **kwargs)

    @staticmethod
    def _state_dim(args):
        features = args.feature_extractor.features
        image_trajectory_features = getattr(features, "image_trajectories", None)
        if image_trajectory_features is not None:
            return len(image_trajectory_features)
        return len(features)

    @staticmethod
    def create_model(args):
        """Factory method to create a TrajectoryPredictor model based on the provided configuration."""
        model_type = args.model.type.lower()
        state_dim = TrajectoryPredictor._state_dim(args)

        if model_type == "linear":
            return base_model(
                state_dim=state_dim,
                num_trajectory_possibilities=args.model.num_trajectory_possibilities,
            )
        elif model_type == "gru":
            return gru_model(
                state_dim=state_dim,
                num_trajectory_possibilities=args.model.num_trajectory_possibilities,
                hidden_dim=getattr(args.model, "hidden_dim", 64),
                refinement_steps=getattr(args.model, "refinement_steps", 3),
            )
        else:
            raise ValueError(f"Unsupported model type: {args.model.type}")

        """Follow this format for inititing your model:"""
        """
        if model_type == "put your model here":
            return your_model_initilization(args.1, args.2, args.3)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        """
