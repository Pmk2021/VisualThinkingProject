import torch.nn as nn
from AnytimeTrajectoryPredictor.models.architectures.base_model import (
    base_model,
)
from AnytimeTrajectoryPredictor.models.architectures.gru_model import gru_model
from AnytimeTrajectoryPredictor.models.architectures.lstm_model import lstm_model


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
    def _require_model_config(args, name):
        if name not in args.model:
            raise ValueError(f"model.{name} must be defined in the configuration file")
        return args.model[name]

    @staticmethod
    def _require_feature_config(args, name):
        if name not in args.feature_extractor:
            raise ValueError(
                f"feature_extractor.{name} must be defined in the configuration file"
            )
        return args.feature_extractor[name]

    @staticmethod
    def _trajectory_distribution_kwargs(args):
        return {
            "polynomial_degree": TrajectoryPredictor._require_model_config(
                args, "polynomial_degree"
            ),
            "trajectory_dims": TrajectoryPredictor._require_feature_config(
                args, "trajectory_dims"
            ),
            "spatial_dims": TrajectoryPredictor._require_model_config(
                args, "spatial_dims"
            ),
        }

    @staticmethod
    def create_model(args):
        """Factory method to create a TrajectoryPredictor model based on the provided configuration."""
        model_type = args.model.type.lower()
        state_dim = TrajectoryPredictor._state_dim(args)
        trajectory_distribution_kwargs = (
            TrajectoryPredictor._trajectory_distribution_kwargs(args)
        )

        if model_type == "linear":
            return base_model(
                state_dim=state_dim,
                num_trajectory_possibilities=args.model.num_trajectory_possibilities,
                **trajectory_distribution_kwargs,
            )
        elif model_type == "gru":
            return gru_model(
                state_dim=state_dim,
                num_trajectory_possibilities=args.model.num_trajectory_possibilities,
                hidden_dim=TrajectoryPredictor._require_model_config(
                    args, "hidden_dim"
                ),
                **trajectory_distribution_kwargs,
            )
        elif model_type == "lstm":
            return lstm_model(
                state_dim=state_dim,
                num_trajectory_possibilities=args.model.num_trajectory_possibilities,
                hidden_dim=TrajectoryPredictor._require_model_config(
                    args, "hidden_dim"
                ),
                **trajectory_distribution_kwargs,
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
