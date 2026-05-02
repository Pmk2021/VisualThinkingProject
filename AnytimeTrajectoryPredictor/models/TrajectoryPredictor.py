import torch
import torch.nn as nn
from AnytimeTrajectoryPredictor.models.architectures.base_model import (
    base_model,
)


class TrajectoryPredictor(nn.Module):
    def __init__(self, model_config):
        pass

    @staticmethod
    def create_model(args):
        """Factory method to create a TrajectoryPredictor model based on the provided configuration."""
        model_type = args.model.type
        if model_type == "Linear":
            return base_model(
                state_dim=len(args.feature_extractor.features),
                num_trajectory_possibilities=args.model.num_trajectory_possibilities,
            )

        """Follow this format for inititing your model:"""
        """
        if model_type == "put your model here":
            return your_model_initilization(args.1, args.2, args.3)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        """
