import torch
import torch.nn as nn
from architectures.linear_model import linear_model


class TrajectoryPredictor(nn.Module):
    def __init__(self, model_config):
        pass

    @staticmethod
    def create_model(args):
        """Factory method to create a TrajectoryPredictor model based on the provided configuration."""
        model_type = args.model.type
        if model_type == "linear":
            return linear_model(
                state_dim=len(args.feature_extractor.features),
                num_trajectory_possibilities=args.model.num_trajectory_possibilities,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
