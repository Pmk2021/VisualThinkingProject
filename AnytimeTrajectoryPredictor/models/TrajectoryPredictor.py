import torch
import torch.nn as nn
from AnytimeTrajectoryPredictor.Data.feature_extractor import FEATURE_DIM
from AnytimeTrajectoryPredictor.models.architectures.base_model import (
    base_model,
)


from AnytimeTrajectoryPredictor.models.architectures.gru import gru_model


class TrajectoryPredictor(nn.Module):
    def __init__(self, model_config):
        pass

    @staticmethod
    def create_model(args):
        """Factory method to create a TrajectoryPredictor model based on the provided configuration."""
        model_type = args.model.type
        state_dim = args.model.state_dim if "state_dim" in args.model else FEATURE_DIM
        if model_type == "linear":
            return base_model(
                state_dim=state_dim,
                num_trajectory_possibilities=args.model.num_trajectory_possibilities,
            )
        if model_type == "GNN":
            from AnytimeTrajectoryPredictor.models.architectures.gnn import GNN
            return GNN(
                state_dim=state_dim,
                num_trajectory_possibilities=args.model.num_trajectory_possibilities,
            )
        if model_type == "gru":
            return gru_model(
                state_dim=state_dim,
                num_trajectory_possibilities=args.model.num_trajectory_possibilities,
                hidden_dim=args.model.hidden_dim if "hidden_dim" in args.model else None,
            )
        if model_type == "astra_edm":
            from AnytimeTrajectoryPredictor.models.architectures.astra_edm_diffusion import (
                ASTRAEDMDiffusionModel,
            )

            return ASTRAEDMDiffusionModel(
                config=args.model,
                num_trajectory_possibilities=args.model.num_trajectory_possibilities,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        """Follow this format for inititing your model:"""
        """
        if model_type == "put your model here":
            return your_model_initilization(args.1, args.2, args.3)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        """
