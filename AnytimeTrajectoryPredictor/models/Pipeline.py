import torch
import torch.nn as nn

from AnytimeTrajectoryPredictor.models.ObjectTracker import ObjectTracker
from AnytimeTrajectoryPredictor.models.TrajectoryPredictor import TrajectoryPredictor

class Pipeline(nn.Module):
    """
    End-to-end pipeline that combines the ObjectTracker and any TrajectoryPredictor to produce trajectory predictions from raw RGB input data.
    """
    def __init__(self, model_args, feature_extractor_args):
        """
        Initialize the Pipeline with the specified model and feature extractor configurations.

        Parameters
        ----------
        model_args : Dict
            Arguments for initializing the TrajectoryPredictor model (e.g. type, num_trajectory_possibilities, etc.)
        feature_extractor_args : Dict
            Arguments for initializing the ObjectTracker feature extractor (e.g. features to extract, etc.)
        """
        super().__init__()
        self.feature_extractor_args = feature_extractor_args
        self.model_args = model_args
        self.feature_extractor = ObjectTracker(**feature_extractor_args)
        self.trajectory_predictor = TrajectoryPredictor.create_model(model_args)

    def forward(self, input_data):
        pass