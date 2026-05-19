import torch.nn as nn
import pyarrow.parquet as pq
from PIL import Image
from io import BytesIO
from box import Box
from typing import List

from AnytimeTrajectoryPredictor.models.ObjectTracker import ObjectTracker
from AnytimeTrajectoryPredictor.models.TrajectoryPredictor import TrajectoryPredictor


class Pipeline(nn.Module):
    """
    End-to-end pipeline that combines the ObjectTracker and any TrajectoryPredictor to produce trajectory predictions from raw RGB input data.
    """

    def __init__(self, model_args, feature_extractor_args, verbose: bool = False):
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
        self.verbose = verbose

    def forward(self, image, f_: List[int]):
        """
        Forward pass through the pipeline: extract features from the input image and then predict trajectories.

        Parameters
        ----------
        image : PIL.Image
            Input RGB image from which to extract features and predict trajectories.
        f_ : Optional[List[int]]]
            List of integers specifying the number of refinement steps for each frame.

        Returns
        -------
        List[torch.Tensor]
            A list of length T (number of frames) where each element is a tensor of shape (B, N, num_trajectory_possibilities, 11) containing the predicted trajectories for that frame.
        """
        fe_output = self.feature_extractor(image)
        features = fe_output["features"]
        mask = fe_output["mask"]
        if self.verbose:
            print(f"Extracted features shape: {features.shape} - (T, B, N, F)")
            print(f"Extracted mask shape: {mask.shape} - (T, B, N, 1)")
        predictions = self.trajectory_predictor(features, f_, object_mask=mask)
        return predictions


if __name__ == "__main__":

    # Data setup
    path_to_local_images = (
        "/Users/nathangromb/Documents/MA4/VI/project/data/waymo/images.parquet"
    )

    parquet_file = pq.read_table(path_to_local_images).to_pandas()
    sample_jpeg = parquet_file["image_jpeg"].sample(1, random_state=42).iloc[0]
    image = Image.open(BytesIO(sample_jpeg))

    # Pipeline setup
    num_trajectory_possibilities = 5
    refinements = [3]

    model_args = Box(
        {
            "feature_extractor": Box(
                {
                    "features": [None]
                    * 391,  # Matches the dim of the FE output features
                }
            ),
            "model": Box(
                {
                    "num_trajectory_possibilities": num_trajectory_possibilities,
                    "type": "GNN",
                }
            ),
        }
    )

    feature_extractor_args = {
        "feature_components": [
            "bboxes",
            "confidences",
            "object_ids",
            "class_ids",
            "latent_features",
            "local_latent_features",
        ],
        "imgsz": 640,
        "verbose": False,
    }

    pipeline = Pipeline(model_args, feature_extractor_args, verbose=True)

    # Run the pipeline on the sample image
    output = pipeline(image, f_=refinements)
    print(
        f"Output: {len(output)} timestep of shape {tuple(set(t.shape for t in output))} - (B, N, {num_trajectory_possibilities}, 11)"
    )

    image.close()
