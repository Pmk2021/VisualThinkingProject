from torch.utils import data
from torch.utils.data import dataset
from torchvision.io import read_video
import pandas as pd
import os
import tqdm

from features.dummy_feature_1 import DummyFeatureExtractor


class FeatureExtractor:
    def __init__(self, args):
        self.args = args

    def compute_feature(self, frames, dt_per_frame):
        """
        Given a list of frames, and dt_per_frame,
        compute the specified features for each frame and return a feature per frame

        ie:

        {
            1: feature_values_for_frame1,
            2: feature_values_for_frame2,
            ...
        }
        """
        pass


FEATUREDICT = {"DUMMY_FEATURE": DummyFeatureExtractor}


class FeatureDataset(dataset.Dataset):
    def __init__(self, feature_path, args, data_path=None):
        """Dataset class for loading pre-extracted features from CSV files.
        If regenerate_features is True, it will extract features from raw video data.

        args:
            feature_path: The file path where extracted features are stored (e.g., CSV file).
            args: Configuration arguments
            data_path: Path to folder containing raw video data (required if regenerate_features is True)
        """

        self.features = args.features
        self.feature_extractors = {
            feature: FEATUREDICT[feature](args) for feature in self.features
        }

        # Regenerate features if specified in the config
        if args.data.regenerate_features or not os.path.exists(feature_path):
            assert data_path is not None, (
                "data_path must be provided if regenerate_features is True or if feature directory does not exist."
            )

            print("Regenerating features from raw data...")
            self.extract_features(data_path, feature_path)
            print(
                "Features successfully extracted and saved to:", feature_path
            )

    def extract_features(self, raw_data_path, output_path) -> None:
        """iterate through every video in the raw data path,
        extract features using the specified feature extractors,
        and save them to the output path.

        Args:
            raw_data_path: Path to folder containing raw video data
            output_path: The file path where extracted features will be saved.
        """

        for file in tqdm.tqdm(os.listdir(raw_data_path)):
            video, audio, info = read_video(
                os.path.join(raw_data_path, file), pts_unit="sec"
            )

            fps = info["video_fps"]

            dt_per_frame = [1 / fps] * (
                video.shape[0]
            )  # Assuming constant frame rate

            # process each frame to extract features using self.feature_extractors
            frame_list = [
                frame.permute(2, 0, 1) for frame in video
            ]  # Convert frames to (C, H, W) format

            feature_dicts = {}
            for feature_name, extrator in self.feature_extractors.items():
                feature_by_frame = extrator.compute_feature(
                    frame_list, dt_per_frame
                )  # Returns a dictionary of features for each frame

                feature_dicts[feature_name] = feature_by_frame

            # Convert feature_dicts into a pandas dataframe and save to output_path
            df = pd.DataFrame(feature_dicts)
            df = df.reset_index().rename(columns={"index": "frame"})
            df["dt"] = 1 / fps  # Add dt column based on video frame rate

            # Save the extracted features to save_path (e.g., as a CSV file)
            save_path = os.path.join(output_path, file.replace(".mp4", ".csv"))

            df.to_csv(save_path, index=False)
