from torch.utils import data
from torch.utils.data import dataset
from torchvision.io import read_video
import pandas as pd
import os
import tqdm
import pyarrow.parquet as pq
import numpy as np
from collections import defaultdict
import random


# from features.dummy_feature_1 import DummyFeatureExtractor
import torch

# FEATUREDICT = {"DUMMY_FEATURE": DummyFeatureExtractor}

# Define important columns
TRAJ_ID = "trajectory_row_id"
TIME = "frame_timestamp_micros"


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


class FeatureDataset(dataset.Dataset):
    def __init__(self, args, window=10, future_frames=5, num_objects=5):
        """Dataset class for loading pre-extracted features from CSV files.
        If regenerate_features is True, it will extract features from raw video data.

        args:
            feature_path: The file path where extracted features are stored (e.g., CSV file).
            args: Configuration arguments
            data_path: Path to folder containing raw video data (required if regenerate_features is True)
        """
        # How many frames can we look at a time
        self.window = window

        # How many frames to look into future to plan trajectory
        self.future_frames = future_frames
        self.num_objects = num_objects

        table = pq.read_table(args.image_trajectories_path)
        print("AAA", table.schema.names)
        # Define datasets
        self.image_trajectory_features = args.features.image_trajectories
        self.img_traj_table = pq.read_table(
            args.image_trajectories_path,
            columns=self.image_trajectory_features + [TRAJ_ID, TIME],
        )

        # build index: trajectory_id -> row indices
        self._traj_to_indices = defaultdict(list)

        traj_ids = self.img_traj_table.column(TRAJ_ID).to_numpy()
        times = self.img_traj_table.column(TIME).to_numpy()

        for i, tid in enumerate(traj_ids):
            self._traj_to_indices[tid].append(i)

        # sort each trajectory by timestamp
        for tid, idxs in self._traj_to_indices.items():
            idxs = np.array(idxs)
            idxs = idxs[np.argsort(times[idxs])]
            self._traj_to_indices[tid] = idxs

        self.valid_traj_ids = []

        min_len = self.window + self.future_frames

        for tid, idxs in self._traj_to_indices.items():
            if len(idxs) >= min_len:
                self.valid_traj_ids.append(tid)

        self._traj_ids = self.valid_traj_ids
        self.table = self.img_traj_table

    def __len__(self):
        return len(self._traj_ids)

    def __getitem__(self, idx):
        """
        Returns a dict with the keys:
        'x': shape (num_frames, num_objects, num_features)
        'y': shape (num_frames, num_objects, 3(x_velocity, y_velocity, log size velocity))
        'mask': shape (num_frames, num_objects, 1)
        """
        traj_id = self._traj_ids[idx]
        row_idxs = self._traj_to_indices[traj_id]

        table = self.img_traj_table.take(row_idxs)
        data = table.to_pandas().sort_values(TIME)

        grouped = list(data.groupby(TIME))

        T = min(len(grouped), self.window + self.future_frames)
        grouped = grouped[:T]

        F = 4
        O = self.num_objects

        x = torch.zeros((T, O, F))
        mask = torch.zeros((T, O, 1))

        for t, (_, frame_df) in enumerate(grouped):
            frame_df = frame_df.head(O)
            n = len(frame_df)

            x[t, :n] = torch.tensor(
                frame_df[
                    [
                        "bbox_center_x",
                        "bbox_center_y",
                        "bbox_width",
                        "bbox_height",
                    ]
                ].values
            )

            mask[t, :n] = 1.0

        # --- target ---
        K = self.future_frames

        area = x[:, :, 2] * x[:, :, 3]
        log_area = torch.log(area + 1e-6)

        vx = (x[K:, :, 0] - x[:-K, :, 0]) / K
        vy = (x[K:, :, 1] - x[:-K, :, 1]) / K
        v_area = (log_area[K:] - log_area[:-K]) / K

        y_valid = torch.stack([vx, vy, v_area], dim=-1)
        pad_len = K

        y_pad = torch.zeros((pad_len, x.shape[1], 3))  # (K, O, 3)

        y = torch.cat([y_valid, y_pad], dim=0)

        # --- window sampling ---
        if len(y) - K > self.window:
            start_index = random.randint(0, len(y) - K - self.window)

            x = x[start_index : start_index + self.window]
            y = y[start_index : start_index + self.window]
            mask = mask[start_index : start_index + self.window]

        else:
            x = x[:-K]
            y = y[:-K]
            mask = mask[:-K]
            pad_len = self.window - len(y)

            # pad tensors along time dimension
            x = torch.cat([x, torch.zeros((pad_len, O, F))], dim=0)

            y = torch.cat([y, torch.zeros((pad_len, O, 3))], dim=0)

            mask = torch.cat([mask, torch.zeros((pad_len, O, 1))], dim=0)

        return {
            "feature": x.float(),
            "trajectory": y.float(),
            "mask": mask.float(),
        }

    """
    def extract_features(self, args, data_path, feature_path):
        # Regenerate features if specified in the config
        # NOTE: Once we start adding extra features will make this save as a parquet file
        if args.data.regenerate_features or not os.path.exists(feature_path):
            self.feature_extractors = {
                feature: FEATUREDICT[feature](args)
                for feature in self.features
            }

            assert data_path is not None, (
                "data_path must be provided if regenerate_features is True or if feature directory does not exist."
            )

            print("Regenerating features from raw data...")
            self.extract_features(data_path, feature_path)
            print(
                "Features successfully extracted and saved to:", feature_path
            )

    def extract_features(self, raw_data_path, output_path) -> None:
        iterate through every video in the raw data path,
        extract features using the specified feature extractors,
        and save them to the output path.

        Args:
            raw_data_path: Path to folder containing raw video data
            output_path: The file path where extracted features will be saved.
        

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
        """
