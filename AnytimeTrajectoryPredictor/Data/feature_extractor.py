from torch.utils import data
from torch.utils.data import dataset
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


def fit_polynomial(signal, K, polynomial_degree):
    """
    Fits a polynomial over sliding windows.

    Args:
        signal: (T, B) tensor
        K: window size
        polynomial_degree: degree of the polynomial basis

    Returns:
        coeffs: (T-K+1, B, polynomial_degree + 1)
    """
    if K <= polynomial_degree:
        raise ValueError(
            "feature_extractor.future_frames must be larger than "
            "feature_extractor.polynomial_degree"
        )

    T, B = signal.shape
    device = signal.device

    # normalized time basis [-1, 1]
    t = torch.linspace(-1, 1, K, device=device, dtype=signal.dtype)

    X = torch.stack(
        [t**degree for degree in range(polynomial_degree + 1)],
        dim=1,
    )

    # sliding windows
    sig_win = signal.unfold(0, K, 1)  # (T-K+1, B, K)

    Tn = sig_win.shape[0]

    # flatten batch + time windows
    sig_flat = sig_win.reshape(Tn * B, K)

    # solve least squares, with each sliding window as one target column
    coeffs = torch.linalg.lstsq(X, sig_flat.T).solution.T

    return coeffs.reshape(Tn, B, polynomial_degree + 1)


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
    def __init__(
        self,
        args,
        window=None,
        future_frames=None,
        num_objects=None,
        polynomial_degree=None,
    ):
        """Dataset class for loading pre-extracted features from CSV files.
        If regenerate_features is True, it will extract features from raw video data.

        args:
            feature_path: The file path where extracted features are stored (e.g., CSV file).
            args: Configuration arguments
            data_path: Path to folder containing raw video data (required if regenerate_features is True)
        """
        # How many frames can we look at a time
        self.window = int(args.window if window is None else window)

        # How many frames to look into future to plan trajectory
        self.future_frames = int(
            args.future_frames if future_frames is None else future_frames
        )
        self.num_objects = int(args.num_objects if num_objects is None else num_objects)
        self.polynomial_degree = int(
            args.polynomial_degree
            if polynomial_degree is None
            else polynomial_degree
        )
        self.trajectory_dims = int(args.trajectory_dims)
        self.target_features = args.target_features

        table = pq.read_table(args.image_trajectories_path)
        """
        if path.is_dir():
            parquet_files = sorted(path.glob("*.parquet"))
            tables = [pq.read_table(f) for f in parquet_files]
            table = pa.concat_tables(tables)
        else:
            table = pq.read_table(path)
        """
        # Define datasets
        self.image_trajectory_features = args.features.image_trajectories
        self.x_feature_idx = self._feature_index(self.target_features.x)
        self.y_feature_idx = self._feature_index(self.target_features.y)
        self.width_feature_idx = self._feature_index(self.target_features.width)
        self.height_feature_idx = self._feature_index(self.target_features.height)
        self.num_coeffs = self.polynomial_degree + 1
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
            self._traj_to_indices[tid] = idxs.tolist()

        self.valid_traj_ids = []

        min_len = self.window + self.future_frames

        for tid, idxs in self._traj_to_indices.items():
            if len(idxs) >= min_len:
                self.valid_traj_ids.append(tid)

        self._traj_ids = self.valid_traj_ids
        self.table = self.img_traj_table

    def _feature_index(self, feature_name):
        if feature_name not in self.image_trajectory_features:
            raise ValueError(
                f"Target feature '{feature_name}' is not listed in "
                "feature_extractor.features.image_trajectories"
            )
        return self.image_trajectory_features.index(feature_name)

    def __len__(self):
        return len(self._traj_ids)

    def __getitem__(self, idx):
        """
        Returns a dict with the keys:
        'x': shape (num_frames, num_objects, num_features)
        'y': shape (num_frames, num_objects, trajectory_dims, num_coeffs)
        'mask': shape (num_frames, num_objects, 1)
        """

        traj_id = self._traj_ids[idx]
        row_idxs = self._traj_to_indices[traj_id]

        # --- load subset from Arrow ---
        table = self.img_traj_table.take(row_idxs)

        # --- extract columns as numpy ---
        times = table.column(TIME).to_numpy()

        features = np.stack(
            [
                table.column(feature_name).to_numpy()
                for feature_name in self.image_trajectory_features
            ],
            axis=1,
        )

        # --- sort by time ---
        order = np.argsort(times)
        times = times[order]
        features = features[order]

        # --- group by time (no pandas) ---
        unique_times, indices = np.unique(times, return_index=True)

        T = min(len(unique_times), self.window + self.future_frames)

        F = len(self.image_trajectory_features)
        O = self.num_objects

        x = torch.zeros((T, O, F))
        mask = torch.zeros((T, O, 1))

        # --- fill tensors ---
        for t in range(T):
            start = indices[t]
            end = indices[t + 1] if t + 1 < len(indices) else len(times)

            frame_feats = features[start:end]

            n = min(len(frame_feats), O)

            x[t, :n] = torch.from_numpy(frame_feats[:n])
            mask[t, :n] = 1.0

        # --- target ---
        K = self.future_frames
        target_length = x.shape[0] - K
        if target_length <= 0:
            raise ValueError("Not enough frames to construct polynomial targets")

        area = x[:, :, self.width_feature_idx] * x[:, :, self.height_feature_idx]
        log_area = torch.log(area + 1e-6)

        x_coeffs = fit_polynomial(
            x[:, :, self.x_feature_idx], K, self.polynomial_degree
        )[:target_length]
        y_coeffs = fit_polynomial(
            x[:, :, self.y_feature_idx], K, self.polynomial_degree
        )[:target_length]
        log_area_coeffs = fit_polynomial(
            log_area, K, self.polynomial_degree
        )[:target_length]

        trajectory_coeffs = [x_coeffs, y_coeffs, log_area_coeffs]
        if len(trajectory_coeffs) != self.trajectory_dims:
            raise ValueError(
                "feature_extractor.trajectory_dims must match the generated "
                "polynomial target components"
            )
        y_valid = torch.stack(trajectory_coeffs, dim=2)

        # Pad all tensors to self.window so we can keep everything the same size

        # First make x,y, and mask the same size
        y = y_valid
        x = x[: len(y)]
        mask = mask[: len(y)]

        # --- window sampling ---
        if len(y) > self.window:
            # If the length of y is greater than the window, cut it down
            start_index = random.randint(0, len(y) - self.window)

            x = x[start_index : start_index + self.window]
            y = y[start_index : start_index + self.window]
            mask = mask[start_index : start_index + self.window]

        else:
            # Otherwise, we add padding to everything
            pad_len = self.window - len(y)

            # pad tensors along time dimension
            x = torch.cat([x, torch.zeros((pad_len, O, F))], dim=0)

            y = torch.cat(
                [
                    y,
                    torch.zeros(
                        (
                            pad_len,
                            O,
                            self.trajectory_dims,
                            self.num_coeffs,
                        )
                    ),
                ],
                dim=0,
            )

            mask = torch.cat([mask, torch.zeros((pad_len, O, 1))], dim=0)

        return {
            "features": x.float(),
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
