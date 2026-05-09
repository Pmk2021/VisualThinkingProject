from torch.utils import data
from torch.utils.data import dataset
import pandas as pd
import os
import tqdm
import pyarrow.parquet as pq
import numpy as np
from collections import defaultdict
import random
from pathlib import Path


# from features.dummy_feature_1 import DummyFeatureExtractor
import torch

# FEATUREDICT = {"DUMMY_FEATURE": DummyFeatureExtractor}

# Define important columns
TRAJ_ID = "trajectory_row_id"
TIME = "frame_timestamp_micros"
SCENE_ID = "scene_id"


class FeatureExtractor:
    def __init__(self, args):
        self.args = args

    def compute_feature(self, frames, dt_per_frame):
        """


# Re-declare the production datasets below. A previous scaffold in this file is
# kept for compatibility with old experiments, but these final definitions are
# the ones imported by the training scripts.
class FeatureDataset(dataset.Dataset):
    def __init__(self, args, window=4, future_frames=5, num_objects=1):
        self.window = window
        self.future_frames = future_frames
        self.num_objects = num_objects
        self.image_trajectory_features = args.features.image_trajectories
        self._camera_columns = ["camera_name"]
        self.img_traj_table = pq.read_table(
            args.image_trajectories_path,
            columns=self.image_trajectory_features
            + [TRAJ_ID, TIME, SCENE_ID]
            + self._camera_columns,
        )
        self._traj_to_indices = defaultdict(list)
        traj_ids = self.img_traj_table.column(TRAJ_ID).to_numpy()
        times = self.img_traj_table.column(TIME).to_numpy()
        for i, tid in enumerate(traj_ids):
            self._traj_to_indices[tid].append(i)
        for tid, idxs in self._traj_to_indices.items():
            idxs = np.array(idxs)
            idxs = idxs[np.argsort(times[idxs])]
            self._traj_to_indices[tid] = idxs
        min_len = self.window + self.future_frames
        self._traj_ids = [
            tid for tid, idxs in self._traj_to_indices.items() if len(idxs) >= min_len
        ]

    def __len__(self):
        return len(self._traj_ids)

    def __getitem__(self, idx):
        traj_id = self._traj_ids[idx]
        row_idxs = self._traj_to_indices[traj_id]
        table = self.img_traj_table.take(row_idxs)
        times = table.column(TIME).to_numpy()
        scenes = table.column(SCENE_ID).to_numpy()
        camera_name = table.column("camera_name").to_numpy()
        features = np.stack(
            [
                table.column("bbox_center_x").to_numpy(),
                table.column("bbox_center_y").to_numpy(),
                table.column("bbox_width").to_numpy(),
                table.column("bbox_height").to_numpy(),
            ],
            axis=1,
        )
        order = np.argsort(times)
        times = times[order]
        features = features[order]
        scenes = scenes[order]
        camera_name = camera_name[order]
        _, indices = np.unique(times, return_index=True)
        total_steps = min(len(indices), self.window + self.future_frames)
        x = torch.zeros((total_steps, self.num_objects, 4))
        mask = torch.zeros((total_steps, self.num_objects, 1))
        for t in range(total_steps):
            start = indices[t]
            end = indices[t + 1] if t + 1 < len(indices) else len(times)
            frame_feats = features[start:end]
            n = min(len(frame_feats), self.num_objects)
            x[t, :n] = torch.from_numpy(frame_feats[:n])
            mask[t, :n] = 1.0

        k = self.future_frames
        area = x[:, :, 2] * x[:, :, 3]
        log_area = torch.log(area + 1e-6)
        vx = (x[k:, :, 0] - x[:-k, :, 0]) / k
        vy = (x[k:, :, 1] - x[:-k, :, 1]) / k
        v_area = (log_area[k:] - log_area[:-k]) / k
        y = torch.cat(
            [torch.stack([vx, vy, v_area], dim=-1), torch.zeros((k, x.shape[1], 3))],
            dim=0,
        )
        x = x[: len(y)]
        if len(y) - k > self.window:
            start_index = random.randint(0, len(y) - k - self.window)
            x = x[start_index : start_index + self.window]
            y = y[start_index : start_index + self.window]
            mask = mask[start_index : start_index + self.window]
        else:
            x = x[:-k]
            y = y[:-k]
            mask = mask[:-k]
            pad_len = self.window - len(y)
            x = torch.cat([x, torch.zeros((pad_len, self.num_objects, 4))], dim=0)
            y = torch.cat([y, torch.zeros((pad_len, self.num_objects, 3))], dim=0)
            mask = torch.cat([mask, torch.zeros((pad_len, self.num_objects, 1))], dim=0)

        image = f"{scenes[0]};{times[-1]};camera_{camera_name[-1]}"
        return {
            "features": x.float(),
            "trajectory": y.float(),
            "mask": mask.float(),
            "image_id": image,
        }


class WaymoPredictionDataset(dataset.Dataset):
    REQUIRED_COLUMNS = [
        "scene_id",
        "trajectory_row_id",
        "object_type",
        "observed_valid",
        "observed_x",
        "observed_y",
        "observed_heading",
        "observed_velocity_x",
        "observed_velocity_y",
        "future_valid",
        "future_x",
        "future_y",
        "has_future_gt",
    ]
    TRAJECTORY_COLUMNS = [
        "scene_id",
        "trajectory_id",
        "trajectory_row_id",
        "object_type",
        "num_steps",
        "x",
        "y",
        "heading",
        "velocity_x",
        "velocity_y",
    ]

    def __init__(self, args, split="train"):
        self.args = args
        self.split = split
        self.root = Path(_cfg_get(args, "waymo_root", "/work/cs-503/santanto/waymo"))
        self.history_steps = int(_cfg_get(args, "history_steps", 11))
        self.future_steps = int(_cfg_get(args, "future_steps", 80))
        self.validation_fraction = float(_cfg_get(args, "validation_fraction", 0.1))
        self.split_seed = int(_cfg_get(args, "split_seed", 503))
        self.max_segments = _cfg_get(args, "max_segments", None)
        self.max_samples = _cfg_get(args, "max_samples", None)
        self.min_future_valid = int(_cfg_get(args, "min_future_valid", 1))
        self.trajectory_stride = int(_cfg_get(args, "trajectory_stride", self.future_steps))
        self.max_windows_per_trajectory = _cfg_get(args, "max_windows_per_trajectory", 3)

        segments = self._select_segments()
        self.samples = self._load_samples(segments)
        if not self.samples:
            self.samples = self._load_trajectory_samples(segments)
        if self.max_samples is not None:
            self.samples = self.samples[: int(self.max_samples)]
        if not self.samples:
            raise RuntimeError(
                f"No Waymo prediction samples found for split={split} under {self.root}"
            )
        self.target_mean, self.target_std = self._compute_target_stats()

    def _select_segments(self):
        if not self.root.exists():
            raise FileNotFoundError(f"Waymo root does not exist: {self.root}")
        segments = sorted(
            path
            for path in self.root.iterdir()
            if path.is_dir() and (path / "prediction_targets.parquet").exists()
        )
        if self.max_segments is not None:
            segments = segments[: int(self.max_segments)]
        rng = random.Random(self.split_seed)
        shuffled = list(segments)
        rng.shuffle(shuffled)
        val_count = (
            max(1, int(round(len(shuffled) * self.validation_fraction)))
            if len(shuffled) > 1
            else 0
        )
        val_segments = set(shuffled[:val_count])
        if self.split in ("val", "validation"):
            return [segment for segment in segments if segment in val_segments]
        if self.split == "train":
            return [segment for segment in segments if segment not in val_segments]
        return segments

    def _load_samples(self, segments):
        samples = []
        for segment in segments:
            prediction_path = segment / "prediction_targets.parquet"
            parquet_file = pq.ParquetFile(prediction_path)
            if parquet_file.metadata.num_rows == 0:
                continue
            schema_names = set(parquet_file.schema_arrow.names)
            if not set(self.REQUIRED_COLUMNS).issubset(schema_names):
                continue
            table = pq.read_table(prediction_path, columns=self.REQUIRED_COLUMNS)
            data_dict = table.to_pydict()
            for row_idx in range(table.num_rows):
                if not data_dict["has_future_gt"][row_idx]:
                    continue
                future_valid = _to_bool_array(data_dict["future_valid"][row_idx], self.future_steps)
                if int(future_valid.sum()) < self.min_future_valid:
                    continue
                samples.append(
                    {column: data_dict[column][row_idx] for column in self.REQUIRED_COLUMNS}
                )
        return samples

    def _load_trajectory_samples(self, segments):
        samples = []
        for segment in segments:
            trajectory_path = segment / "trajectories.parquet"
            parquet_file = pq.ParquetFile(trajectory_path)
            if parquet_file.metadata.num_rows == 0:
                continue
            table = pq.read_table(trajectory_path, columns=self.TRAJECTORY_COLUMNS)
            data_dict = table.to_pydict()
            for row_idx in range(table.num_rows):
                num_steps = int(data_dict["num_steps"][row_idx] or 0)
                min_total_steps = self.history_steps + self.min_future_valid
                if num_steps < min_total_steps:
                    continue
                max_start = max(0, num_steps - self.history_steps - self.min_future_valid)
                starts = list(range(0, max_start + 1, max(1, self.trajectory_stride)))
                if not starts:
                    starts = [0]
                if self.max_windows_per_trajectory is not None:
                    starts = starts[: int(self.max_windows_per_trajectory)]
                for start in starts:
                    observed_slice = slice(start, start + self.history_steps)
                    future_slice = slice(start + self.history_steps, start + self.history_steps + self.future_steps)
                    future_len = max(0, min(num_steps - (start + self.history_steps), self.future_steps))
                    if future_len < self.min_future_valid:
                        continue
                    sample = {
                        "scene_id": data_dict["scene_id"][row_idx],
                        "trajectory_row_id": data_dict["trajectory_row_id"][row_idx],
                        "object_type": data_dict["object_type"][row_idx],
                        "observed_valid": [True] * min(self.history_steps, num_steps - start),
                        "observed_x": data_dict["x"][row_idx][observed_slice],
                        "observed_y": data_dict["y"][row_idx][observed_slice],
                        "observed_heading": data_dict["heading"][row_idx][observed_slice],
                        "observed_velocity_x": data_dict["velocity_x"][row_idx][observed_slice],
                        "observed_velocity_y": data_dict["velocity_y"][row_idx][observed_slice],
                        "future_valid": [True] * future_len,
                        "future_x": data_dict["x"][row_idx][future_slice],
                        "future_y": data_dict["y"][row_idx][future_slice],
                        "has_future_gt": True,
                    }
                    samples.append(sample)
        return samples

    def _compute_target_stats(self):
        values = []
        for sample in self.samples:
            _, trajectory, _, future_mask = self._build_tensors(sample)
            valid = future_mask.astype(bool)
            if valid.any():
                values.append(trajectory[valid])
        if not values:
            return [0.0, 0.0], [1.0, 1.0]
        stacked = np.concatenate(values, axis=0)
        mean = stacked.mean(axis=0).astype(np.float32)
        std = np.maximum(stacked.std(axis=0).astype(np.float32), 1e-3)
        return mean.tolist(), std.tolist()

    def __len__(self):
        return len(self.samples)

    def _build_tensors(self, sample):
        obs_valid = _to_bool_array(sample["observed_valid"], self.history_steps)
        obs_x = _to_float_array(sample["observed_x"], self.history_steps)
        obs_y = _to_float_array(sample["observed_y"], self.history_steps)
        obs_heading = _to_float_array(sample["observed_heading"], self.history_steps)
        obs_vx = _to_float_array(sample["observed_velocity_x"], self.history_steps)
        obs_vy = _to_float_array(sample["observed_velocity_y"], self.history_steps)
        fut_x = _to_float_array(sample["future_x"], self.future_steps)
        fut_y = _to_float_array(sample["future_y"], self.future_steps)
        fut_valid = _to_bool_array(sample["future_valid"], self.future_steps)

        valid_indices = np.flatnonzero(obs_valid)
        anchor_idx = valid_indices[-1] if len(valid_indices) else self.history_steps - 1
        anchor_x = obs_x[anchor_idx]
        anchor_y = obs_y[anchor_idx]
        anchor_heading = obs_heading[anchor_idx]
        local_obs_x, local_obs_y = _rotate_xy(obs_x - anchor_x, obs_y - anchor_y, anchor_heading)
        local_vx, local_vy = _rotate_xy(obs_vx, obs_vy, anchor_heading)
        rel_heading = _wrap_angle_np(obs_heading - anchor_heading)
        features = np.stack(
            [
                local_obs_x,
                local_obs_y,
                rel_heading,
                local_vx,
                local_vy,
                obs_valid.astype(np.float32),
            ],
            axis=-1,
        ).astype(np.float32)
        local_fut_x, local_fut_y = _rotate_xy(fut_x - anchor_x, fut_y - anchor_y, anchor_heading)
        trajectory = np.stack([local_fut_x, local_fut_y], axis=-1).astype(np.float32)
        return features, trajectory, obs_valid.astype(np.float32), fut_valid.astype(np.float32)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features, trajectory, observed_mask, future_mask = self._build_tensors(sample)
        return {
            "features": torch.from_numpy(features).unsqueeze(0).float(),
            "trajectory": torch.from_numpy(trajectory).unsqueeze(0).float(),
            "observed_mask": torch.from_numpy(observed_mask).unsqueeze(0).float(),
            "future_mask": torch.from_numpy(future_mask).unsqueeze(0).float(),
            "scene_id": str(sample["scene_id"]),
            "trajectory_row_id": str(sample["trajectory_row_id"]),
            "object_type": int(sample["object_type"]),
        }
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
    def __init__(self, args, window=4, future_frames=5, num_objects=1):
        """Dataset class for loading pre-extracted features from CSV files.
        If regenerate_features is True, it will extract features from raw video data.

        args:
            feature_path: The file path where extracted features are stored (e.g., CSV file).
            args: Configuration arguments
            data_path: Path to folder containing raw video data (required if regenerate_features is True)
        """


def _cfg_get(config, key, default=None):
    return getattr(config, key, default) if config is not None else default


def _to_float_array(values, length, fill_value=0.0):
    arr = np.asarray(values if values is not None else [], dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    out = np.full((length,), fill_value, dtype=np.float32)
    n = min(len(arr), length)
    if n:
        out[:n] = arr[:n]
    return out


def _to_bool_array(values, length):
    arr = np.asarray(values if values is not None else [], dtype=bool)
    out = np.zeros((length,), dtype=bool)
    n = min(len(arr), length)
    if n:
        out[:n] = arr[:n]
    return out


def _rotate_xy(x, y, heading):
    cos_h = np.cos(-heading)
    sin_h = np.sin(-heading)
    return cos_h * x - sin_h * y, sin_h * x + cos_h * y


def _wrap_angle_np(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


class WaymoPredictionDataset(dataset.Dataset):
    """Waymo prediction-target dataset backed by per-segment parquet folders."""

    REQUIRED_COLUMNS = [
        "scene_id",
        "trajectory_row_id",
        "object_type",
        "observed_valid",
        "observed_x",
        "observed_y",
        "observed_heading",
        "observed_velocity_x",
        "observed_velocity_y",
        "future_valid",
        "future_x",
        "future_y",
        "has_future_gt",
    ]

    def __init__(self, args, split="train"):
        self.args = args
        self.split = split
        self.root = Path(_cfg_get(args, "waymo_root", "/work/cs-503/santanto/waymo"))
        self.history_steps = int(_cfg_get(args, "history_steps", 11))
        self.future_steps = int(_cfg_get(args, "future_steps", 80))
        self.validation_fraction = float(_cfg_get(args, "validation_fraction", 0.1))
        self.split_seed = int(_cfg_get(args, "split_seed", 503))
        self.max_segments = _cfg_get(args, "max_segments", None)
        self.max_samples = _cfg_get(args, "max_samples", None)
        self.min_future_valid = int(_cfg_get(args, "min_future_valid", 1))

        segments = self._select_segments()
        self.samples = self._load_samples(segments)
        if self.max_samples is not None:
            self.samples = self.samples[: int(self.max_samples)]
        if not self.samples:
            raise RuntimeError(
                f"No Waymo prediction samples found for split={split} under {self.root}"
            )
        self.target_mean, self.target_std = self._compute_target_stats()

    def _select_segments(self):
        if not self.root.exists():
            raise FileNotFoundError(f"Waymo root does not exist: {self.root}")

        segments = sorted(
            path
            for path in self.root.iterdir()
            if path.is_dir() and (path / "prediction_targets.parquet").exists()
        )
        if self.max_segments is not None:
            segments = segments[: int(self.max_segments)]

        rng = random.Random(self.split_seed)
        shuffled = list(segments)
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * self.validation_fraction))) if len(shuffled) > 1 else 0
        val_segments = set(shuffled[:val_count])
        if self.split in ("val", "validation"):
            selected = [segment for segment in segments if segment in val_segments]
        elif self.split == "train":
            selected = [segment for segment in segments if segment not in val_segments]
        else:
            selected = segments
        return selected

    def _load_samples(self, segments):
        samples = []
        for segment in segments:
            table = pq.read_table(segment / "prediction_targets.parquet", columns=self.REQUIRED_COLUMNS)
            data_dict = table.to_pydict()
            num_rows = table.num_rows
            for row_idx in range(num_rows):
                if not data_dict["has_future_gt"][row_idx]:
                    continue
                future_valid = _to_bool_array(data_dict["future_valid"][row_idx], self.future_steps)
                if int(future_valid.sum()) < self.min_future_valid:
                    continue
                samples.append(
                    {
                        column: data_dict[column][row_idx]
                        for column in self.REQUIRED_COLUMNS
                    }
                )
        return samples

    def _compute_target_stats(self):
        values = []
        for sample in self.samples:
            _, trajectory, _, future_mask = self._build_tensors(sample)
            valid = future_mask.astype(bool)
            if valid.any():
                values.append(trajectory[valid])
        if not values:
            return [0.0, 0.0], [1.0, 1.0]
        stacked = np.concatenate(values, axis=0)
        mean = stacked.mean(axis=0).astype(np.float32)
        std = stacked.std(axis=0).astype(np.float32)
        std = np.maximum(std, 1e-3)
        return mean.tolist(), std.tolist()

    def __len__(self):
        return len(self.samples)

    def _build_tensors(self, sample):
        obs_valid = _to_bool_array(sample["observed_valid"], self.history_steps)
        obs_x = _to_float_array(sample["observed_x"], self.history_steps)
        obs_y = _to_float_array(sample["observed_y"], self.history_steps)
        obs_heading = _to_float_array(sample["observed_heading"], self.history_steps)
        obs_vx = _to_float_array(sample["observed_velocity_x"], self.history_steps)
        obs_vy = _to_float_array(sample["observed_velocity_y"], self.history_steps)
        fut_x = _to_float_array(sample["future_x"], self.future_steps)
        fut_y = _to_float_array(sample["future_y"], self.future_steps)
        fut_valid = _to_bool_array(sample["future_valid"], self.future_steps)

        valid_indices = np.flatnonzero(obs_valid)
        anchor_idx = valid_indices[-1] if len(valid_indices) else self.history_steps - 1
        anchor_x = obs_x[anchor_idx]
        anchor_y = obs_y[anchor_idx]
        anchor_heading = obs_heading[anchor_idx]

        local_obs_x, local_obs_y = _rotate_xy(obs_x - anchor_x, obs_y - anchor_y, anchor_heading)
        local_vx, local_vy = _rotate_xy(obs_vx, obs_vy, anchor_heading)
        rel_heading = _wrap_angle_np(obs_heading - anchor_heading)
        features = np.stack(
            [
                local_obs_x,
                local_obs_y,
                rel_heading,
                local_vx,
                local_vy,
                obs_valid.astype(np.float32),
            ],
            axis=-1,
        ).astype(np.float32)

        local_fut_x, local_fut_y = _rotate_xy(fut_x - anchor_x, fut_y - anchor_y, anchor_heading)
        trajectory = np.stack([local_fut_x, local_fut_y], axis=-1).astype(np.float32)
        return features, trajectory, obs_valid.astype(np.float32), fut_valid.astype(np.float32)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features, trajectory, observed_mask, future_mask = self._build_tensors(sample)
        return {
            "features": torch.from_numpy(features).unsqueeze(0).float(),
            "trajectory": torch.from_numpy(trajectory).unsqueeze(0).float(),
            "observed_mask": torch.from_numpy(observed_mask).unsqueeze(0).float(),
            "future_mask": torch.from_numpy(future_mask).unsqueeze(0).float(),
            "scene_id": str(sample["scene_id"]),
            "trajectory_row_id": str(sample["trajectory_row_id"]),
            "object_type": int(sample["object_type"]),
        }
        # How many frames can we look at a time
        self.window = window

        # How many frames to look into future to plan trajectory
        self.future_frames = future_frames
        self.num_objects = num_objects

        # Define datasets
        self.image_trajectory_features = args.features.image_trajectories
        self._camera_columns = ["camera_name"]

        self.img_traj_table = pq.read_table(
            args.image_trajectories_path,
            columns=self.image_trajectory_features + [TRAJ_ID, TIME, SCENE_ID] + self._camera_columns,
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

        # --- load subset from Arrow ---
        table = self.img_traj_table.take(row_idxs)

        # --- extract columns as numpy ---
        times = table.column(TIME).to_numpy()
        scenes = table.column(SCENE_ID).to_numpy()
        camera_name = table.column("camera_name").to_numpy()

        features = np.stack(
            [
                table.column("bbox_center_x").to_numpy(),
                table.column("bbox_center_y").to_numpy(),
                table.column("bbox_width").to_numpy(),
                table.column("bbox_height").to_numpy(),
            ],
            axis=1,
        )

        # --- sort by time ---
        order = np.argsort(times)
        times = times[order]
        features = features[order]
        scenes = scenes[order]
        camera_name = camera_name[order]

        # --- group by time (no pandas) ---
        unique_times, indices = np.unique(times, return_index=True)

        T = min(len(unique_times), self.window + self.future_frames)

        F = 4
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

        area = x[:, :, 2] * x[:, :, 3]
        log_area = torch.log(area + 1e-6)

        vx = (x[K:, :, 0] - x[:-K, :, 0]) / K
        vy = (x[K:, :, 1] - x[:-K, :, 1]) / K
        v_area = (log_area[K:] - log_area[:-K]) / K

        y_valid = torch.stack([vx, vy, v_area], dim=-1)
        pad_len = K

        y_pad = torch.zeros((pad_len, x.shape[1], 3))  # (K, O, 3)

        y = torch.cat([y_valid, y_pad], dim=0)
        x = x[: len(y)]
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

        image = f"{scenes[0]};{times[-1]};camera_{camera_name[-1]}"
        
        return {
            "features": x.float(),
            "trajectory": y.float(),
            "mask": mask.float(),
            "image_id": image,
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


class WaymoPredictionDataset(dataset.Dataset):
    REQUIRED_COLUMNS = [
        "scene_id",
        "trajectory_row_id",
        "object_type",
        "observed_valid",
        "observed_x",
        "observed_y",
        "observed_heading",
        "observed_velocity_x",
        "observed_velocity_y",
        "future_valid",
        "future_x",
        "future_y",
        "has_future_gt",
    ]
    TRAJECTORY_COLUMNS = [
        "scene_id",
        "trajectory_id",
        "trajectory_row_id",
        "object_type",
        "num_steps",
        "x",
        "y",
        "heading",
        "velocity_x",
        "velocity_y",
    ]

    def __init__(self, args, split="train"):
        self.args = args
        self.split = split
        self.root = Path(_cfg_get(args, "waymo_root", "/work/cs-503/santanto/waymo"))
        self.history_steps = int(_cfg_get(args, "history_steps", 11))
        self.future_steps = int(_cfg_get(args, "future_steps", 80))
        self.validation_fraction = float(_cfg_get(args, "validation_fraction", 0.1))
        self.split_seed = int(_cfg_get(args, "split_seed", 503))
        self.max_segments = _cfg_get(args, "max_segments", None)
        self.max_samples = _cfg_get(args, "max_samples", None)
        self.min_future_valid = int(_cfg_get(args, "min_future_valid", 1))
        self.trajectory_stride = int(_cfg_get(args, "trajectory_stride", self.future_steps))
        self.max_windows_per_trajectory = _cfg_get(args, "max_windows_per_trajectory", 3)

        segments = self._select_segments()
        self.samples = self._load_prediction_samples(segments)
        if not self.samples:
            self.samples = self._load_trajectory_samples(segments)
        if self.max_samples is not None:
            self.samples = self.samples[: int(self.max_samples)]
        if not self.samples:
            raise RuntimeError(
                f"No Waymo prediction samples found for split={split} under {self.root}"
            )
        self.target_mean, self.target_std = self._compute_target_stats()

    def _select_segments(self):
        if not self.root.exists():
            raise FileNotFoundError(f"Waymo root does not exist: {self.root}")
        segments = sorted(
            path
            for path in self.root.iterdir()
            if path.is_dir() and (path / "prediction_targets.parquet").exists()
        )
        if self.max_segments is not None:
            segments = segments[: int(self.max_segments)]
        rng = random.Random(self.split_seed)
        shuffled = list(segments)
        rng.shuffle(shuffled)
        val_count = (
            max(1, int(round(len(shuffled) * self.validation_fraction)))
            if len(shuffled) > 1
            else 0
        )
        val_segments = set(shuffled[:val_count])
        if self.split in ("val", "validation"):
            return [segment for segment in segments if segment in val_segments]
        if self.split == "train":
            return [segment for segment in segments if segment not in val_segments]
        return segments

    def _load_prediction_samples(self, segments):
        samples = []
        required = set(self.REQUIRED_COLUMNS)
        for segment in segments:
            prediction_path = segment / "prediction_targets.parquet"
            parquet_file = pq.ParquetFile(prediction_path)
            if parquet_file.metadata.num_rows == 0:
                continue
            if not required.issubset(set(parquet_file.schema_arrow.names)):
                continue
            table = pq.read_table(prediction_path, columns=self.REQUIRED_COLUMNS)
            data_dict = table.to_pydict()
            for row_idx in range(table.num_rows):
                if not data_dict["has_future_gt"][row_idx]:
                    continue
                future_valid = _to_bool_array(data_dict["future_valid"][row_idx], self.future_steps)
                if int(future_valid.sum()) < self.min_future_valid:
                    continue
                samples.append(
                    {column: data_dict[column][row_idx] for column in self.REQUIRED_COLUMNS}
                )
        return samples

    def _load_trajectory_samples(self, segments):
        samples = []
        for segment in segments:
            trajectory_path = segment / "trajectories.parquet"
            parquet_file = pq.ParquetFile(trajectory_path)
            if parquet_file.metadata.num_rows == 0:
                continue
            table = pq.read_table(trajectory_path, columns=self.TRAJECTORY_COLUMNS)
            data_dict = table.to_pydict()
            for row_idx in range(table.num_rows):
                num_steps = int(data_dict["num_steps"][row_idx] or 0)
                if num_steps < self.history_steps + self.min_future_valid:
                    continue
                max_start = max(0, num_steps - self.history_steps - self.min_future_valid)
                starts = list(range(0, max_start + 1, max(1, self.trajectory_stride))) or [0]
                if self.max_windows_per_trajectory is not None:
                    starts = starts[: int(self.max_windows_per_trajectory)]
                for start in starts:
                    future_start = start + self.history_steps
                    future_len = max(0, min(num_steps - future_start, self.future_steps))
                    if future_len < self.min_future_valid:
                        continue
                    observed_slice = slice(start, start + self.history_steps)
                    future_slice = slice(future_start, future_start + self.future_steps)
                    samples.append(
                        {
                            "scene_id": data_dict["scene_id"][row_idx],
                            "trajectory_row_id": data_dict["trajectory_row_id"][row_idx],
                            "object_type": data_dict["object_type"][row_idx],
                            "observed_valid": [True] * min(self.history_steps, num_steps - start),
                            "observed_x": data_dict["x"][row_idx][observed_slice],
                            "observed_y": data_dict["y"][row_idx][observed_slice],
                            "observed_heading": data_dict["heading"][row_idx][observed_slice],
                            "observed_velocity_x": data_dict["velocity_x"][row_idx][observed_slice],
                            "observed_velocity_y": data_dict["velocity_y"][row_idx][observed_slice],
                            "future_valid": [True] * future_len,
                            "future_x": data_dict["x"][row_idx][future_slice],
                            "future_y": data_dict["y"][row_idx][future_slice],
                            "has_future_gt": True,
                        }
                    )
        return samples

    def _compute_target_stats(self):
        values = []
        for sample in self.samples:
            _, trajectory, _, future_mask = self._build_tensors(sample)
            valid = future_mask.astype(bool)
            if valid.any():
                values.append(trajectory[valid])
        if not values:
            return [0.0, 0.0], [1.0, 1.0]
        stacked = np.concatenate(values, axis=0)
        mean = stacked.mean(axis=0).astype(np.float32)
        std = np.maximum(stacked.std(axis=0).astype(np.float32), 1e-3)
        return mean.tolist(), std.tolist()

    def __len__(self):
        return len(self.samples)

    def _build_tensors(self, sample):
        obs_valid = _to_bool_array(sample["observed_valid"], self.history_steps)
        obs_x = _to_float_array(sample["observed_x"], self.history_steps)
        obs_y = _to_float_array(sample["observed_y"], self.history_steps)
        obs_heading = _to_float_array(sample["observed_heading"], self.history_steps)
        obs_vx = _to_float_array(sample["observed_velocity_x"], self.history_steps)
        obs_vy = _to_float_array(sample["observed_velocity_y"], self.history_steps)
        fut_x = _to_float_array(sample["future_x"], self.future_steps)
        fut_y = _to_float_array(sample["future_y"], self.future_steps)
        fut_valid = _to_bool_array(sample["future_valid"], self.future_steps)

        valid_indices = np.flatnonzero(obs_valid)
        anchor_idx = valid_indices[-1] if len(valid_indices) else self.history_steps - 1
        anchor_x = obs_x[anchor_idx]
        anchor_y = obs_y[anchor_idx]
        anchor_heading = obs_heading[anchor_idx]
        local_obs_x, local_obs_y = _rotate_xy(obs_x - anchor_x, obs_y - anchor_y, anchor_heading)
        local_vx, local_vy = _rotate_xy(obs_vx, obs_vy, anchor_heading)
        rel_heading = _wrap_angle_np(obs_heading - anchor_heading)
        features = np.stack(
            [
                local_obs_x,
                local_obs_y,
                rel_heading,
                local_vx,
                local_vy,
                obs_valid.astype(np.float32),
            ],
            axis=-1,
        ).astype(np.float32)
        local_fut_x, local_fut_y = _rotate_xy(fut_x - anchor_x, fut_y - anchor_y, anchor_heading)
        trajectory = np.stack([local_fut_x, local_fut_y], axis=-1).astype(np.float32)
        return features, trajectory, obs_valid.astype(np.float32), fut_valid.astype(np.float32)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features, trajectory, observed_mask, future_mask = self._build_tensors(sample)
        return {
            "features": torch.from_numpy(features).unsqueeze(0).float(),
            "trajectory": torch.from_numpy(trajectory).unsqueeze(0).float(),
            "observed_mask": torch.from_numpy(observed_mask).unsqueeze(0).float(),
            "future_mask": torch.from_numpy(future_mask).unsqueeze(0).float(),
            "scene_id": str(sample["scene_id"]),
            "trajectory_row_id": str(sample["trajectory_row_id"]),
            "object_type": int(sample["object_type"]),
        }
