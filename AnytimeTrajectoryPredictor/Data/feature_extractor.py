from torch.utils import data
from torch.utils.data import dataset
import pandas as pd
import os
import tqdm
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
from collections import defaultdict
import random
import pickle
import random
from collections import defaultdict, OrderedDict
from io import BytesIO
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm


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


def _read_existing(path, columns):
    schema = pq.read_schema(path)
    present = [column for column in columns if column in schema.names]
    return pq.read_table(path, columns=present)


def _official_split(path):
    name = path.name
    if name.startswith("training__"):
        return "train"
    if name.startswith("validation__"):
        return "val"
    if name.startswith("testing__"):
        return "test"
    return None


def _select_segments(root, split, max_segments=None, validation_fraction=0.1, split_seed=503):
    if not root.exists():
        raise FileNotFoundError(f"Waymo root does not exist: {root}")
    segments = sorted(path for path in root.iterdir() if path.is_dir())
    official = [path for path in segments if _official_split(path) is not None]
    if official:
        split_source = "official_filename_prefix"
        if split in ("val", "validation"):
            selected = [path for path in official if _official_split(path) == "val"]
        elif split == "train":
            selected = [path for path in official if _official_split(path) == "train"]
        elif split == "test":
            selected = [path for path in official if _official_split(path) == "test"]
        else:
            selected = official
    else:
        split_source = "seeded_fraction"
        rng = random.Random(split_seed)
        shuffled = list(segments)
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * validation_fraction))) if len(shuffled) > 1 else 0
        val_segments = set(shuffled[:val_count])
        if split in ("val", "validation"):
            selected = [segment for segment in segments if segment in val_segments]
        elif split == "train":
            selected = [segment for segment in segments if segment not in val_segments]
        else:
            selected = segments
    if max_segments is not None:
        selected = selected[: int(max_segments)]
    return selected, split_source

def _get_config_value(args, key, default=None):
    return getattr(args, key, default)


    def compute_feature(self, frames, dt_per_frame):
        raise NotImplementedError


class FeatureDataset(dataset.Dataset):
    """Legacy bbox-feature dataset kept for old configs."""

    def __init__(self, args, window=4, future_frames=5, num_objects=1):
        self.window = window
        self.future_frames = future_frames
        self.num_objects = num_objects
        self.image_trajectory_features = args.features.image_trajectories
        columns = self.image_trajectory_features + [
            "trajectory_row_id",
            "frame_timestamp_micros",
            "scene_id",
            "camera_name",
        ]
        self.img_traj_table = pq.read_table(args.image_trajectories_path, columns=columns)
        self._traj_to_indices = defaultdict(list)
        traj_ids = self.img_traj_table.column("trajectory_row_id").to_numpy()
        times = self.img_traj_table.column("frame_timestamp_micros").to_numpy()
        for i, tid in enumerate(traj_ids):
            self._traj_to_indices[tid].append(i)
        for tid, idxs in self._traj_to_indices.items():
            idxs = np.array(idxs)
            idxs = idxs[np.argsort(times[idxs])]
            self._traj_to_indices[tid] = idxs
        min_len = self.window + self.future_frames
        self._traj_ids = [tid for tid, idxs in self._traj_to_indices.items() if len(idxs) >= min_len]

    def __len__(self):
        return len(self._traj_ids)

    def __getitem__(self, idx):
        traj_id = self._traj_ids[idx]
        row_idxs = self._traj_to_indices[traj_id]
        table = self.img_traj_table.take(row_idxs)
        times = table.column("frame_timestamp_micros").to_numpy()
        scenes = table.column("scene_id").to_numpy()
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
        traj = traj[order]
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
        y = torch.cat([torch.stack([vx, vy, v_area], dim=-1), torch.zeros((k, x.shape[1], 3))], dim=0)
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
        return {"features": x.float(), "trajectory": y.float(), "mask": mask.float(), "image_id": image}


class WaymoPredictionDataset(dataset.Dataset):
    """World/local-coordinate Waymo prediction dataset used by the original ASTRA-EDM baseline."""

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
        self.segments, self.split_source = _select_segments(
            self.root,
            split,
            self.max_segments,
            self.validation_fraction,
            self.split_seed,
        )
        self.segments = [segment for segment in self.segments if (segment / "prediction_targets.parquet").exists()]
        self.samples = self._load_samples(self.segments)
        if not self.samples:
            self.samples = self._load_trajectory_samples(self.segments)
        if self.max_samples is not None:
            self.samples = self.samples[: int(self.max_samples)]
        if not self.samples:
            raise RuntimeError(f"No Waymo prediction samples found for split={split} under {self.root}")
        self.target_mean, self.target_std = self._compute_target_stats()

    def _load_samples(self, segments):
        samples = []
        for segment in tqdm(segments, desc=f"Scanning prediction parquet ({self.split})", unit="seg"):
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
                samples.append({column: data_dict[column][row_idx] for column in self.REQUIRED_COLUMNS})
        return samples

    def _load_trajectory_samples(self, segments):
        samples = []
        for segment in tqdm(segments, desc=f"Scanning trajectory parquet ({self.split})", unit="seg"):
            trajectory_path = segment / "trajectories.parquet"
            if not trajectory_path.exists() or pq.ParquetFile(trajectory_path).metadata.num_rows == 0:
                continue
            table = _read_existing(trajectory_path, self.TRAJECTORY_COLUMNS)
            data_dict = table.to_pydict()
            for row_idx in range(table.num_rows):
                num_steps = int(data_dict["num_steps"][row_idx] or 0)
                min_total_steps = self.history_steps + self.min_future_valid
                if num_steps < min_total_steps:
                    continue
                max_start = max(0, num_steps - self.history_steps - self.min_future_valid)
                starts = list(range(0, max_start + 1, max(1, self.trajectory_stride))) or [0]
                if self.max_windows_per_trajectory is not None:
                    starts = starts[: int(self.max_windows_per_trajectory)]
                for start in starts:
                    future_len = max(0, min(num_steps - (start + self.history_steps), self.future_steps))
                    if future_len < self.min_future_valid:
                        continue
                    observed_slice = slice(start, start + self.history_steps)
                    future_slice = slice(start + self.history_steps, start + self.history_steps + self.future_steps)
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
        for sample in tqdm(self.samples, desc=f"Computing target stats ({self.split})", unit="sample"):
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
            [local_obs_x, local_obs_y, rel_heading, local_vx, local_vy, obs_valid.astype(np.float32)],
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


class WaymoImagePlaneDataset(dataset.Dataset):
    """RGB + image-plane bbox dataset. Targets are future bbox centers in camera pixels normalized to [-1, 1]."""

    LINK_COLUMNS = [
        "image_id",
        "scene_id",
        "frame_timestamp_micros",
        "camera_name",
        "camera_name_text",
        "trajectory_row_id",
        "object_type",
        "bbox_center_x",
        "bbox_center_y",
        "bbox_width",
        "bbox_height",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
    ]
    IMAGE_COLUMNS = [
        "image_id",
        "scene_id",
        "frame_timestamp_micros",
        "camera_name",
        "camera_name_text",
        "image_jpeg",
        "image_width",
        "image_height",
    ]

    def __init__(self, args, split="train"):
        self.args = args
        self.split = split
        self.root = Path(_cfg_get(args, "waymo_root", "/work/cs-503/santanto/waymo"))
        self.history_steps = int(_cfg_get(args, "history_steps", 5))
        self.future_steps = int(_cfg_get(args, "future_steps", 20))
        self.validation_fraction = float(_cfg_get(args, "validation_fraction", 0.1))
        self.split_seed = int(_cfg_get(args, "split_seed", 503))
        self.max_segments = _cfg_get(args, "max_segments", None)
        self.max_samples = _cfg_get(args, "max_samples", None)
        self.min_future_valid = int(_cfg_get(args, "min_future_valid", 1))
        self.window_stride = int(_cfg_get(args, "window_stride", self.future_steps))
        self.max_windows_per_track = _cfg_get(args, "max_windows_per_track", 4)
        self.camera_name = _cfg_get(args, "camera_name", 1)
        self.image_height = int(_cfg_get(args, "image_height", 256))
        self.image_width = int(_cfg_get(args, "image_width", 384))
        self.cache_segments = int(_cfg_get(args, "cache_segments", 2))
        self.sample_cache_dir = _cfg_get(args, "sample_cache_dir", None)
        self.use_sample_cache = bool(_cfg_get(args, "use_sample_cache", False))
        self.image_cache_dir = _cfg_get(args, "image_cache_dir", self.sample_cache_dir)
        self.use_image_cache = bool(_cfg_get(args, "use_image_cache", False))
        self.target_mean = [0.0, 0.0]
        self.target_std = [1.0, 1.0]
        self.segments, self.split_source = _select_segments(
            self.root,
            split,
            self.max_segments,
            self.validation_fraction,
            self.split_seed,
        )
        self.segments = [
            segment
            for segment in self.segments
            if (segment / "images.parquet").exists() and (segment / "image_trajectories.parquet").exists()
        ]
        self.samples = None
        if self.use_sample_cache and self.sample_cache_dir is not None:
            self.samples = self._load_sample_cache()
        if self.samples is None:
            self.samples = self._load_samples(self.segments)
        if self.max_samples is not None:
            self.samples = self.samples[: int(self.max_samples)]
        if not self.samples:
            raise RuntimeError(f"No image-plane Waymo samples found for split={split} under {self.root}")
        self._image_cache = OrderedDict()
        self._image_array = None
        self._history_id_to_index = None
        self._image_size_table = None
        if self.use_image_cache and self.image_cache_dir is not None:
            self._load_image_cache()

    def _cache_metadata(self):
        return {
            "version": 1,
            "split": self.split,
            "waymo_root": str(self.root),
            "history_steps": self.history_steps,
            "future_steps": self.future_steps,
            "min_future_valid": self.min_future_valid,
            "window_stride": self.window_stride,
            "max_windows_per_track": self.max_windows_per_track,
            "camera_name": self.camera_name,
            "split_source": self.split_source,
            "segment_names": [segment.name for segment in self.segments],
        }

    def _sample_cache_path(self, output_dir=None):
        cache_dir = output_dir if output_dir is not None else self.sample_cache_dir
        if cache_dir is None:
            return None
        camera = "all" if self.camera_name is None else str(self.camera_name)
        name = (
            f"{self.split}_cam{camera}_H{self.history_steps}_T{self.future_steps}"
            f"_min{self.min_future_valid}_stride{self.window_stride}.pkl"
        )
        return Path(cache_dir) / name

    def _load_sample_cache(self):
        cache_path = self._sample_cache_path()
        if cache_path is None or not cache_path.exists():
            return None
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        metadata = payload.get("metadata", {})
        expected = self._cache_metadata()
        comparable_keys = [
            "version",
            "split",
            "waymo_root",
            "history_steps",
            "future_steps",
            "min_future_valid",
            "window_stride",
            "max_windows_per_track",
            "camera_name",
            "segment_names",
        ]
        mismatches = [key for key in comparable_keys if metadata.get(key) != expected.get(key)]
        if mismatches:
            print(
                f"Ignoring stale image-plane sample cache {cache_path}; "
                f"metadata mismatch: {', '.join(mismatches)}"
            )
            return None
        samples = payload.get("samples")
        if samples is None:
            return None
        self.target_mean = payload.get("target_mean", self.target_mean)
        self.target_std = payload.get("target_std", self.target_std)
        print(f"Loaded image-plane sample cache: {cache_path} ({len(samples)} samples)")
        return samples

    def save_sample_cache(self, output_dir=None):
        cache_path = self._sample_cache_path(output_dir=output_dir)
        if cache_path is None:
            raise ValueError("sample_cache_dir/output_dir must be set before saving a sample cache.")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": self._cache_metadata(),
            "samples": self.samples,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
        }
        with open(cache_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return cache_path

    def _image_cache_paths(self, output_dir=None):
        cache_dir = output_dir if output_dir is not None else self.image_cache_dir
        if cache_dir is None:
            return None, None
        camera = "all" if self.camera_name is None else str(self.camera_name)
        stem = (
            f"{self.split}_cam{camera}_H{self.history_steps}_T{self.future_steps}"
            f"_{self.image_height}x{self.image_width}"
        )
        cache_dir = Path(cache_dir)
        return cache_dir / f"{stem}_history_images.npy", cache_dir / f"{stem}_image_meta.pkl"

    def save_image_cache(self, output_dir=None):
        npy_path, meta_path = self._image_cache_paths(output_dir=output_dir)
        if npy_path is None:
            raise ValueError("image_cache_dir/output_dir must be set before saving an image cache.")
        npy_path.parent.mkdir(parents=True, exist_ok=True)

        history_ids_per_segment = defaultdict(set)
        future_ids_per_segment = defaultdict(set)
        for sample in self.samples:
            segment = str(sample["segment"])
            for row in sample["history"]:
                history_ids_per_segment[segment].add(str(row["image_id"]))
            for row in sample["future"][: self.future_steps]:
                future_ids_per_segment[segment].add(str(row["image_id"]))

        history_id_to_index = {}
        for segment in sorted(history_ids_per_segment):
            for image_id in sorted(history_ids_per_segment[segment]):
                if image_id not in history_id_to_index:
                    history_id_to_index[image_id] = len(history_id_to_index)

        h, w = self.image_height, self.image_width
        arr = np.zeros((len(history_id_to_index), 3, h, w), dtype=np.uint8)
        image_size_table = {}

        segment_keys = sorted(set(history_ids_per_segment) | set(future_ids_per_segment))
        for segment in tqdm(segment_keys, desc=f"Caching images ({self.split})", unit="seg"):
            wanted_history = history_ids_per_segment.get(segment, set())
            wanted_meta = wanted_history | future_ids_per_segment.get(segment, set())
            if not wanted_meta:
                continue
            table = _read_existing(Path(segment) / "images.parquet", self.IMAGE_COLUMNS)
            data = table.to_pydict()
            for row_idx, image_id in enumerate(data["image_id"]):
                image_id = str(image_id)
                if image_id not in wanted_meta:
                    continue
                image_size_table[image_id] = (
                    int(data["image_width"][row_idx]),
                    int(data["image_height"][row_idx]),
                )
                if image_id in wanted_history:
                    image = Image.open(BytesIO(data["image_jpeg"][row_idx])).convert("RGB")
                    image = image.resize((w, h), Image.BILINEAR)
                    arr[history_id_to_index[image_id]] = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)

        np.save(npy_path, arr)
        payload = {
            "metadata": self._cache_metadata(),
            "history_id_to_index": history_id_to_index,
            "image_size_table": image_size_table,
            "image_height": h,
            "image_width": w,
            "shape": tuple(arr.shape),
            "dtype": str(arr.dtype),
        }
        with open(meta_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return npy_path, meta_path

    def _load_image_cache(self):
        npy_path, meta_path = self._image_cache_paths()
        if npy_path is None or not npy_path.exists() or not meta_path.exists():
            return False
        with open(meta_path, "rb") as f:
            payload = pickle.load(f)
        if payload.get("image_height") != self.image_height or payload.get("image_width") != self.image_width:
            print(
                f"Ignoring stale image cache {npy_path}: size mismatch "
                f"({payload.get('image_height')}x{payload.get('image_width')} vs "
                f"{self.image_height}x{self.image_width})"
            )
            return False
        self._image_array = np.load(npy_path, mmap_mode="r")
        self._history_id_to_index = payload["history_id_to_index"]
        self._image_size_table = payload["image_size_table"]
        print(
            f"Loaded image cache: {npy_path} "
            f"({self._image_array.shape[0]} history frames, {len(self._image_size_table)} sized frames)"
        )
        return True

    def _load_samples(self, segments):
        samples = []
        for segment in tqdm(segments, desc=f"Scanning image-plane parquet ({self.split})", unit="seg"):
            link_path = segment / "image_trajectories.parquet"
            if not link_path.exists() or pq.ParquetFile(link_path).metadata.num_rows == 0:
                continue
            table = _read_existing(link_path, self.LINK_COLUMNS)
            data = table.to_pydict()
            groups = defaultdict(list)
            rows = table.num_rows
            for row_idx in range(rows):
                if self.camera_name is not None and int(data["camera_name"][row_idx]) != int(self.camera_name):
                    continue
                key = (int(data["camera_name"][row_idx]), str(data["trajectory_row_id"][row_idx]))
                groups[key].append(row_idx)

            for (camera_name, trajectory_row_id), idxs in groups.items():
                idxs = sorted(idxs, key=lambda i: int(data["frame_timestamp_micros"][i]))
                if len(idxs) < self.history_steps + self.min_future_valid:
                    continue
                max_start = len(idxs) - self.history_steps - self.min_future_valid
                starts = list(range(0, max_start + 1, max(1, self.window_stride))) or [0]
                if self.max_windows_per_track is not None:
                    starts = starts[: int(self.max_windows_per_track)]
                for start in starts:
                    hist_idxs = idxs[start : start + self.history_steps]
                    fut_idxs = idxs[start + self.history_steps : start + self.history_steps + self.future_steps]
                    if len(fut_idxs) < self.min_future_valid:
                        continue
                    samples.append(
                        {
                            "segment": segment,
                            "scene_id": str(data["scene_id"][hist_idxs[-1]]),
                            "camera_name": int(camera_name),
                            "camera_name_text": str(data.get("camera_name_text", [""] * rows)[hist_idxs[-1]]),
                            "trajectory_row_id": trajectory_row_id,
                            "object_type": int(data["object_type"][hist_idxs[-1]]),
                            "history": [self._row_dict(data, i) for i in hist_idxs],
                            "future": [self._row_dict(data, i) for i in fut_idxs],
                        }
                    )
        return samples

    def _row_dict(self, data, row_idx):
        return {column: data[column][row_idx] for column in data}

    def __len__(self):
        return len(self.samples)

    def _load_segment_images(self, segment):
        segment = Path(segment)
        key = str(segment)
        if key in self._image_cache:
            self._image_cache.move_to_end(key)
            return self._image_cache[key]
        table = _read_existing(segment / "images.parquet", self.IMAGE_COLUMNS)
        data = table.to_pydict()
        images = {}
        for i, image_id in enumerate(data["image_id"]):
            images[str(image_id)] = {column: data[column][i] for column in data}
        self._image_cache[key] = images
        while len(self._image_cache) > self.cache_segments:
            self._image_cache.popitem(last=False)
        return images

    def _decode_image(self, image_row):
        image = Image.open(BytesIO(image_row["image_jpeg"])).convert("RGB")
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(self.image_height, self.image_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return tensor

    @staticmethod
    def _center_to_unit(row, image_row):
        width = max(float(image_row["image_width"]), 1.0)
        height = max(float(image_row["image_height"]), 1.0)
        u = 2.0 * float(row["bbox_center_x"]) / width - 1.0
        v = 2.0 * float(row["bbox_center_y"]) / height - 1.0
        return u, v

    @staticmethod
    def _box_to_unit(row, image_row):
        u, v = WaymoImagePlaneDataset._center_to_unit(row, image_row)
        width = max(float(image_row["image_width"]), 1.0)
        height = max(float(image_row["image_height"]), 1.0)
        bw = float(row["bbox_width"]) / width
        bh = float(row["bbox_height"]) / height
        return [u, v, bw, bh]

    def _lookup_image_row(self, sample, image_id, segment_images_cache):
        if self._image_size_table is not None and image_id in self._image_size_table:
            width, height = self._image_size_table[image_id]
            return {"image_width": width, "image_height": height}
        if segment_images_cache["images"] is None:
            segment_images_cache["images"] = self._load_segment_images(sample["segment"])
        return segment_images_cache["images"][image_id]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        segment_images_cache = {"images": None}

        rgb_history = []
        box_history = []
        observed_mask = []
        history_image_ids = []
        for row in sample["history"]:
            image_id = str(row["image_id"])
            if self._image_array is not None and image_id in self._history_id_to_index:
                tensor = torch.from_numpy(
                    np.array(self._image_array[self._history_id_to_index[image_id]], copy=True)
                ).to(torch.float32).div_(255.0)
            else:
                if segment_images_cache["images"] is None:
                    segment_images_cache["images"] = self._load_segment_images(sample["segment"])
                tensor = self._decode_image(segment_images_cache["images"][image_id])
            image_row = self._lookup_image_row(sample, image_id, segment_images_cache)
            rgb_history.append(tensor)
            box_history.append(self._box_to_unit(row, image_row))
            observed_mask.append(1.0)
            history_image_ids.append(image_id)

        trajectory = np.zeros((self.future_steps, 2), dtype=np.float32)
        future_mask = np.zeros((self.future_steps,), dtype=np.float32)
        future_image_ids = []
        for t, row in enumerate(sample["future"][: self.future_steps]):
            image_id = str(row["image_id"])
            image_row = self._lookup_image_row(sample, image_id, segment_images_cache)
            trajectory[t] = np.asarray(self._center_to_unit(row, image_row), dtype=np.float32)
            future_mask[t] = 1.0
            future_image_ids.append(image_id)

        while len(future_image_ids) < self.future_steps:
            future_image_ids.append("")

        return {
            "rgb_history": torch.stack(rgb_history).float(),
            "box_history": torch.tensor(box_history, dtype=torch.float32).unsqueeze(0),
            "features": torch.tensor(box_history, dtype=torch.float32).unsqueeze(0),
            "trajectory": torch.from_numpy(trajectory).unsqueeze(0).float(),
            "observed_mask": torch.tensor(observed_mask, dtype=torch.float32).unsqueeze(0),
            "future_mask": torch.from_numpy(future_mask).unsqueeze(0).float(),
            "camera_name": torch.tensor(int(sample["camera_name"]), dtype=torch.long),
            "object_type": torch.tensor(int(sample["object_type"]), dtype=torch.long),
            "image_size": torch.tensor([self.image_height, self.image_width], dtype=torch.long),
            "scene_id": sample["scene_id"],
            "trajectory_row_id": sample["trajectory_row_id"],
            "history_image_ids": history_image_ids,
            "future_image_ids": future_image_ids,
        }


class WaymoKeypointDataset(dataset.Dataset):
    """One sample per image. Targets are Gaussian heatmaps at bbox centers; used to pretrain a U-Net backbone."""

    LINK_COLUMNS = [
        "image_id",
        "camera_name",
        "bbox_center_x",
        "bbox_center_y",
    ]
    IMAGE_COLUMNS = [
        "image_id",
        "image_jpeg",
        "image_width",
        "image_height",
        "camera_name",
    ]

    def __init__(self, args, split="train"):
        self.args = args
        self.split = split
        self.root = Path(_cfg_get(args, "waymo_root", "/work/cs-503/santanto/waymo"))
        self.validation_fraction = float(_cfg_get(args, "validation_fraction", 0.1))
        self.split_seed = int(_cfg_get(args, "split_seed", 503))
        self.max_segments = _cfg_get(args, "max_segments", None)
        self.max_images = _cfg_get(args, "max_images", None)
        self.camera_name = _cfg_get(args, "camera_name", 1)
        self.image_height = int(_cfg_get(args, "image_height", 256))
        self.image_width = int(_cfg_get(args, "image_width", 384))
        self.heatmap_sigma = float(_cfg_get(args, "heatmap_sigma", 4.0))
        self.cache_segments = int(_cfg_get(args, "cache_segments", 1))
        self.segments, self.split_source = _select_segments(
            self.root,
            split,
            self.max_segments,
            self.validation_fraction,
            self.split_seed,
        )
        self.segments = [
            segment
            for segment in self.segments
            if (segment / "images.parquet").exists() and (segment / "image_trajectories.parquet").exists()
        ]
        self.samples = self._load_samples(self.segments)
        if self.max_images is not None:
            self.samples = self.samples[: int(self.max_images)]
        if not self.samples:
            raise RuntimeError(f"No keypoint samples found for split={split} under {self.root}")
        self._image_cache = OrderedDict()

    def _load_samples(self, segments):
        samples = []
        for segment in tqdm(segments, desc=f"Scanning keypoint parquet ({self.split})", unit="seg"):
            link_path = segment / "image_trajectories.parquet"
            if not link_path.exists() or pq.ParquetFile(link_path).metadata.num_rows == 0:
                continue
            table = _read_existing(link_path, self.LINK_COLUMNS)
            data = table.to_pydict()
            per_image = defaultdict(list)
            for row_idx in range(table.num_rows):
                if self.camera_name is not None and int(data["camera_name"][row_idx]) != int(self.camera_name):
                    continue
                image_id = str(data["image_id"][row_idx])
                per_image[image_id].append(
                    (float(data["bbox_center_x"][row_idx]), float(data["bbox_center_y"][row_idx]))
                )
            for image_id, centers in per_image.items():
                samples.append({"segment": segment, "image_id": image_id, "centers": centers})
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_segment_images(self, segment):
        segment = Path(segment)
        key = str(segment)
        if key in self._image_cache:
            self._image_cache.move_to_end(key)
            return self._image_cache[key]
        table = _read_existing(segment / "images.parquet", self.IMAGE_COLUMNS)
        data = table.to_pydict()
        images = {}
        for i, image_id in enumerate(data["image_id"]):
            if self.camera_name is not None and int(data["camera_name"][i]) != int(self.camera_name):
                continue
            images[str(image_id)] = {column: data[column][i] for column in data}
        self._image_cache[key] = images
        while len(self._image_cache) > self.cache_segments:
            self._image_cache.popitem(last=False)
        return images

    def _decode_image(self, image_row):
        image = Image.open(BytesIO(image_row["image_jpeg"])).convert("RGB")
        image = image.resize((self.image_width, self.image_height), Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def _build_heatmap(self, centers, orig_w, orig_h):
        heatmap = np.zeros((self.image_height, self.image_width), dtype=np.float32)
        if not centers:
            return heatmap
        sx = self.image_width / max(orig_w, 1.0)
        sy = self.image_height / max(orig_h, 1.0)
        sigma = max(self.heatmap_sigma, 1e-3)
        radius = int(np.ceil(3.0 * sigma))
        yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1].astype(np.float32)
        kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
        for cx_pix, cy_pix in centers:
            cx = int(round(cx_pix * sx))
            cy = int(round(cy_pix * sy))
            if cx < 0 or cy < 0 or cx >= self.image_width or cy >= self.image_height:
                continue
            x0, x1 = max(0, cx - radius), min(self.image_width, cx + radius + 1)
            y0, y1 = max(0, cy - radius), min(self.image_height, cy + radius + 1)
            kx0, ky0 = x0 - (cx - radius), y0 - (cy - radius)
            kx1, ky1 = kx0 + (x1 - x0), ky0 + (y1 - y0)
            np.maximum(heatmap[y0:y1, x0:x1], kernel[ky0:ky1, kx0:kx1], out=heatmap[y0:y1, x0:x1])
        return heatmap

    def __getitem__(self, idx):
        sample = self.samples[idx]
        images = self._load_segment_images(sample["segment"])
        image_row = images[sample["image_id"]]
        rgb = self._decode_image(image_row)
        heatmap = self._build_heatmap(
            sample["centers"],
            float(image_row["image_width"]),
            float(image_row["image_height"]),
        )
        return {
            "image": rgb,
            "heatmap": torch.from_numpy(heatmap).unsqueeze(0),
            "image_id": sample["image_id"],
            "num_objects": torch.tensor(len(sample["centers"]), dtype=torch.long),
        }
