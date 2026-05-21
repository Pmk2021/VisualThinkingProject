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
from pathlib import Path


# from features.dummy_feature_1 import DummyFeatureExtractor
import torch

# FEATUREDICT = {"DUMMY_FEATURE": DummyFeatureExtractor}

# Define important columns
TRAJ_ID = "trajectory_row_id"
TIME = "frame_timestamp_micros"
LATENT_FEATURE_COLS = [f"local_latent_features_{i}" for i in range(192)]
DATAFOLDERCOL = "DATAFOLDERCOL"

def _get_config_value(args, key, default=None):
    return getattr(args, key, default)


def _get_table_filename(args, table_name):
    tables = _get_config_value(args, "tables")
    if tables is not None:
        table_filename = _get_config_value(
            tables, table_name, f"{table_name}.parquet"
        )
    else:
        table_filename = f"{table_name}.parquet"

    return str(table_filename)


def _resolve_table_files(args, dataset_root, table_name, legacy_path_key, split=None):
    table_filename = _get_table_filename(args, table_name)
    

    if dataset_root is not None:
        root = Path(str(dataset_root))

        if split is not None and root.exists():
            split_files = sorted(
                path / table_filename
                for path in root.iterdir()
                if path.is_dir() and path.name.startswith(f"{split}__")
            )
            split_files = [path for path in split_files if path.exists()]
            if split_files:
                return split_files

        flat_file = root / table_filename
        if flat_file.exists():
            return [flat_file]

        recursive_files = sorted(root.rglob(table_filename))
        if recursive_files:
            return recursive_files

        raise FileNotFoundError(
            f"No {table_filename} files found under dataset_root={root}"
        )

    legacy_path = _get_config_value(args, legacy_path_key)
    if legacy_path is None:
        raise ValueError(
            f"Expected either dataset_root or {legacy_path_key} in feature_extractor config"
        )

    path = Path(str(legacy_path))
    if path.is_dir():
        if split is not None:
            split_files = sorted(
                file
                for file in path.rglob(table_filename)
                if any(parent.name.startswith(f"{split}__") for parent in file.parents)
            )
            if split_files:
                return split_files

        parquet_files = sorted(path.rglob(table_filename))
        if parquet_files:
            return parquet_files

        raise FileNotFoundError(f"No {table_filename} files found under {path}")

    return [path]



def _read_table_files(files, columns=None, save=False):

    tables = []

    for file in files:
        table = pq.read_table(file, columns=columns)

        folder_name = Path(file).parent.name

        # FORCE consistent type (string, not null)
        folder_col = pa.array(
            [folder_name] * table.num_rows,
            type=pa.string()
        )

        table = table.append_column(DATAFOLDERCOL, folder_col)

        tables.append(table)

    if len(tables) == 1:
        return tables[0]

    return pa.concat_tables(tables)


def fit_cubic(signal, K):
    """
    Fits a cubic polynomial over sliding windows of size K.

    Args:
        signal: (T, B) tensor
        K: window size

    Returns:
        coeffs: (T-K+1, B, 4)
    """

    T, B = signal.shape
    device = signal.device

    # normalized time basis [-1, 1]
    t = torch.linspace(-1, 1, K, device=device)

    X = torch.stack(
        [t**0, t**1, t**2, t**3],
        dim=1,
    )  # (K, 4)

    # sliding windows
    sig_win = signal.unfold(0, K, 1)  # (T-K+1, B, K)

    Tn = sig_win.shape[0]

    # flatten batch + time windows
    sig_flat = sig_win.reshape(Tn * B, K).T  # (K, Tn*B)

    # solve least squares
    coeffs = torch.linalg.lstsq(X, sig_flat).solution.T  # (Tn*B, 4)

    return coeffs.reshape(Tn, B, 4)




class FeatureDataset(dataset.Dataset):
    def __init__(self, args, split=None, window=4, future_frames=5, num_objects=5):
        """Dataset class for loading pre-extracted features from CSV files.
        If regenerate_features is True, it will extract features from raw video data.

        args:
            feature_path: The file path where extracted features are stored (e.g., CSV file).
            args: Configuration arguments
            data_path: Path to folder containing raw video data (required if regenerate_features is True)
        """
        # How many frames can we look at a time
        self.window = window
        self.args = args
        # How many frames to look into future to plan trajectory
        self.future_frames = future_frames
        self.num_objects = args.num_objects if hasattr(args, "num_objects") else num_objects
        print("A")
        # Define datasets
        self.image_trajectory_features = args.features.image_trajectories
        image_trajectory_files = _resolve_table_files(
            args,
            dataset_root=args.dataset_root,
            table_name="image_trajectories",
            legacy_path_key="image_trajectories_path",
            split=split,
        )
        print("B")
        latent_feature_files = _resolve_table_files(
            args,
            dataset_root=args.feature_root,
            table_name="fe_gt_local_latent_features",
            legacy_path_key="fe_gt_local_latent_features_path",
            split=split,
        )
        print("C")
        self.img_traj_table = _read_table_files(
            image_trajectory_files,
            columns=self.image_trajectory_features + [TRAJ_ID, TIME],
        )


        print("E")
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
        print("F")
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
        'y': shape (num_frames, num_objects, 3, 4) the last two dimentions represent x, y, z x 4 coefficients of a 3 degree polynomial
        'mask': shape (num_frames, num_objects, 1)
        """

        traj_id = self._traj_ids[idx]
        row_idxs = self._traj_to_indices[traj_id]

        # --- load subset from Arrow ---
        table = self.img_traj_table.take(row_idxs)
        table = table.set_column(
            table.schema.get_field_index(TRAJ_ID),
            TRAJ_ID,
            table[TRAJ_ID].cast(pa.string())
        )
        # --- extract columns as numpy ---
        times = table.column(TIME).to_numpy()

        bbox_cols = [
            "bbox_center_x",
            "bbox_center_y",
            "bbox_width",
            "bbox_height",
        ]

        # --- load latent features from folder-specific parquet ---
        datafolder = table.column(DATAFOLDERCOL)[0].as_py()

        latent_path = Path(self.args.feature_root) / datafolder / "fe_gt_local_latent_features.parquet"

        latent_table = pq.read_table(
            latent_path,
            columns=[TRAJ_ID, TIME] + LATENT_FEATURE_COLS
        )
        latent_table = latent_table.set_column(
            latent_table.schema.get_field_index(TRAJ_ID),
            TRAJ_ID,
            latent_table[TRAJ_ID].cast(pa.string())
        )

        # --- merge on TRAJ_ID, TIME ---
        merged = (
            table.to_pandas()
            .merge(
                latent_table.to_pandas(),
                on=[TRAJ_ID, TIME],
                how="inner"
            )
            .sort_values(TIME)   
        )

        feature_cols = bbox_cols + LATENT_FEATURE_COLS

        times = merged[TIME].to_numpy()
        traj = merged[TRAJ_ID].to_numpy()
        features = merged[feature_cols].to_numpy()

        

        # --- sort by time ---
        order = np.argsort(times)
        times = times[order]
        traj = traj[order]
        features = features[order]

        # --- group by time (no pandas) ---
        unique_times, indices = np.unique(times, return_index=True)

        T = min(len(unique_times), self.window + self.future_frames)

        F = 196  # 4 bbox + 192 latent features
        O = self.num_objects

        x = torch.zeros((T, O, F))
        mask = torch.zeros((T, O, 1))

        # --- tracking state ---
        active_map = {}  # object_id -> slot
        free_slots = list(range(O))

        for t in range(T):

            start = indices[t]
            end = indices[t + 1] if t + 1 < len(indices) else len(times)

            frame_feats = features[start:end]
            frame_ids = traj[start:end]  # MUST exist

            for feat, obj_id in zip(frame_feats, frame_ids):

                # assign slot if new object
                if obj_id not in active_map:
                    if len(free_slots) == 0:
                        continue  # drop object if no space
                    active_map[obj_id] = free_slots.pop(0)

                slot = active_map[obj_id]

                x[t, slot] = torch.from_numpy(feat)
                mask[t, slot] = 1.0

        # --- target ---
        K = self.future_frames

        area = x[:, :, 2] * x[:, :, 3]
        log_area = torch.log(area + 1e-6)

        vx = fit_cubic(x[:, :, 0], K)
        vy = fit_cubic(x[:, :, 1], K)
        v_area = fit_cubic(log_area, K)

        y_valid = torch.stack(
            [vx, vy, v_area], dim=2
        )  # num_frames, num_objects, (x,y,z), poly_coefficients

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

        elif len(y) < self.window:
            pad_len = self.window - len(y)

            # pad tensors along time dimension
            x = torch.cat([x, torch.zeros((pad_len, O, F))], dim=0)

            y = torch.cat([y, torch.zeros((pad_len, O, 3, 4))], dim=0)

            mask = torch.cat([mask, torch.zeros((pad_len, O, 1))], dim=0)

        return {
            "features": x.float(),
            "trajectory": y.float(),
            "mask": mask.float(),
        }

   