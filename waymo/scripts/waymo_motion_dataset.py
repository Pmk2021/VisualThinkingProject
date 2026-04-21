#!/usr/bin/env python3
"""Small PyTorch Dataset for extracted Waymo motion trajectory tables.

This loader consumes the tables produced by scripts/extract_waymo_motion.py.
Pass either a processed directory containing states.parquet/states.jsonl, or a
direct path to one of those files.

Example:
    from torch.utils.data import DataLoader
    from scripts.waymo_motion_dataset import WaymoMotionDataset, pad_motion_batch

    ds = WaymoMotionDataset("dataset/processed_motion", only_tracks_to_predict=True)
    loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=pad_motion_batch)
    batch = next(iter(loader))
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


DEFAULT_STATE_COLUMNS = (
    "center_x",
    "center_y",
    "center_z",
    "length",
    "width",
    "height",
    "heading",
    "velocity_x",
    "velocity_y",
)


class WaymoMotionDataset(Dataset):
    """Trajectory-level PyTorch dataset backed by extracted Waymo state rows.

    Each item is one object track from one scenario. The item includes the full
    trajectory, a history/current slice, and a future target slice.
    """

    def __init__(
        self,
        data_path: str | Path,
        *,
        state_columns: tuple[str, ...] = DEFAULT_STATE_COLUMNS,
        include_invalid: bool = True,
        only_tracks_to_predict: bool = False,
        tracks_to_predict_path: str | Path | None = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.state_columns = state_columns
        states_path = resolve_states_path(self.data_path)
        states = read_table(states_path)

        required = {
            "scenario_id",
            "track_index",
            "track_id",
            "object_type",
            "object_type_name",
            "is_sdc",
            "time_index",
            "timestamp_seconds",
            "split",
            "valid",
            *state_columns,
        }
        missing = sorted(required.difference(states.columns))
        if missing:
            raise ValueError(f"{states_path} is missing required columns: {missing}")

        if not include_invalid:
            states = states[states["valid"].astype(bool)].copy()

        if only_tracks_to_predict:
            predict_path = resolve_tracks_to_predict_path(
                self.data_path, tracks_to_predict_path
            )
            predict = read_table(predict_path)
            keys = predict[["scenario_id", "track_index"]].drop_duplicates()
            states = states.merge(keys, on=["scenario_id", "track_index"], how="inner")

        sort_cols = ["scenario_id", "track_index", "time_index"]
        states = states.sort_values(sort_cols).reset_index(drop=True)

        self.states_path = states_path
        self.states = states
        self.groups = list(states.groupby(["scenario_id", "track_index"], sort=False).indices.items())

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict[str, Any]:
        (scenario_id, track_index), row_indices = self.groups[index]
        rows = self.states.iloc[row_indices]

        states = torch.tensor(
            rows.loc[:, self.state_columns].to_numpy(dtype="float32"),
            dtype=torch.float32,
        )
        valid = torch.tensor(rows["valid"].to_numpy(dtype="bool"), dtype=torch.bool)
        timestamps = torch.tensor(
            rows["timestamp_seconds"].to_numpy(dtype="float32"),
            dtype=torch.float32,
        )
        time_index = torch.tensor(rows["time_index"].to_numpy(dtype="int64"), dtype=torch.long)

        split = rows["split"].astype(str)
        history_mask = torch.tensor(split.isin(["past", "current"]).to_numpy(), dtype=torch.bool)
        future_mask = torch.tensor((split == "future").to_numpy(), dtype=torch.bool)

        first = rows.iloc[0]
        return {
            "scenario_id": str(scenario_id),
            "track_index": int(track_index),
            "track_id": int(first["track_id"]),
            "object_type": int(first["object_type"]),
            "object_type_name": str(first["object_type_name"]),
            "is_sdc": bool(first["is_sdc"]),
            "time_index": time_index,
            "timestamps": timestamps,
            "states": states,
            "valid": valid,
            "history_states": states[history_mask],
            "history_valid": valid[history_mask],
            "future_states": states[future_mask],
            "future_valid": valid[future_mask],
        }


def pad_motion_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate variable-length track samples into padded tensors."""

    tensor_keys = (
        "states",
        "valid",
        "time_index",
        "timestamps",
        "history_states",
        "history_valid",
        "future_states",
        "future_valid",
    )
    out: dict[str, Any] = {
        "scenario_id": [item["scenario_id"] for item in batch],
        "track_index": torch.tensor([item["track_index"] for item in batch], dtype=torch.long),
        "track_id": torch.tensor([item["track_id"] for item in batch], dtype=torch.long),
        "object_type": torch.tensor([item["object_type"] for item in batch], dtype=torch.long),
        "object_type_name": [item["object_type_name"] for item in batch],
        "is_sdc": torch.tensor([item["is_sdc"] for item in batch], dtype=torch.bool),
    }
    for key in tensor_keys:
        padding_value = False if batch[0][key].dtype == torch.bool else 0.0
        out[key] = pad_sequence(
            [item[key] for item in batch],
            batch_first=True,
            padding_value=padding_value,
        )
        out[f"{key}_lengths"] = torch.tensor([item[key].shape[0] for item in batch], dtype=torch.long)
    return out


def resolve_states_path(path: Path) -> Path:
    if path.is_file():
        return path
    parquet_path = path / "states.parquet"
    jsonl_path = path / "states.jsonl"
    if parquet_path.exists():
        return parquet_path
    if jsonl_path.exists():
        return jsonl_path
    raise FileNotFoundError(f"Could not find states.parquet or states.jsonl in {path}")


def resolve_tracks_to_predict_path(
    data_path: Path,
    explicit_path: str | Path | None,
) -> Path:
    if explicit_path is not None:
        path = Path(explicit_path)
        if path.exists():
            return path
        raise FileNotFoundError(path)
    base = data_path if data_path.is_dir() else data_path.parent
    parquet_path = base / "tracks_to_predict.parquet"
    jsonl_path = base / "tracks_to_predict.jsonl"
    if parquet_path.exists():
        return parquet_path
    if jsonl_path.exists():
        return jsonl_path
    raise FileNotFoundError(
        "only_tracks_to_predict=True requires tracks_to_predict.parquet or "
        f"tracks_to_predict.jsonl next to states table in {base}"
    )


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported table format: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the Waymo PyTorch dataset loader.")
    parser.add_argument(
        "data_path",
        nargs="?",
        default="dataset/processed_motion",
        help="Processed directory or states.parquet/states.jsonl path.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--only-tracks-to-predict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = WaymoMotionDataset(
        args.data_path,
        only_tracks_to_predict=args.only_tracks_to_predict,
    )
    print(f"Loaded {len(dataset)} object tracks from {dataset.states_path}")
    item = dataset[0]
    print(
        "First item:",
        {
            "scenario_id": item["scenario_id"],
            "track_id": item["track_id"],
            "states_shape": tuple(item["states"].shape),
            "history_shape": tuple(item["history_states"].shape),
            "future_shape": tuple(item["future_states"].shape),
        },
    )

    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pad_motion_batch,
    )
    batch = next(iter(loader))
    print(
        "Batch:",
        {
            "states_shape": tuple(batch["states"].shape),
            "future_shape": tuple(batch["future_states"].shape),
            "track_ids": batch["track_id"].tolist(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
