#!/usr/bin/env python3
"""Extract Waymo Open Motion Dataset Scenario TFRecords.

The shard in ./dataset is a TFRecord of serialized Scenario protos. This script
uses a small schema-aware protobuf decoder so it does not require the
waymo-open-dataset Python wheel. It writes normalized tables that are friendly
to PyTorch data loaders:

  - scenarios: one row per scenario
  - tracks: one row per object track
  - states: one row per object per timestep (the main trajectory table)
  - tracks_to_predict: challenge/training prediction targets
  - objects_of_interest: interactive object ids
  - traffic_signals: dynamic traffic light states
  - map_features: static map summaries
  - map_points: lane/line/edge/polyline/polygon points
  - camera_tokens: WOMD camera-token data, when present
  - lidar: compressed lidar byte-size metadata, when present

If pyarrow is installed, each table is also written as parquet. CSV and JSONL
are always written.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


OBJECT_TYPES = {
    0: "TYPE_UNSET",
    1: "TYPE_VEHICLE",
    2: "TYPE_PEDESTRIAN",
    3: "TYPE_CYCLIST",
    4: "TYPE_OTHER",
}

DIFFICULTY_LEVELS = {0: "NONE", 1: "LEVEL_1", 2: "LEVEL_2"}

LANE_SIGNAL_STATES = {
    0: "LANE_STATE_UNKNOWN",
    1: "LANE_STATE_ARROW_STOP",
    2: "LANE_STATE_ARROW_CAUTION",
    3: "LANE_STATE_ARROW_GO",
    4: "LANE_STATE_STOP",
    5: "LANE_STATE_CAUTION",
    6: "LANE_STATE_GO",
    7: "LANE_STATE_FLASHING_STOP",
    8: "LANE_STATE_FLASHING_CAUTION",
}

CAMERA_NAMES = {
    0: "UNKNOWN",
    1: "FRONT",
    2: "FRONT_LEFT",
    3: "FRONT_RIGHT",
    4: "SIDE_LEFT",
    5: "SIDE_RIGHT",
    6: "REAR_LEFT",
    7: "REAR",
    8: "REAR_RIGHT",
}

LASER_NAMES = {
    0: "UNKNOWN",
    1: "TOP",
    2: "FRONT",
    3: "SIDE_LEFT",
    4: "SIDE_RIGHT",
    5: "REAR",
}

LANE_TYPES = {
    0: "TYPE_UNDEFINED",
    1: "TYPE_FREEWAY",
    2: "TYPE_SURFACE_STREET",
    3: "TYPE_BIKE_LANE",
}

ROAD_EDGE_TYPES = {
    0: "TYPE_UNKNOWN",
    1: "TYPE_ROAD_EDGE_BOUNDARY",
    2: "TYPE_ROAD_EDGE_MEDIAN",
}

ROAD_LINE_TYPES = {
    0: "TYPE_UNKNOWN",
    1: "TYPE_BROKEN_SINGLE_WHITE",
    2: "TYPE_SOLID_SINGLE_WHITE",
    3: "TYPE_SOLID_DOUBLE_WHITE",
    4: "TYPE_BROKEN_SINGLE_YELLOW",
    5: "TYPE_BROKEN_DOUBLE_YELLOW",
    6: "TYPE_SOLID_SINGLE_YELLOW",
    7: "TYPE_SOLID_DOUBLE_YELLOW",
    8: "TYPE_PASSING_DOUBLE_YELLOW",
}


class ProtoDecodeError(RuntimeError):
    pass


@dataclass
class ProtoReader:
    data: bytes
    pos: int = 0

    def eof(self) -> bool:
        return self.pos >= len(self.data)

    def read_varint(self) -> int:
        shift = 0
        value = 0
        while True:
            if self.pos >= len(self.data):
                raise ProtoDecodeError("unexpected EOF while reading varint")
            byte = self.data[self.pos]
            self.pos += 1
            value |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return value
            shift += 7
            if shift >= 70:
                raise ProtoDecodeError("varint is too long")

    def read_fixed32(self) -> bytes:
        if self.pos + 4 > len(self.data):
            raise ProtoDecodeError("unexpected EOF while reading fixed32")
        raw = self.data[self.pos : self.pos + 4]
        self.pos += 4
        return raw

    def read_fixed64(self) -> bytes:
        if self.pos + 8 > len(self.data):
            raise ProtoDecodeError("unexpected EOF while reading fixed64")
        raw = self.data[self.pos : self.pos + 8]
        self.pos += 8
        return raw

    def read_len(self) -> bytes:
        size = self.read_varint()
        if self.pos + size > len(self.data):
            raise ProtoDecodeError("unexpected EOF while reading bytes field")
        raw = self.data[self.pos : self.pos + size]
        self.pos += size
        return raw

    def read_field(self) -> tuple[int, int, Any]:
        key = self.read_varint()
        field_no = key >> 3
        wire_type = key & 7
        if wire_type == 0:
            return field_no, wire_type, self.read_varint()
        if wire_type == 1:
            return field_no, wire_type, self.read_fixed64()
        if wire_type == 2:
            return field_no, wire_type, self.read_len()
        if wire_type == 5:
            return field_no, wire_type, self.read_fixed32()
        raise ProtoDecodeError(f"unsupported protobuf wire type {wire_type}")


def as_double(raw: bytes) -> float:
    return struct.unpack("<d", raw)[0]


def as_float(raw: bytes) -> float:
    return struct.unpack("<f", raw)[0]


def read_packed_varints(raw: bytes) -> list[int]:
    reader = ProtoReader(raw)
    values = []
    while not reader.eof():
        values.append(reader.read_varint())
    return values


def parse_message(raw: bytes) -> dict[int, list[tuple[int, Any]]]:
    reader = ProtoReader(raw)
    fields: dict[int, list[tuple[int, Any]]] = defaultdict(list)
    while not reader.eof():
        field_no, wire_type, value = reader.read_field()
        fields[field_no].append((wire_type, value))
    return fields


def get_varint(fields: dict[int, list[tuple[int, Any]]], field_no: int, default: int = 0) -> int:
    values = fields.get(field_no)
    return int(values[-1][1]) if values else default


def get_bool(fields: dict[int, list[tuple[int, Any]]], field_no: int, default: bool = False) -> bool:
    return bool(get_varint(fields, field_no, int(default)))


def get_string(fields: dict[int, list[tuple[int, Any]]], field_no: int, default: str = "") -> str:
    values = fields.get(field_no)
    if not values:
        return default
    return values[-1][1].decode("utf-8", errors="replace")


def get_double(fields: dict[int, list[tuple[int, Any]]], field_no: int, default: float = math.nan) -> float:
    values = fields.get(field_no)
    return as_double(values[-1][1]) if values else default


def get_float(fields: dict[int, list[tuple[int, Any]]], field_no: int, default: float = math.nan) -> float:
    values = fields.get(field_no)
    return as_float(values[-1][1]) if values else default


def get_message_values(fields: dict[int, list[tuple[int, Any]]], field_no: int) -> list[bytes]:
    return [value for wire_type, value in fields.get(field_no, []) if wire_type == 2]


def parse_map_point(raw: bytes) -> dict[str, float]:
    fields = parse_message(raw)
    return {
        "x": get_double(fields, 1),
        "y": get_double(fields, 2),
        "z": get_double(fields, 3),
    }


def parse_object_state(raw: bytes) -> dict[str, Any]:
    fields = parse_message(raw)
    return {
        "center_x": get_double(fields, 2),
        "center_y": get_double(fields, 3),
        "center_z": get_double(fields, 4),
        "length": get_float(fields, 5),
        "width": get_float(fields, 6),
        "height": get_float(fields, 7),
        "heading": get_float(fields, 8),
        "velocity_x": get_float(fields, 9),
        "velocity_y": get_float(fields, 10),
        "valid": get_bool(fields, 11),
    }


def parse_track(raw: bytes) -> dict[str, Any]:
    fields = parse_message(raw)
    object_type = get_varint(fields, 2)
    return {
        "id": get_varint(fields, 1),
        "object_type": object_type,
        "object_type_name": OBJECT_TYPES.get(object_type, f"UNKNOWN_{object_type}"),
        "states": [parse_object_state(v) for v in get_message_values(fields, 3)],
    }


def parse_required_prediction(raw: bytes) -> dict[str, Any]:
    fields = parse_message(raw)
    difficulty = get_varint(fields, 2)
    return {
        "track_index": get_varint(fields, 1),
        "difficulty": difficulty,
        "difficulty_name": DIFFICULTY_LEVELS.get(difficulty, f"UNKNOWN_{difficulty}"),
    }


def parse_traffic_signal_lane_state(raw: bytes) -> dict[str, Any]:
    fields = parse_message(raw)
    state = get_varint(fields, 2)
    stop_point_values = get_message_values(fields, 3)
    stop_point = parse_map_point(stop_point_values[-1]) if stop_point_values else {}
    return {
        "lane": get_varint(fields, 1),
        "state": state,
        "state_name": LANE_SIGNAL_STATES.get(state, f"UNKNOWN_{state}"),
        "stop_point_x": stop_point.get("x", math.nan),
        "stop_point_y": stop_point.get("y", math.nan),
        "stop_point_z": stop_point.get("z", math.nan),
    }


def parse_dynamic_map_state(raw: bytes) -> list[dict[str, Any]]:
    fields = parse_message(raw)
    return [parse_traffic_signal_lane_state(v) for v in get_message_values(fields, 1)]


def parse_boundary_segment(raw: bytes) -> dict[str, Any]:
    fields = parse_message(raw)
    boundary_type = get_varint(fields, 4)
    return {
        "lane_start_index": get_varint(fields, 1),
        "lane_end_index": get_varint(fields, 2),
        "boundary_feature_id": get_varint(fields, 3),
        "boundary_type": boundary_type,
        "boundary_type_name": ROAD_LINE_TYPES.get(boundary_type, f"UNKNOWN_{boundary_type}"),
    }


def parse_lane_neighbor(raw: bytes) -> dict[str, Any]:
    fields = parse_message(raw)
    return {
        "feature_id": get_varint(fields, 1),
        "self_start_index": get_varint(fields, 2),
        "self_end_index": get_varint(fields, 3),
        "neighbor_start_index": get_varint(fields, 4),
        "neighbor_end_index": get_varint(fields, 5),
        "boundaries": [parse_boundary_segment(v) for v in get_message_values(fields, 6)],
    }


def parse_lane_center(raw: bytes) -> dict[str, Any]:
    fields = parse_message(raw)
    lane_type = get_varint(fields, 2)
    entry_lanes: list[int] = []
    exit_lanes: list[int] = []
    for wire_type, value in fields.get(9, []):
        entry_lanes.extend(read_packed_varints(value) if wire_type == 2 else [int(value)])
    for wire_type, value in fields.get(10, []):
        exit_lanes.extend(read_packed_varints(value) if wire_type == 2 else [int(value)])
    return {
        "speed_limit_mph": get_double(fields, 1),
        "lane_type": lane_type,
        "lane_type_name": LANE_TYPES.get(lane_type, f"UNKNOWN_{lane_type}"),
        "interpolating": get_bool(fields, 3),
        "polyline": [parse_map_point(v) for v in get_message_values(fields, 8)],
        "entry_lanes": entry_lanes,
        "exit_lanes": exit_lanes,
        "left_boundaries": [parse_boundary_segment(v) for v in get_message_values(fields, 13)],
        "right_boundaries": [parse_boundary_segment(v) for v in get_message_values(fields, 14)],
        "left_neighbors": [parse_lane_neighbor(v) for v in get_message_values(fields, 11)],
        "right_neighbors": [parse_lane_neighbor(v) for v in get_message_values(fields, 12)],
    }


def parse_road_edge(raw: bytes) -> dict[str, Any]:
    fields = parse_message(raw)
    edge_type = get_varint(fields, 1)
    return {
        "type": edge_type,
        "type_name": ROAD_EDGE_TYPES.get(edge_type, f"UNKNOWN_{edge_type}"),
        "polyline": [parse_map_point(v) for v in get_message_values(fields, 2)],
    }


def parse_road_line(raw: bytes) -> dict[str, Any]:
    fields = parse_message(raw)
    line_type = get_varint(fields, 1)
    return {
        "type": line_type,
        "type_name": ROAD_LINE_TYPES.get(line_type, f"UNKNOWN_{line_type}"),
        "polyline": [parse_map_point(v) for v in get_message_values(fields, 2)],
    }


def parse_stop_sign(raw: bytes) -> dict[str, Any]:
    fields = parse_message(raw)
    lanes: list[int] = []
    for wire_type, value in fields.get(1, []):
        lanes.extend(read_packed_varints(value) if wire_type == 2 else [int(value)])
    position_values = get_message_values(fields, 2)
    return {
        "lanes": lanes,
        "position": parse_map_point(position_values[-1]) if position_values else {},
    }


def parse_polygon_feature(raw: bytes) -> dict[str, Any]:
    fields = parse_message(raw)
    return {"polygon": [parse_map_point(v) for v in get_message_values(fields, 1)]}


def parse_map_feature(raw: bytes) -> dict[str, Any]:
    fields = parse_message(raw)
    result: dict[str, Any] = {"id": get_varint(fields, 1)}
    feature_parsers: dict[int, tuple[str, Callable[[bytes], dict[str, Any]]]] = {
        3: ("lane", parse_lane_center),
        4: ("road_line", parse_road_line),
        5: ("road_edge", parse_road_edge),
        7: ("stop_sign", parse_stop_sign),
        8: ("crosswalk", parse_polygon_feature),
        9: ("speed_bump", parse_polygon_feature),
        10: ("driveway", parse_polygon_feature),
    }
    for field_no, (feature_type, parser) in feature_parsers.items():
        values = get_message_values(fields, field_no)
        if values:
            result["feature_type"] = feature_type
            result["feature"] = parser(values[-1])
            break
    if "feature_type" not in result:
        result["feature_type"] = "unknown"
        result["feature"] = {}
    return result


def parse_camera_tokens(raw: bytes) -> list[dict[str, Any]]:
    fields = parse_message(raw)
    rows = []
    for camera_raw in get_message_values(fields, 1):
        camera_fields = parse_message(camera_raw)
        camera_name = get_varint(camera_fields, 1)
        tokens: list[int] = []
        for wire_type, value in camera_fields.get(2, []):
            tokens.extend(read_packed_varints(value) if wire_type == 2 else [int(value)])
        rows.append(
            {
                "camera_name": camera_name,
                "camera_name_text": CAMERA_NAMES.get(camera_name, f"UNKNOWN_{camera_name}"),
                "tokens": tokens,
                "token_count": len(tokens),
            }
        )
    return rows


def parse_compressed_lidar(raw: bytes) -> list[dict[str, Any]]:
    fields = parse_message(raw)
    rows = []
    for laser_raw in get_message_values(fields, 1):
        laser_fields = parse_message(laser_raw)
        laser_name = get_varint(laser_fields, 1)
        row = {
            "laser_name": laser_name,
            "laser_name_text": LASER_NAMES.get(laser_name, f"UNKNOWN_{laser_name}"),
            "ri_return1_bytes": 0,
            "ri_return2_bytes": 0,
            "ri_return1_pose_bytes": 0,
            "ri_return2_pose_bytes": 0,
        }
        for prefix, field_no in (("ri_return1", 2), ("ri_return2", 3)):
            values = get_message_values(laser_fields, field_no)
            if values:
                ri_fields = parse_message(values[-1])
                ranges = get_message_values(ri_fields, 1)
                poses = get_message_values(ri_fields, 4)
                row[f"{prefix}_bytes"] = len(ranges[-1]) if ranges else 0
                row[f"{prefix}_pose_bytes"] = len(poses[-1]) if poses else 0
        rows.append(row)
    return rows


def parse_scenario(raw: bytes) -> dict[str, Any]:
    fields = parse_message(raw)
    timestamps = [as_double(value) for wire_type, value in fields.get(1, []) if wire_type == 1]
    objects_of_interest: list[int] = []
    for wire_type, value in fields.get(4, []):
        objects_of_interest.extend(read_packed_varints(value) if wire_type == 2 else [int(value)])
    return {
        "scenario_id": get_string(fields, 5),
        "timestamps_seconds": timestamps,
        "tracks": [parse_track(v) for v in get_message_values(fields, 2)],
        "dynamic_map_states": [parse_dynamic_map_state(v) for v in get_message_values(fields, 7)],
        "map_features": [parse_map_feature(v) for v in get_message_values(fields, 8)],
        "sdc_track_index": get_varint(fields, 6, -1),
        "objects_of_interest": objects_of_interest,
        "current_time_index": get_varint(fields, 10, -1),
        "tracks_to_predict": [parse_required_prediction(v) for v in get_message_values(fields, 11)],
        "compressed_frame_laser_data": [parse_compressed_lidar(v) for v in get_message_values(fields, 12)],
        "frame_camera_tokens": [parse_camera_tokens(v) for v in get_message_values(fields, 13)],
    }


def iter_tfrecord_records(path: Path) -> Any:
    with path.open("rb") as f:
        while True:
            header = f.read(12)
            if not header:
                return
            if len(header) != 12:
                raise EOFError(f"incomplete TFRecord header in {path}")
            length = struct.unpack("<Q", header[:8])[0]
            payload = f.read(length)
            footer = f.read(4)
            if len(payload) != length or len(footer) != 4:
                raise EOFError(f"incomplete TFRecord payload in {path}")
            yield payload


def scenario_to_tables(scenario: dict[str, Any], record_index: int) -> dict[str, list[dict[str, Any]]]:
    scenario_id = scenario["scenario_id"] or f"record_{record_index:06d}"
    current_time_index = scenario["current_time_index"]
    timestamps = scenario["timestamps_seconds"]
    tables: dict[str, list[dict[str, Any]]] = defaultdict(list)

    tables["scenarios"].append(
        {
            "record_index": record_index,
            "scenario_id": scenario_id,
            "num_timestamps": len(timestamps),
            "current_time_index": current_time_index,
            "sdc_track_index": scenario["sdc_track_index"],
            "num_tracks": len(scenario["tracks"]),
            "num_map_features": len(scenario["map_features"]),
            "num_dynamic_map_states": len(scenario["dynamic_map_states"]),
            "num_tracks_to_predict": len(scenario["tracks_to_predict"]),
            "num_objects_of_interest": len(scenario["objects_of_interest"]),
            "num_camera_token_frames": len(scenario["frame_camera_tokens"]),
            "num_lidar_frames": len(scenario["compressed_frame_laser_data"]),
        }
    )

    for rank, object_id in enumerate(scenario["objects_of_interest"]):
        tables["objects_of_interest"].append(
            {"scenario_id": scenario_id, "rank": rank, "object_id": object_id}
        )

    for rank, pred in enumerate(scenario["tracks_to_predict"]):
        track_index = pred["track_index"]
        track_id = scenario["tracks"][track_index]["id"] if 0 <= track_index < len(scenario["tracks"]) else None
        tables["tracks_to_predict"].append(
            {
                "scenario_id": scenario_id,
                "rank": rank,
                "track_index": track_index,
                "track_id": track_id,
                "difficulty": pred["difficulty"],
                "difficulty_name": pred["difficulty_name"],
            }
        )

    for track_index, track in enumerate(scenario["tracks"]):
        is_sdc = track_index == scenario["sdc_track_index"]
        tables["tracks"].append(
            {
                "scenario_id": scenario_id,
                "track_index": track_index,
                "track_id": track["id"],
                "object_type": track["object_type"],
                "object_type_name": track["object_type_name"],
                "is_sdc": is_sdc,
                "num_states": len(track["states"]),
                "num_valid_states": sum(1 for s in track["states"] if s["valid"]),
            }
        )
        for time_index, state in enumerate(track["states"]):
            row = {
                "scenario_id": scenario_id,
                "track_index": track_index,
                "track_id": track["id"],
                "object_type": track["object_type"],
                "object_type_name": track["object_type_name"],
                "is_sdc": is_sdc,
                "time_index": time_index,
                "timestamp_seconds": timestamps[time_index] if time_index < len(timestamps) else math.nan,
                "split": split_name(time_index, current_time_index),
            }
            row.update(state)
            tables["states"].append(row)

    for time_index, lane_states in enumerate(scenario["dynamic_map_states"]):
        for state_index, lane_state in enumerate(lane_states):
            row = {
                "scenario_id": scenario_id,
                "time_index": time_index,
                "timestamp_seconds": timestamps[time_index] if time_index < len(timestamps) else math.nan,
                "state_index": state_index,
            }
            row.update(lane_state)
            tables["traffic_signals"].append(row)

    for map_index, feature in enumerate(scenario["map_features"]):
        feature_type = feature["feature_type"]
        data = feature["feature"]
        tables["map_features"].append(
            {
                "scenario_id": scenario_id,
                "map_index": map_index,
                "feature_id": feature["id"],
                "feature_type": feature_type,
                "speed_limit_mph": data.get("speed_limit_mph", math.nan),
                "type": data.get("lane_type", data.get("type", None)),
                "type_name": data.get("lane_type_name", data.get("type_name", "")),
                "interpolating": data.get("interpolating", None),
                "entry_lanes_json": json.dumps(data.get("entry_lanes", data.get("lanes", []))),
                "exit_lanes_json": json.dumps(data.get("exit_lanes", [])),
                "left_boundaries_json": json.dumps(data.get("left_boundaries", [])),
                "right_boundaries_json": json.dumps(data.get("right_boundaries", [])),
                "left_neighbors_json": json.dumps(data.get("left_neighbors", [])),
                "right_neighbors_json": json.dumps(data.get("right_neighbors", [])),
            }
        )
        add_feature_points(tables["map_points"], scenario_id, feature, map_index)

    for time_index, frame in enumerate(scenario["frame_camera_tokens"]):
        for camera in frame:
            tables["camera_tokens"].append(
                {
                    "scenario_id": scenario_id,
                    "time_index": time_index,
                    "timestamp_seconds": timestamps[time_index] if time_index < len(timestamps) else math.nan,
                    "camera_name": camera["camera_name"],
                    "camera_name_text": camera["camera_name_text"],
                    "token_count": camera["token_count"],
                    "tokens_json": json.dumps(camera["tokens"]),
                }
            )

    for time_index, frame in enumerate(scenario["compressed_frame_laser_data"]):
        for laser in frame:
            row = {
                "scenario_id": scenario_id,
                "time_index": time_index,
                "timestamp_seconds": timestamps[time_index] if time_index < len(timestamps) else math.nan,
            }
            row.update(laser)
            tables["lidar"].append(row)

    return tables


def split_name(time_index: int, current_time_index: int) -> str:
    if current_time_index < 0:
        return "unknown"
    if time_index < current_time_index:
        return "past"
    if time_index == current_time_index:
        return "current"
    return "future"


def add_feature_points(
    rows: list[dict[str, Any]],
    scenario_id: str,
    feature: dict[str, Any],
    map_index: int,
) -> None:
    data = feature["feature"]
    if "polyline" in data:
        points = data["polyline"]
        geometry = "polyline"
    elif "polygon" in data:
        points = data["polygon"]
        geometry = "polygon"
    elif "position" in data and data["position"]:
        points = [data["position"]]
        geometry = "position"
    else:
        return
    for point_index, point in enumerate(points):
        rows.append(
            {
                "scenario_id": scenario_id,
                "map_index": map_index,
                "feature_id": feature["id"],
                "feature_type": feature["feature_type"],
                "geometry": geometry,
                "point_index": point_index,
                "x": point.get("x", math.nan),
                "y": point.get("y", math.nan),
                "z": point.get("z", math.nan),
            }
        )


def merge_tables(dst: dict[str, list[dict[str, Any]]], src: dict[str, list[dict[str, Any]]]) -> None:
    for name, rows in src.items():
        dst[name].extend(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(clean_value(row), allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in columns})


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return value


def clean_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: clean_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_value(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def write_parquet_if_available(path: Path, rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        return False

    columns = sorted({key for row in rows for key in row})
    normalized = [{key: clean_value(row.get(key)) for key in columns} for row in rows]
    table = pa.Table.from_pylist(normalized)
    pq.write_table(table, path)
    return True


def write_tables(out_dir: Path, tables: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, Any]] = {}
    for name, rows in sorted(tables.items()):
        jsonl_path = out_dir / f"{name}.jsonl"
        csv_path = out_dir / f"{name}.csv"
        parquet_path = out_dir / f"{name}.parquet"
        write_jsonl(jsonl_path, rows)
        write_csv(csv_path, rows)
        parquet_written = write_parquet_if_available(parquet_path, rows)
        manifest[name] = {
            "rows": len(rows),
            "jsonl": str(jsonl_path),
            "csv": str(csv_path),
            "parquet": str(parquet_path) if parquet_written else None,
        }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def print_summary(tables: dict[str, list[dict[str, Any]]], manifest: dict[str, dict[str, Any]], sample_rows: int) -> None:
    print("\nExtraction summary")
    print("==================")
    for name, info in sorted(manifest.items()):
        parquet_note = " + parquet" if info["parquet"] else ""
        print(f"{name:22s} {info['rows']:10d} rows -> {info['jsonl']} / {info['csv']}{parquet_note}")

    scenarios = tables.get("scenarios", [])
    tracks = tables.get("tracks", [])
    states = tables.get("states", [])
    map_features = tables.get("map_features", [])
    camera_tokens = tables.get("camera_tokens", [])

    print("\nDataset contents")
    print("================")
    print(f"Scenarios decoded: {len(scenarios)}")
    print(f"Tracks decoded: {len(tracks)}")
    print(f"Trajectory state rows: {len(states)}")
    print(f"Map features decoded: {len(map_features)}")
    print(f"Camera token rows: {len(camera_tokens)}")
    print("RGB image rows: 0")

    if camera_tokens:
        print(
            "Camera data in this shard is tokenized WOMD camera data, not raw RGB JPEG/PNG bytes. "
            "Decode tokens with Waymo's camera-token codebook/model before feeding RGB-like images."
        )
    else:
        print(
            "No camera token frames or RGB images were present in the decoded scenarios. "
            "The standard motion Scenario shard contains trajectories/maps; RGB requires the WOMD camera-token release or another Waymo perception/e2e shard."
        )

    object_counts = Counter(row["object_type_name"] for row in tracks)
    if object_counts:
        print("\nObject types")
        print("============")
        for name, count in object_counts.most_common():
            print(f"{name:18s} {count:8d}")

    feature_counts = Counter(row["feature_type"] for row in map_features)
    if feature_counts:
        print("\nMap feature types")
        print("=================")
        for name, count in feature_counts.most_common():
            print(f"{name:18s} {count:8d}")

    for table_name in ("scenarios", "tracks", "states", "tracks_to_predict", "traffic_signals", "map_features", "camera_tokens", "lidar"):
        rows = tables.get(table_name, [])
        if not rows:
            continue
        print(f"\nSample {table_name}")
        print("-" * (7 + len(table_name)))
        for row in rows[:sample_rows]:
            print(json.dumps(clean_value(row), indent=2, allow_nan=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="dataset/uncompressed_scenario_training_training.tfrecord-00000-of-01000",
        help="Path to a Waymo motion Scenario TFRecord shard.",
    )
    parser.add_argument(
        "--output",
        default="dataset/processed_motion",
        help="Directory for extracted tables.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=3,
        help="Maximum number of scenarios to decode. Use 0 for the full shard.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=2,
        help="Number of sample rows to print per table.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 2

    max_records = None if args.max_records == 0 else args.max_records
    all_tables: dict[str, list[dict[str, Any]]] = defaultdict(list)
    decoded = 0
    for record_index, payload in enumerate(iter_tfrecord_records(input_path)):
        if max_records is not None and decoded >= max_records:
            break
        scenario = parse_scenario(payload)
        merge_tables(all_tables, scenario_to_tables(scenario, record_index))
        decoded += 1
        if decoded % 25 == 0:
            print(f"Decoded {decoded} scenarios...", file=sys.stderr)

    manifest = write_tables(Path(args.output), all_tables)
    print_summary(all_tables, manifest, args.sample_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
