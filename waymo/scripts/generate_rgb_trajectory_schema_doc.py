#!/usr/bin/env python3
"""Generate a Markdown interface document for RGB trajectory parquet outputs.

The generator inspects ``scripts/build_waymo_rgb_trajectory_dataset.py`` instead
of duplicating column lists by hand. It finds the ``write_parquet`` calls in
``main()``, follows the row-builder functions that feed each output list, and
extracts constant keys from row dictionaries.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


TABLE_DESCRIPTIONS = {
    "images": "One row per standalone RGB camera image.",
    "ego_poses": "One row per ego/vehicle pose timestamp.",
    "trajectories": "One row per scene/object 3D trajectory.",
    "image_trajectories": "Visible object links and 2D projected boxes per image.",
    "prediction_targets": "WOMD tracks_to_predict rows with observed history and optional future labels.",
}

COLUMN_NOTES = {
    "split": "Dataset split name.",
    "image_id": "Unique image key.",
    "scene_id": "Waymo scene or segment id.",
    "frame_timestamp_micros": "Frame timestamp in microseconds.",
    "camera_name": "Waymo camera enum id.",
    "camera_name_text": "Human-readable camera name.",
    "image_jpeg": "JPEG-encoded RGB image bytes.",
    "image_format": "Image encoding label.",
    "image_width": "Decoded image width in pixels.",
    "image_height": "Decoded image height in pixels.",
    "visible_trajectory_ids": "Trajectory ids visible in this image.",
    "num_visible_trajectories": "Number of visible trajectory ids.",
    "trajectory_id": "Object trajectory id within a scene.",
    "trajectory_row_id": "Unique trajectory row key.",
    "object_type": "Waymo object type enum.",
    "num_steps": "Number of trajectory timesteps.",
    "timestamps_micros": "Trajectory timestamps in microseconds.",
    "x": "Object x positions over time.",
    "y": "Object y positions over time.",
    "z": "Object z positions over time.",
    "length": "Object box lengths over time.",
    "width": "Object box widths over time.",
    "height": "Object box heights over time.",
    "heading": "Object headings over time.",
    "ego_pose_transform": "4x4 row-major world_from_vehicle transform.",
    "ego_x": "Ego x position from vehicle pose.",
    "ego_y": "Ego y position from vehicle pose.",
    "ego_z": "Ego z position from vehicle pose.",
    "ego_roll": "Ego roll from the vehicle pose rotation matrix.",
    "ego_pitch": "Ego pitch from the vehicle pose rotation matrix.",
    "ego_yaw": "Ego yaw/heading from the vehicle pose rotation matrix.",
    "ego_velocity_source": "Source for preferred ego linear velocity.",
    "bbox_center_x": "Projected 2D box center x in image pixels.",
    "bbox_center_y": "Projected 2D box center y in image pixels.",
    "bbox_width": "Projected 2D box width in pixels.",
    "bbox_height": "Projected 2D box height in pixels.",
    "bbox_x1": "Projected 2D box left coordinate in pixels.",
    "bbox_y1": "Projected 2D box top coordinate in pixels.",
    "bbox_x2": "Projected 2D box right coordinate in pixels.",
    "bbox_y2": "Projected 2D box bottom coordinate in pixels.",
    "track_index": "Track index in the WOMD scenario.",
    "track_id": "WOMD track id.",
    "rank": "Rank/order in tracks_to_predict.",
    "difficulty": "WOMD prediction target difficulty enum.",
    "difficulty_name": "Human-readable difficulty label.",
    "object_type_name": "Human-readable object type label.",
    "current_time_index": "WOMD current timestep index.",
    "has_future_gt": "Whether future labels were present in the source scenario.",
}

JOIN_HINTS = [
    "`images.image_id` -> `image_trajectories.image_id`",
    "`images.(split, scene_id, frame_timestamp_micros)` -> `ego_poses.(split, scene_id, frame_timestamp_micros)`",
    "`trajectories.trajectory_row_id` -> `image_trajectories.trajectory_row_id`",
]


@dataclass
class FunctionSchema:
    columns: list[str] = field(default_factory=list)
    types: dict[str, str] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)


@dataclass
class ConverterSchema:
    table_to_collection: dict[str, str]
    collection_to_builders: dict[str, list[str]]
    functions: dict[str, FunctionSchema]


class ConverterVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.table_to_collection: dict[str, str] = {}
        self.assignments: dict[str, str] = {}
        self.extends: dict[str, list[str]] = {}
        self.functions: dict[str, FunctionSchema] = {}
        self._current_function: str | None = None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        previous = self._current_function
        self._current_function = node.name
        self.functions.setdefault(node.name, FunctionSchema())
        self.generic_visit(node)
        self._current_function = previous

    def visit_Assign(self, node: ast.Assign) -> Any:
        if isinstance(node.value, ast.Call):
            call_name = dotted_name(node.value.func)
            if call_name:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.assignments[target.id] = call_name
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        call_name = dotted_name(node.func)
        if self._current_function and call_name:
            self.functions.setdefault(self._current_function, FunctionSchema()).calls.append(call_name)

        if call_name == "write_parquet" and len(node.args) >= 2:
            table = parquet_table_name(node.args[0])
            collection = name_of(node.args[1])
            if table and collection:
                self.table_to_collection[table] = collection

        if isinstance(node.func, ast.Attribute) and node.func.attr == "extend":
            collection = name_of(node.func.value)
            source = name_of(node.args[0]) if node.args else None
            if collection and source:
                self.extends.setdefault(collection, []).append(source)

        if isinstance(node.func, ast.Attribute) and node.func.attr == "append":
            schema = self.functions.setdefault(self._current_function or "", FunctionSchema())
            if node.args:
                extract_dict_columns(node.args[0], schema)

        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        self.generic_visit(node)

    def visit_Module(self, node: ast.Module) -> Any:
        self.generic_visit(node)
        self._extract_subscript_assignments(node)

    def _extract_subscript_assignments(self, node: ast.AST) -> None:
        for func in [n for n in ast.walk(node) if isinstance(n, ast.FunctionDef)]:
            schema = self.functions.setdefault(func.name, FunctionSchema())
            for child in ast.walk(func):
                if not isinstance(child, ast.Assign):
                    continue
                for target in child.targets:
                    key = subscript_string_key(target)
                    if key:
                        add_column(schema, key, infer_type(key, child.value))

    def result(self) -> ConverterSchema:
        collection_to_builders: dict[str, list[str]] = {}
        for collection in self.table_to_collection.values():
            builders = self.resolve_collection_builders(collection)
            collection_to_builders[collection] = builders
        return ConverterSchema(self.table_to_collection, collection_to_builders, self.functions)

    def resolve_collection_builders(self, collection: str) -> list[str]:
        builders: list[str] = []
        seen: set[str] = set()

        def add_builder(func_name: str) -> None:
            if func_name in seen:
                return
            seen.add(func_name)
            builders.append(func_name)
            for nested in nested_row_builders(func_name, self.functions):
                add_builder(nested)

        if collection in self.assignments:
            add_builder(self.assignments[collection])
        for source in self.extends.get(collection, []):
            if source in self.assignments:
                add_builder(self.assignments[source])
        return builders


def nested_row_builders(func_name: str, functions: dict[str, FunctionSchema]) -> list[str]:
    nested = []
    for call in functions.get(func_name, FunctionSchema()).calls:
        if call in functions and call != func_name:
            if functions[call].columns:
                nested.append(call)
            elif call.startswith("build_") or call.endswith("_from_scenario"):
                nested.append(call)
    return nested


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def name_of(node: ast.AST) -> str | None:
    return node.id if isinstance(node, ast.Name) else None


def parquet_table_name(node: ast.AST) -> str | None:
    filename = string_leaf(node)
    if filename and filename.endswith(".parquet"):
        return filename.removesuffix(".parquet")
    return None


def string_leaf(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp):
        return string_leaf(node.right) or string_leaf(node.left)
    if isinstance(node, ast.Call):
        for arg in reversed(node.args):
            value = string_leaf(arg)
            if value:
                return value
    return None


def extract_dict_columns(node: ast.AST, schema: FunctionSchema) -> None:
    if not isinstance(node, ast.Dict):
        return
    for key_node, value_node in zip(node.keys, node.values):
        if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
            continue
        key = key_node.value
        add_column(schema, key, infer_type(key, value_node))


def subscript_string_key(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Subscript):
        return None
    if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        return node.slice.value
    return None


def add_column(schema: FunctionSchema, column: str, type_name: str) -> None:
    if column not in schema.columns:
        schema.columns.append(column)
    schema.types.setdefault(column, type_name)


def infer_type(column: str, node: ast.AST | None = None) -> str:
    if column == "image_jpeg":
        return "binary"
    if column == "ego_pose_transform":
        return "list<float64>"
    if column == "visible_trajectory_ids":
        return "list<string>"
    if column.endswith("_timestamps_seconds"):
        return "list<float64>"
    if column.endswith("_timestamps_micros") or column == "timestamps_micros":
        return "list<int64>"
    if column.endswith("_valid"):
        return "list<bool>"
    if column in {"x", "y", "z", "length", "width", "height", "heading", "velocity_x", "velocity_y", "velocity_z"}:
        return "list<float64>"
    if column.startswith(("observed_", "future_")) and column not in {
        "observed_valid",
        "future_valid",
        "observed_timestamps_seconds",
        "future_timestamps_seconds",
    }:
        return "list<float64>"
    if column.startswith(("bbox_", "ego_", "ego_fd_")) and column != "ego_velocity_source":
        return "float64"
    if column.startswith("is_") or column.startswith("has_"):
        return "bool"
    if column in {
        "frame_timestamp_micros",
        "camera_name",
        "image_width",
        "image_height",
        "num_visible_trajectories",
        "object_type",
        "num_steps",
        "track_index",
        "track_id",
        "rank",
        "difficulty",
        "current_time_index",
    }:
        return "int64"
    if column.endswith("_id") or column.endswith("_row_id") or column.endswith("_name") or column.endswith("_text"):
        return "string"
    if column in {"split", "scene_id", "image_format", "ego_velocity_source"}:
        return "string"

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return "bool"
        if isinstance(node.value, int):
            return "int64"
        if isinstance(node.value, float):
            return "float64"
        if isinstance(node.value, str):
            return "string"
        if isinstance(node.value, bytes):
            return "binary"
    if isinstance(node, ast.Call):
        call = dotted_name(node.func)
        if call == "int":
            return "int64"
        if call == "float":
            return "float64"
        if call == "str":
            return "string"
        if call == "bool":
            return "bool"
        if call == "bytes":
            return "binary"
    if isinstance(node, ast.List):
        return "list<unknown>"
    return "unknown"


def load_converter_schema(converter: Path) -> ConverterSchema:
    tree = ast.parse(converter.read_text(encoding="utf-8"), filename=str(converter))
    visitor = ConverterVisitor()
    visitor.visit(tree)
    return visitor.result()


def merged_table_columns(
    schema: ConverterSchema,
    collection: str,
) -> tuple[list[str], dict[str, str], list[str]]:
    columns: list[str] = []
    types: dict[str, str] = {}
    builders = schema.collection_to_builders.get(collection, [])
    for builder in builders:
        function_schema = schema.functions.get(builder, FunctionSchema())
        for column in function_schema.columns:
            if column not in columns:
                columns.append(column)
            types.setdefault(column, function_schema.types.get(column, "unknown"))
    return columns, types, builders


def render_markdown(schema: ConverterSchema, converter: Path) -> str:
    lines = [
        "# Generated RGB Trajectory Parquet Interface",
        "",
        "This file is generated from the converter implementation.",
        "",
        "```bash",
        f"python scripts/generate_rgb_trajectory_schema_doc.py --converter {converter}",
        "```",
        "",
        "Do not edit this file by hand. Update the converter or the generator, then regenerate it.",
        "",
        "## Source",
        "",
        f"- Converter: `{converter}`",
        "- Extraction method: `write_parquet(...)` calls plus constant keys in row dictionaries.",
        "",
        "## Tables",
        "",
    ]

    for table, collection in schema.table_to_collection.items():
        description = TABLE_DESCRIPTIONS.get(table, "")
        lines.append(f"### {table}.parquet")
        lines.append("")
        if description:
            lines.append(description)
            lines.append("")
        columns, types, builders = merged_table_columns(schema, collection)
        lines.append(f"- Output collection: `{collection}`")
        lines.append(f"- Builder functions: {', '.join(f'`{name}`' for name in builders) or '`unknown`'}")
        lines.append("")
        lines.append("| Column | Inferred Type | Note |")
        lines.append("| --- | --- | --- |")
        for column in columns:
            lines.append(f"| `{column}` | `{types.get(column, 'unknown')}` | {column_note(column)} |")
        lines.append("")

    lines.extend(
        [
            "## Join Hints",
            "",
            *[f"- {hint}" for hint in JOIN_HINTS],
            "",
            "## Type Inference Notes",
            "",
            "Types are inferred statically from the converter code and column naming conventions.",
            "Use `pyarrow.parquet.read_schema(...)` on a concrete output file when you need the exact physical Arrow type for a particular run.",
            "",
        ]
    )
    return "\n".join(lines)


def column_note(column: str) -> str:
    if column in COLUMN_NOTES:
        return COLUMN_NOTES[column]
    if column.startswith("observed_"):
        return "Observed history array from WOMD current/past timesteps."
    if column.startswith("future_"):
        return "Future label array from WOMD timesteps after current_time_index, when present."
    if column.startswith("ego_velocity_"):
        return "Preferred ego linear velocity component."
    if column.startswith("ego_angular_velocity_"):
        return "Ego angular velocity component."
    if column.startswith("ego_fd_velocity_"):
        return "Finite-difference ego velocity component derived from pose translation."
    if column.startswith("velocity_"):
        return "Object velocity component."
    return "Generated by the converter."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--converter",
        type=Path,
        default=Path("scripts/build_waymo_rgb_trajectory_dataset.py"),
        help="Path to the RGB trajectory dataset converter.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/rgb_trajectory_parquet_interface.md"),
        help="Markdown file to write.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    schema = load_converter_schema(args.converter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(schema, args.converter), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
