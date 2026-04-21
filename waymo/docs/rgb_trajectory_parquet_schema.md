# Waymo RGB Trajectory Parquet Schema

This document describes the Parquet files produced by:

```bash
python scripts/build_waymo_rgb_trajectory_dataset.py
```

The converted dataset is stored by default in:

```text
dataset/rgb_trajectory_dataset/
```

The main tables are:

```text
images.parquet
ego_poses.parquet
trajectories.parquet
image_trajectories.parquet
prediction_targets.parquet
manifest.json
```

`images`, `ego_poses`, `trajectories`, and `image_trajectories` form the
RGB-to-visible-trajectory supervised dataset.
`prediction_targets.parquet` stores WOMD `tracks_to_predict` entries, mainly
useful for test-time inference targets.

The exact table/column interface can be regenerated from the converter code:

```bash
python scripts/generate_rgb_trajectory_schema_doc.py
```

This writes:

```text
docs/rgb_trajectory_parquet_interface.md
```

Use this generated file as the source of truth for column lists after changing
`scripts/build_waymo_rgb_trajectory_dataset.py`.

## Coordinate Frames

The `trajectories.parquet` and `prediction_targets.parquet` coordinates are in
Waymo scenario/world coordinates, not ego coordinates. They are absolute within
the local scenario frame.

The web viewer converts trajectories into an egocentric visualization by placing
ego at the canvas center. That is a visualization transform, not the storage
format.

For ego-relative modeling, transform each object state with the SDC pose at the
same timestep:

```python
dx = object_x - ego_x
dy = object_y - ego_y

cos_h = cos(-ego_heading)
sin_h = sin(-ego_heading)

x_ego = cos_h * dx - sin_h * dy
y_ego = sin_h * dx + cos_h * dy
heading_ego = object_heading - ego_heading
```

## Ego Agent Pose and Velocity

The ego vehicle is provided in the source datasets, but it appears in different
places depending on the source.

### WOMD Motion TFRecords

In Waymo Open Motion Dataset `Scenario` TFRecords, the ego vehicle is the SDC
track:

```text
Scenario.sdc_track_index
```

This index points into:

```text
Scenario.tracks[Scenario.sdc_track_index]
```

The ego track has the same per-timestep fields as every other object:

```text
center_x
center_y
center_z
heading
velocity_x
velocity_y
length
width
height
valid
```

In tables produced by `scripts/extract_waymo_motion.py`, ego rows are marked by:

```text
is_sdc == True
```

Example:

```python
import pandas as pd

states = pd.read_parquet("dataset/processed_motion/states.parquet")

ego = (
    states[states["is_sdc"]]
    .sort_values(["scenario_id", "time_index"])
)

ego[[
    "scenario_id",
    "time_index",
    "timestamp_seconds",
    "center_x",
    "center_y",
    "center_z",
    "heading",
    "velocity_x",
    "velocity_y",
]]
```

So the WOMD motion dataset provides ego position, heading, and planar velocity.
It does not include `velocity_z` in `ObjectState`.

### RGB/Sensory v2 Component Parquet

The v2 sensory component parquet data provides ego/vehicle pose separately from
object labels.

Relevant components:

```text
vehicle_pose/
camera_image/
```

`camera_image` includes camera pose and camera/vehicle motion fields such as:

```text
[CameraImageComponent].pose.transform
[CameraImageComponent].velocity.linear_velocity.x
[CameraImageComponent].velocity.linear_velocity.y
[CameraImageComponent].velocity.linear_velocity.z
[CameraImageComponent].velocity.angular_velocity.x
[CameraImageComponent].velocity.angular_velocity.y
[CameraImageComponent].velocity.angular_velocity.z
```

The converted RGB-trajectory dataset writes this information to:

```text
ego_poses.parquet
```

This table uses `vehicle_pose` for position/orientation and uses averaged
`camera_image` linear/angular velocity fields per timestamp when available. It
also stores finite-difference velocities derived from the pose translation.

With `ego_poses.parquet`, every RGB image and object trajectory can be
transformed into an ego frame consistently by joining on `split`, `scene_id`,
and `frame_timestamp_micros`.

## Join Keys

Common keys:

| Key | Meaning |
| --- | --- |
| `split` | Dataset split such as `training`, `validation`, or `testing`. |
| `scene_id` | Waymo scene/segment identifier. |
| `image_id` | Unique RGB camera-frame id: `scene_id;timestamp;camera_<id>`. |
| `trajectory_id` | Object id within a scene. For sensory labels this is the Waymo laser object id string. For WOMD prediction targets this is the track id converted to string. |
| `trajectory_row_id` | Unique trajectory row key: `scene_id;trajectory_id`. |

Primary joins:

```text
images.image_id
  -> image_trajectories.image_id

images.(split, scene_id, frame_timestamp_micros)
  -> ego_poses.(split, scene_id, frame_timestamp_micros)

trajectories.trajectory_row_id
  -> image_trajectories.trajectory_row_id

prediction_targets.trajectory_row_id
  -> prediction target row identity
```

`prediction_targets` may not join to `trajectories` for test data because the
RGB sensory v2 labels and WOMD motion test scenarios can be different samples,
and Waymo test labels/future GT are withheld.

## images.parquet

One row per standalone RGB camera image.

| Column | Type | Meaning |
| --- | --- | --- |
| `split` | string | Dataset split. |
| `image_id` | string | Unique image key: `scene_id;frame_timestamp_micros;camera_<camera_name>`. |
| `scene_id` | string | Waymo segment/scene id from the sensory component table. |
| `frame_timestamp_micros` | int64 | Frame timestamp in microseconds. |
| `camera_name` | int64 | Waymo camera enum id. Common values: `1=FRONT`, `2=FRONT_LEFT`, `3=FRONT_RIGHT`, `4=SIDE_LEFT`, `5=SIDE_RIGHT`. |
| `camera_name_text` | string | Human-readable camera name. |
| `image_jpeg` | binary | JPEG-encoded RGB image bytes. |
| `image_format` | string | Currently `jpeg`. |
| `image_width` | int64 | Decoded image width in pixels. |
| `image_height` | int64 | Decoded image height in pixels. |
| `visible_trajectory_ids` | list<string> | Trajectory ids visible in this image according to `projected_lidar_box`. |
| `num_visible_trajectories` | int64 | Number of visible trajectory ids. |

PyTorch usage:

```python
from io import BytesIO
from PIL import Image

image = Image.open(BytesIO(row["image_jpeg"])).convert("RGB")
```

## ego_poses.parquet

One row per vehicle pose timestep from the v2 `vehicle_pose` component.

| Column | Type | Meaning |
| --- | --- | --- |
| `split` | string | Dataset split. |
| `scene_id` | string | Scene/segment id. |
| `frame_timestamp_micros` | int64 | Frame timestamp in microseconds. |
| `ego_pose_transform` | list<float64> | 4x4 row-major `world_from_vehicle` transform. |
| `ego_x` | float64 | Ego position x from transform translation. |
| `ego_y` | float64 | Ego position y from transform translation. |
| `ego_z` | float64 | Ego position z from transform translation. |
| `ego_roll` | float64 | Ego roll estimated from the rotation matrix. |
| `ego_pitch` | float64 | Ego pitch estimated from the rotation matrix. |
| `ego_yaw` | float64 | Ego yaw/heading estimated from the rotation matrix. |
| `ego_velocity_x` | float64 | Preferred ego linear velocity x. Uses camera-image velocity when available, otherwise finite-difference pose velocity. |
| `ego_velocity_y` | float64 | Preferred ego linear velocity y. |
| `ego_velocity_z` | float64 | Preferred ego linear velocity z. |
| `ego_angular_velocity_x` | float64 | Ego angular velocity x from `camera_image`, when available. |
| `ego_angular_velocity_y` | float64 | Ego angular velocity y from `camera_image`, when available. |
| `ego_angular_velocity_z` | float64 | Ego angular velocity z from `camera_image`, when available. |
| `ego_velocity_source` | string | `camera_image` or `finite_difference`. |
| `ego_fd_velocity_x` | float64 | Finite-difference velocity x computed from `ego_x`. |
| `ego_fd_velocity_y` | float64 | Finite-difference velocity y computed from `ego_y`. |
| `ego_fd_velocity_z` | float64 | Finite-difference velocity z computed from `ego_z`. |

Example join from an RGB image to ego pose:

```python
images = pd.read_parquet("dataset/rgb_trajectory_dataset/images.parquet")
ego = pd.read_parquet("dataset/rgb_trajectory_dataset/ego_poses.parquet")

images_with_ego = images.merge(
    ego,
    on=["split", "scene_id", "frame_timestamp_micros"],
    how="left",
)
```

## trajectories.parquet

One row per object trajectory from the v2 sensory label components. The source
is `lidar_box`, used as label/trajectory ground truth. Lidar point clouds are
not stored or used as model input.

| Column | Type | Meaning |
| --- | --- | --- |
| `split` | string | Dataset split. |
| `scene_id` | string | Waymo sensory segment id. |
| `trajectory_id` | string | Waymo laser object id. |
| `trajectory_row_id` | string | Unique key: `scene_id;trajectory_id`. |
| `object_type` | int64 | Waymo object type enum from labels. |
| `num_steps` | int64 | Number of timesteps in this trajectory. |
| `timestamps_micros` | list<int64> | Timestamps for each trajectory state. |
| `x` | list<float64> | Object center x at each timestep, in scenario/world frame. |
| `y` | list<float64> | Object center y at each timestep, in scenario/world frame. |
| `z` | list<float64> | Object center z at each timestep, in scenario/world frame. |
| `length` | list<float64> | 3D box length at each timestep. |
| `width` | list<float64> | 3D box width at each timestep. |
| `height` | list<float64> | 3D box height at each timestep. |
| `heading` | list<float64> | Box yaw/heading in radians. |
| `velocity_x` | list<float64> | Velocity x in scenario/world frame. |
| `velocity_y` | list<float64> | Velocity y in scenario/world frame. |
| `velocity_z` | list<float64> | Velocity z in scenario/world frame. |

Array convention: all list columns in a row have length `num_steps` and are
aligned by index.

Example:

```python
xy = torch.tensor([row["x"], row["y"]], dtype=torch.float32).T
```

## image_trajectories.parquet

Many-to-many mapping between RGB images and trajectories visible in those
images. The source is `projected_lidar_box`.

One row means:

```text
trajectory_id is visible in image_id at frame_timestamp_micros
```

| Column | Type | Meaning |
| --- | --- | --- |
| `split` | string | Dataset split. |
| `image_id` | string | Foreign key into `images.parquet`. |
| `scene_id` | string | Scene id. |
| `frame_timestamp_micros` | int64 | Frame timestamp. |
| `camera_name` | int64 | Camera enum id. |
| `camera_name_text` | string | Camera name. |
| `trajectory_id` | string | Visible object trajectory id. |
| `trajectory_row_id` | string | Foreign key into `trajectories.parquet`. |
| `object_type` | int64 | Waymo object type enum. |
| `bbox_center_x` | float64 | Projected 2D box center x in pixels. |
| `bbox_center_y` | float64 | Projected 2D box center y in pixels. |
| `bbox_width` | float64 | Projected 2D box width in pixels. |
| `bbox_height` | float64 | Projected 2D box height in pixels. |
| `bbox_x1` | float64 | Left box edge in pixels. |
| `bbox_y1` | float64 | Top box edge in pixels. |
| `bbox_x2` | float64 | Right box edge in pixels. |
| `bbox_y2` | float64 | Bottom box edge in pixels. |

Use this table to select which object trajectories are visible from a camera
frame and to draw 2D boxes on the RGB image.

## prediction_targets.parquet

One row per WOMD `tracks_to_predict` target extracted from motion `Scenario`
TFRecords. This is separate from sensory v2 label trajectories.

For test data, this table contains observed history/current states and target
metadata, but usually no future ground truth.

| Column | Type | Meaning |
| --- | --- | --- |
| `split` | string | Motion split, e.g. `testing`. |
| `scene_id` | string | WOMD motion scenario id. |
| `trajectory_id` | string | Track id converted to string. |
| `trajectory_row_id` | string | Unique key: `scene_id;trajectory_id`. |
| `track_index` | int64 | Index into the WOMD `Scenario.tracks` array. |
| `track_id` | int64 | Waymo track id. |
| `rank` | int64 | Rank/order within `tracks_to_predict`. |
| `difficulty` | int64 | Waymo difficulty enum. |
| `difficulty_name` | string | Difficulty label. |
| `object_type` | int64 | Waymo object type enum. |
| `object_type_name` | string | Object type label. |
| `current_time_index` | int64 | WOMD current timestep index. |
| `observed_timestamps_seconds` | list<float64> | Observed timestamps up to current time. |
| `observed_valid` | list<bool> | Validity mask for observed states. |
| `observed_x` | list<float64> | Observed center x in scenario/world frame. |
| `observed_y` | list<float64> | Observed center y in scenario/world frame. |
| `observed_z` | list<float64> | Observed center z in scenario/world frame. |
| `observed_heading` | list<float64> | Observed heading/yaw. |
| `observed_velocity_x` | list<float64> | Observed velocity x. |
| `observed_velocity_y` | list<float64> | Observed velocity y. |
| `future_timestamps_seconds` | list<float64> | Future timestamps when future GT is available. Empty for held-out test GT. |
| `future_valid` | list<bool> | Future validity mask when available. |
| `future_x` | list<float64> | Future center x when available. |
| `future_y` | list<float64> | Future center y when available. |
| `future_z` | list<float64> | Future center z when available. |
| `future_heading` | list<float64> | Future heading when available. |
| `future_velocity_x` | list<float64> | Future velocity x when available. |
| `future_velocity_y` | list<float64> | Future velocity y when available. |
| `has_future_gt` | bool | Whether future arrays contain ground truth labels. |

Important caveat: `prediction_targets.scene_id` comes from WOMD motion
scenarios. It may not match `images.scene_id` from the v2 sensory component
sample unless you have downloaded a corresponding linked release or have an
explicit association table.

## manifest.json

Small metadata file with row counts and paths:

```json
{
  "splits": ["training", "validation", "testing"],
  "tables": {
    "images": {"path": "...", "rows": 2810},
    "ego_poses": {"path": "...", "rows": 562},
    "trajectories": {"path": "...", "rows": 219},
    "image_trajectories": {"path": "...", "rows": 10375},
    "prediction_targets": {"path": "...", "rows": 1251}
  }
}
```

## Typical Training Join

For supervised RGB-conditioned trajectory prediction on train/validation:

```python
import pandas as pd

images = pd.read_parquet("dataset/rgb_trajectory_dataset/images.parquet")
links = pd.read_parquet("dataset/rgb_trajectory_dataset/image_trajectories.parquet")
trajectories = pd.read_parquet("dataset/rgb_trajectory_dataset/trajectories.parquet")

sample = (
    links
    .merge(images, on=["split", "scene_id", "image_id", "frame_timestamp_micros", "camera_name", "camera_name_text"])
    .merge(trajectories, on=["split", "scene_id", "trajectory_id", "trajectory_row_id", "object_type"])
)
```

For each `image_id`, the rows in `image_trajectories` identify visible targets.
You can group by `image_id` to build a multi-target sample.

## Test-Time Inference

For WOMD testing:

```python
targets = pd.read_parquet("dataset/rgb_trajectory_dataset/prediction_targets.parquet")
test_targets = targets[targets["split"] == "testing"]
```

These rows provide observed histories and track ids to predict. Future GT is
withheld, so `has_future_gt` is normally `False`.

## Viewer Files

The web viewer is generated separately by:

```bash
python scripts/build_waymo_web_viewer.py \
  --dataset-root dataset/rgb_trajectory_dataset
```

Viewer output:

```text
dataset/rgb_trajectory_dataset/viewer/index.html
dataset/rgb_trajectory_dataset/viewer/viewer_data.js
dataset/rgb_trajectory_dataset/viewer/images/
```

The viewer intentionally hides RGB testing scenes with no linked trajectories
and prediction-target-only scenes. The Parquet tables still contain that data.
