#!/usr/bin/env python3
"""Build a static HTML viewer for the converted Waymo RGB trajectory dataset."""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    build_web_viewer_from_tables(
        dataset_root=dataset_root,
        viewer_dir=Path(args.output) if args.output else dataset_root / "viewer",
        max_scenes=args.max_scenes,
        max_images_per_scene=args.max_images_per_scene,
    )
    return 0


def build_web_viewer_from_tables(
    dataset_root: Path,
    viewer_dir: Path | None = None,
    max_scenes: int = 24,
    max_images_per_scene: int = 8,
) -> None:
    viewer_dir = viewer_dir or dataset_root / "viewer"
    images = read_records(dataset_root / "images.parquet")
    trajectories = read_records(dataset_root / "trajectories.parquet")
    links = read_records(dataset_root / "image_trajectories.parquet")
    prediction_targets = read_records(dataset_root / "prediction_targets.parquet")
    build_web_viewer(
        viewer_dir=viewer_dir,
        images=images,
        trajectories=trajectories,
        links=links,
        prediction_targets=prediction_targets,
        max_scenes=max_scenes,
        max_images_per_scene=max_images_per_scene,
    )


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pd.read_parquet(path).to_dict("records")


def build_web_viewer(
    viewer_dir: Path,
    images: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
    links: list[dict[str, Any]],
    prediction_targets: list[dict[str, Any]],
    max_scenes: int,
    max_images_per_scene: int,
) -> None:
    image_dir = viewer_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    links_by_image: dict[str, list[dict[str, Any]]] = {}
    for row in links:
        links_by_image.setdefault(row["image_id"], []).append(row)

    images_by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in images:
        images_by_scene.setdefault(row["scene_id"], []).append(row)

    trajectories_by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in trajectories:
        trajectories_by_scene.setdefault(row["scene_id"], []).append(row)

    targets_by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in prediction_targets:
        targets_by_scene.setdefault(row["scene_id"], []).append(row)

    # The viewer is for inspecting RGB-to-visible-trajectory supervision. Keep
    # prediction-target-only and unlabeled RGB test scenes in the parquet tables,
    # but hide them from the interactive visualization until they can be linked.
    linked_image_ids = {row["image_id"] for row in links}
    scene_ids = sorted(
        scene_id
        for scene_id, scene_images in images_by_scene.items()
        if scene_id in trajectories_by_scene
        and any(image["image_id"] in linked_image_ids for image in scene_images)
    )
    if max_scenes:
        scene_ids = scene_ids[:max_scenes]

    viewer_scenes = []
    for scene_id in scene_ids:
        scene_images = sorted(
            [
                image
                for image in images_by_scene.get(scene_id, [])
                if image["image_id"] in linked_image_ids
            ],
            key=lambda r: (r["frame_timestamp_micros"], r["camera_name"]),
        )
        if max_images_per_scene:
            scene_images = scene_images[:max_images_per_scene]

        image_items = []
        for image_row in scene_images:
            rel_path = f"images/{safe_name(image_row['image_id'])}.jpg"
            out_path = viewer_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not out_path.exists():
                with Image.open(BytesIO(image_row["image_jpeg"])) as image:
                    image.convert("RGB").save(out_path, quality=90)
            image_items.append(
                {
                    "image_id": image_row["image_id"],
                    "src": rel_path,
                    "timestamp": int(image_row["frame_timestamp_micros"]),
                    "camera_name": int(image_row["camera_name"]),
                    "camera_name_text": image_row["camera_name_text"],
                    "width": none_to_zero(image_row.get("image_width")),
                    "height": none_to_zero(image_row.get("image_height")),
                    "visible_trajectory_ids": list_or_empty(image_row.get("visible_trajectory_ids")),
                    "boxes": [
                        link_to_viewer_box(link)
                        for link in links_by_image.get(image_row["image_id"], [])
                    ],
                }
            )

        viewer_scenes.append(
            {
                "scene_id": scene_id,
                "images": image_items,
                "trajectories": [
                    trajectory_to_viewer(row)
                    for row in trajectories_by_scene.get(scene_id, [])
                ],
                "prediction_targets": [
                    prediction_target_to_viewer(row)
                    for row in targets_by_scene.get(scene_id, [])
                ],
            }
        )

    data = {"scenes": viewer_scenes}
    (viewer_dir / "viewer_data.js").write_text(
        "window.WAYMO_VIEWER_DATA = " + json.dumps(data) + ";\n",
        encoding="utf-8",
    )
    (viewer_dir / "index.html").write_text(viewer_html(), encoding="utf-8")
    print(f"[web-viewer] {viewer_dir / 'index.html'} scenes={len(viewer_scenes)}")


def link_to_viewer_box(link: dict[str, Any]) -> dict[str, Any]:
    return {
        "trajectory_id": link["trajectory_id"],
        "object_type": int(link["object_type"]),
        "x1": float(link["bbox_x1"]),
        "y1": float(link["bbox_y1"]),
        "x2": float(link["bbox_x2"]),
        "y2": float(link["bbox_y2"]),
    }


def trajectory_to_viewer(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "trajectory_id": row["trajectory_id"],
        "object_type": int(row["object_type"]),
        "x": [float(v) for v in list_or_empty(row.get("x"))],
        "y": [float(v) for v in list_or_empty(row.get("y"))],
        "z": [float(v) for v in list_or_empty(row.get("z"))],
        "timestamps": [int(v) for v in list_or_empty(row.get("timestamps_micros"))],
    }


def prediction_target_to_viewer(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "trajectory_id": row["trajectory_id"],
        "track_id": int(row["track_id"]),
        "track_index": int(row["track_index"]),
        "rank": int(row["rank"]),
        "object_type": int(row["object_type"]),
        "has_future_gt": bool(row["has_future_gt"]),
        "timestamps": [float(v) for v in list_or_empty(row.get("observed_timestamps_seconds"))],
        "x": [float(v) for v in list_or_empty(row.get("observed_x"))],
        "y": [float(v) for v in list_or_empty(row.get("observed_y"))],
        "z": [float(v) for v in list_or_empty(row.get("observed_z"))],
        "future_x": [float(v) for v in list_or_empty(row.get("future_x"))],
        "future_y": [float(v) for v in list_or_empty(row.get("future_y"))],
        "future_z": [float(v) for v in list_or_empty(row.get("future_z"))],
    }


def list_or_empty(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def none_to_zero(value: Any) -> int:
    if value is None:
        return 0
    return int(value)


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def viewer_html() -> str:
    return r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Waymo RGB Trajectory Viewer</title>
  <style>
    :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #f5f7fa; color: #17202a; }
    header { padding: 16px 22px; background: #0e1b2a; color: #fff; display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
    h1 { font-size: 20px; margin: 0; font-weight: 700; }
    select, button, input { font: inherit; }
    select, button { border: 1px solid #ccd4df; background: #fff; border-radius: 6px; padding: 7px 10px; }
    main { display: grid; grid-template-columns: 340px minmax(0, 1fr); gap: 16px; padding: 16px; }
    aside { display: flex; flex-direction: column; gap: 12px; min-width: 0; }
    section { background: #fff; border: 1px solid #dde4ed; border-radius: 8px; padding: 12px; box-shadow: 0 1px 2px rgba(20, 35, 55, .04); }
    h2 { font-size: 15px; margin: 0 0 10px; }
    .scene-meta { font-size: 13px; line-height: 1.45; color: #526070; }
    .list { max-height: 260px; overflow: auto; display: grid; gap: 6px; }
    .item { text-align: left; border: 1px solid #d7dfeb; background: #fbfcfe; border-radius: 6px; padding: 8px; cursor: pointer; width: 100%; }
    .item.active { border-color: #136f63; background: #e8f5f1; }
    .item strong { display: block; font-size: 12px; overflow-wrap: anywhere; }
    .item span { color: #607084; font-size: 12px; }
    .workspace { display: grid; grid-template-rows: minmax(260px, 44vh) minmax(260px, 42vh); gap: 16px; min-width: 0; }
    .image-wrap, .traj-wrap { position: relative; min-width: 0; min-height: 0; }
    canvas { width: 100%; height: 100%; display: block; background: #101820; border-radius: 6px; }
    #rgbCanvas { background: #101820; }
    .time-controls { display: grid; grid-template-columns: auto minmax(120px, 1fr) auto auto; gap: 10px; align-items: center; margin-bottom: 10px; font-size: 13px; color: #526070; }
    .time-controls input { width: 100%; accent-color: #136f63; }
    .hint { color: #657386; font-size: 13px; }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      .workspace { grid-template-rows: 360px 360px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Waymo RGB Trajectory Viewer</h1>
    <label>Scene <select id="sceneSelect"></select></label>
  </header>
  <main>
    <aside>
      <section>
        <h2>Scene Data</h2>
        <div id="sceneMeta" class="scene-meta"></div>
      </section>
      <section>
        <h2>RGB Images</h2>
        <div id="imageList" class="list"></div>
      </section>
      <section>
        <h2>Trajectories</h2>
        <div id="trajectoryList" class="list"></div>
      </section>
      <section>
        <h2>Prediction Targets</h2>
        <div id="targetList" class="list"></div>
      </section>
    </aside>
    <div class="workspace">
      <section class="image-wrap">
        <h2>RGB With Visible Targets</h2>
        <canvas id="rgbCanvas"></canvas>
      </section>
      <section class="traj-wrap">
        <h2>Egocentric Trajectory Evolution</h2>
        <div class="time-controls">
          <span>Time</span>
          <input id="timeSlider" type="range" min="0" max="0" value="0" step="1">
          <span id="timeLabel">0 / 0</span>
          <button id="autoplayButton" type="button">Play</button>
        </div>
        <canvas id="trajectoryCanvas"></canvas>
      </section>
    </div>
  </main>
  <script src="viewer_data.js"></script>
  <script>
    const data = window.WAYMO_VIEWER_DATA;
    let currentScene = null;
    let currentImage = null;
    let highlightedTrajectoryId = null;
    let imageBitmap = null;
    let sceneTimeline = [];
    let currentTimeIndex = 0;
    let autoplayTimer = null;
    let clickTargets = [];

    const sceneSelect = document.getElementById('sceneSelect');
    const sceneMeta = document.getElementById('sceneMeta');
    const imageList = document.getElementById('imageList');
    const trajectoryList = document.getElementById('trajectoryList');
    const targetList = document.getElementById('targetList');
    const rgbCanvas = document.getElementById('rgbCanvas');
    const trajCanvas = document.getElementById('trajectoryCanvas');
    const timeSlider = document.getElementById('timeSlider');
    const timeLabel = document.getElementById('timeLabel');
    const autoplayButton = document.getElementById('autoplayButton');
    const rgbCtx = rgbCanvas.getContext('2d');
    const trajCtx = trajCanvas.getContext('2d');

    function init() {
      data.scenes.forEach((scene, i) => {
        const option = document.createElement('option');
        option.value = String(i);
        option.textContent = scene.scene_id;
        sceneSelect.appendChild(option);
      });
      sceneSelect.addEventListener('change', () => selectScene(Number(sceneSelect.value)));
      timeSlider.addEventListener('input', () => {
        setTimeIndex(Number(timeSlider.value), true);
      });
      autoplayButton.addEventListener('click', toggleAutoplay);
      trajCanvas.addEventListener('click', handleTrajectoryCanvasClick);
      window.addEventListener('resize', drawAll);
      selectScene(0);
    }

    function selectScene(index) {
      currentScene = data.scenes[index];
      currentImage = currentScene.images[0] || null;
      highlightedTrajectoryId = null;
      imageBitmap = null;
      stopAutoplay();
      sceneTimeline = buildSceneTimeline(currentScene);
      currentTimeIndex = currentImage ? nearestTimelineIndex(currentImage.timestamp) : 0;
      configureTimeSlider();
      renderLists();
      loadCurrentImage();
      drawAll();
    }

    function renderLists() {
      sceneMeta.innerHTML = `
        <div><strong>${escapeHtml(currentScene.scene_id)}</strong></div>
        <div>RGB images: ${currentScene.images.length}</div>
        <div>3D trajectories: ${currentScene.trajectories.length}</div>
        <div>Prediction targets: ${currentScene.prediction_targets.length}</div>
      `;
      renderImages();
      renderTrajectories();
      renderPredictionTargets();
    }

    function renderImages() {
      imageList.innerHTML = '';
      currentScene.images.forEach((img) => {
        const button = document.createElement('button');
        button.className = 'item' + (currentImage && img.image_id === currentImage.image_id ? ' active' : '');
        button.innerHTML = `<strong>${escapeHtml(img.camera_name_text)} @ ${img.timestamp}</strong><span>${img.visible_trajectory_ids.length} visible trajectories</span>`;
        button.onclick = () => {
          currentImage = img;
          highlightedTrajectoryId = null;
          setTimeIndex(nearestTimelineIndex(img.timestamp), false);
          renderLists();
          loadCurrentImage();
        };
        imageList.appendChild(button);
      });
      if (!currentScene.images.length) imageList.innerHTML = '<div class="hint">No RGB images for this scene.</div>';
    }

    function renderTrajectories() {
      trajectoryList.innerHTML = '';
      currentScene.trajectories.forEach((traj) => {
        const button = document.createElement('button');
        button.className = 'item' + (traj.trajectory_id === highlightedTrajectoryId ? ' active' : '');
        button.innerHTML = `<strong>${escapeHtml(traj.trajectory_id)}</strong><span>${traj.x.length} steps, type ${traj.object_type}</span>`;
        button.onclick = () => { highlightedTrajectoryId = traj.trajectory_id; renderLists(); drawAll(); };
        trajectoryList.appendChild(button);
      });
      if (!currentScene.trajectories.length) trajectoryList.innerHTML = '<div class="hint">No label trajectories for this scene.</div>';
    }

    function renderPredictionTargets() {
      targetList.innerHTML = '';
      currentScene.prediction_targets.forEach((target) => {
        const button = document.createElement('button');
        button.className = 'item' + (target.trajectory_id === highlightedTrajectoryId ? ' active' : '');
        button.innerHTML = `<strong>track ${target.track_id} rank ${target.rank}</strong><span>${target.x.length} observed steps, future GT: ${target.has_future_gt}</span>`;
        button.onclick = () => { highlightedTrajectoryId = target.trajectory_id; renderLists(); drawAll(); };
        targetList.appendChild(button);
      });
      if (!currentScene.prediction_targets.length) targetList.innerHTML = '<div class="hint">No WOMD prediction targets for this scene.</div>';
    }

    function loadCurrentImage() {
      if (!currentImage) { imageBitmap = null; drawAll(); return; }
      const img = new Image();
      img.onload = () => { imageBitmap = img; drawAll(); };
      img.src = currentImage.src;
    }

    function drawAll() { drawRgb(); drawTrajectories(); }

    function buildSceneTimeline(scene) {
      const values = [];
      scene.trajectories.forEach((traj) => (traj.timestamps || []).forEach((t) => values.push(Number(t))));
      scene.prediction_targets.forEach((target) => (target.timestamps || []).forEach((t) => values.push(Number(t))));
      const unique = Array.from(new Set(values.filter(Number.isFinite))).sort((a, b) => a - b);
      if (unique.length) return unique;
      let maxLen = 0;
      scene.trajectories.forEach((traj) => { maxLen = Math.max(maxLen, traj.x.length); });
      scene.prediction_targets.forEach((target) => { maxLen = Math.max(maxLen, target.x.length); });
      return Array.from({ length: maxLen }, (_, i) => i);
    }

    function nearestTimelineIndex(value) {
      if (!sceneTimeline.length || !Number.isFinite(Number(value))) return 0;
      let bestIndex = 0;
      let bestDistance = Infinity;
      sceneTimeline.forEach((t, i) => {
        const distance = Math.abs(Number(t) - Number(value));
        if (distance < bestDistance) {
          bestIndex = i;
          bestDistance = distance;
        }
      });
      return bestIndex;
    }

    function configureTimeSlider() {
      const maxIndex = Math.max(0, sceneTimeline.length - 1);
      currentTimeIndex = Math.min(Math.max(0, currentTimeIndex), maxIndex);
      timeSlider.min = '0';
      timeSlider.max = String(maxIndex);
      timeSlider.value = String(currentTimeIndex);
      timeSlider.disabled = sceneTimeline.length <= 1;
      updateTimeLabel();
    }

    function setTimeIndex(index, syncRgbImage) {
      const maxIndex = Math.max(0, sceneTimeline.length - 1);
      currentTimeIndex = Math.min(Math.max(0, index), maxIndex);
      timeSlider.value = String(currentTimeIndex);
      updateTimeLabel();
      if (syncRgbImage) syncImageToCurrentTime();
      renderLists();
      drawAll();
    }

    function syncImageToCurrentTime() {
      if (!currentImage) return;
      const replacement = nearestImageForCameraAtTime(currentImage.camera_name, sceneTimeline[currentTimeIndex]);
      if (replacement && replacement.image_id !== currentImage.image_id) {
        currentImage = replacement;
        loadCurrentImage();
      }
    }

    function updateTimeLabel() {
      const value = sceneTimeline[currentTimeIndex];
      const display = value === undefined ? 'n/a' : formatTime(value);
      timeLabel.textContent = `${currentTimeIndex + 1} / ${Math.max(1, sceneTimeline.length)} · ${display}`;
    }

    function formatTime(value) {
      if (Math.abs(value) > 1000000) return `${Math.round(value / 1000) / 1000}s`;
      return `${Math.round(value * 100) / 100}s`;
    }

    function toggleAutoplay() {
      if (autoplayTimer) {
        stopAutoplay();
        return;
      }
      autoplayButton.textContent = 'Pause';
      autoplayTimer = window.setInterval(() => {
        if (!sceneTimeline.length) return;
        const next = currentTimeIndex >= sceneTimeline.length - 1 ? 0 : currentTimeIndex + 1;
        setTimeIndex(next, true);
      }, 350);
    }

    function stopAutoplay() {
      if (!autoplayTimer) return;
      window.clearInterval(autoplayTimer);
      autoplayTimer = null;
      autoplayButton.textContent = 'Play';
    }

    function nearestImageForCameraAtTime(cameraName, timeValue) {
      const candidates = currentScene.images.filter((img) => img.camera_name === cameraName);
      if (!candidates.length) return null;
      const tolerance = imageTimeTolerance(candidates);
      let best = null;
      let bestDistance = Infinity;
      candidates.forEach((img) => {
        const distance = Math.abs(Number(img.timestamp) - Number(timeValue));
        if (distance < bestDistance) {
          best = img;
          bestDistance = distance;
        }
      });
      return bestDistance <= tolerance ? best : null;
    }

    function imageTimeTolerance(images) {
      const times = images.map((img) => Number(img.timestamp)).filter(Number.isFinite).sort((a, b) => a - b);
      const deltas = times.slice(1).map((time, i) => Math.abs(time - times[i])).filter((delta) => delta > 0).sort((a, b) => a - b);
      return (deltas[0] || timelineTolerance()) * 0.6;
    }

    function timelineTolerance() {
      const deltas = sceneTimeline.slice(1).map((time, i) => Math.abs(Number(time) - Number(sceneTimeline[i]))).filter((delta) => delta > 0).sort((a, b) => a - b);
      return deltas[0] || 1;
    }

    function fitCanvas(canvas) {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const w = Math.max(1, Math.floor(rect.width * dpr));
      const h = Math.max(1, Math.floor(rect.height * dpr));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      return { w, h };
    }

    function drawRgb() {
      const { w, h } = fitCanvas(rgbCanvas);
      rgbCtx.clearRect(0, 0, w, h);
      rgbCtx.fillStyle = '#101820';
      rgbCtx.fillRect(0, 0, w, h);
      if (!currentImage || !imageBitmap) {
        drawCentered(rgbCtx, w, h, 'No RGB image selected');
        return;
      }
      const scale = Math.min(w / imageBitmap.width, h / imageBitmap.height);
      const dw = imageBitmap.width * scale;
      const dh = imageBitmap.height * scale;
      const ox = (w - dw) / 2;
      const oy = (h - dh) / 2;
      rgbCtx.drawImage(imageBitmap, ox, oy, dw, dh);
      currentImage.boxes.forEach((box) => {
        const active = !highlightedTrajectoryId || box.trajectory_id === highlightedTrajectoryId;
        rgbCtx.strokeStyle = active ? '#ffdf3d' : 'rgba(255,90,90,.65)';
        rgbCtx.lineWidth = active ? 4 : 2;
        rgbCtx.strokeRect(ox + box.x1 * scale, oy + box.y1 * scale, (box.x2 - box.x1) * scale, (box.y2 - box.y1) * scale);
        if (active) {
          rgbCtx.fillStyle = '#ffdf3d';
          rgbCtx.font = '14px sans-serif';
          rgbCtx.fillText(box.trajectory_id.slice(0, 8), ox + box.x1 * scale + 4, oy + box.y1 * scale - 5);
        }
      });
    }

    function drawTrajectories() {
      const { w, h } = fitCanvas(trajCanvas);
      clickTargets = [];
      trajCtx.clearRect(0, 0, w, h);
      trajCtx.fillStyle = '#f8fafc';
      trajCtx.fillRect(0, 0, w, h);
      const all = [
        ...currentScene.trajectories.map(t => ({...t, kind: 'label'})),
        ...currentScene.prediction_targets.map(t => ({...t, kind: 'target'})),
      ];
      if (!all.length || !sceneTimeline.length) {
        drawCentered(trajCtx, w, h, 'No 3D trajectories for this scene', '#526070');
        return;
      }

      const currentTime = sceneTimeline[currentTimeIndex];
      const activeStates = [];
      all.forEach((traj) => {
        const idx = trajectoryIndexAtTime(traj, currentTime);
        if (idx >= 0) activeStates.push({ traj, idx });
      });
      if (!activeStates.length) {
        drawEgoFrame(w, h, 10);
        drawCentered(trajCtx, w, h, 'No objects visible at this timestep', '#526070');
        return;
      }

      const extents = [];
      activeStates.forEach(({ traj, idx }) => {
        for (let i = 0; i <= idx; i++) {
          if (Number.isFinite(traj.x[i]) && Number.isFinite(traj.y[i])) {
            extents.push(Math.abs(traj.x[i]), Math.abs(traj.y[i]));
          }
        }
      });
      const maxMeters = Math.max(15, Math.min(120, percentile(extents, 0.92) * 1.25 || 40));
      const scale = Math.min(w, h) * 0.42 / maxMeters;
      drawEgoFrame(w, h, scale);

      function toCanvas(x, y) {
        return [w / 2 - y * scale, h / 2 - x * scale];
      }

      all.forEach((traj, idx) => {
        const currentIdx = trajectoryIndexAtTime(traj, currentTime);
        if (currentIdx < 0) return;
        const highlighted = traj.trajectory_id === highlightedTrajectoryId;
        const visible = currentImage && currentImage.visible_trajectory_ids.includes(traj.trajectory_id);
        const color = highlighted ? '#d7263d' : visible ? '#136f63' : traj.kind === 'target' ? '#3859c7' : palette(idx);
        trajCtx.strokeStyle = color;
        trajCtx.lineWidth = highlighted ? 5 : visible ? 4 : 2;
        trajCtx.globalAlpha = highlighted || visible || !highlightedTrajectoryId ? 1 : .2;
        trajCtx.beginPath();
        for (let i = 0; i <= currentIdx; i++) {
          const [cx, cy] = toCanvas(traj.x[i], traj.y[i] || 0);
          if (i === 0) trajCtx.moveTo(cx, cy); else trajCtx.lineTo(cx, cy);
        }
        trajCtx.stroke();
        const [ex, ey] = toCanvas(traj.x[currentIdx], traj.y[currentIdx] || 0);
        clickTargets.push({
          trajectory_id: traj.trajectory_id,
          x: ex,
          y: ey,
          radius: highlighted ? 14 : visible ? 12 : 10,
        });
        trajCtx.fillStyle = color;
        trajCtx.beginPath();
        trajCtx.arc(ex, ey, highlighted ? 8 : visible ? 6 : 5, 0, Math.PI * 2);
        trajCtx.fill();
        if (highlighted || visible) {
          trajCtx.font = '13px sans-serif';
          trajCtx.fillText(traj.trajectory_id.slice(0, 10), ex + 6, ey - 4);
        }
        trajCtx.globalAlpha = 1;
      });
      trajCtx.fillStyle = '#526070';
      trajCtx.font = '13px sans-serif';
      trajCtx.fillText('Ego is fixed at center. Red: selected, green: visible in selected RGB, blue: WOMD prediction target', 16, 22);
    }

    function handleTrajectoryCanvasClick(event) {
      if (!clickTargets.length) return;
      const rect = trajCanvas.getBoundingClientRect();
      const sx = trajCanvas.width / rect.width;
      const sy = trajCanvas.height / rect.height;
      const x = (event.clientX - rect.left) * sx;
      const y = (event.clientY - rect.top) * sy;
      let best = null;
      let bestDistance = Infinity;
      clickTargets.forEach((target) => {
        const distance = Math.hypot(target.x - x, target.y - y);
        if (distance < bestDistance && distance <= target.radius) {
          best = target;
          bestDistance = distance;
        }
      });
      if (!best) return;
      highlightedTrajectoryId = best.trajectory_id;
      const image = randomImageForTrajectoryAtCurrentTime(best.trajectory_id);
      if (image) {
        currentImage = image;
        loadCurrentImage();
      }
      renderLists();
      drawAll();
    }

    function randomImageForTrajectoryAtCurrentTime(trajectoryId) {
      const currentTime = Number(sceneTimeline[currentTimeIndex]);
      const candidates = currentScene.images.filter((img) => {
        if (!img.visible_trajectory_ids.includes(trajectoryId)) return false;
        return Math.abs(Number(img.timestamp) - currentTime) <= imageTimeTolerance(currentScene.images);
      });
      if (!candidates.length) return null;
      return candidates[Math.floor(Math.random() * candidates.length)];
    }

    function trajectoryIndexAtTime(traj, timeValue) {
      if (!traj.x.length) return -1;
      const times = traj.timestamps || [];
      if (!times.length) {
        const idx = Math.min(currentTimeIndex, traj.x.length - 1);
        return idx >= 0 ? idx : -1;
      }
      let bestIdx = -1;
      let bestDistance = Infinity;
      times.forEach((time, i) => {
        const distance = Math.abs(Number(time) - Number(timeValue));
        if (distance < bestDistance) {
          bestDistance = distance;
          bestIdx = i;
        }
      });
      if (bestIdx < 0) return -1;
      const sortedDeltas = times.slice(1).map((time, i) => Math.abs(Number(time) - Number(times[i]))).filter(d => d > 0).sort((a, b) => a - b);
      const tolerance = (sortedDeltas[0] || 1) * 0.55;
      return bestDistance <= tolerance ? bestIdx : -1;
    }

    function drawEgoFrame(w, h, scale) {
      const cx = w / 2;
      const cy = h / 2;
      trajCtx.strokeStyle = '#dce4ef';
      trajCtx.lineWidth = 1;
      trajCtx.beginPath();
      trajCtx.moveTo(cx, 0);
      trajCtx.lineTo(cx, h);
      trajCtx.moveTo(0, cy);
      trajCtx.lineTo(w, cy);
      trajCtx.stroke();
      trajCtx.strokeStyle = '#b8c5d6';
      [10, 20, 40, 80].forEach((meters) => {
        const r = meters * scale;
        if (r < Math.max(w, h)) {
          trajCtx.beginPath();
          trajCtx.arc(cx, cy, r, 0, Math.PI * 2);
          trajCtx.stroke();
        }
      });
      trajCtx.fillStyle = '#111827';
      trajCtx.beginPath();
      trajCtx.arc(cx, cy, 6, 0, Math.PI * 2);
      trajCtx.fill();
      trajCtx.fillStyle = '#111827';
      trajCtx.font = '13px sans-serif';
      trajCtx.fillText('ego', cx + 8, cy - 8);
      trajCtx.strokeStyle = '#111827';
      trajCtx.lineWidth = 2;
      trajCtx.beginPath();
      trajCtx.moveTo(cx, cy - 18);
      trajCtx.lineTo(cx - 6, cy - 6);
      trajCtx.moveTo(cx, cy - 18);
      trajCtx.lineTo(cx + 6, cy - 6);
      trajCtx.stroke();
    }

    function percentile(values, p) {
      const finite = values.filter(Number.isFinite).sort((a, b) => a - b);
      if (!finite.length) return 0;
      return finite[Math.min(finite.length - 1, Math.floor((finite.length - 1) * p))];
    }

    function drawCentered(ctx, w, h, text, color = '#dfe7f1') {
      ctx.fillStyle = color;
      ctx.font = '16px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(text, w / 2, h / 2);
      ctx.textAlign = 'left';
    }

    function palette(i) {
      const colors = ['#2f80ed', '#f2994a', '#27ae60', '#9b51e0', '#eb5757', '#00a7a7', '#8f6b32'];
      return colors[i % colors.length];
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[ch]));
    }

    init();
  </script>
</body>
</html>
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="dataset/rgb_trajectory_dataset",
        help="Converted dataset directory containing the parquet tables.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Viewer output directory. Default: <dataset-root>/viewer.",
    )
    parser.add_argument("--max-scenes", type=int, default=24)
    parser.add_argument("--max-images-per-scene", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
