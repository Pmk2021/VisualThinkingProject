#!/usr/bin/env python3
"""Stream Waymo shards through local conversion and copy them to Izar.

This script is designed for a machine with little free disk. It processes small
chunks:

  1. Discover component parquet files in GCS.
  2. Download a small group of matching segment files into a local staging dir.
  3. Convert that staged subset with build_waymo_rgb_trajectory_dataset.py.
  4. Copy the converted chunk to Izar with rsync over ssh.
  5. Mark the chunk done in a checkpoint and delete local staging/output.

It can resume after interruption because each chunk has a durable state file.

Example:
  python scripts/stream_waymo_to_izar.py \
    --remote izar:/scratch/$USER/waymo_rgb_trajectory \
    --target-gb 10 \
    --max-local-gb 3 \
    --splits training,validation \
    --chunk-workers 2
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_COMPONENTS = (
    "camera_image",
    "vehicle_pose",
    "projected_lidar_box",
    "lidar_box",
)


@dataclass(frozen=True)
class SegmentChunk:
    split: str
    segment: str
    files: dict[str, str]

    @property
    def chunk_id(self) -> str:
        return f"{self.split}__{self.segment}"


def main() -> int:
    args = parse_args()
    work_dir = Path(args.work_dir)
    checkpoint_dir = work_dir / "checkpoints"
    staging_root = work_dir / "staging"
    converted_root = work_dir / "converted"
    for path in (checkpoint_dir, staging_root, converted_root):
        path.mkdir(parents=True, exist_ok=True)

    remote_root = args.remote.rstrip("/")
    require_command(args.gcs_cli)
    if not args.dry_run:
        require_command("rsync")
        require_command("ssh")
        run(["ssh", remote_host(remote_root), f"mkdir -p {shell_quote(remote_path(remote_root))}"])

    components = tuple(c.strip() for c in args.components.split(",") if c.strip())
    splits = [normalize_split(s.strip()) for s in args.splits.split(",") if s.strip()]

    print("Discovering GCS component files...")
    chunks = discover_chunks(
        base_uri=args.base_uri.rstrip("/"),
        splits=splits,
        components=components,
        gcs_cli=args.gcs_cli,
        max_chunks=args.max_chunks,
    )
    print(f"Discovered {len(chunks)} complete segment chunks.")
    if args.dry_run:
        for chunk in chunks[: args.max_chunks or 20]:
            print(f"[dry-run] {chunk.chunk_id}")
            for component, uri in chunk.files.items():
                print(f"  {component}: {uri}")
        print("Dry run complete; no downloads, conversion, or remote copies performed.")
        return 0

    copied_bytes = remote_copied_bytes(checkpoint_dir)
    target_bytes = int(args.target_gb * (1024**3))
    print(f"Already copied according to checkpoint: {format_bytes(copied_bytes)}")
    print(f"Target copy volume: {format_bytes(target_bytes)}")

    pending = [
        chunk
        for chunk in chunks
        if read_state(checkpoint_dir / f"{chunk.chunk_id}.json").get("status") != "copied"
    ]

    copied_lock = threading.Lock()
    submitted_index = 0
    failures = 0

    def should_submit_more() -> bool:
        with copied_lock:
            return copied_bytes < target_bytes

    def worker(chunk: SegmentChunk) -> tuple[SegmentChunk, int]:
        state_path = checkpoint_dir / f"{chunk.chunk_id}.json"
        wait_for_space(Path(args.work_dir), args.min_free_gb)
        print(f"\n=== {chunk.chunk_id} ===")
        bytes_copied = process_chunk(
            chunk=chunk,
            args=args,
            components=components,
            staging_root=staging_root,
            converted_root=converted_root,
            state_path=state_path,
            remote_root=remote_root,
        )
        return chunk, bytes_copied

    max_workers = max(1, args.chunk_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        while submitted_index < len(pending) and len(futures) < max_workers and should_submit_more():
            chunk = pending[submitted_index]
            submitted_index += 1
            futures[executor.submit(worker, chunk)] = chunk

        while futures:
            for future in as_completed(list(futures)):
                chunk = futures.pop(future)
                try:
                    _, bytes_done = future.result()
                    with copied_lock:
                        copied_bytes += bytes_done
                        total_text = format_bytes(copied_bytes)
                    print(f"[copied] {chunk.chunk_id}; total copied: {total_text}")
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    failures += 1
                    write_state(
                        checkpoint_dir / f"{chunk.chunk_id}.json",
                        {
                            "chunk_id": chunk.chunk_id,
                            "status": "failed",
                            "error": repr(exc),
                            "updated_at": time.time(),
                        },
                    )
                    print(f"[failed] {chunk.chunk_id}: {exc}", file=sys.stderr)
                    if not args.keep_going:
                        return 1

                while submitted_index < len(pending) and len(futures) < max_workers and should_submit_more():
                    next_chunk = pending[submitted_index]
                    submitted_index += 1
                    futures[executor.submit(worker, next_chunk)] = next_chunk
                break

    if failures:
        print(f"Completed with {failures} failed chunks.", file=sys.stderr)
        return 1 if not args.keep_going else 0

    print("\nDone.")
    print(f"Checkpoint dir: {checkpoint_dir}")
    return 0


def process_chunk(
    chunk: SegmentChunk,
    args: argparse.Namespace,
    components: tuple[str, ...],
    staging_root: Path,
    converted_root: Path,
    state_path: Path,
    remote_root: str,
) -> int:
    chunk_stage = staging_root / chunk.chunk_id
    chunk_out = converted_root / chunk.chunk_id
    if chunk_stage.exists():
        shutil.rmtree(chunk_stage)
    if chunk_out.exists():
        shutil.rmtree(chunk_out)
    chunk_stage.mkdir(parents=True)
    chunk_out.mkdir(parents=True)

    write_state(state_path, {"chunk_id": chunk.chunk_id, "status": "downloading", "updated_at": time.time()})
    download_chunk(chunk, components, chunk_stage, args.gcs_cli, args.download_workers)
    enforce_local_budget(Path(args.work_dir), args.max_local_gb)

    write_state(state_path, {"chunk_id": chunk.chunk_id, "status": "converting", "updated_at": time.time()})
    converter = Path(args.converter)
    cmd = [
        sys.executable,
        str(converter),
        "--sensory-root",
        str(chunk_stage),
        "--output",
        str(chunk_out),
        "--splits",
        chunk.split,
        "--visualize",
        "0",
        "--no-web-viewer",
        "--motion-scenario-root",
        str(args.motion_scenario_root),
        "--prediction-target-splits",
        "",
        "--motion-max-records-per-split",
        "0",
    ]
    run(cmd)
    enforce_local_budget(Path(args.work_dir), args.max_local_gb)

    size = directory_size(chunk_out)
    wait_for_space(Path(args.work_dir), args.min_free_gb)

    write_state(
        state_path,
        {
            "chunk_id": chunk.chunk_id,
            "status": "copying",
            "converted_bytes": size,
            "updated_at": time.time(),
        },
    )
    remote_chunk = f"{remote_root}/{chunk.chunk_id}/"
    run(["rsync", "-a", "--partial", "--inplace", f"{chunk_out}/", remote_chunk])

    write_state(
        state_path,
        {
            "chunk_id": chunk.chunk_id,
            "status": "copied",
            "converted_bytes": size,
            "remote": remote_chunk,
            "updated_at": time.time(),
        },
    )

    if not args.keep_local:
        shutil.rmtree(chunk_stage, ignore_errors=True)
        shutil.rmtree(chunk_out, ignore_errors=True)
    return size


def discover_chunks(
    base_uri: str,
    splits: list[str],
    components: tuple[str, ...],
    gcs_cli: str,
    max_chunks: int,
) -> list[SegmentChunk]:
    chunks = []
    for split in splits:
        by_segment: dict[str, dict[str, str]] = {}
        for component in components:
            uri = f"{base_uri}/{split}/{component}/*.parquet"
            files = list_gcs(uri, gcs_cli)
            for file_uri in files:
                segment = Path(file_uri).name.removesuffix(".parquet")
                by_segment.setdefault(segment, {})[component] = file_uri
        for segment, files in sorted(by_segment.items()):
            if all(component in files for component in components):
                chunks.append(SegmentChunk(split=split, segment=segment, files=files))
                if max_chunks and len(chunks) >= max_chunks:
                    return chunks
    return chunks


def download_chunk(
    chunk: SegmentChunk,
    components: tuple[str, ...],
    chunk_stage: Path,
    gcs_cli: str,
    workers: int,
) -> None:
    tasks = []
    for component in components:
        source = chunk.files[component]
        dest_dir = chunk_stage / chunk.split / component
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / Path(source).name
        tasks.append((source, dest))

    def copy_one(task: tuple[str, Path]) -> None:
        source, dest = task
        if gcs_cli == "gcloud":
            run(["gcloud", "storage", "cp", source, str(dest)])
        else:
            run(["gsutil", "cp", source, str(dest)])

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(copy_one, task) for task in tasks]
        for future in as_completed(futures):
            future.result()


def list_gcs(uri: str, gcs_cli: str) -> list[str]:
    if gcs_cli == "gcloud":
        cmd = ["gcloud", "storage", "ls", uri]
    else:
        cmd = ["gsutil", "ls", uri]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip().endswith(".parquet")]


def remote_copied_bytes(checkpoint_dir: Path) -> int:
    total = 0
    for path in checkpoint_dir.glob("*.json"):
        state = read_state(path)
        if state.get("status") == "copied":
            total += int(state.get("converted_bytes", 0))
    return total


def wait_for_space(path: Path, min_free_gb: float) -> None:
    while True:
        free = shutil.disk_usage(path).free
        if free >= min_free_gb * (1024**3):
            return
        print(f"Waiting for local free space: have {format_bytes(free)}, need {min_free_gb} GB")
        time.sleep(30)


def enforce_local_budget(work_dir: Path, max_local_gb: float) -> None:
    if max_local_gb <= 0:
        return
    used = directory_size(work_dir)
    limit = int(max_local_gb * (1024**3))
    if used > limit:
        raise RuntimeError(
            f"Local work dir exceeded budget: {format_bytes(used)} > {max_local_gb} GiB. "
            "Use smaller chunks/components or increase --max-local-gb."
        )


def read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(path)


def directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def require_command(command: str) -> None:
    executable = command.split()[0]
    if shutil.which(executable) is None:
        raise SystemExit(f"Required command not found: {executable}")


def remote_host(remote: str) -> str:
    return remote.split(":", 1)[0]


def remote_path(remote: str) -> str:
    if ":" not in remote:
        raise ValueError("--remote must look like user@host:/path or host:/path")
    return remote.split(":", 1)[1]


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def normalize_split(split: str) -> str:
    aliases = {"train": "training", "val": "validation", "valid": "validation", "test": "testing"}
    return aliases.get(split, split)


def format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{value} B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-uri", default="gs://waymo_open_dataset_v_2_0_1")
    parser.add_argument("--splits", default="training,validation")
    parser.add_argument("--components", default=",".join(DEFAULT_COMPONENTS))
    parser.add_argument("--remote", required=True, help="Remote destination, e.g. izar:/scratch/$USER/waymo")
    parser.add_argument("--target-gb", type=float, default=10.0)
    parser.add_argument("--max-local-gb", type=float, default=3.0, help="Hard budget for local work-dir usage.")
    parser.add_argument("--min-free-gb", type=float, default=0.5)
    parser.add_argument("--work-dir", default="dataset/stream_to_izar_work")
    parser.add_argument("--converter", default="scripts/build_waymo_rgb_trajectory_dataset.py")
    parser.add_argument(
        "--motion-scenario-root",
        default="dataset/trajectory_dataset/uncompressed/scenario",
        help="Passed to converter; prediction target export is disabled for streaming chunks.",
    )
    parser.add_argument("--gcs-cli", choices=("gcloud", "gsutil"), default="gcloud")
    parser.add_argument("--chunk-workers", type=int, default=1, help="Number of chunks processed in parallel.")
    parser.add_argument("--download-workers", type=int, default=3)
    parser.add_argument("--max-chunks", type=int, default=0, help="0 means no discovery cap.")
    parser.add_argument("--keep-local", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
