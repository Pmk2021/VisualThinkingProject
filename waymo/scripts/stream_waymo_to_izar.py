#!/usr/bin/env python3
"""Orchestrate Waymo download and conversion directly on izar.

Runs on the LOCAL machine. All data I/O (download + conversion) happens on izar.
A gcloud OAuth2 bearer token is minted locally and passed to izar so it can wget
Waymo parquet files from GCS without needing a gcloud installation.

Flow per chunk:
  1. Discover GCS segment files locally (gcloud/gsutil ls).
  2. Mint a fresh OAuth2 bearer token locally (gcloud auth print-access-token).
  3. SSH to izar: wget each component parquet using the bearer token.
  4. SSH to izar: run the converter on the downloaded files.
  5. Mark the chunk done in a local checkpoint file and clean up izar staging.

Token refresh: tokens last ~60 minutes; the local cache refreshes every 45 minutes.
Resume: any chunk recorded as "done" in the local checkpoint dir is skipped on restart.
Early stopping: processing halts once accumulated converted output reaches --target-gb.

Example:
  python waymo/scripts/stream_waymo_to_izar.py \\
    --izar user@izar.epfl.ch \\
    --izar-work-dir /scratch/$USER/waymo_work \\
    --izar-output-dir /scratch/$USER/waymo_rgb_trajectory \\
    --izar-converter ~/VisualThinkingProject/waymo/scripts/build_waymo_rgb_trajectory_dataset.py \\
    --target-gb 100 \\
    --max-buffer-gb 5 \\
    --splits training,validation
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

GCS_HTTPS_BASE = "https://storage.googleapis.com"
TOKEN_LIFETIME = 45 * 60  # seconds; gcloud tokens last ~60 min, refresh at 45

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
    files: dict[str, str]  # component -> gs:// URI

    @property
    def chunk_id(self) -> str:
        return f"{self.split}__{self.segment}"


class TokenCache:
    """Thread-safe gcloud bearer token with automatic refresh before expiry."""

    def __init__(self) -> None:
        self._token = ""
        self._fetched_at = 0.0
        self._lock = threading.Lock()

    def get(self) -> str:
        with self._lock:
            if time.monotonic() - self._fetched_at >= TOKEN_LIFETIME:
                print("[auth] Refreshing gcloud access token...")
                self._token = _fetch_access_token()
                self._fetched_at = time.monotonic()
            return self._token


def _fetch_access_token() -> str:
    result = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout.strip()


def gcs_uri_to_https(uri: str) -> str:
    """Convert gs://bucket/path to https://storage.googleapis.com/bucket/path."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    return GCS_HTTPS_BASE + "/" + uri[5:]


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    local_checkpoint_dir = Path(args.work_dir) / "checkpoints"
    local_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    izar = args.izar
    izar_staging = args.izar_work_dir.rstrip("/") + "/staging"
    izar_output = args.izar_output_dir.rstrip("/")

    require_command("gcloud")
    require_command("ssh")

    if not args.dry_run:
        ssh(izar, f"mkdir -p {sq(izar_staging)} {sq(izar_output)}")

    components = tuple(c.strip() for c in args.components.split(",") if c.strip())
    splits = [normalize_split(s) for s in args.splits.split(",") if s.strip()]

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
            for comp, uri in chunk.files.items():
                print(f"  {comp}: {gcs_uri_to_https(uri)}")
        return 0

    token_cache = TokenCache()

    converted_bytes = _checkpoint_total(local_checkpoint_dir)
    target_bytes = int(args.target_gb * 1024**3)
    print(f"Already done (checkpointed): {format_bytes(converted_bytes)}")
    print(f"Target converted volume:     {format_bytes(target_bytes)}")

    pending = [
        chunk
        for chunk in chunks
        if read_state(local_checkpoint_dir / f"{chunk.chunk_id}.json").get("status") != "done"
    ]
    print(f"Pending chunks: {len(pending)} / {len(chunks)}")

    converted_lock = threading.Lock()
    submitted_index = 0
    failures = 0

    def should_submit_more() -> bool:
        with converted_lock:
            return converted_bytes < target_bytes

    def worker(chunk: SegmentChunk) -> tuple[SegmentChunk, int]:
        state_path = local_checkpoint_dir / f"{chunk.chunk_id}.json"
        bytes_done = process_chunk(
            chunk, args, components, izar, izar_staging, izar_output, state_path, token_cache
        )
        return chunk, bytes_done

    max_workers = max(1, args.chunk_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: dict = {}

        def _submit_next() -> None:
            nonlocal submitted_index
            while (
                submitted_index < len(pending)
                and len(futures) < max_workers
                and should_submit_more()
            ):
                c = pending[submitted_index]
                submitted_index += 1
                futures[executor.submit(worker, c)] = c

        _submit_next()

        while futures:
            for future in as_completed(list(futures)):
                chunk = futures.pop(future)
                try:
                    _, bytes_done = future.result()
                    with converted_lock:
                        converted_bytes += bytes_done
                    print(f"[done] {chunk.chunk_id}; total: {format_bytes(converted_bytes)}")
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    failures += 1
                    write_state(
                        local_checkpoint_dir / f"{chunk.chunk_id}.json",
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
                _submit_next()
                break

    if failures:
        print(f"\nCompleted with {failures} failed chunks.", file=sys.stderr)
    else:
        print("\nDone.")
    print(f"Local checkpoint dir: {local_checkpoint_dir}")
    return 1 if failures and not args.keep_going else 0


# ── chunk processing ──────────────────────────────────────────────────────────

def process_chunk(
    chunk: SegmentChunk,
    args: argparse.Namespace,
    components: tuple[str, ...],
    izar: str,
    izar_staging: str,
    izar_output: str,
    state_path: Path,
    token_cache: TokenCache,
) -> int:
    chunk_stage = f"{izar_staging}/{chunk.chunk_id}"
    chunk_out = f"{izar_output}/{chunk.chunk_id}"

    print(f"\n=== {chunk.chunk_id} ===")

    # Clean up any leftover staging from a previous failed attempt.
    ssh(izar, f"rm -rf {sq(chunk_stage)} && mkdir -p {sq(chunk_stage)}")

    # Hold off if izar is running low on free space.
    wait_for_izar_space(izar, args.izar_work_dir, args.min_izar_free_gb)

    write_state(
        state_path, {"chunk_id": chunk.chunk_id, "status": "downloading", "updated_at": time.time()}
    )
    token = token_cache.get()
    download_chunk_on_izar(chunk, components, chunk_stage, izar, token)

    write_state(
        state_path, {"chunk_id": chunk.chunk_id, "status": "converting", "updated_at": time.time()}
    )
    # Re-check the token: downloads can be slow and the token might have aged.
    token = token_cache.get()  # noqa: F841  (unused after this; kept to trigger refresh)
    convert_on_izar(chunk, chunk_stage, chunk_out, izar, args)

    size = izar_dir_size(izar, chunk_out)
    write_state(
        state_path,
        {
            "chunk_id": chunk.chunk_id,
            "status": "done",
            "converted_bytes": size,
            "izar_path": chunk_out,
            "updated_at": time.time(),
        },
    )

    if not args.keep_staging:
        ssh(izar, f"rm -rf {sq(chunk_stage)}")

    return size


def download_chunk_on_izar(
    chunk: SegmentChunk,
    components: tuple[str, ...],
    chunk_stage: str,
    izar: str,
    token: str,
) -> None:
    """Stream a bash download script to izar via SSH stdin.

    Using stdin avoids embedding the token in the process argument list visible
    to ps(1) on the local machine. The remote shell receives it as script text.
    """
    # Launch all wget calls in parallel using bash background jobs, then wait for all.
    lines = ["set -euo pipefail", "pids=()"]
    for component in components:
        uri = chunk.files[component]
        url = gcs_uri_to_https(uri)
        dest_dir = f"{chunk_stage}/{chunk.split}/{component}"
        dest = f"{dest_dir}/{PurePosixPath(uri).name}"
        # GCS tokens are URL-safe base64 (A-Za-z0-9._-) — safe inside double quotes.
        lines += [
            f"mkdir -p {sq(dest_dir)}",
            f'wget -q --header="Authorization: Bearer {token}" {sq(url)} -O {sq(dest)} & pids+=($!)',
        ]
    lines += [
        'for pid in "${pids[@]}"; do',
        '  wait "$pid" || { echo "wget failed (pid $pid)" >&2; exit 1; }',
        "done",
    ]
    script = "\n".join(lines) + "\n"
    print(f"+[izar] downloading {len(components)} component(s) in parallel for {chunk.chunk_id}")
    _run_script_on_izar(izar, script)


def convert_on_izar(
    chunk: SegmentChunk,
    chunk_stage: str,
    chunk_out: str,
    izar: str,
    args: argparse.Namespace,
) -> None:
    parts = [
        args.izar_python,
        sq(args.izar_converter),
        "--sensory-root", sq(chunk_stage),
        "--output", sq(chunk_out),
        "--splits", chunk.split,
        "--workers", str(args.izar_workers),
        "--visualize", "0",
        "--no-web-viewer",
        "--prediction-target-splits", "''",
        "--motion-max-records-per-split", "0",
    ]
    if args.izar_motion_scenario_root:
        parts += ["--motion-scenario-root", sq(args.izar_motion_scenario_root)]
    cmd = " ".join(parts)
    print(f"+[izar] convert {chunk.chunk_id}")
    ssh(izar, cmd)


# ── remote helpers ────────────────────────────────────────────────────────────

def _run_script_on_izar(izar: str, script: str) -> None:
    subprocess.run(["ssh", izar, "bash -s"], input=script.encode(), check=True)


def ssh(izar: str, cmd: str) -> None:
    print(f"+ssh {izar} {cmd[:120]}")
    subprocess.run(["ssh", izar, cmd], check=True)


def izar_dir_size(izar: str, path: str) -> int:
    result = subprocess.run(
        ["ssh", izar, f"du -sb {sq(path)} 2>/dev/null | awk '{{print $1}}'"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


def wait_for_izar_space(izar: str, path: str, min_free_gb: float) -> None:
    needed = int(min_free_gb * 1024**3)
    while True:
        result = subprocess.run(
            ["ssh", izar, f"df -B1 {sq(path)} | tail -1 | awk '{{print $4}}'"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            free = int(result.stdout.strip())
        except ValueError:
            free = 0
        if free >= needed:
            return
        print(f"Waiting for izar free space: have {format_bytes(free)}, need {min_free_gb} GiB")
        time.sleep(30)


# ── GCS discovery ─────────────────────────────────────────────────────────────

def discover_chunks(
    base_uri: str,
    splits: list[str],
    components: tuple[str, ...],
    gcs_cli: str,
    max_chunks: int,
) -> list[SegmentChunk]:
    chunks: list[SegmentChunk] = []
    for split in splits:
        by_segment: dict[str, dict[str, str]] = {}
        for component in components:
            uri = f"{base_uri}/{split}/{component}/*.parquet"
            for file_uri in list_gcs(uri, gcs_cli):
                segment = PurePosixPath(file_uri).stem
                by_segment.setdefault(segment, {})[component] = file_uri
        for segment, files in sorted(by_segment.items()):
            if all(c in files for c in components):
                chunks.append(SegmentChunk(split=split, segment=segment, files=files))
                if max_chunks and len(chunks) >= max_chunks:
                    return chunks
    return chunks


def list_gcs(uri: str, gcs_cli: str) -> list[str]:
    cmd = (
        ["gcloud", "storage", "ls", uri]
        if gcs_cli == "gcloud"
        else ["gsutil", "ls", uri]
    )
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return []
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip().endswith(".parquet")
    ]


# ── checkpoints ───────────────────────────────────────────────────────────────

def _checkpoint_total(checkpoint_dir: Path) -> int:
    total = 0
    for path in checkpoint_dir.glob("*.json"):
        state = read_state(path)
        if state.get("status") == "done":
            total += int(state.get("converted_bytes", 0))
    return total


def read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(path)


# ── misc ──────────────────────────────────────────────────────────────────────

def normalize_split(split: str) -> str:
    aliases = {
        "train": "training",
        "val": "validation",
        "valid": "validation",
        "test": "testing",
    }
    return aliases.get(split.strip(), split.strip())


def format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{value} B"


def require_command(cmd: str) -> None:
    import shutil

    if shutil.which(cmd.split()[0]) is None:
        raise SystemExit(f"Required command not found: {cmd}")


def sq(value: str) -> str:
    """Single-quote a value for safe use in a remote shell command."""
    return "'" + value.replace("'", "'\\''") + "'"


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # GCS source
    p.add_argument("--base-uri", default="gs://waymo_open_dataset_v_2_0_1")
    p.add_argument("--splits", default="training,validation")
    p.add_argument("--components", default=",".join(DEFAULT_COMPONENTS))
    p.add_argument(
        "--gcs-cli",
        choices=("gcloud", "gsutil"),
        default="gcloud",
        help="CLI used for GCS file listing on the local machine. "
        "Downloads use wget on izar and do not require gcloud there.",
    )

    # Izar target
    p.add_argument("--izar", required=True, help="SSH target, e.g. user@izar.epfl.ch.")
    p.add_argument(
        "--izar-work-dir",
        required=True,
        help="Work directory on izar. Staging subdirs are created here.",
    )
    p.add_argument(
        "--izar-output-dir",
        required=True,
        help="Output directory on izar where converted chunks are written.",
    )
    p.add_argument(
        "--izar-converter",
        required=True,
        help="Absolute path to build_waymo_rgb_trajectory_dataset.py on izar.",
    )
    p.add_argument(
        "--izar-python",
        default="python3",
        help="Python interpreter on izar (use absolute path for a venv).",
    )
    p.add_argument(
        "--izar-motion-scenario-root",
        default="",
        help="Passed to the converter on izar for prediction targets. "
        "Leave blank to skip prediction target export.",
    )

    # Budgets / stopping criteria
    p.add_argument(
        "--target-gb",
        type=float,
        default=100.0,
        help="Stop submitting new chunks once this many GiB of converted output "
        "have been recorded in the local checkpoint (cumulative across runs).",
    )
    p.add_argument(
        "--max-buffer-gb",
        type=float,
        default=5.0,
        help="Expected max staging footprint per chunk on izar (informational). "
        "Set --chunk-workers so that chunk_workers * max-buffer-gb fits izar's disk.",
    )
    p.add_argument(
        "--min-izar-free-gb",
        type=float,
        default=2.0,
        help="Pause before starting a new chunk if izar has less than this many "
        "GiB free in --izar-work-dir.",
    )

    # Orchestration
    p.add_argument(
        "--work-dir",
        default="dataset/izar_stream_work",
        help="Local directory for checkpoint JSON files.",
    )
    p.add_argument(
        "--chunk-workers",
        type=int,
        default=1,
        help="Number of chunks processed in parallel. Each holds one SSH session "
        "and up to --max-buffer-gb of staging data on izar.",
    )
    p.add_argument(
        "--izar-workers",
        type=int,
        default=4,
        help="Thread-pool size passed to the converter on izar (--workers). "
        "Controls parallel component loading and JPEG decoding.",
    )
    p.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="Cap the number of chunks discovered. 0 = no cap.",
    )
    p.add_argument(
        "--keep-staging",
        action="store_true",
        help="Do not delete izar staging dirs after successful conversion.",
    )
    p.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue processing after a failed chunk instead of aborting.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover chunks and print what would be done; make no SSH calls.",
    )
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
