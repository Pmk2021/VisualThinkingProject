#!/usr/bin/env python3
"""Orchestrate Waymo download and conversion directly on izar.

Runs on the LOCAL machine. All data I/O (download + conversion) happens on izar
via persistent SSH bash sessions — conda init in .bashrc runs once per session,
not once per command.

Pipeline: download workers and convert workers run concurrently so conversion
starts on each chunk as soon as it finishes downloading.

         pending chunks
              │
    ┌─────────┴──────────┐
    │   download workers  │  --download-workers sessions, parallel wget per chunk
    └─────────┬──────────┘
              │  staging queue  (at most --pipeline-buffer chunks in staging at once)
    ┌─────────┴──────────┐
    │   convert workers   │  --convert-workers sessions, parallel component loading
    └─────────┬──────────┘
              │
          checkpoint + staging cleanup

Token refresh: tokens last ~60 min; the local cache refreshes every 45 min.
Resume: any chunk recorded as "done" in the local checkpoint dir is skipped.

Example:
  python waymo/scripts/stream_waymo_to_izar.py \\
    --izar user@izar.epfl.ch \\
    --izar-work-dir /scratch/$USER/waymo_work \\
    --izar-output-dir /scratch/$USER/waymo_rgb_trajectory \\
    --izar-converter ~/VisualThinkingProject/waymo/scripts/build_waymo_rgb_trajectory_dataset.py \\
    --izar-python /home/$USER/miniconda3/envs/vtp/bin/python \\
    --target-gb 100 \\
    --download-workers 3 \\
    --convert-workers 2
"""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
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

_STOP = object()  # sentinel value pushed into the pipeline queue to stop convert workers


# ── token cache ───────────────────────────────────────────────────────────────

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
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    return result.stdout.strip()


# ── chunk descriptor ──────────────────────────────────────────────────────────

class SegmentChunk:
    def __init__(self, split: str, segment: str, files: dict[str, str]) -> None:
        self.split = split
        self.segment = segment
        self.files = files  # component → gs:// URI

    @property
    def chunk_id(self) -> str:
        return f"{self.split}__{self.segment}"


# ── persistent SSH session ────────────────────────────────────────────────────

class PersistentShell:
    """One long-lived bash session on a remote host via SSH stdin/stdout.

    bash -l sources .bash_profile → .bashrc so conda init runs once at open,
    not once per command. Stderr is drained by a background thread.
    """

    def __init__(self, host: str, tag: str = "") -> None:
        self._host = host
        self._tag = tag or host
        self._seq = 0
        self._lock = threading.Lock()
        self._proc = subprocess.Popen(
            ["ssh", host, "bash", "-l"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        # Explicitly source .bashrc in case .bash_profile skips it for
        # non-interactive shells (common pattern: [ -z "$PS1" ] && return).
        marker = self._next_marker()
        self._write(f"source ~/.bashrc 2>/dev/null || true\necho '{marker}0'\n")
        self._drain_until(marker)

    # ── internal ─────────────────────────────────────────────────────────────

    def _next_marker(self) -> str:
        m = f"__WM{self._seq}__"
        self._seq += 1
        return m

    def _write(self, text: str) -> None:
        assert self._proc.stdin
        self._proc.stdin.write(text.encode())
        self._proc.stdin.flush()

    def _drain_until(self, marker: str) -> int:
        """Read stdout lines until one starts with marker. Return the exit code."""
        assert self._proc.stdout
        while True:
            raw = self._proc.stdout.readline()
            if not raw:
                raise RuntimeError(f"[{self._tag}] SSH session closed unexpectedly")
            line = raw.decode(errors="replace").rstrip("\n")
            if line.startswith(marker):
                return int(line[len(marker):] or "0")
            if line:
                print(f"  [{self._tag}] {line}")

    def _drain_stderr(self) -> None:
        assert self._proc.stderr
        for raw in self._proc.stderr:
            line = raw.decode(errors="replace").rstrip()
            if line:
                print(f"  [{self._tag}|err] {line}", file=sys.stderr)

    # ── public API ───────────────────────────────────────────────────────────

    def run(self, cmd: str) -> None:
        """Run a shell command; raise CalledProcessError on non-zero exit."""
        with self._lock:
            marker = self._next_marker()
            self._write(f"( {cmd} )\n_rc=$?\necho '{marker}'\"$_rc\"\n")
            rc = self._drain_until(marker)
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)

    def run_script(self, script: str) -> None:
        """Send a multi-line bash script via heredoc; raise on non-zero exit."""
        with self._lock:
            marker = self._next_marker()
            tag = f"_WS{self._seq}_"
            self._write(
                f"bash <<'{tag}'\n{script}\n{tag}\n"
                f"_rc=$?\necho '{marker}'\"$_rc\"\n"
            )
            rc = self._drain_until(marker)
        if rc != 0:
            raise subprocess.CalledProcessError(rc, "script")

    def capture(self, cmd: str) -> str:
        """Run a command and return its stdout as a string."""
        with self._lock:
            marker = self._next_marker()
            end = f"__WME{self._seq}__"
            # Capture into a bash variable so the markers stay on their own lines.
            self._write(
                f"_cap=$( {cmd} 2>/dev/null )\n"
                f"_rc=$?\n"
                f"echo '{marker}'\"$_rc\"\n"
                f"printf '%s\\n' \"$_cap\"\n"
                f"echo '{end}'\n"
            )
            assert self._proc.stdout
            rc = 0
            lines: list[str] = []
            got_marker = False
            while True:
                raw = self._proc.stdout.readline()
                if not raw:
                    raise RuntimeError(f"[{self._tag}] SSH session closed")
                line = raw.decode(errors="replace").rstrip("\n")
                if not got_marker:
                    if line.startswith(marker):
                        rc = int(line[len(marker):] or "0")
                        got_marker = True
                else:
                    if line == end:
                        break
                    lines.append(line)
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)
        return "\n".join(lines)

    def close(self) -> None:
        try:
            assert self._proc.stdin
            self._proc.stdin.write(b"exit\n")
            self._proc.stdin.flush()
            self._proc.wait(timeout=10)
        except Exception:
            self._proc.kill()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    izar = args.izar
    izar_staging = args.izar_work_dir.rstrip("/") + "/staging"
    izar_output = args.izar_output_dir.rstrip("/")
    local_checkpoint_dir = Path(args.work_dir) / "checkpoints"
    local_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    require_command("gcloud")
    require_command("ssh")

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

    pending = [
        c for c in chunks
        if read_state(local_checkpoint_dir / f"{c.chunk_id}.json").get("status") != "done"
    ]
    converted_bytes = _checkpoint_total(local_checkpoint_dir)
    target_bytes = int(args.target_gb * 1024**3)
    print(f"Already done (checkpointed): {format_bytes(converted_bytes)}")
    print(f"Target:                      {format_bytes(target_bytes)}")
    print(f"Pending chunks:              {len(pending)} / {len(chunks)}")

    if not pending or converted_bytes >= target_bytes:
        print("Nothing to do.")
        return 0

    n_dl = args.download_workers
    n_cv = args.convert_workers
    print(f"Opening {n_dl} download + {n_cv} convert SSH sessions on {izar}…")
    print("(Each session sources .bashrc once — conda init will appear below.)")

    dl_shells = [PersistentShell(izar, f"dl{i}") for i in range(n_dl)]
    cv_shells = [PersistentShell(izar, f"cv{i}") for i in range(n_cv)]

    # Ensure remote directories exist using the first download shell.
    dl_shells[0].run(f"mkdir -p {sq(izar_staging)} {sq(izar_output)}")

    try:
        return _pipeline(
            pending=pending,
            dl_shells=dl_shells,
            cv_shells=cv_shells,
            components=components,
            args=args,
            checkpoint_dir=local_checkpoint_dir,
            izar_staging=izar_staging,
            izar_output=izar_output,
            converted_bytes=converted_bytes,
            target_bytes=target_bytes,
        )
    finally:
        print("Closing SSH sessions…")
        for shell in dl_shells + cv_shells:
            shell.close()


# ── pipeline ──────────────────────────────────────────────────────────────────

def _pipeline(
    pending: list[SegmentChunk],
    dl_shells: list[PersistentShell],
    cv_shells: list[PersistentShell],
    components: tuple[str, ...],
    args: argparse.Namespace,
    checkpoint_dir: Path,
    izar_staging: str,
    izar_output: str,
    converted_bytes: int,
    target_bytes: int,
) -> int:
    token_cache = TokenCache()

    # Queue connecting download → convert workers.
    # maxsize limits how many chunks are staged on izar at once.
    pipeline_buffer = args.pipeline_buffer if args.pipeline_buffer > 0 else len(dl_shells)
    ready_q: queue.Queue = queue.Queue(maxsize=pipeline_buffer)

    # Shared mutable state (use lists so inner functions can rebind).
    converted = [converted_bytes]
    converted_lock = threading.Lock()
    pending_iter = iter(pending)
    pending_lock = threading.Lock()
    failures: list[tuple[SegmentChunk, Exception]] = []
    failures_lock = threading.Lock()
    stop_flag = threading.Event()

    def next_chunk() -> SegmentChunk | None:
        with pending_lock:
            if stop_flag.is_set():
                return None
            with converted_lock:
                if converted[0] >= target_bytes:
                    return None
            try:
                return next(pending_iter)
            except StopIteration:
                return None

    def download_worker(shell: PersistentShell) -> None:
        while True:
            chunk = next_chunk()
            if chunk is None:
                return
            state_path = checkpoint_dir / f"{chunk.chunk_id}.json"
            chunk_stage = f"{izar_staging}/{chunk.chunk_id}"
            try:
                print(f"\n[{shell._tag}] downloading {chunk.chunk_id}")
                write_state(state_path, {
                    "chunk_id": chunk.chunk_id, "status": "downloading",
                    "updated_at": time.time(),
                })
                shell.run(f"rm -rf {sq(chunk_stage)} && mkdir -p {sq(chunk_stage)}")
                _wait_for_space(shell, args.izar_work_dir, args.min_izar_free_gb)
                download_chunk_on_izar(chunk, components, chunk_stage, shell, token_cache.get())
                write_state(state_path, {
                    "chunk_id": chunk.chunk_id, "status": "downloaded",
                    "updated_at": time.time(),
                })
                ready_q.put(chunk)  # blocks if pipeline_buffer chunks are already staged
            except Exception as exc:
                with failures_lock:
                    failures.append((chunk, exc))
                write_state(state_path, {
                    "chunk_id": chunk.chunk_id, "status": "failed",
                    "error": repr(exc), "updated_at": time.time(),
                })
                print(f"[dl-failed] {chunk.chunk_id}: {exc}", file=sys.stderr)
                if not args.keep_going:
                    stop_flag.set()
                    return

    def convert_worker(shell: PersistentShell) -> None:
        while True:
            item = ready_q.get()
            if item is _STOP:
                return
            chunk: SegmentChunk = item
            state_path = checkpoint_dir / f"{chunk.chunk_id}.json"
            chunk_stage = f"{izar_staging}/{chunk.chunk_id}"
            chunk_out = f"{izar_output}/{chunk.chunk_id}"
            try:
                print(f"\n[{shell._tag}] converting {chunk.chunk_id}")
                write_state(state_path, {
                    "chunk_id": chunk.chunk_id, "status": "converting",
                    "updated_at": time.time(),
                })
                convert_on_izar(chunk, chunk_stage, chunk_out, shell, args)
                size = _dir_size(shell, chunk_out)
                write_state(state_path, {
                    "chunk_id": chunk.chunk_id, "status": "done",
                    "converted_bytes": size, "izar_path": chunk_out,
                    "updated_at": time.time(),
                })
                with converted_lock:
                    converted[0] += size
                print(f"[done] {chunk.chunk_id}; total: {format_bytes(converted[0])}")
                if not args.keep_staging:
                    shell.run(f"rm -rf {sq(chunk_stage)}")
            except Exception as exc:
                with failures_lock:
                    failures.append((chunk, exc))
                write_state(state_path, {
                    "chunk_id": chunk.chunk_id, "status": "failed",
                    "error": repr(exc), "updated_at": time.time(),
                })
                print(f"[cv-failed] {chunk.chunk_id}: {exc}", file=sys.stderr)

    dl_threads = [
        threading.Thread(target=download_worker, args=(s,), name=f"dl-{i}", daemon=True)
        for i, s in enumerate(dl_shells)
    ]
    cv_threads = [
        threading.Thread(target=convert_worker, args=(s,), name=f"cv-{i}", daemon=True)
        for i, s in enumerate(cv_shells)
    ]

    for t in cv_threads + dl_threads:
        t.start()

    for t in dl_threads:
        t.join()

    # All downloads done (or stopped); signal each convert worker to exit.
    for _ in cv_threads:
        ready_q.put(_STOP)

    for t in cv_threads:
        t.join()

    n_fail = len(failures)
    if n_fail:
        print(f"\nCompleted with {n_fail} failure(s).", file=sys.stderr)
    else:
        print("\nDone.")
    print(f"Local checkpoint dir: {checkpoint_dir}")
    return 1 if failures and not args.keep_going else 0


# ── chunk operations ──────────────────────────────────────────────────────────

def download_chunk_on_izar(
    chunk: SegmentChunk,
    components: tuple[str, ...],
    chunk_stage: str,
    shell: PersistentShell,
    token: str,
) -> None:
    """Download all component files in parallel using bash background jobs."""
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
    shell.run_script("\n".join(lines))


def convert_on_izar(
    chunk: SegmentChunk,
    chunk_stage: str,
    chunk_out: str,
    shell: PersistentShell,
    args: argparse.Namespace,
) -> None:
    parts = [
        "PYTHONUNBUFFERED=1",
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
    shell.run(" ".join(parts))


# ── remote helpers ────────────────────────────────────────────────────────────

def _dir_size(shell: PersistentShell, path: str) -> int:
    try:
        output = shell.capture(f"du -sb {sq(path)}")
        return int(output.split()[0])
    except (ValueError, IndexError, subprocess.CalledProcessError):
        return 0


def _wait_for_space(shell: PersistentShell, path: str, min_free_gb: float) -> None:
    needed = int(min_free_gb * 1024**3)
    while True:
        try:
            output = shell.capture(f"df -B1 {sq(path)} | tail -1 | awk '{{print $4}}'")
            free = int(output.strip())
        except (ValueError, subprocess.CalledProcessError):
            free = 0
        if free >= needed:
            return
        print(f"Waiting for izar free space: {format_bytes(free)} free, need {min_free_gb} GiB")
        time.sleep(30)


# ── GCS discovery ─────────────────────────────────────────────────────────────

def gcs_uri_to_https(uri: str) -> str:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    return GCS_HTTPS_BASE + "/" + uri[5:]


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


# ── utilities ─────────────────────────────────────────────────────────────────

def normalize_split(split: str) -> str:
    aliases = {
        "train": "training", "val": "validation",
        "valid": "validation", "test": "testing",
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
        "--gcs-cli", choices=("gcloud", "gsutil"), default="gcloud",
        help="CLI used for GCS file listing on the local machine.",
    )

    # Izar target
    p.add_argument("--izar", required=True, help="SSH target, e.g. user@izar.epfl.ch.")
    p.add_argument("--izar-work-dir", required=True,
                   help="Work directory on izar; staging subdirs are created here.")
    p.add_argument("--izar-output-dir", required=True,
                   help="Output directory on izar where converted chunks are written.")
    p.add_argument("--izar-converter", required=True,
                   help="Absolute path to build_waymo_rgb_trajectory_dataset.py on izar.")
    p.add_argument("--izar-python", default="python3",
                   help="Python interpreter on izar (use absolute path for a venv).")
    p.add_argument("--izar-motion-scenario-root", default="",
                   help="Passed to the converter for prediction targets. Leave blank to skip.")
    p.add_argument("--izar-workers", type=int, default=4,
                   help="Thread-pool size passed to the converter (component loading + JPEG decode).")

    # Budgets
    p.add_argument("--target-gb", type=float, default=100.0,
                   help="Stop once this many GiB of converted output are checkpointed.")
    p.add_argument("--max-buffer-gb", type=float, default=5.0,
                   help="Expected max staging size per chunk (informational). "
                        "Set pipeline-buffer × max-buffer-gb ≤ izar free space.")
    p.add_argument("--min-izar-free-gb", type=float, default=2.0,
                   help="Pause before downloading a chunk if izar free space drops below this.")

    # Pipeline / parallelism
    p.add_argument("--download-workers", type=int, default=2,
                   help="Number of parallel download sessions. Each opens one SSH bash session.")
    p.add_argument("--convert-workers", type=int, default=1,
                   help="Number of parallel convert sessions. Each opens one SSH bash session.")
    p.add_argument("--pipeline-buffer", type=int, default=0,
                   help="Max chunks staged on izar waiting for conversion. "
                        "0 = auto (equals --download-workers).")

    # Orchestration
    p.add_argument("--work-dir", default="dataset/izar_stream_work",
                   help="Local directory for checkpoint JSON files.")
    p.add_argument("--max-chunks", type=int, default=0,
                   help="Cap on discovered chunks. 0 = no cap.")
    p.add_argument("--keep-staging", action="store_true",
                   help="Do not delete izar staging dirs after successful conversion.")
    p.add_argument("--keep-going", action="store_true",
                   help="Continue after failed chunks instead of aborting.")
    p.add_argument("--dry-run", action="store_true",
                   help="Discover chunks and print what would be done; open no SSH sessions.")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
