#!/usr/bin/env bash
set -euo pipefail

# Serve the Waymo HTML viewer either from this machine or from a remote host
# such as Izar through an SSH tunnel.
#
# Local:
#   scripts/serve_waymo_viewer.sh --local dataset/rgb_trajectory_dataset/viewer
#
# Remote:
#   scripts/serve_waymo_viewer.sh \
#     --remote izar:/scratch/$USER/waymo_rgb_trajectory/some_chunk/viewer
#
# Then open:
#   http://127.0.0.1:8000/

LOCAL_DIR="dataset/rgb_trajectory_dataset/viewer"
REMOTE_SPEC=""
LOCAL_PORT="8000"
REMOTE_PORT="18000"
PYTHON_BIN="python3"

usage() {
  cat <<'USAGE'
Serve the Waymo HTML viewer either from this machine or from a remote host
such as Izar through an SSH tunnel.

Local:
  scripts/serve_waymo_viewer.sh --local dataset/rgb_trajectory_dataset/viewer

Remote:
  scripts/serve_waymo_viewer.sh \
    --remote izar:/scratch/$USER/waymo_rgb_trajectory/some_chunk/viewer

Then open:
  http://127.0.0.1:8000/

Options:
  --local DIR          Serve a local viewer directory. Default: dataset/rgb_trajectory_dataset/viewer
  --remote HOST:DIR    Serve viewer from a remote host through SSH, e.g. izar:/scratch/me/viewer
  --local-port PORT    Browser port on this machine. Default: 8000
  --remote-port PORT   HTTP server port on remote host. Default: 18000
  --python-bin BIN     Python executable to run http.server. Default: python3
  -h, --help           Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local)
      LOCAL_DIR="${2:?missing value for --local}"
      REMOTE_SPEC=""
      shift 2
      ;;
    --remote)
      REMOTE_SPEC="${2:?missing value for --remote}"
      shift 2
      ;;
    --local-port)
      LOCAL_PORT="${2:?missing value for --local-port}"
      shift 2
      ;;
    --remote-port)
      REMOTE_PORT="${2:?missing value for --remote-port}"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="${2:?missing value for --python-bin}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "$REMOTE_SPEC" ]]; then
  if [[ "$REMOTE_SPEC" != *:* ]]; then
    echo "--remote must look like HOST:DIR, e.g. izar:/scratch/me/viewer" >&2
    exit 2
  fi

  REMOTE_HOST="${REMOTE_SPEC%%:*}"
  REMOTE_DIR="${REMOTE_SPEC#*:}"

  echo "Serving remote viewer:"
  echo "  remote: ${REMOTE_HOST}:${REMOTE_DIR}"
  echo "  local:  http://127.0.0.1:${LOCAL_PORT}/"
  echo
  echo "Press Ctrl-C to stop the tunnel and remote server."

  # One SSH session both starts the remote server and forwards local traffic.
  # The server binds to remote localhost only, so it is not exposed on Izar.
  exec ssh \
    -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
    "$REMOTE_HOST" \
    "cd $(printf '%q' "$REMOTE_DIR") && exec ${PYTHON_BIN} -m http.server ${REMOTE_PORT} --bind 127.0.0.1"
fi

if [[ ! -d "$LOCAL_DIR" ]]; then
  echo "Local viewer directory not found: $LOCAL_DIR" >&2
  exit 1
fi

echo "Serving local viewer:"
echo "  directory: $LOCAL_DIR"
echo "  url:       http://127.0.0.1:${LOCAL_PORT}/"
echo
echo "Press Ctrl-C to stop the server."

cd "$LOCAL_DIR"
exec "$PYTHON_BIN" -m http.server "$LOCAL_PORT" --bind 127.0.0.1
