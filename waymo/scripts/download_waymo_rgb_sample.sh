#!/usr/bin/env bash
set -euo pipefail

# Download a tiny recursive sample from Waymo Open Dataset v2 on GCS.
#
# Default behavior:
#   - Uses gs://waymo_open_dataset_v_2_0_1
#   - Recursively lists concrete objects under the dataset root
#   - Infers leaf directories from those object paths
#   - Downloads exactly one concrete object from each leaf directory
#   - Refuses to overwrite existing local files unless --force is passed
#
# Requirements:
#   - gsutil from Google Cloud SDK, authenticated if the bucket requires it
#   - Enough local disk for the sampled objects
#
# Examples:
#   scripts/download_waymo_rgb_sample.sh --dry-run
#   scripts/download_waymo_rgb_sample.sh --max-dirs 3
#   scripts/download_waymo_rgb_sample.sh --base-uri gs://waymo_open_dataset_v_2_0_1 --out-dir dataset/waymo_rgb_sample

BASE_URI="gs://waymo_open_dataset_v_2_0_1"
OUT_DIR="dataset/waymo_rgb_sample"
MAX_DIRS=0
DRY_RUN=0
FORCE=0

usage() {
  sed -n '2,31p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'USAGE'

Options:
  --base-uri URI      GCS dataset root. Default: gs://waymo_open_dataset_v_2_0_1
  --out-dir DIR       Local output directory. Default: dataset/waymo_rgb_sample
  --max-dirs N        Stop after N sampled leaf directories. 0 means no explicit limit. Default: 0.
  --dry-run           Print planned downloads without downloading.
  --force             Overwrite existing local files.
  -h, --help          Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-uri)
      BASE_URI="${2:?missing value for --base-uri}"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="${2:?missing value for --out-dir}"
      shift 2
      ;;
    --max-dirs)
      MAX_DIRS="${2:?missing value for --max-dirs}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --force)
      FORCE=1
      shift
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

if ! command -v gsutil >/dev/null 2>&1; then
  echo "gsutil was not found. Install the Google Cloud SDK, then rerun this script." >&2
  exit 127
fi

if ! [[ "$MAX_DIRS" =~ ^[0-9]+$ ]]; then
  echo "--max-dirs must be a non-negative integer." >&2
  exit 2
fi

BASE_URI="${BASE_URI%/}"
mkdir -p "$OUT_DIR"

echo "Dataset root: $BASE_URI"
echo "Output dir:   $OUT_DIR"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Mode:         dry run"
fi
echo

# GCS paths are virtual directories. To avoid accidentally copying a whole
# prefix, list concrete objects first, infer leaf directories from object paths,
# and copy exactly one object URI per inferred leaf.
mapfile -t ALL_OBJECTS < <(
  gsutil ls -r "${BASE_URI}/**" \
    | sed '/:$/d; /\/$/d; /^$/d' \
    | sort
)

if [[ "${#ALL_OBJECTS[@]}" -eq 0 ]]; then
  echo "No GCS objects found under the dataset root." >&2
  echo "Try a dry listing with: gsutil ls -r '${BASE_URI}/**' | less" >&2
  exit 1
fi

# One object per true leaf directory. We first group objects by their containing
# directory, then skip any directory that is a prefix of another file-containing
# directory.
mapfile -t SAMPLE_OBJECTS < <(
  printf '%s\n' "${ALL_OBJECTS[@]}" | awk '
    {
      object = $0
      dir = object
      sub(/[^/]+$/, "", dir)
      if (!(dir in first_object_by_dir)) {
        first_object_by_dir[dir] = object
        dirs[++n] = dir
      }
    }
    END {
      for (i = 1; i <= n; i++) {
        has_child = 0
        for (j = i + 1; j <= n; j++) {
          if (index(dirs[j], dirs[i]) == 1) {
            has_child = 1
            break
          } else {
            break
          }
        }
        if (!has_child) {
          print first_object_by_dir[dirs[i]]
        }
      }
    }
  '
)

if [[ "${#SAMPLE_OBJECTS[@]}" -eq 0 ]]; then
  echo "No sample objects found under the dataset root." >&2
  exit 1
fi

downloaded=0
planned=0

for first_object in "${SAMPLE_OBJECTS[@]}"; do
  if [[ "$MAX_DIRS" -gt 0 && "$planned" -ge "$MAX_DIRS" ]]; then
    break
  fi

  if [[ -z "$first_object" ]]; then
    continue
  fi
  if [[ "$first_object" == */ || "$first_object" == *: ]]; then
    echo "[skip not object] $first_object" >&2
    continue
  fi

  relative_path="${first_object#${BASE_URI}/}"
  local_path="${OUT_DIR}/${relative_path}"
  planned=$((planned + 1))

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] $first_object"
    echo "          -> $local_path"
    continue
  fi

  if [[ -e "$local_path" && "$FORCE" -ne 1 ]]; then
    echo "[skip exists] $local_path"
    continue
  fi

  mkdir -p "$(dirname "$local_path")"
  echo "[download] $first_object"
  echo "           -> $local_path"
  if [[ "$FORCE" -eq 1 ]]; then
    gsutil cp "$first_object" "$local_path"
  else
    gsutil cp -n "$first_object" "$local_path"
  fi
  downloaded=$((downloaded + 1))
done

echo
echo "Planned sample objects: $planned"
if [[ "$DRY_RUN" -eq 0 ]]; then
  echo "Downloaded objects:     $downloaded"
  echo "Saved under:            $OUT_DIR"
fi
