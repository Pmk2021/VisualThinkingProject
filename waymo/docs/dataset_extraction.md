# Waymo Dataset Extraction

Converts Waymo Open Dataset v2 parquet files into a compact RGB-trajectory dataset.
All downloading and conversion runs on **izar**; your local machine only orchestrates.

---

## How it works

```
local machine                          izar
─────────────────                      ──────────────────────────────────────
gcloud ls  ──── discovers segments ─►
gcloud auth ─── mints bearer token ──► wget GCS parquet files (parallel)
                                       python build_waymo_rgb_trajectory_dataset.py
                                         ├─ load components in parallel
                                         └─ decode images in parallel
                                       output written to --izar-output-dir
checkpoint JSON saved locally ◄──────
```

---

## Prerequisites

### Local machine
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) with access to `gs://waymo_open_dataset_v_2_0_1`
- SSH key configured for izar (`ssh user@izar.epfl.ch` works without a password prompt)

```bash
gcloud auth login
gcloud auth print-access-token   # should return a token
```

### Izar
- Repo cloned
- Conda environment created and dependencies installed:

```bash
# on izar
git clone <repo-url> ~/VisualThinkingProject
cd ~/VisualThinkingProject
conda env create -f environment.yml
conda activate vtp
pip install -e .
```

- `wget` available (standard on most Linux systems)

---

## Running the extraction

All commands below run **on your local machine**.

### Dry run — verify discovery and SSH connectivity

```bash
python waymo/scripts/stream_waymo_to_izar.py \
  --izar user@izar.epfl.ch \
  --izar-work-dir /scratch/$USER/waymo_work \
  --izar-output-dir /scratch/$USER/waymo_rgb_trajectory \
  --izar-converter ~/VisualThinkingProject/waymo/scripts/build_waymo_rgb_trajectory_dataset.py \
  --izar-python /home/$USER/miniconda3/envs/vtp/bin/python \
  --splits training,validation \
  --dry-run
```

### Full extraction (example: 100 GiB target)

```bash
python waymo/scripts/stream_waymo_to_izar.py \
  --izar user@izar.epfl.ch \
  --izar-work-dir /scratch/$USER/waymo_work \
  --izar-output-dir /scratch/$USER/waymo_rgb_trajectory \
  --izar-converter ~/VisualThinkingProject/waymo/scripts/build_waymo_rgb_trajectory_dataset.py \
  --izar-python /home/$USER/miniconda3/envs/vtp/bin/python \
  --splits training,validation \
  --target-gb 100 \
  --max-buffer-gb 5 \
  --min-izar-free-gb 2 \
  --izar-workers 8 \
  --work-dir dataset/izar_stream_work
```

### Resuming after interruption

Re-run the exact same command. Chunks recorded as `done` in `--work-dir/checkpoints/` are skipped automatically.

### Reset a failed chunk

```bash
rm dataset/izar_stream_work/checkpoints/<split>__<segment>.json
```

---

## Key arguments

| Argument | Default | Description |
|---|---|---|
| `--izar` | *(required)* | SSH target, e.g. `user@izar.epfl.ch` |
| `--izar-work-dir` | *(required)* | Staging directory on izar |
| `--izar-output-dir` | *(required)* | Converted output directory on izar |
| `--izar-converter` | *(required)* | Absolute path to `build_waymo_rgb_trajectory_dataset.py` on izar |
| `--izar-python` | `python3` | Python interpreter on izar — use the conda env's absolute path |
| `--splits` | `training,validation` | Comma-separated splits to process |
| `--target-gb` | `100.0` | Stop once this many GiB of converted output are checkpointed |
| `--max-buffer-gb` | `5.0` | Expected staging size per chunk; size `chunk-workers` accordingly |
| `--min-izar-free-gb` | `2.0` | Pause before a new chunk if izar free space drops below this |
| `--chunk-workers` | `1` | Parallel chunks; each uses up to `--max-buffer-gb` of izar staging |
| `--izar-workers` | `4` | Threads on izar for parallel component loading and JPEG decoding |
| `--work-dir` | `dataset/izar_stream_work` | Local directory for checkpoint JSON files |
| `--keep-going` | off | Continue after failed chunks instead of aborting |
| `--dry-run` | off | Discover and print chunks; make no SSH calls |

### Parallelism guide

| Knob | What it parallelizes | Recommended |
|---|---|---|
| `--izar-workers` | Component loading (4 parquet files at once) + JPEG decoding threads | Set to izar core count (e.g. `8`) |
| `--chunk-workers` | Whole chunks end-to-end (download + convert simultaneously) | `2`–`4` if izar disk allows; each needs `--max-buffer-gb` of free staging |

Downloads within each chunk always run in parallel (all 4 components fetched simultaneously via `wget` background jobs).

### Finding the conda Python path on izar

```bash
ssh user@izar.epfl.ch "conda run -n vtp which python"
```

---

## Output layout on izar

```
<izar-output-dir>/
  <split>__<segment>/
    images.parquet            # one row per RGB camera frame
    trajectories.parquet      # one row per tracked object trajectory
    image_trajectories.parquet# image ↔ trajectory visibility links
    ego_poses.parquet         # vehicle pose per timestep
    prediction_targets.parquet# WOMD tracks_to_predict (empty unless --izar-motion-scenario-root set)
    manifest.json             # row counts and table descriptions
```

See `waymo/docs/rgb_trajectory_parquet_schema.md` for column-level documentation.

---

## Authentication note

The local `gcloud auth print-access-token` token is valid for ~60 minutes and is
refreshed automatically every 45 minutes. For runs spanning many hours the script
handles token rotation transparently — no manual intervention needed.
