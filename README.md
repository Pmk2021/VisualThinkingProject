# Anytime Trajectory Prediction

Code for preparing Waymo image-plane trajectory data, training ASTRA-EDM, training baselines, evaluating metrics, and rendering qualitative results.

Main replication flow:

1. Convert Waymo v2 parquet data to RGB trajectory parquet segments.
2. Build image-plane sample/image caches.
3. Finetune or provide the U-Net/keypoint RGB feature extractor.
4. Train ASTRA-EDM.
5. Train ASTRA and MLP baselines.
6. Evaluate minADE, maxADE, minFDE, NLL, and latency.
7. Render qualitative overlays/GIFs.

## Setup

```bash
conda env create -f environment.yml
conda activate vtp
pip install -e .
```

Use a CUDA GPU for training. Edit `feature_extractor.waymo_root` in configs before running on a new machine; the checked-in configs use local paths such as `/work/cs-503/santanto/waymo`.

## Waymo Data

Primary files:

- `waymo/docs/dataset_extraction.md`: full extraction notes.
- `waymo/scripts/stream_waymo_to_izar.py`: downloads/streams Waymo chunks and runs conversion remotely.
- `waymo/scripts/build_waymo_rgb_trajectory_dataset.py`: converts Waymo parquet components.
- `waymo/docs/rgb_trajectory_parquet_schema.md`: converted dataset schema.
- `AnytimeTrajectoryPredictor/scripts/build_image_plane_dataset_cache.py`: builds training caches.

Expected converted layout:

```text
<waymo_root>/<split>__<segment>/
  images.parquet
  trajectories.parquet
  image_trajectories.parquet
  ego_poses.parquet
  prediction_targets.parquet
  manifest.json
```

Dry run extraction:

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

Full extraction example:

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
  --izar-workers 8 \
  --work-dir dataset/izar_stream_work
```

Re-run the same command to resume; completed chunks are checkpointed.

## Build Caches

```bash
python AnytimeTrajectoryPredictor/scripts/build_image_plane_dataset_cache.py \
  --config configs/astra_edm_diffusion_waymo_image_plane.yml \
  --splits train,val
```

Default cache path: `cache/waymo_image_plane/`.

## Feature Extractor

```bash
python AnytimeTrajectoryPredictor/scripts/train_unet_keypoints.py \
  --config configs/unet_keypoints_waymo.yml
```

Outputs:

```text
checkpoints/unet_keypoints_waymo_latest.pth
checkpoints/unet_keypoints_waymo_latest.best.pth
```

Used by ASTRA-EDM RGB context and the ASTRA baseline.

## Main Model

```bash
python AnytimeTrajectoryPredictor/scripts/train_trajectory_model.py \
  --config configs/astra_edm_diffusion_waymo_image_plane.yml
```

Key files:

- `AnytimeTrajectoryPredictor/scripts/train_trajectory_model.py`: main trainer entrypoint.
- `AnytimeTrajectoryPredictor/trainer.py`: training/validation loop.
- `AnytimeTrajectoryPredictor/models/architectures/astra_edm_diffusion.py`: ASTRA-EDM model.
- `configs/astra_edm_diffusion_waymo_image_plane.yml`: main config.

Sanity run:

```bash
python AnytimeTrajectoryPredictor/scripts/train_trajectory_model.py \
  --config configs/astra_edm_diffusion_waymo_image_plane_sanity.yml
```

## Baselines

ASTRA baseline:

```bash
python AnytimeTrajectoryPredictor/scripts/train_astra_image_plane.py \
  --config configs/astra_waymo_image_plane_baseline.yml
```

MLP GMM baseline:

```bash
python AnytimeTrajectoryPredictor/scripts/train_image_plane_mlp_baseline.py \
  --config configs/image_plane_mlp_gmm_baseline.yml
```

Baseline files:

- `AnytimeTrajectoryPredictor/scripts/train_astra_image_plane.py`: ASTRA adapter/trainer.
- `ASTRA/models/astra_model.py`: upstream ASTRA model.
- `ASTRA/utils/losses.py`: ASTRA losses.
- `AnytimeTrajectoryPredictor/models/architectures/image_plane_mlp_baseline.py`: MLP GMM baseline.

## Evaluation

ASTRA-EDM, Euler steps 1/4/8:

```bash
python AnytimeTrajectoryPredictor/scripts/diagnose_image_plane_gmm.py \
  --config configs/astra_edm_diffusion_waymo_image_plane.yml \
  --split val \
  --sampling_steps_list 1,4,8 \
  --samplers euler \
  --summary_csv visualizations/image_plane_training/gmm_eval_summary.csv \
  --table_png visualizations/image_plane_training/gmm_eval_table.png
```

Full table with baselines:

```bash
python AnytimeTrajectoryPredictor/scripts/diagnose_image_plane_gmm.py \
  --full_baseline_eval \
  --config configs/astra_edm_diffusion_waymo_image_plane.yml \
  --astra_config configs/astra_waymo_image_plane_baseline.yml \
  --mlp_config configs/image_plane_mlp_gmm_baseline.yml \
  --split val \
  --summary_csv visualizations/image_plane_training/full_baseline_eval_summary.csv \
  --table_png visualizations/image_plane_training/full_baseline_eval_summary_table.png
```

Metrics: `minADE`, `maxADE`, `minFDE`, `NLL`, `latency_ms`. ASTRA baseline has no `NLL` because it does not output a GMM.

## Visualizations

Main model batch render:

```bash
python visualizations/visualize_image_plane_batch.py \
  --config configs/astra_edm_diffusion_waymo_image_plane.yml \
  --indices 1090-1100 \
  --outs_subdir samples_1090_1100 \
  --num_steps 8 \
  --mode_selection best \
  --gmm_heatmap all \
  --save_gmm_png
```

Single refinement GIF:

```bash
python visualizations/visualize_image_plane_diffusion.py \
  --config configs/astra_edm_diffusion_waymo_image_plane.yml \
  --sample_index 1090 \
  --num_steps 8 \
  --output visualizations/outs/sample_1090_refinement.gif
```

ASTRA baseline render:

```bash
python visualizations/visualize_astra_image_plane.py \
  --config configs/astra_waymo_image_plane_baseline.yml \
  --indices 1090-1100 \
  --output_dir visualizations/outs/astra_baseline_1090_1100
```

## Repository Map

Source/config/docs only.

```text
.
|-- .gitignore - Ignore rules.
|-- README.md - Replication guide.
|-- MODEL_TRAINING.md - Legacy model-training notes.
|-- Makefile - Convenience targets.
|-- environment.yml - Conda environment.
|-- exploredata.py - Data inspection helper.
|-- setup.py - Package setup.
|-- configs/
|   |-- astra_edm.yml - Legacy ASTRA-EDM config.
|   |-- astra_edm_diffusion_waymo.yml - Waymo non-image-plane ASTRA-EDM config.
|   |-- astra_edm_diffusion_waymo_image_plane.yml - Main ASTRA-EDM config.
|   |-- astra_edm_diffusion_waymo_image_plane_sanity.yml - Main-model sanity config.
|   |-- astra_edm_diffusion_waymo_sanity.yml - Waymo sanity config.
|   |-- astra_waymo_image_plane_baseline.yml - ASTRA baseline config.
|   |-- astra_waymo_image_plane_quick.yml - Quick ASTRA baseline config.
|   |-- astra_waymo_image_plane_smoke.yml - ASTRA baseline smoke config.
|   |-- basic_config.yml - Basic model config.
|   |-- gnn.yml - GNN config.
|   |-- gru_config.yml - GRU config.
|   |-- image_plane_mlp_gmm_baseline.yml - MLP GMM baseline config.
|   `-- unet_keypoints_waymo.yml - U-Net/keypoint config.
|-- data/
|   `-- waymo/manifest.json - Example Waymo manifest.
|-- notebooks/
|   |-- explore_scenes.ipynb - Scene exploration notebook.
|   `-- test_object_tracker.ipynb - Tracker notebook.
|-- AnytimeTrajectoryPredictor/
|   |-- __init__.py - Package marker.
|   |-- trainer.py - Training loop.
|   |-- Data/
|   |   |-- __init__.py - Data package marker.
|   |   |-- feature_extractor.py - Dataset/feature loaders.
|   |   |-- feature_extractor copy.py - Legacy copy.
|   |   `-- features/dummy_feature_1.py - Placeholder feature.
|   |-- evaluation/
|   |   |-- diversity.py - Diversity metrics.
|   |   `-- latency.py - Latency helpers.
|   |-- models/
|   |   |-- __init__.py - Models package marker.
|   |   |-- ObjectTracker.py - YOLO/ByteTrack wrapper.
|   |   |-- Pipeline.py - Inference pipeline.
|   |   |-- TrajectoryPredictor.py - Model factory.
|   |   |-- unet_keypoint.py - U-Net/keypoint model.
|   |   `-- architectures/
|   |       |-- DiT.py - DiT components.
|   |       |-- astra_edm_diffusion.py - Main ASTRA-EDM model.
|   |       |-- base_model.py - Base model class.
|   |       |-- gnn.py - GNN model.
|   |       |-- gnn copy.py - Legacy GNN copy.
|   |       |-- gru.py - GRU model.
|   |       |-- image_plane_mlp_baseline.py - MLP GMM baseline.
|   |       `-- linear_model.py - Linear baseline.
|   `-- scripts/
|       |-- build_image_plane_dataset_cache.py - Cache builder.
|       |-- compute_fe_stats.py - Feature stats.
|       |-- convert_dataset_for_yolo.py - YOLO data conversion.
|       |-- diagnose_image_plane_gmm.py - Metrics/table eval.
|       |-- finetune_yolo.py - YOLO finetuning.
|       |-- gpu_preflight.sbatch - GPU preflight job.
|       |-- gt_fe_health_check.py - GT feature check.
|       |-- run_astra_edm_diffusion_waymo.sbatch - ASTRA-EDM Slurm job.
|       |-- run_unet_keypoints_waymo.sbatch - U-Net Slurm job.
|       |-- sanity_astra_edm_diffusion.py - Diffusion sanity check.
|       |-- sanity_astra_edm_image_plane.py - Image-plane sanity check.
|       |-- save_fe_features.py - Save detector features.
|       |-- save_fe_features_gt.py - Save GT-box features.
|       |-- test_trajectory_model.py - Legacy model test.
|       |-- train_astra_image_plane.py - ASTRA baseline trainer.
|       |-- train_gnn.sh - GNN train wrapper.
|       |-- train_image_plane_mlp_baseline.py - MLP baseline trainer.
|       |-- train_trajectory_model.py - Main trainer.
|       |-- train_unet_keypoints.py - U-Net trainer.
|       |-- visualize_diffusion.py - Legacy diffusion viz.
|       `-- yolo_conversion_health_check.py - YOLO conversion check.
|-- waymo/
|   |-- docs/
|   |   |-- dataset_extraction.md - Extraction guide.
|   |   |-- rgb_trajectory_parquet_interface.md - Dataset interface.
|   |   `-- rgb_trajectory_parquet_schema.md - Dataset schema.
|   `-- scripts/
|       |-- build_waymo_rgb_trajectory_dataset.py - Waymo converter.
|       |-- build_waymo_web_viewer.py - Web viewer builder.
|       |-- download_waymo_rgb_sample.sh - Sample downloader.
|       |-- extract_waymo_motion.py - Motion extractor.
|       |-- generate_rgb_trajectory_schema_doc.py - Schema doc generator.
|       |-- serve_waymo_viewer.sh - Viewer server.
|       |-- stream_waymo_to_izar.py - Remote extraction orchestrator.
|       |-- waymo_motion_dataset.py - Motion data utilities.
|       `-- waymo_smoke.py - Waymo smoke test.
|-- visualizations/
|   |-- batch_rgb_refinement.py - Batch RGB refinement viz.
|   |-- image_plane_turn_diagnostic.py - Turn diagnostic viz.
|   |-- render_selected_tracks_refinement_gif.py - Curated GIF renderer.
|   |-- visualize_astra_image_plane.py - ASTRA baseline viz.
|   |-- visualize_diffusion.py - Legacy scene viz.
|   |-- visualize_gru_gnn.py - GRU/GNN viz.
|   |-- visualize_image_plane_batch.py - Main batch viz.
|   |-- visualize_image_plane_diffusion.py - Main single-sample viz.
|   |-- visualize_polynomial_gmm.py - Polynomial/GMM viz.
|   `-- visualize_refinement.py - Legacy refinement viz.
`-- ASTRA/
    |-- LICENSE - Upstream license.
    |-- README.md - Upstream README.
    |-- requirements.txt - Upstream requirements.
    |-- main.py - Upstream entrypoint.
    |-- train_ETH.py - ETH/UCY trainer.
    |-- test_ETH.py - ETH/UCY evaluator.
    |-- train_PIE.py - PIE trainer.
    |-- test_PIE.py - PIE evaluator.
    |-- pretrain_unet_eth.py - ETH U-Net pretraining.
    |-- configs/
    |   |-- eth.yaml - ETH config.
    |   |-- hotel.yaml - Hotel config.
    |   |-- pie.yaml - PIE config/template.
    |   |-- univ.yaml - Univ config.
    |   |-- zara01.yaml - Zara01 config.
    |   `-- zara02.yaml - Zara02 config.
    |-- data/
    |   |-- PIE_origin.py - PIE reader.
    |   |-- eth.py - ETH/UCY reader.
    |   |-- eth_preprocessor.py - ETH/UCY preprocessing.
    |   |-- pie_data_layer.py - PIE data layer.
    |   |-- process_eth.py - ETH/UCY processor.
    |   `-- process_nuscenes.py - NuScenes processor.
    |-- models/
    |   |-- astra_model.py - ASTRA model.
    |   |-- keypoint_model.py - ASTRA keypoint model.
    |   `-- vae.py - VAE modules.
    |-- scripts/
    |   |-- build_env_data_process.sh - Upstream setup helper.
    |   |-- down_pretrained_astra_models.bash - ASTRA weight downloader.
    |   |-- down_pretrained_unet_models.bash - U-Net weight downloader.
    |   |-- down_process_PIE.bash - PIE setup.
    |   |-- down_process_eth.bash - ETH setup.
    |   |-- down_process_sdd.bash - SDD setup.
    |   |-- pretrain_unet_eth.py - ETH U-Net wrapper.
    |   `-- pretrain_unet_pie.py - PIE U-Net wrapper.
    `-- utils/
        |-- generatePIEAnnotation.py - PIE annotation utility.
        |-- logger.py - Logger helpers.
        |-- losses.py - ASTRA losses.
        |-- metrics.py - ASTRA metrics.
        |-- misc.py - Misc helpers.
        |-- video2images.py - Video frame extractor.
        |-- visualization.ipynb - Upstream viz notebook.
        `-- visualizer.py - Plotting helpers.
```
