# Image-Plane ASTRA-EDM Diffusion Handoff

## Current Goal

This model predicts future object-center trajectories directly in camera image coordinates from RGB history and image-derived past boxes. It intentionally avoids BEV/world-coordinate targets and conditioning. The current main config is `configs/astra_edm_diffusion_waymo_image_plane.yml`.

The active path is a single-agent image-plane setup (`A=1`) for Waymo front-camera tracks. Future bbox-size prediction is not implemented; the target is center-only `(u, v)`.

## Data And Batch Contract

Dataset: `WaymoImagePlaneDataset` in `AnytimeTrajectoryPredictor/Data/feature_extractor.py`.

Expected training batch fields:

- `rgb_history`: `[B, H, 3, image_height, image_width]`, float RGB frames.
- `box_history`: `[B, 1, H, 4]`, normalized image-plane boxes `(cx, cy, w, h)` where centers are in `[-1, 1]` and sizes are normalized by image size.
- `features`: same tensor as `box_history`, kept for compatibility with older model interfaces.
- `trajectory`: `[B, 1, T, 2]`, future normalized image-plane centers `(u, v)` in `[-1, 1]`.
- `observed_mask`: `[B, 1, H]`, valid observed boxes.
- `future_mask`: `[B, 1, T]`, valid future center labels.
- Metadata: `scene_id`, `camera_name`, `trajectory_row_id`, image size fields, timestamps/object type when available.

The dataset can load prebuilt sample/image caches from `cache/waymo_image_plane`. The training config currently uses those caches.

## Architecture

Main class: `ASTRAEDMDiffusionModel` in `AnytimeTrajectoryPredictor/models/architectures/astra_edm_diffusion.py`.

High-level flow:

1. `RGBBoxContextEncoder` encodes the past.
2. A diffusion transformer denoises noisy future control points or trajectory points.
3. `GMMHead` produces K trajectory modes with Gaussian covariance and mode logits.
4. If control points are enabled, knot outputs are expanded into the full future horizon using Catmull-Rom interpolation.

RGB/box context encoder:

- Uses `rgb_backbone: unet_resnet18` in the current config.
- Loads encoder weights from `astra/pretrained_weights/unet/pie_unet_model_best.pt`.
- Uses only a selected encoder feature stage (`rgb_backbone_stage: -2`).
- The RGB backbone is frozen by default with `freeze_rgb_backbone: true`.
- Frozen backbone behavior:
  - all backbone parameters have `requires_grad=False`;
  - the backbone is kept in eval mode during forward;
  - forward through the backbone runs under `torch.no_grad()`.
- ROI-like visual tokens are extracted from the observed bbox region with `grid_sample`.
- Past box tokens are projected separately.
- Visual and box tokens are fused, then passed through a small transformer context encoder.

Diffusion transformer:

- Input state shape is `[B, K, A, N, 2]`, where `K` is number of modes and `N` is either full horizon or number of knots.
- Uses mode, agent, future-time, and noise embeddings.
- Supports self-conditioning.
- Cross-attends denoising tokens to the RGB/box context tokens.

Control-point trajectory representation:

- Enabled by `use_control_points: true`.
- Current main config uses `num_knots: 6` over `future_horizon_T: 80`.
- The model predicts knot/control-point trajectories, then expands to all future timesteps with Catmull-Rom basis interpolation.
- This was introduced because full per-timestep diffusion was producing tangled, non-curve-like trajectories.

Prior+correction mode:

- Enabled by `use_prior_correction: true`.
- A constant-velocity image-plane prior is built from past box centers.
- The denoiser predicts residual corrections around that prior instead of absolute future centers.
- At output time, predicted residuals are added back to the prior.

Sampler:

- Default sampler is Heun (`sampler_type: heun`).
- Euler is still supported by config.
- Sampling starts from `_initial_noise`.
- Current default uses `initial_noise_type: radial_control_points`: modes start as radial residual fans around the prior center, with radius growing over future knots.

## Outputs

The model returns `GMMParams`:

- `mode_logits`: `[B, K]`.
- `mode_probs`: `[B, K]` after softmax.
- `mu`: `[B, K, A, T, 2]` after knot expansion and denormalization when using normal forward inference.
- `cov_cholesky`: `[B, K, A, T, 2, 2]` Cholesky factor of the per-step 2D Gaussian covariance.
- `cov_raw`: raw covariance head output, mostly used for regularization/diagnostics.

For training loss internals, `mu` is kept normalized. Visualization/evaluation usually denormalizes only for rendering or pixel conversion.

## Losses And Why They Exist

The total training loss is a weighted sum of the following terms. Current weights live in `configs/astra_edm_diffusion_waymo_image_plane.yml`.

- `loss_diff`: all-mode denoising loss on noisy states.
  - Purpose: standard diffusion reconstruction signal.
  - Current weight is reduced (`lambda_diff: 0.25`) so it does not force all modes to reconstruct the same target too strongly.

- `loss_wta_diff`: winner-take-all denoising loss on the minADE-winning mode.
  - Purpose: make mode-specific denoising useful and reduce mode collapse/spaghetti from forcing all modes toward the same trajectory.
  - Current weight: `lambda_wta_diff: 1.0`.

- `loss_rollout_diff`: denoising loss from states produced by 1-2 sampler rollout steps.
  - Purpose: expose the model to imperfect sampler states, not only clean target plus Gaussian noise.
  - Current weight multiplier: `lambda_rollout_diff`.

- `loss_nll`: Gaussian mixture negative log likelihood.
  - Purpose: train covariance and probabilistic calibration.
  - Kept mild because diagnostics showed covariance was not the main collapse source.

- `loss_minade` and `loss_minfde`: min-over-modes displacement losses.
  - Purpose: directly optimize best-mode trajectory accuracy and preserve multimodal hypotheses.
  - `best_mode` from minADE is also used for mode CE and WTA denoising.

- `loss_mode`: cross-entropy on mode logits using the minADE-winning mode as label.
  - Purpose: make the model assign probability mass to the geometrically best mode.
  - Increased because diagnostics showed mode probabilities were nearly uniform and top mode was correct only at chance.

- `loss_mode_margin`: best-vs-rest logit margin loss.
  - Purpose: explicitly separate the best mode logit from other modes.

- `loss_cov`: covariance range regularization.
  - Purpose: prevent degenerate covariance head outputs outside configured log-std bounds.

- `loss_bounds`: image-coordinate bounds penalty.
  - Purpose: softly discourage future center predictions outside plausible normalized image coordinates.

- `loss_smooth`, `loss_accel`, `loss_jerk`: smoothness/acceleration/jerk penalties.
  - Purpose: discourage physically implausible zig-zag trajectories in image space.
  - Introduced after visualizations showed tangled trajectories.

- `loss_prior_residual`: residual magnitude penalty around the kinematic prior.
  - Purpose: keep corrections from drifting too far from the constant-velocity prior unless data/losses support it.

- `loss_speed`: soft speed cap in normalized image coordinates.
  - Purpose: penalize unrealistic per-timestep jumps.

- `loss_entropy`: entropy regularization on mode probabilities.
  - Purpose: originally used to prevent early mode collapse.
  - Currently disabled (`lambda_entropy: 0.0`) because diagnostics showed probabilities were already too uniform.

- `loss_diversity`: pairwise trajectory repulsion hinge.
  - Purpose: keep modes geometrically distinct.
  - Should be monitored because too much diversity can worsen spaghetti if mode assignment is weak.

## Important Current Config Choices

Current main model settings:

- `use_rgb_context: true`
- `rgb_backbone: unet_resnet18`
- `rgb_backbone_weights: astra/pretrained_weights/unet/pie_unet_model_best.pt`
- `freeze_rgb_backbone: true`
- `num_modes_K: 6`
- `future_horizon_T` comes from the dataset config, currently `80`.
- `use_control_points: true`
- `num_knots: 6`
- `interpolation: catmull_rom`
- `use_prior_correction: true`
- `initial_noise_type: radial_control_points`
- `radial_train_noise_prob: 0.25`
- `sampler_type: heun`
- `num_sampling_steps: 12`

Current training output path:

- `checkpoints/astra_edm_diffusion_waymo_image_plane_prior_correction_knots_regularized_latest.pth`

## Useful Commands And Scripts

Train image-plane model:

```bash
conda run -n vtp-gpu python -u AnytimeTrajectoryPredictor/scripts/train_trajectory_model.py --config configs/astra_edm_diffusion_waymo_image_plane.yml
```

Sanity check model/data/loss/gradients on tiny config:

```bash
conda run -n vtp-gpu python -u AnytimeTrajectoryPredictor/scripts/sanity_astra_edm_image_plane.py --config configs/astra_edm_diffusion_waymo_image_plane_sanity.yml
```

Build dataset caches before training:

```bash
conda run -n vtp-gpu python -u AnytimeTrajectoryPredictor/scripts/build_image_plane_dataset_cache.py --config configs/astra_edm_diffusion_waymo_image_plane.yml
```

Run inference diagnostics for ADE/FDE/NLL/covariance/mode assignment/spread:

```bash
conda run -n vtp-gpu python -u AnytimeTrajectoryPredictor/scripts/diagnose_image_plane_gmm.py \
  --config configs/astra_edm_diffusion_waymo_image_plane.yml \
  --checkpoint checkpoints/astra_edm_diffusion_waymo_image_plane_prior_correction_knots_regularized_latest.best.pth \
  --split val \
  --batch_size 8 \
  --max_batches 16 \
  --sampling_steps_list 4,8,16 \
  --device cuda \
  --output_csv visualizations/image_plane_training/gmm_diagnostics.csv
```

Visualize a batch using defaults from the training config:

```bash
conda run -n vtp-gpu python -u visualizations/visualize_image_plane_batch.py --config configs/astra_edm_diffusion_waymo_image_plane.yml
```

Useful visualization options:

```bash
conda run -n vtp-gpu python -u visualizations/visualize_image_plane_batch.py \
  --config configs/astra_edm_diffusion_waymo_image_plane.yml \
  --random_indices 8 \
  --random_scene \
  --moving_only \
  --num_steps 8 \
  --mode_selection best \
  --gmm_heatmap all \
  --save_gmm_png
```

To inspect the most informative agents from one scene, rank candidates by the model's top-probability mode error:

```bash
conda run -n vtp-gpu python -u visualizations/visualize_image_plane_batch.py \
  --config configs/astra_edm_diffusion_waymo_image_plane.yml \
  --outs_subdir scene_quality_debug \
  --bad=4 \
  --good=4 \
  --moving_only \
  --num_steps 8
```

`--bad=K` renders the K highest top-mode ADE samples in the selected scene; `--good=J` renders the J lowest. When these flags are used without an explicit `--mode_selection`, the visualizer shows the top-probability mode so the GIF reflects the prediction that was ranked. `--outs_subdir NAME` writes to `visualizations/outs/NAME`; use it instead of `--output_dir`.

Single-sample refinement GIF script:

```bash
conda run -n vtp-gpu python -u visualizations/visualize_image_plane_diffusion.py \
  --config configs/astra_edm_diffusion_waymo_image_plane.yml \
  --checkpoint checkpoints/astra_edm_diffusion_waymo_image_plane_prior_correction_knots_regularized_latest.best.pth \
  --output visualizations/image_plane_results/sample_refinement.gif
```

Training-loop visualizations:

- Controlled by `training.visualization_enabled` and related `visualization_*` keys in the main config.
- Current output directory: `visualizations/image_plane_training`.
- The visualizer currently shows the best-fitting mode rather than all modes, includes GT-only frames, overlays the tracked bbox, and can save a separate GMM heatmap PNG.

## What To Watch During Training

Primary indicators:

- `loss_minade`, `loss_minfde`: best-mode geometry.
- `loss_mode`, `loss_mode_margin`: mode assignment learning.
- `mode_entropy`: should not stay exactly uniform forever; for `K=6`, uniform entropy is about `1.79`.
- Diagnostics `top_is_best_rate`: should rise above random chance (`~1/K`).
- Diagnostics `best_mode_prob_mean`: should rise above `1/K` if logits are learning.
- Diagnostics `mode_pair_fde_mean`: confirms whether modes are geometrically separated.
- Covariance stats: watch for `tiny_std_rate_lt_0p02` and `huge_std_rate_gt_0p5`.
- Visualizations: trajectories should become curve-like and consistent across denoising steps rather than tangled.

Known failure patterns:

- Uniform `mode_probs` with decent minADE means geometry exists but mode logits are not calibrated.
- Increasing NLL with stable ADE often points to covariance/calibration rather than center prediction.
- Spaghetti trajectories usually mean the representation/loss is allowing high-frequency motion; control points, acceleration/jerk, and WTA denoising were added to fight this.
- If training startup appears hung, it is often importing `torchvision`/`segmentation_models_pytorch` or loading U-Net weights. This can take tens of seconds in this environment.

## Recent Implementation Notes

- The U-Net encoder load path handles SMP encoders whose `load_state_dict` returns `None`.
- The frozen U-Net backbone was verified with `backbone_trainable 0`.
- `diagnose_image_plane_gmm.py` supports `--sampling_steps_list` and no longer crashes when a config `save_to` checkpoint is missing.
- The current model requires retraining/fine-tuning after loss/noise changes; old checkpoints are not a clean comparison for the new WTA/radial setup.
