# ASTRA-EDM Diffusion Trajectory Model Handoff

This document describes the ASTRA-conditioned EDM diffusion trajectory predictor implemented in this repository. It is written as a working handoff for a human engineer or another coding agent.

## What The Model Is

The model predicts multi-modal future trajectories for Waymo objects. It uses observed object history as context, samples noisy future trajectory hypotheses, and denoises those hypotheses with an EDM-preconditioned transformer.

High-level flow:

```text
Waymo observed history
  -> target-local features
  -> ASTRA-style context encoder
  -> context tokens C
  -> EDM diffusion denoiser over K trajectory modes
  -> GMM head with mode logits, means, and 2D Cholesky covariances
```

The first implementation predicts one target agent at a time:

```text
features:   [B, A=1, H, 6]
trajectory: [B, A=1, T, 2]
samples:    [B, K, A=1, T, 2]
```

The six observed features are:

```text
local_x, local_y, relative_heading, local_velocity_x, local_velocity_y, valid
```

The future target is normalized local-frame `(x, y)`.

## Model Components

The main model is `ASTRAEDMDiffusionModel`.

It contains:

- `TrajectoryNormalizer`: stores train-set target mean/std and normalizes future trajectories before diffusion training.
- `ASTRAStyleContextEncoder`: projects observed history tokens and applies a transformer encoder to produce context tokens.
- `DiffusionTransformer`: denoises trajectory tokens with mode, agent, future-time, and noise embeddings.
- EDM wrapper: applies `c_skip`, `c_out`, `c_in`, and `c_noise` around the transformer, so training and sampling call the same denoiser interface.
- `GMMHead`: predicts mode logits, mean refinements, and full-rank 2D Cholesky covariance per mode/timestep.
- `compute_loss`: combines denoising loss, GMM NLL, minADE, minFDE, mode classification, and covariance regularization.
- `forward`: runs iterative EDM Euler sampling from Gaussian noise.

The total training loss is:

```text
L = lambda_diff * Ldiff
  + lambda_nll  * LNLL
  + lambda_ade  * LminADE
  + lambda_fde  * LminFDE
  + lambda_mode * Lmode
  + lambda_cov  * Lcov
```

Note: `LNLL` and even the total loss can become negative because this is a continuous density model. A Gaussian density can exceed `1` when covariance is small. Watch `loss_diff`, `loss_minade`, and `loss_minfde` alongside total loss.

## Code Organization

Primary files:

| Path | Purpose |
| --- | --- |
| `AnytimeTrajectoryPredictor/models/architectures/astra_edm_diffusion.py` | Main model, EDM schedule/wrapper, transformer denoiser, GMM head, losses, sampler. |
| `AnytimeTrajectoryPredictor/models/TrajectoryPredictor.py` | Registers `model.type: astra_edm_diffusion`. |
| `AnytimeTrajectoryPredictor/Data/feature_extractor.py` | Contains `WaymoPredictionDataset` for Waymo parquet loading and target-local transforms. |
| `AnytimeTrajectoryPredictor/trainer.py` | Batch-native training loop, validation, checkpointing, AMP, gradient clipping, W&B logging. |
| `AnytimeTrajectoryPredictor/scripts/train_trajectory_model.py` | Training entrypoint. Builds datasets, injects normalizer stats into model config, creates optimizer/trainer. |
| `AnytimeTrajectoryPredictor/scripts/sanity_astra_edm_diffusion.py` | Fast model/data/loss/sampler sanity check. |
| `AnytimeTrajectoryPredictor/scripts/run_astra_edm_diffusion_waymo.sbatch` | Full GPU Slurm training job. Includes CUDA fail-fast check. |
| `AnytimeTrajectoryPredictor/scripts/gpu_preflight.sbatch` | Short GPU environment test job. |
| `configs/astra_edm_diffusion_waymo.yml` | Full training config. |
| `configs/astra_edm_diffusion_waymo_sanity.yml` | Small CPU-friendly sanity config. |

Existing baseline code is still present. The trainer checks `model.expects_batch`; the ASTRA-EDM model sets this to `True`, while older baseline models keep the old `(features, trajectory, f_)` style.

## Data Path And Dataset Behavior

The intended dataset root is:

```text
/work/cs-503/santanto/waymo
```

The loader first tries to use:

```text
*/prediction_targets.parquet
```

In the current data snapshot, those files are empty. The loader therefore falls back to supervised sliding windows from:

```text
*/trajectories.parquet
```

For each trajectory row, it builds:

- observed history of length `history_steps`
- future target of length `future_steps`
- target-local coordinates anchored at the last valid observed pose
- deterministic train/validation segment split using `split_seed` and `validation_fraction`

The train dataset computes `target_mean` and `target_std`, and `train_trajectory_model.py` injects them into `args.model` before constructing the model.

## Environment

Use `vtp-gpu` for GPU training:

```bash
conda activate vtp-gpu
```

This environment was created because the older `vtp` environment had CPU-only PyTorch. The Slurm script now activates `vtp-gpu` and exits immediately if CUDA is not visible.

Before submitting a long run, test the GPU environment:

```bash
sbatch AnytimeTrajectoryPredictor/scripts/gpu_preflight.sbatch
```

Then inspect:

```bash
tail -n 80 logs/gpu_preflight_<job_id>.out
tail -n 80 logs/gpu_preflight_<job_id>.err
```

Expected signs of success:

```text
cuda_available True
device_count 1
device_name ...
matmul_ok ...
```

## How To Run

Sanity check:

```bash
conda run -n vtp-gpu python -u AnytimeTrajectoryPredictor/scripts/sanity_astra_edm_diffusion.py \
  --config configs/astra_edm_diffusion_waymo_sanity.yml \
  --batch-size 2 \
  --sampling-steps 2
```

Small training run:

```bash
conda run -n vtp-gpu python -u AnytimeTrajectoryPredictor/scripts/train_trajectory_model.py \
  --config configs/astra_edm_diffusion_waymo_sanity.yml
```

Full training run:

```bash
sbatch AnytimeTrajectoryPredictor/scripts/run_astra_edm_diffusion_waymo.sbatch
```

Monitor Slurm logs:

```bash
tail -f logs/astra_edm_waymo_<job_id>.out
tail -f logs/astra_edm_waymo_<job_id>.err
```

## Monitoring And Checkpoints

The trainer saves:

```text
checkpoints/astra_edm_diffusion_waymo_latest.pth
checkpoints/astra_edm_diffusion_waymo_latest.best.pth
```

Each checkpoint contains:

```python
{
    "epoch": ...,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "metrics": ...,
    "config": ...,
}
```

Inspect checkpoint metrics:

```bash
conda run -n vtp-gpu python -c "import torch; ckpt=torch.load('checkpoints/astra_edm_diffusion_waymo_latest.best.pth', map_location='cpu'); print(ckpt['epoch']); print(ckpt['metrics'])"
```

W&B is configured in `configs/astra_edm_diffusion_waymo.yml`:

```yaml
wandb_project: astra-edm-waymo
wandb_mode: online
```

If online logging is enabled and W&B credentials are present, the Slurm stderr log prints the W&B run URL.

Useful metrics to watch:

- `train/loss`, `val/loss`
- `train/loss_diff`, `val/loss_diff`
- `train/loss_minade`, `val/loss_minade`
- `train/loss_minfde`, `val/loss_minfde`
- `train/loss_nll`, `val/loss_nll`
- `train/sigma_mean`, `train/sigma_max`
- `train/grad_norm`

Do not judge the run only by total loss. For this model, total loss can be negative because the GMM NLL term is a continuous density objective.

## Common Failure Modes

### Job starts but appears stuck at 0%

Check whether PyTorch sees CUDA:

```text
cuda_available True
```

If false, the job is on CPU. The current Slurm script should now fail fast instead of training.

### Negative loss

This can be valid because the GMM NLL is a continuous density. If ADE/FDE do not improve while NLL becomes very negative, the covariance head may be collapsing. Try:

```yaml
log_std_min: -2.0
lambda_nll: 0.01
```

### Training is slow before first epoch progress

Dataset indexing and target-stat computation happen before the model is constructed. If this becomes a bottleneck, add a cached index/stat file for `WaymoPredictionDataset`.

### Empty prediction targets

Current `prediction_targets.parquet` files under `/work/cs-503/santanto/waymo` are empty. This is expected for now; the loader falls back to `trajectories.parquet`.

## Suggested Next Work

High-value next steps:

1. Add cached dataset indexing and cached target normalization stats.
2. Log covariance diagnostics directly: mean/std of Cholesky diagonal logs and covariance off-diagonal.
3. Add an evaluation script that loads `.best.pth`, runs sampling on validation examples, and writes ADE/FDE plus plots.
4. Compare against a simple deterministic baseline on the same local-frame Waymo windows.
5. Add image or map context after the trajectory-only model is stable.
6. Consider lower `lambda_nll` and higher `log_std_min` if covariance collapse appears.

## Agent Checklist For Future Changes

When modifying this model:

1. Run `python -m py_compile` on touched Python files.
2. Run `sanity_astra_edm_diffusion.py` with the sanity config.
3. Run one tiny training job with the sanity config.
4. Confirm checkpoint creation and finite metrics.
5. If changing GPU/runtime behavior, run `gpu_preflight.sbatch`.
6. Keep old baseline model behavior intact unless intentionally refactoring the shared pipeline.
