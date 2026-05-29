## Instructions for training model

### Step 1: Add your model

First, go to `AnytimeTrajectoryPredictor/models/architectures` and create a new file for your model. For simplicity,
you can import `AnytimeTrjectoryPredictor.base_model.base_model` and use it as a parent class. The only function that should be defined is `forward`.

To add your model to the pipeline, open `AnytimeTrajectoryPredictor/models/TrajectoryPredictor` and follow the examle to add your model. Ideally, let your model parameters be loaded from a config file

### Step 2: Create config file

You can copy `configs/base_config.yml` as a guide.

#### Data

Replace the paths in `feature_extractor` with the path to wherever your data is

#### Training

Set the training parameters to whatever you choose. If you're loading from a path, set `from_checkpoint` to your checkpoint. And set `save_to` to wherever you'd like to save your model.

**Important** Please make your checkpoint name unique(ie. don't do model.pth or weights.pth). This is just to avoid people accidentally overriding checkpoint files when testing out different models.

#### Model

The model field is relatively open. The only thing you need to ensure is that `model.type` matches the model type you defined in Step 1 within `AnytimeTrajectoryPredictor/models/TrajectoryPredictor`

To avoid any overriding, name your config `[model_name].config` or `[run_name].config` or something along those lines(pretty much anything except `model.config`).

### Step 3 Training:

To train your model, call:

```bash
    python AnytimeTrajectoryPredictor/scripts/train_trajectory_model.py --config path_to_config
```

where `path_to_config` is your config path

Try to keep your code reproducable since it might be run on different machines and different folders. So do your best to only change the code blocks above. If you'd like to add some model-specific code, it's probably better to make a separate folder for your model or something instead of directly changing the pipeline code. That way we can merge pr's without having to worry about anything breaking

1. Update config file with right paths
2. Update feature extractor lists with what features you want
