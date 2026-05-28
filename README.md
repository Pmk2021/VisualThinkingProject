# Anytime Trajectory Prediction

## Environment Setup

A conda environment is provided in `environment.yaml`. To create the environment, run:

```bash
    conda env create -f environment.yaml
```

It contains the following dependencies:

```
name: vtp
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - numpy>=2.4.6
  - matplotlib>=3.10.9
  - scipy>=1.17.1
  - pytorch>=2.3
  - torchvision>=0.26.0
  - ipykernel>=6.29.5
  - tqdm>=4.67.3
  # astra
  - python-box>=7.4.1
  - segmentation-models-pytorch
  - icecream
  - opencv
  # waymo converter
  - pandas>=3.0.3
  - pyarrow>=24.0.0
  - pillow>=12.2.0
  - pip
  - torch_geometric
  - pip:
      - opencv-python
      - ultralytics
      - black
      - -e .
```

## How to run the code

### Step 1: Download and format the Waymo dataset

TODO

### Step 2: Extract features for the dataset

Once the dataset is downloaded and formatted, extract YOLO features for the dataset by running:

```bash
    python AnytimeTrajectoryPredictor/scripts/save_fe_features.py
```

or, if you want to extract features for the GT boxes instead of the predicted boxes, run:

```bash
    python AnytimeTrajectoryPredictor/scripts/save_fe_features_gt.py
```

### Step 3: Train your model

TODO

### Step 4: Evaluate your model

TODO

## File Hierarchy

Here is a description of the key files and folders in the repository:

```
./AnytimeTrajectoryPredictor: # Main codebase (models, training scripts, evaluation scripts, etc.)
Data/ # Data processing and feature extraction code
evaluation/ # Evaluation metrics
models/ # Model architectures and pipeline code
scripts/ # Various scripts for training, testing, and feature extraction
trainer.py # Main training loop

./AnytimeTrajectoryPredictor/Data:
feature_extractor.py # Dataset definition, allows to load saved parquet files and extract features for training

./AnytimeTrajectoryPredictor/evaluation:
diversity.py # Mode diversity metric
latency.py # Latency metric

./AnytimeTrajectoryPredictor/models:
ObjectTracker.py # Object tracking code (YOLO, ByteTrack)
Pipeline.py # End-to-end pipeline code, takes model and feature extractor args to allow end-to-end inference.

./AnytimeTrajectoryPredictor/models/architectures:
astra_edm_diffusion.py # ASTRA model
base_model.py # Base model class
DiT.py # DiT model
gnn.py # Graph Neural Network
gru.py # GRU

./AnytimeTrajectoryPredictor/scripts:
compute_fe_stats.py # Computes performance stats for feature extractor (mAP, IoU, number of objects detected, etc.) to compare to the Waymo ground truth
convert_dataset_for_yolo.py # Converts Waymo dataset to YOLO format for training the object detector
finetune_yolo.py # Script for finetuning YOLO on Waymo data (ran into crashing issues)
gt_fe_health_check.py # Script for checking the saved feature extractor features on Waymo samples.
save_fe_features_gt.py # Extracts and saves feature extractor features for the Waymo dataset ground truth (ie. using the GT boxes instead of the predicted boxes)
save_fe_features.py # Extracts and saves feature extractor features by running the feature extractor on the Waymo dataset and saving the features.
test_trajectory_model.py # Plotting convenience script for model health check
train_gnn.sh # GNN training script
train_trajectory_model.py # Main training script for trajectory prediction models
yolo_conversion_health_check.py # Health check for convert_dataset_for_yolo.py

./configs: # Contains the config files for training.

./notebooks:
explore_scenes.ipynb # Notebook for visualizing Waymo scenes and trajectories to extract intersting ones for qualitative evaluation
13056 test_object_tracker.ipynb # Notebook for testing the object tracker on Waymo data and visualizing the results

./outputs: # Contains images saved from the notebooks and scripts for visualization purposes

./visualizations: # Various visualization scripts

./waymo/docs: # Documentation for the Waymo dataset conversion code, which allows to understand how we can use the dataset for our task.

./waymo/scripts: # Scripts for downloading and formatting the Waymo dataset.
```
