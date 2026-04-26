#!/bin/bash

# Pull AirSim image
mkdir /home/$USER/airsim_container
cd /home/$USER/airsim_container
singularity pull airsim.sif docker://nathangromb/airsim:1.7.0-linux

# Setup UE Blocks test environment
mkdir -p /home/$USER/airsim_test/
cd /home/$USER/airsim_test
wget https://github.com/microsoft/AirSim/releases/download/v1.7.0-linux/Blocks.zip
unzip Blocks.zip
rm Blocks.zip
chmod +x /home/$USER/airsim_test/Blocks/LinuxNoEditor/Blocks.sh
mkdir -p /home/$USER/airsim_test/Settings
cat <<EOF > /home/$USER/airsim_test//Settings/settings.json
{
  "SettingsVersion": 1.2,
  "SimMode": "ComputerVision",
  "CameraDefaults": {
    "CaptureSettings": [
      {
        "ImageType": 0,           // 0=Scene, 1=DepthPlanar, 2=DepthPerspective, 3=DepthVis, 4=DisparityNormalized, 5=Segmentation, 6=SurfaceNormals
        "Width": 256,
        "Height": 144,
        "FOV_Degrees": 90,
        "AutoExposureSpeed": 100,
        "AutoExposureBias": 0,
        "AutoExposureMaxBrightness": 0.64,
        "AutoExposureMinBrightness": 0.03,
        "MotionBlurAmount": 0,
        "TargetGamma": 1.0
      }
    ],
    "X": 0.0,
    "Y": 0.0,
    "Z": 0.0,
    "Pitch": 0.0,
    "Roll": 0.0,
    "Yaw": 0.0
  },
  "Cameras": {
    "0": {
      "CaptureSettings": [
        {
          "ImageType": 0,
          "Width": 512,
          "Height": 288,
          "FOV_Degrees": 90
        }
      ],
      "X": 0.0,
      "Y": 0.0,
      "Z": 0.0,
      "Pitch": 0.0,
      "Roll": 0.0,
      "Yaw": 0.0
    }
  }
}
EOF

# Setup python environment (manual fix needed for airsim)
conda create -n airsimtest python=3.7 -y
conda activate airsimtest
pip install numpy msgpack-rpc-python opencv-contrib-python==4.5.1.48

# If using airsim=1.8.1, need manual fix:
git clone https://github.com/microsoft/AirSim.git /home/$USER/airsim_python
cd /home/$USER/airsim_python/PythonClient
# MANUAL FIX HERE: Update setup.py to match https://github.com/microsoft/AirSim/issues/4920#issuecomment-1989402420
pip install . --no-deps
# Otherwise, use airsim=1.5.0 from pip
pip install airsim==1.7.0


# Start AirSim on UE Blocks in container
cd /home/$USER/airsim_test
singularity exec --nv \
  --bind /home/$USER/airsim_test/Blocks:/app \
  --bind /home/$USER/airsim_test/Settings:/home/airsim_user/Documents/AirSim \
  /home/$USER/airsim_container/airsim.sif \
  /app/LinuxNoEditor/Blocks.sh -opengl4 -windowed -ResX=640 -ResY=480 -noaudio -novideo -nohmd -noxr

  # Once AirSim is running, test using airsim_test.py