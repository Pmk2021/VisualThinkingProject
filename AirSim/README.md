# AirSim setup attempts

This directory contains parts of the scripts that were used in the process of trying to install AirSim and make it work on Izar.

Critical steps included :
1. Create an AirSim image based on our image of choice, on our local machines
2. Publish it to the Docker Hub to easily pull it from Izar
3. On Izar, create a set-up script that pulls the image, install the Unreal Engine environment, and create the conda env used to interact with the simulation

Two versions of the set-up script are present : 
- `setup.sh` is the main, generic script that has been updated often to try different versions and setups
- `setup_1_7_0.sh` is an example of the script where all dependencies are aligned on the AirSim version 1.7.0

A python script `test.py` is included to test the connection between the simulation and the python code, to run after starting the image.

The set-up failed due to errors in the compiled Unreal Engine environment that we did not manage to resolve : everything started smoothly, but as soon as the python code ran, the AirSim simulation crashed. This happened across several versions and UE environments and seems to be a frequent issue.