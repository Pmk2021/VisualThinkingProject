# Script adapted from :
# - https://github.com/IzzeddinTeeti/ASTRA/blob/main/scripts/down_pretrained_astra_models.bash
# - https://github.com/IzzeddinTeeti/ASTRA/blob/main/scripts/down_pretrained_unet_models.bash

# Download Pretrained ASTRA Models
mkdir -p astra/pretrained_weights/astra
cd astra/pretrained_weights/astra
gdown 1k5XclP7XRwiJOXkB7QJUn9OSDuRWEd8c -O pretrained_astra_weights.zip
unzip pretrained_astra_weights.zip
rm pretrained_astra_weights.zip

# Download U-Net Pretrained Embeddings for ASTRA model
mkdir -p ../unet
cd ../unet
gdown 1ygi7-XtVn_24MfUxZ1-OrswSsm3z1eQ1 -O pretrained_unet_weights.zip
unzip pretrained_unet_weights.zip
rm pretrained_unet_weights.zip