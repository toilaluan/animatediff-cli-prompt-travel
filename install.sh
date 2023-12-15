python -m pip install --upgrade pip
# Torch installation must be modified to suit the environment. (https://pytorch.org/get-started/locally/)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e .

# If you want to use the 'stylize' command, you will also need
python -m pip install -e .[stylize]

# If you want to use use dwpose as a preprocessor for controlnet_openpose, you will also need
python -m pip install -e .[dwpose]
# (DWPose is a more powerful version of Openpose)

# If you want to use the 'stylize create-mask' and 'stylize composite' command, you will also need
python -m pip install -e .[stylize_mask]