#!/bin/bash

# Create a new conda environment named 'sam2' with Python 3.10
conda create -n sam2 python=3.10 -y

# Activate the new environment
conda activate sam2

# Navigate to the SAM-2 directory
cd ./sam2

# Install the SAM-2 package in editable mode
pip install -e .

# Install any additional dependencies (if needed)
# pip install -r requirements.txt

echo "SAM-2 environment setup complete. Activate it with 'conda activate sam2'"
cd ../