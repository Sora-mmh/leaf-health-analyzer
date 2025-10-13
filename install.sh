#!/bin/bash

echo "🚀 Starting installation with Conda..."

# Check for requirements.txt
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found!"
    exit 1
fi

# Check for conda
if ! command -v conda &> /dev/null; then
    echo "❌ Error: Conda not found. Install Miniconda/Anaconda first."
    exit 1
fi

# Create and activate environment
conda create -n dinov3 python=3.10 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dinov3

# Install requirements
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "✅ Installation complete! Use: conda activate dinov3"